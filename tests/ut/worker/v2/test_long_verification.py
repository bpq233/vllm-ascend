# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU regression checks for MRV2's cached-prefill routing boundary."""

import ast
import importlib.util
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4] / "vllm_ascend"
spec = importlib.util.spec_from_file_location("_long_query", ROOT / "attention/spec_decode.py")
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


def config(width=32, enabled=True):
    return NS(
        use_v2_model_runner=True,
        speculative_config=NS(num_speculative_tokens=width, method="dflash"),
        additional_config={"multi_stage_speculative": {"intermediate": {"num_rounds": 4}}} if enabled else {},
        model_config=NS(runner_type="generate", max_model_len=256, enforce_eager=False),
        compilation_config=NS(cudagraph_mode=GraphMode.FULL, splitting_ops=[], cudagraph_capture_sizes=[16, 64]),
        scheduler_config=NS(enable_chunked_prefill=False, max_num_batched_tokens=256, max_num_seqs=4),
    )


class GraphMode(Enum):
    NONE = 0
    FULL = 1
    FULL_AND_PIECEWISE = 2
    PIECEWISE = 3

    def requires_piecewise_compilation(self):
        return self in (self.FULL_AND_PIECEWISE, self.PIECEWISE)

    def has_full_cudagraphs(self):
        return self in (self.FULL, self.FULL_AND_PIECEWISE)


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("mode", ["plain", "dflash", "multi_stage"])
@pytest.mark.parametrize("empty", [False, True])
def test_progress_boundary_preserves_reordered_new_and_inactive_rows(mode, empty):
    tree = ast.parse((ROOT / "worker/v2/model_runner.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("postprocess_sampled", "_update_seq_lens_cpu")
    ]
    base = type("Base", (), {"postprocess_sampled": Mock()})
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    ns = dict(Base=base, np=np)
    exec(compile(ast.fix_missing_locations(module), "progress", "exec"), ns)
    runner = ns["NPUModelRunner"]()
    runner.speculator = None if mode == "plain" else NS(updates_computed_tokens_cpu=mode == "multi_stage")
    runner._copy_num_computed_tokens_to_cpu = Mock()
    runner.num_computed_tokens_event = NS(synchronize=Mock())
    # Multi-stage's combined D2H already published the committed progress.
    runner.req_states = NS(
        req_id_to_index={"a": 0, "new": 1, "b": 2, "idle": 3},
        num_computed_tokens_cpu=torch.tensor(
            [7, 3, 9, 99] if mode == "multi_stage" else [-7, 3, -9, 99], dtype=torch.int32
        ),
        num_computed_tokens_np=np.array([7, 300, 9, 999], dtype=np.int32),
    )
    runner.num_computed_tokens_cpu = torch.tensor([7, 300, 9, 999], dtype=torch.int32)
    runner.input_buffers = NS(seq_lens_cpu=torch.full((5,), -1, dtype=torch.int32))
    runner.postprocess_sampled(None, None, None, None)
    base.postprocess_sampled.assert_called_once()
    assert runner._copy_num_computed_tokens_to_cpu.call_count == int(mode == "dflash")
    scheduler = NS(
        scheduled_cached_reqs=NS(req_ids=[] if empty else ["a", "b"]),
        num_scheduled_tokens={"b": 4, "new": 2, "a": 1},
    )
    runner._update_seq_lens_cpu(scheduler, [] if empty else ["b", "new", "a"])
    assert runner.num_computed_tokens_event.synchronize.call_count == int(mode == "dflash")
    expected_progress = [-7, 3, -9, 99] if empty and mode != "multi_stage" else [7, 3, 9, 99]
    assert runner.req_states.num_computed_tokens_cpu.tolist() == expected_progress
    assert runner.input_buffers.seq_lens_cpu.tolist() == ([-1] * 5 if empty else [13, 5, 8, -1, -1])


STATES = NS(
    PrefillNoCache="new", PrefillCacheHit="cached", DecodeOnly="decode", SpecDecoding="spec", ChunkedPrefill="extend"
)


@pytest.mark.parametrize("width,expected", [(4, 5), (15, 16), (16, 16), (32, 16)])
def test_decode_boundary_does_not_limit_candidate_storage(width, expected):
    cfg = config(width)
    assert routing.speculative_decode_threshold(cfg) == expected
    assert cfg.speculative_config.num_speculative_tokens == width
    assert routing.speculative_decode_threshold(config(width, enabled=False)) == width + 1


@pytest.mark.parametrize(
    "lengths,expected", [([1], "decode"), ([16], "extend"), ([17], "extend"), ([1, 5, 17, 33], "extend")]
)
def test_actual_query_lengths_route_cached_speculation(lengths, expected):
    build = function(
        "worker/v2/attn_utils.py",
        "build_attn_state",
        dict(VllmConfig=object, np=np, AscendAttentionState=STATES, **vars(routing)),
    )
    scheduled = np.array(lengths)
    assert build(config(), scheduled + 40, len(lengths), scheduled, np.ones(len(lengths))) == expected
    # Pure initial prefill must not read uninitialized cache pages.
    assert build(config(), scheduled, len(lengths), scheduled, scheduled) == "new"


def test_metadata_decode_assertion_remains_for_ordinary_speculation():
    tree = ast.parse((ROOT / "attention/attention_v1.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendAttentionMetadataBuilder")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )

    class Base:
        def __init__(self, *args):
            pass

    ns = dict(
        Base=Base,
        cdiv=lambda a, b: (a + b - 1) // b,
        AscendAttentionBackend=NS(get_supported_kernel_block_sizes=lambda: [128]),
        AttentionMaskBuilder=lambda device: None,
        **vars(routing),
    )
    exec(compile(ast.fix_missing_locations(module), "builder", "exec"), ns)
    builder = ns["AscendAttentionMetadataBuilder"]
    assert builder(None, [], config(), "cpu").decode_threshold == 16
    with pytest.raises(AssertionError, match="limit of 16"):
        builder(None, [], config(enabled=False), "cpu")


def test_cached_prefill_reads_existing_blocks_and_only_visible_lengths():
    tree = ast.parse((ROOT / "attention/attention_v1.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendAttentionBackendImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_fia_params")
    ns = dict(torch=torch, AscendAttentionState=STATES, AscendMetadata=object)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "fia_params", "exec"), ns)
    cache = torch.arange(4 * 128 * 2 * 8).reshape(4, 128, 2, 8)
    impl = NS(key_cache=cache, value_cache=cache.clone())
    blocks = torch.tensor([[2, 1], [3, 0]])
    metadata = NS(attn_state=STATES.ChunkedPrefill, block_tables=blocks, seq_lens_list=[73, 105])
    key, value, block_size, actual_blocks, lengths = ns["_get_fia_params"](impl, None, None, metadata)
    assert key.data_ptr() == cache.data_ptr()
    assert value.data_ptr() == impl.value_cache.data_ptr()
    assert actual_blocks is blocks and block_size == 128 and lengths == [73, 105]
    # Rejected speculative tail remains physically allocated but is hidden.
    metadata.seq_lens_list = [57, 80]
    assert ns["_get_fia_params"](impl, None, None, metadata)[-1] == [57, 80]


def test_both_graph_manager_versions_capture_long_target_piecewise():
    tree = ast.parse((ROOT / "worker/v2/aclgraph_utils.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelAclGraphManager")
    constructors = [n for n in ast.walk(cls) if isinstance(n, ast.FunctionDef) and n.name == "__init__"]

    class Base:
        def __init__(self, config, device, mode, width, **kwargs):
            self.mode = mode
            self._capture_descs = {}

        def needs_capture(self):
            return False

    for constructor in constructors:
        wrapper = ast.ClassDef(
            name="Manager",
            bases=[ast.Name(id="Base", ctx=ast.Load())],
            keywords=[],
            body=[constructor],
            decorator_list=[],
        )
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), wrapper],
            type_ignores=[],
        )
        ns = dict(
            Base=Base,
            CUDAGraphMode=GraphMode,
            uses_long_speculative_queries=routing.uses_long_speculative_queries,
            collect_sorted_captured_token_sizes=lambda desc: [],
        )
        ns["target_graph_mode"] = function("worker/v2/aclgraph_utils.py", "target_graph_mode", ns)
        exec(compile(ast.fix_missing_locations(module), "graph_manager", "exec"), ns)
        assert ns["Manager"](config(), "cpu", GraphMode.FULL, 33, NS(update_stream=None)).mode == GraphMode.FULL
        assert ns["Manager"](config(4), "cpu", GraphMode.FULL, 5, NS(update_stream=None)).mode == GraphMode.FULL
        assert ns["Manager"](config(), "cpu", GraphMode.NONE, 33, NS(update_stream=None)).mode == GraphMode.NONE
        cfg = config()
        cfg.compilation_config.cudagraph_mode = GraphMode.PIECEWISE
        for mode in GraphMode:
            assert ns["Manager"](cfg, "cpu", mode, 33, NS(update_stream=None)).mode == mode


def test_both_runner_versions_sort_short_queries_before_long_candidates():
    tree = ast.parse((ROOT / "worker/v2/model_runner.py").read_text(encoding="utf-8"))
    methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "prepare_inputs"]
    for method in methods:
        end = next(
            i
            for i, n in enumerate(method.body)
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "_update_seq_lens_cpu"
        )
        method.body = method.body[:end] + [ast.Return(value=ast.Name(id="req_ids", ctx=ast.Load()))]
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
            type_ignores=[],
        )
        ns = dict(sort_batch_req_ids=lambda counts, *args: list(counts), **vars(routing))
        exec(compile(ast.fix_missing_locations(module), "prepare_inputs", "exec"), ns)
        runner = NS(vllm_config=config(), decode_query_len=33)
        scheduler = NS(
            total_num_scheduled_tokens=57,
            num_scheduled_tokens={"long": 33, "decode": 1, "short": 5, "extend": 18},
            scheduled_spec_decode_tokens={"long": [1] * 32, "short": [1] * 4},
        )
        desc = NS(num_tokens=57)
        args = [runner, scheduler]
        if len(method.args.args) == 4:
            args.append(NS())
        assert ns["prepare_inputs"](*args, desc) == ["decode", "short", "long", "extend"]
