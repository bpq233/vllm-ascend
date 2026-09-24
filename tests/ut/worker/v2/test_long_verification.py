# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU regression checks for MRV2's cached-prefill routing boundary."""

import ast
import importlib.util
from contextlib import nullcontext
from dataclasses import dataclass
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

        def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras):
            return NS(cg_mode=GraphMode.NONE)

        def _add_long_verification_graphs(self, config):
            return ()

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
            signature=__import__("inspect").signature,
            uses_long_speculative_queries=routing.uses_long_speculative_queries,
            collect_sorted_captured_token_sizes=lambda desc: [],
            _normalize_capture_config=lambda config: None,
            _scalar_int=lambda value, name: value[0] if isinstance(value, list) else value,
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


def test_full_decode_only_has_sparse_target_gears_for_long_verification():
    fn = function(
        "worker/v2/aclgraph_utils.py",
        "long_verification_capture_shapes",
        dict(
        CUDAGraphMode=NS(FULL_DECODE_ONLY="full_decode_only"),
        MAX_DECODE_QUERY_LEN=16,
        MAX_LONG_VERIFICATION_CAPTURE_TOKENS=256,
        uses_long_speculative_queries=lambda cfg: True,
        _normalize_capture_config=lambda config: None,
        _scalar_int=lambda value, name: value[0] if isinstance(value, list) else value,
        ),
    )
    cfg = NS(
        model_config=NS(enforce_eager=False),
        compilation_config=NS(
            cudagraph_mode="full_decode_only",
            max_cudagraph_capture_size=512,
            cudagraph_capture_sizes=[16, 32, 64, 128],
        ),
        scheduler_config=NS(max_num_batched_tokens=512, max_num_seqs=4),
        speculative_config=NS(num_speculative_tokens=53),
        parallel_config=NS(tensor_parallel_size=1),
    )
    assert fn(cfg) == [(1, 54), (2, 108), (3, 162), (4, 216)]

    cfg.parallel_config.tensor_parallel_size = 8
    assert fn(cfg) == [(1, 56), (2, 112), (3, 168), (4, 216)]

    cfg.scheduler_config.max_num_batched_tokens = 200
    assert fn(cfg) == [(1, 56), (2, 112), (3, 168)]

    cfg.scheduler_config.max_num_batched_tokens = 512
    cfg.compilation_config.max_cudagraph_capture_size = 128
    # The ordinary 128-token gear cannot cover four 54-token queries; the
    # unsupported long batch must use eager execution instead of adding 216.
    assert fn(cfg) == [(1, 56), (2, 112)]

    cfg.compilation_config.max_cudagraph_capture_size = 0
    cfg.compilation_config.cudagraph_capture_sizes = []
    assert fn(cfg) == []

    cfg.compilation_config.max_cudagraph_capture_size = 512
    cfg.compilation_config.cudagraph_capture_sizes = [16, 32, 64, 128]
    cfg.model_config.enforce_eager = True
    assert fn(cfg) == []


def test_long_verification_shape_inputs_must_be_scalar():
    fn = function(
        "worker/v2/aclgraph_utils.py",
        "long_verification_capture_shapes",
        dict(
            CUDAGraphMode=NS(FULL_DECODE_ONLY="full_decode_only"),
            MAX_DECODE_QUERY_LEN=16,
            MAX_LONG_VERIFICATION_CAPTURE_TOKENS=256,
            uses_long_speculative_queries=lambda cfg: True,
            _normalize_capture_config=lambda config: setattr(
                config.compilation_config, "max_cudagraph_capture_size", 256
            ),
            _scalar_int=lambda value, name: value[0] if isinstance(value, list) else value,
        ),
    )
    cfg = NS(
        model_config=NS(enforce_eager=False),
        compilation_config=NS(
            cudagraph_mode="full_decode_only",
            max_cudagraph_capture_size=[256],
            cudagraph_capture_sizes=[16, 32, 64, 128],
        ),
        scheduler_config=NS(max_num_batched_tokens=512, max_num_seqs=4),
        speculative_config=NS(num_speculative_tokens=53),
        parallel_config=NS(tensor_parallel_size=1),
    )
    assert fn(cfg) == [(1, 54), (2, 108), (3, 162), (4, 216)]


def test_long_target_graph_dispatch_is_opt_in_and_uses_compatible_bucket():
    tree = ast.parse((ROOT / "worker/v2/aclgraph_utils.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelAclGraphManager")
    dispatch = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "dispatch")

    class Base:
        def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras):
            return NS(cg_mode="full", num_tokens=64)

        def _resolve_effective_loras(self, count):
            return count

    namespace = dict(
        Base=Base,
        CUDAGraphMode=NS(NONE="none", FULL="full"),
        MAX_DECODE_QUERY_LEN=16,
        logger=NS(warning_once=lambda *args: None),
        select_long_verification_graph=lambda *args: next(
            (desc for desc in args[0] if desc.num_tokens >= args[2]), None
        ),
    )

    class Descriptor:
        def __init__(
            self,
            cg_mode,
            num_tokens,
            num_reqs=None,
            uniform_token_count=None,
            max_query_len=None,
        ):
            self.cg_mode = cg_mode
            self.num_tokens = num_tokens
            self.num_reqs = num_reqs
            self.uniform_token_count = uniform_token_count
            self.max_query_len = max_query_len

    namespace["BatchExecutionDescriptor"] = Descriptor
    namespace["signature"] = __import__("inspect").signature
    helper_namespace = {
        "BatchExecutionDescriptor": Descriptor,
        "CUDAGraphMode": namespace["CUDAGraphMode"],
        "signature": namespace["signature"],
        "_normalize_descriptor": lambda desc: desc,
    }
    namespace["eager_execution_descriptor"] = function(
        "worker/v2/aclgraph_utils.py",
        "eager_execution_descriptor",
        helper_namespace,
    )
    cls_copy = ast.ClassDef(
        name="Manager",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[dispatch],
        decorator_list=[],
    )
    module = ast.Module(body=[cls_copy], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "dispatch", "exec"), namespace)
    manager = namespace["Manager"]()
    manager.long_verification_graphs = [NS(num_reqs=1, num_tokens=32), NS(num_reqs=1, num_tokens=64)]
    manager._dispatch_parameters = {"num_reqs": None, "num_tokens": None}
    manager.long_verification_active = False
    eager = manager.dispatch(1, 20, None, 0, max_query_len=20)
    assert eager.num_tokens == 64
    manager.long_verification_active = True
    graph = manager.dispatch(1, 20, None, 0, max_query_len=20)
    assert graph.num_tokens == 32
    assert not hasattr(graph, "num_ubatches")

    # Eager mode has no long-verification graph descriptors and must not enter
    # the graph-only protection path.
    manager.long_verification_graphs = []
    eager = manager.dispatch(1, 20, None, 0, max_query_len=20)
    assert eager.num_tokens == 64

    # A long batch that exceeds the captured buckets must also use the normal
    # dispatch result instead of stalling or raising from the graph manager.
    manager.long_verification_graphs = [NS(num_reqs=1, num_tokens=32), NS(num_reqs=1, num_tokens=64)]
    eager = manager.dispatch(2, 80, None, 0, max_query_len=36)
    assert eager.cg_mode == "none"
    assert eager.num_tokens == 80


def test_graph_dispatch_normalizes_list_values_before_upstream_hash_lookup():
    tree = ast.parse((ROOT / "worker/v2/aclgraph_utils.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelAclGraphManager")
    dispatch = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "dispatch")

    @dataclass(frozen=True)
    class Descriptor:
        cg_mode: str
        num_tokens: int
        num_reqs: int
        uniform_token_count: int | None = None
        max_query_len: int | None = None
        num_active_loras: int = 0

    class Base:
        def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None):
            # Mirror upstream's tuple/dict cache lookup to fail if normalization
            # happens after the base dispatch call.
            self.lookup[(num_tokens, num_active_loras)] = True
            return Descriptor("none", num_tokens, num_reqs, uniform_token_count, max_query_len, num_active_loras)

        def _resolve_effective_loras(self, count):
            return count

    namespace = dict(
        Base=Base,
        BatchExecutionDescriptor=Descriptor,
        CUDAGraphMode=NS(NONE="none", FULL="full"),
        MAX_DECODE_QUERY_LEN=16,
        logger=NS(warning_once=lambda *args: None),
        select_long_verification_graph=lambda *_args: None,
        _scalar_int=lambda value, name: int(value[0]) if isinstance(value, list) and len(value) == 1 else int(value),
        _optional_scalar_int=lambda value, name: None
        if value is None
        else int(value[0] if isinstance(value, list) else value),
        _normalize_descriptor=lambda desc: desc,
        signature=__import__("inspect").signature,
        eager_execution_descriptor=lambda desc, *args: desc,
    )
    manager_copy = ast.ClassDef(
        name="Manager",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[dispatch],
        decorator_list=[],
    )
    module = ast.Module(body=[manager_copy], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "dispatch_scalar_keys", "exec"), namespace)

    manager = namespace["Manager"]()
    manager.lookup = {}
    manager._dispatch_parameters = {"max_query_len": None}
    manager.long_verification_active = False
    manager.long_verification_graphs = []
    desc = manager.dispatch([2], [8], [4], [0], max_query_len=[4], num_ubatches=[1])
    assert manager.lookup == {(8, 0): True}
    assert (desc.num_reqs, desc.num_tokens, desc.uniform_token_count, desc.max_query_len) == (2, 8, 4, 4)


def test_acl_graph_batch_descriptor_list_fields_are_normalized_before_hashing():
    helper = function(
        "compilation/acl_graph.py",
        "_normalize_batch_descriptor",
        {"dataclasses": __import__("dataclasses")},
    )

    @dataclass(frozen=True)
    class Descriptor:
        num_tokens: int
        num_reqs: int | None = None
        uniform: bool = False
        has_lora: bool = False
        num_active_loras: int = 0

    desc = helper(Descriptor([8], [2], num_active_loras=[0]))
    assert desc == Descriptor(8, 2, num_active_loras=0)
    assert {desc: "cached"}[desc] == "cached"


def test_long_target_graph_reuses_capped_bucket_for_dynamic_query_width():
    select = function(
        "worker/v2/aclgraph_utils.py",
        "select_long_verification_graph",
        {},
    )
    graph = NS(num_reqs=4, num_tokens=128, num_active_loras=0)
    assert select([graph], 4, 96, 0) is graph


def test_long_target_graph_covers_actual_ragged_token_count():
    select = function(
        "worker/v2/aclgraph_utils.py",
        "select_long_verification_graph",
        {},
    )
    graph = NS(num_reqs=4, num_tokens=144, num_active_loras=0)
    assert select([graph], 4, 131, 0) is graph


def test_long_graph_descriptors_pin_query_shape_when_supported():
    tree = ast.parse((ROOT / "worker/v2/aclgraph_utils.py").read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelAclGraphManager"
    )
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_add_long_verification_graphs"
    )

    class Descriptor:
        def __init__(self, cg_mode, num_tokens, num_reqs, uniform_token_count=None, max_query_len=None):
            self.cg_mode = cg_mode
            self.num_tokens = num_tokens
            self.num_reqs = num_reqs
            self.uniform_token_count = uniform_token_count
            self.max_query_len = max_query_len

        def __hash__(self):
            return id(self)

    class Base:
        def __init__(self):
            self.max_num_reqs = 4
            self._capture_descs = {}

    manager_cls = ast.ClassDef(
        name="Manager",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), manager_cls],
        type_ignores=[],
    )
    namespace = dict(
        Base=Base,
        BatchExecutionDescriptor=Descriptor,
        CUDAGraphMode=NS(FULL="full"),
        signature=__import__("inspect").signature,
        long_verification_capture_shapes=lambda _: [(2, 112)],
    )
    exec(compile(ast.fix_missing_locations(module), "long_descriptor", "exec"), namespace)
    manager = namespace["Manager"]()
    manager._add_long_verification_graphs(NS(speculative_config=NS(num_speculative_tokens=55)))
    desc = manager._capture_descs["full"][0]
    assert desc.uniform_token_count is None
    assert desc.max_query_len == 56


def test_long_full_replay_uses_full_mode_fia_query_boundaries():
    method = next(
        node
        for node in ast.walk(ast.parse((ROOT / "worker/v2/model_runner.py").read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "_pad_query_start_loc_for_fia"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = dict(np=np, CUDAGraphMode=NS(FULL="full"))
    exec(compile(ast.fix_missing_locations(module), "fia_graph_boundaries", "exec"), namespace)
    runner = NS(
        compilation_config=NS(cudagraph_mode="full_decode_only"),
        cudagraph_manager=NS(long_verification_active=True),
        decode_query_len=5,
    )
    starts, padded_reqs = namespace["_pad_query_start_loc_for_fia"](
        runner,
        num_tokens_padded=32,
        num_reqs_padded=4,
        num_reqs=1,
        query_start_loc_np=np.array([0, 20, 32, 32, 32, 32], dtype=np.int32),
        cudagraph_runtime_mode="full",
        batch_desc_num_reqs=4,
    )
    assert padded_reqs == 2
    assert starts[2] == 32


@pytest.mark.parametrize(
    "query_starts,num_reqs,padded_tokens,padded_reqs,expected_reqs,expected_boundary",
    [
        ([0, 40, 80], 2, 108, 2, 3, 108),
        ([0, 54, 108, 162, 216], 4, 216, 4, 4, 216),
        ([0, 40, 80], 2, 80, 2, 2, 80),
    ],
)
def test_long_full_replay_preserves_ragged_fia_boundaries(
    query_starts, num_reqs, padded_tokens, padded_reqs, expected_reqs, expected_boundary
):
    method = next(
        node
        for node in ast.walk(ast.parse((ROOT / "worker/v2/model_runner.py").read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "_pad_query_start_loc_for_fia"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = dict(np=np, CUDAGraphMode=NS(FULL="full"))
    exec(compile(ast.fix_missing_locations(module), "fia_ragged_boundaries", "exec"), namespace)
    runner = NS(
        compilation_config=NS(cudagraph_mode="full_decode_only"),
        cudagraph_manager=NS(long_verification_active=True),
        decode_query_len=54,
    )
    starts = np.array(query_starts + [padded_tokens], dtype=np.int32)
    starts, result_reqs = namespace["_pad_query_start_loc_for_fia"](
        runner,
        num_tokens_padded=padded_tokens,
        num_reqs_padded=padded_reqs,
        num_reqs=num_reqs,
        query_start_loc_np=starts,
        cudagraph_runtime_mode="full",
        batch_desc_num_reqs=padded_reqs,
    )
    assert result_reqs == expected_reqs
    assert starts[expected_reqs] == expected_boundary


def test_capture_model_preserves_long_verification_capture_state():
    tree = ast.parse((ROOT / "worker/v2/model_runner.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "capture_model")
    cls_copy = ast.ClassDef(
        name="Runner",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )

    class Base:
        def capture_model(self):
            assert self.cudagraph_manager.long_verification_active
            return [17, 32]

    namespace = dict(Base=Base, torch=NS(inference_mode=lambda: lambda function: function))
    module = ast.Module(body=[cls_copy], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "capture_state", "exec"), namespace)
    manager = NS(long_verification_graphs=[object()], long_verification_active=False)
    runner = namespace["Runner"]()
    runner.cudagraph_manager = manager
    assert runner.capture_model() == [17, 32]
    assert manager.long_verification_active is False


@pytest.mark.parametrize(
    "scheduled,drafts,expected",
    [
        ({"a": 131, "b": 102}, {"a": [0] * 130, "b": [0] * 101}, True),
        ({"a": 131, "b": 102}, {"a": [0] * 130}, False),
        ({"a": 131, "b": 1}, {"a": [0] * 130, "b": []}, False),
        ({"a": 16}, {"a": [0] * 15}, False),
    ],
)
def test_only_pure_long_verification_batches_enable_target_long_graphs(scheduled, drafts, expected):
    tree = ast.parse((ROOT / "worker/v2/model_runner.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_is_long_verification_batch"
    )
    cls_copy = ast.ClassDef(
        name="Runner",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls_copy],
        type_ignores=[],
    )
    namespace = dict(MAX_DECODE_QUERY_LEN=16, SchedulerOutput=object)
    exec(compile(ast.fix_missing_locations(module), "long_verify_gate", "exec"), namespace)
    output = NS(num_scheduled_tokens=scheduled, scheduled_spec_decode_tokens=drafts)
    assert namespace["Runner"]._is_long_verification_batch(output) is expected


def test_target_graph_parameter_update_precedes_replay():
    tree = ast.parse((ROOT / "worker/v2/aclgraph_utils.py").read_text(encoding="utf-8"))
    manager_cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelAclGraphManager"
    )
    method = next(
        node for node in manager_cls.body if isinstance(node, ast.FunctionDef) and node.name == "run_fullgraph"
    )
    calls = []

    class Base:
        def run_fullgraph(self, desc):
            calls.append("replay")
            return "output"

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, stream):
            calls.append(f"{self.name}.wait({stream.name})")

    class Log:
        def info_once(self, *args):
            pass

        def debug(self, *args):
            pass

    torch_stub = NS(
        npu=NS(current_stream=lambda: current_stream),
        full=lambda *args, **kwargs: object(),
    )
    namespace = dict(
        Base=Base,
        torch=torch_stub,
        logger=Log(),
        set_current_vllm_config=lambda *_: nullcontext(),
        set_forward_context=lambda *_args, **_kwargs: nullcontext(),
        get_forward_context=lambda: object(),
        _get_graph_update_backend=lambda _: object(),
        update_full_graph_params=lambda *_args, **_kwargs: calls.append("update"),
    )
    manager_copy = ast.ClassDef(
        name="Manager",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), manager_copy],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "graph_update_order", "exec"), namespace)
    current_stream = Stream("current")
    manager = namespace["Manager"]()
    manager.update_stream = Stream("update")
    manager.device = "npu:0"
    manager.model_runner = NS(
        dp_size=1,
        model_state=NS(attn_metadata={}),
        attn_groups=[],
        speculative_config=None,
    )
    manager.vllm_config = object()
    assert manager.run_fullgraph(NS(num_tokens=128, cg_mode="full")) == "output"
    assert calls == ["update.wait(current)", "replay", "update"]


@pytest.mark.parametrize("capture_succeeds", [True, False])
def test_long_graph_capture_validation_supports_old_manager_without_profile_field(capture_succeeds):
    tree = ast.parse((ROOT / "worker/v2/aclgraph_utils.py").read_text(encoding="utf-8"))
    manager_cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelAclGraphManager"
    )
    method = next(node for node in manager_cls.body if isinstance(node, ast.FunctionDef) and node.name == "capture")
    expected = type("Descriptor", (), {"num_tokens": 216})()

    class Base:
        def capture(self, *args, **kwargs):
            if capture_succeeds:
                self.graphs[expected] = object()

    class CaptureContext:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    manager_cls_copy = ast.ClassDef(
        name="Manager",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), manager_cls_copy],
        type_ignores=[],
    )
    namespace = dict(
        Base=Base,
        nn=NS(Module=object),
        ModelState=object,
        InputBuffers=object,
        IntermediateTensors=object,
        BlockTables=object,
        AttentionGroup=object,
        KVCacheConfig=object,
        Callable=object,
        ModelWithContext=lambda model: model,
        communicator_switch=CaptureContext,
        CUDAGraphMode=NS(FULL="full"),
    )
    exec(compile(ast.fix_missing_locations(module), "old_manager_capture", "exec"), namespace)
    manager = namespace["Manager"]()
    manager.long_verification_graphs = [expected]
    manager.graphs = {}
    args = (object(), object(), object(), None, object(), [], object())
    if capture_succeeds:
        manager.capture(*args)
    else:
        with pytest.raises(RuntimeError, match="Long Target verification graph capture is incomplete"):
            manager.capture(*args)


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
