# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run real target metadata helpers and both input-preparation versions on CPU."""

import ast
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/model_runner.py"
MODES = NS(FULL="full", NONE="none")


@pytest.fixture(params=[0, 1], ids=["0.27.1", "main"])
def runner(request):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    prepare = [node for node in ast.walk(cls) if isinstance(node, ast.FunctionDef) and node.name == "prepare_inputs"]
    keep = {
        "_copy_spec_metadata",
        "_invalidate_spec_metadata",
        "_pad_query_start_loc_for_fia",
        "execute_model",
        "capture_model",
        "initialize_kv_cache",
    }
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.Assign) or (isinstance(node, ast.FunctionDef) and node.name in keep)
    ]
    cls.body.append(prepare[request.param])
    # Run only execute_model's real cache-lifetime preamble, before model work.
    execute = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "execute_model")
    stop = next(
        i
        for i, node in enumerate(execute.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "_cpp_execution_time_ms" for t in node.targets)
    )
    execute.body = execute.body[:stop] + [ast.Return(value=ast.Constant(None))]
    capture = Mock(return_value=object())
    initialize = Mock()

    class Base:
        def capture_model(self, *args, **kwargs):
            assert not self._spec_metadata and self._query_metadata is None
            self.input_buffers.query_start_loc.fill_(-99)
            return capture(*args, **kwargs)

        def initialize_kv_cache(self, config):
            assert not self._spec_metadata and self._query_metadata is None
            self.input_buffers.query_start_loc.fill_(-99)
            initialize(config)

    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )

    def copy(values, out=None, device=None):
        # A real H2D owns storage; avoid numpy aliases concealing stale copies.
        value = torch.from_numpy(values.copy())
        if out is None:
            return value
        out.copy_(value)
        return out

    copies = Mock(side_effect=copy)

    def expand(indices, total, cumulative, maximum):
        widths = cumulative[1:] - cumulative[:-1]
        return indices.repeat_interleave(widths.long()), torch.cat([torch.arange(int(w)) for w in widths])

    def positions(indices, boundaries, computed, out_positions, out_lengths):
        for row, index in enumerate(indices.tolist()):
            begin, end = boundaries[row : row + 2].tolist()
            out_positions[begin:end] = torch.arange(end - begin) + computed[index]
            out_lengths[row] = computed[index] + end - begin

    def combine(out, indices, sampled, boundaries, lengths, prefill, drafts, cumulative, total, bonus):
        for row, index in enumerate(indices.tolist()):
            begin, end = boundaries[row : row + 2].tolist()
            out[begin] = sampled[index]
            out[begin + 1 : end] = drafts[index, : end - begin - 1]
        return torch.arange(total, dtype=torch.int64)

    stream = NS(current="compute")
    torch_api = NS(inference_mode=torch.inference_mode, npu=NS(current_stream=lambda: stream.current))
    for name in ("arange", "zeros", "from_numpy", "int32"):
        setattr(torch_api, name, getattr(torch, name))
    namespace = dict(
        Base=Base,
        graph_manager_wrapper=lambda obj: nullcontext(),
        OrderedDict=OrderedDict,
        np=np,
        torch=torch_api,
        async_copy_to_gpu=copies,
        sort_batch_req_ids=lambda scheduled, *args: list(scheduled),
        uses_long_speculative_queries=lambda cfg: False,
        build_attn_state=lambda *args: "spec",
        CUDAGraphMode=MODES,
        expand_idx_mapping=expand,
        prepare_pos_seq_lens=positions,
        combine_sampled_and_draft_tokens=combine,
        prepare_prefill_inputs=Mock(side_effect=AssertionError("unexpected prefill")),
        AscendInputBatch=NS,
        vllm_model_runner=NS(pcp=NS(maybe_partition_pcp_batch=lambda manager, batch: batch)),
        update_cos_sin=Mock(),
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    obj = namespace["NPUModelRunner"]()
    obj.device = "cpu"
    obj.speculator = NS(updates_computed_tokens_cpu=True)
    obj.max_num_reqs, obj.decode_query_len = 4, 4
    obj.vllm_config = NS()
    obj.model_config = NS(rswa_window=None, enable_return_routed_experts=False)
    obj.model_state = NS(num_new_sampled_tokens_per_step=1)
    obj.use_pp, obj.pcp_manager = False, None
    obj.cudagraph_manager = NS(cudagraph_mode=MODES.FULL)
    obj.eplb = NS(set_batch_phase=Mock())
    computed = np.array([10, 20], dtype=np.int32)
    obj.req_states = NS(
        req_id_to_index={"a": 0, "b": 1},
        num_computed_tokens_np=computed,
        num_computed_tokens=NS(gpu=torch.from_numpy(computed)),
        num_computed_prefill_tokens=np.array([2, 2], dtype=np.int32),
        prefill_len=NS(np=np.array([2, 2], dtype=np.int32), gpu=torch.tensor([2, 2])),
        last_sampled_tokens=torch.tensor([100, 200]),
        draft_tokens=torch.tensor([[101, 102, 103], [201, 202, 203]]),
    )
    obj.input_buffers = NS(
        query_start_loc=torch.empty(6, dtype=torch.int32),
        positions=torch.zeros(16, dtype=torch.int64),
        seq_lens=torch.zeros(6, dtype=torch.int32),
        seq_lens_np=np.zeros(6, dtype=np.int32),
        input_ids=torch.zeros(16, dtype=torch.int32),
        is_padding=torch.zeros(16, dtype=torch.bool),
    )

    def update_lengths(scheduler, ids):
        for row, req_id in enumerate(ids):
            obj.input_buffers.seq_lens_np[row] = (
                computed[obj.req_states.req_id_to_index[req_id]] + scheduler.num_scheduled_tokens[req_id]
            )

    obj._update_seq_lens_cpu = update_lengths

    def run(order=("a", "b"), widths=(3, 2), padding=0):
        scheduler = NS(
            total_num_scheduled_tokens=sum(widths),
            num_scheduled_tokens=dict(zip(order, widths)),
            scheduled_spec_decode_tokens={key: list(range(width - 1)) for key, width in zip(order, widths)},
            has_structured_output_requests=False,
        )
        desc = NS(num_tokens=sum(widths) + padding, num_reqs=0, cg_mode=MODES.FULL if padding else MODES.NONE)
        if request.param == 0:
            return obj.prepare_inputs(scheduler, desc)
        return obj.prepare_inputs(scheduler, NS(), desc)

    return NS(obj=obj, copies=copies, run=run, stream=stream, capture=capture, initialize=initialize)


@pytest.mark.parametrize("padding", [0, 3])
def test_stable_metadata_removes_three_copies_but_positions_stay_live(runner, padding):
    obj = runner.obj
    first = runner.run(padding=padding)
    assert runner.copies.call_count == 3
    pointers = (first.idx_mapping.data_ptr(), first.cu_num_logits.data_ptr(), first.query_start_loc.data_ptr())
    assert first.positions[:5].tolist() == [10, 11, 12, 20, 21]
    obj.req_states.num_computed_tokens_np[:] = [13, 22]
    runner.copies.reset_mock()
    second = runner.run(padding=padding)
    runner.copies.assert_not_called()
    assert pointers == (
        second.idx_mapping.data_ptr(),
        second.cu_num_logits.data_ptr(),
        second.query_start_loc.data_ptr(),
    )
    assert second.query_start_loc.data_ptr() == obj.input_buffers.query_start_loc.data_ptr()
    assert second.query_start_loc.tolist() == ([0, 3, 5, 8] if padding else [0, 3, 5])
    assert second.positions[:5].tolist() == [13, 14, 15, 22, 23]
    assert second.seq_lens[:2].tolist() == [16, 24]
    assert second.seq_lens_cpu_upper_bound[:2].tolist() == [16, 24]
    assert second.input_ids[:5].tolist() == [100, 101, 102, 200, 201]


def test_reordering_and_changed_candidate_lengths_refresh_the_right_metadata(runner):
    runner.run()
    runner.copies.reset_mock()
    reordered = runner.run(order=("b", "a"))
    assert runner.copies.call_count == 1  # Only slot order changed.
    assert reordered.idx_mapping.tolist() == [1, 0]
    assert reordered.positions[:5].tolist() == [20, 21, 22, 10, 11]
    assert reordered.input_ids[:5].tolist() == [200, 201, 202, 100, 101]
    runner.copies.reset_mock()
    changed = runner.run(order=("b", "a"), widths=(2, 2))
    assert runner.copies.call_count == 2  # Both cumulative boundaries changed.
    assert changed.cu_num_logits.tolist() == [0, 2, 4]
    assert changed.query_start_loc.tolist() == [0, 2, 4]
    assert changed.positions[:4].tolist() == [20, 21, 10, 11]


def test_ordinary_speculation_preserves_unconditional_transfer(runner):
    runner.obj.speculator = NS()
    runner.run()
    runner.copies.reset_mock()
    runner.run()
    assert runner.copies.call_count == 3


def test_cache_snapshots_mutable_arrays_and_distinguishes_dtype_shape_and_name(runner):
    copy = runner.obj._copy_spec_metadata
    values = np.array([0, 3, 5], dtype=np.int32)
    original = copy("indices", values)
    values[1] = 4
    changed = copy("indices", values)
    assert original.tolist() == [0, 3, 5] and changed.tolist() == [0, 4, 5]
    assert copy("indices", np.array([0, 3, 5], dtype=np.int32)) is original
    copy("other", values)
    assert copy("indices", values.astype(np.int64)).dtype == torch.int64
    assert copy("indices", values.reshape(1, 3)).shape == (1, 3)
    assert runner.copies.call_count == 5


def test_query_scratch_is_refreshed_when_values_or_output_storage_changes(runner):
    copy = runner.obj._copy_spec_metadata
    values = np.array([0, 2, 4], dtype=np.int32)
    out = torch.empty(3, dtype=torch.int32)
    assert copy("query", values, out=out) is out
    assert copy("query", values.copy(), out=out) is out
    assert runner.copies.call_count == 1
    values[-1] = 5
    assert copy("query", values, out=out).tolist() == [0, 2, 5]
    replacement = torch.empty_like(out)
    assert copy("query", values, out=replacement).tolist() == [0, 2, 5]
    assert runner.copies.call_count == 3


def test_metadata_cache_is_bounded_and_retains_recently_used_entries(runner):
    obj = runner.obj
    assert obj.MAX_SPEC_METADATA_SHAPES == 16
    arrays = [np.array([i], dtype=np.int32) for i in range(17)]
    first = obj._copy_spec_metadata("indices", arrays[0])
    for values in arrays[1:16]:
        obj._copy_spec_metadata("indices", values)
    assert obj._copy_spec_metadata("indices", arrays[0]) is first
    obj._copy_spec_metadata("indices", arrays[16])
    assert len(obj._spec_metadata) == 16
    runner.copies.reset_mock()
    assert obj._copy_spec_metadata("indices", arrays[0]) is first
    obj._copy_spec_metadata("indices", arrays[1])
    runner.copies.assert_called_once()  # The least-recent entry was evicted.


def test_execute_boundary_invalidates_after_dummy_and_stream_changes(runner):
    obj = runner.obj
    obj.execute_model(None)
    runner.run()
    runner.copies.reset_mock()
    obj.execute_model(None)
    runner.run()
    runner.copies.assert_not_called()
    obj.execute_model(None, dummy_run=True)
    # Capture/dummy work may overwrite the fixed query scratch.
    obj.input_buffers.query_start_loc.fill_(-1)
    runner.run()
    assert runner.copies.call_count == 3
    runner.copies.reset_mock()
    runner.stream.current = "another-compute-stream"
    obj.execute_model(None)
    runner.run()
    assert runner.copies.call_count == 3
    assert obj.input_buffers.query_start_loc[:3].tolist() == [0, 3, 5]


@pytest.mark.parametrize("action", ["capture", "initialize"])
def test_direct_capture_and_kv_initialization_invalidate_before_parent_work(runner, action):
    runner.run()
    runner.copies.reset_mock()
    if action == "capture":
        assert runner.obj.capture_model("warmup", profile=True) is runner.capture.return_value
        runner.capture.assert_called_once_with("warmup", profile=True)
    else:
        config = object()
        runner.obj.initialize_kv_cache(config)
        runner.initialize.assert_called_once_with(config)
    result = runner.run()
    assert runner.copies.call_count == 3
    assert result.query_start_loc.tolist() == [0, 3, 5]
