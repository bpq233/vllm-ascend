# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU interface tests for sparse runtime; these do not execute NPU graphs.

The graph stub reproduces native persistent input/output buffer ownership and
method signatures. It executes Python on replay solely to verify staging,
metadata and ownership contracts, never as hardware capture/replay evidence.
"""

import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


class Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2

    def has_full_cudagraphs(self):
        return self is Mode.FULL


@dataclass(frozen=True)
class Descriptor:
    cg_mode: Mode
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = None
    max_query_len: int | None = None
    num_active_loras: int = 0
    num_ubatches: int = 1


ACTIVE_OWNER = ContextVar("test_sparse_graph_owner", default=None)


@contextmanager
def graph_scope(owner):
    token = ACTIVE_OWNER.set(owner)
    try:
        yield
    finally:
        ACTIVE_OWNER.reset(token)


class Buffers:
    def __init__(self, max_num_reqs, max_num_tokens, device):
        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)
        self.query_start_loc = torch.zeros(max_num_reqs + 2, dtype=torch.int32, device=device)
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
        self.seq_lens_np = self.seq_lens_cpu.numpy()


@dataclass
class Batch:
    req_ids: list = field(default_factory=lambda: ["original-a", "original-b"])
    num_reqs: int = 2
    num_reqs_after_padding: int = 2
    idx_mapping: object = None
    idx_mapping_np: object = None
    num_scheduled_tokens: object = None
    num_tokens: int = 5
    num_tokens_after_padding: int = 5
    num_draft_tokens: int = 0
    num_draft_tokens_per_req: object = None
    query_start_loc: object = None
    query_start_loc_np: object = None
    seq_lens: object = None
    seq_lens_cpu_upper_bound: object = None
    num_computed_tokens_np: object = None
    is_prefilling_np: object = None
    has_prefill: bool = False
    decode_graph_eligible: bool = False
    input_ids: object = None
    positions: object = None
    is_padding: object = None
    seq_lens_np: object = None
    attn_state: object = None
    max_query_len: object = None
    fast_prefill: object = None
    pcp: object = None
    prompt_lens: object = None
    dcp_local_seq_lens: object = None


class PointerBlockTables:
    """Pinned BlockTables gather uses the cached destination pointer tensor."""

    def __init__(self):
        self.input_block_tables = [torch.zeros(16, 32, dtype=torch.int32)]
        self.slot_mappings = torch.full((1, 128), 77, dtype=torch.int32)
        self.kernel_block_sizes = [4]
        self.block_tables = [torch.arange(16 * 32, dtype=torch.int32).reshape(16, 32) + 1]
        self.block_table_ptrs = self._make_ptr_tensor(self.block_tables)
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)
        self._num_dirty_rows = 3

    def _make_ptr_tensor(self, tables):
        # Retaining tensor objects here emulates the ownership represented by
        # cached device pointers, without dereferencing addresses on the CPU.
        return tuple(tables)

    def init_block_table_layout_tensors(self):
        self.block_table_ptrs = self._make_ptr_tensor(self.block_tables)
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

    def gather_block_tables(self, idx_mapping, num_reqs_padded):
        for group, destination in enumerate(self.input_block_table_ptrs):
            destination.zero_()
            destination[: idx_mapping.numel()].copy_(self.block_tables[group][idx_mapping])
        self._num_dirty_rows = idx_mapping.numel()
        return tuple(table[:num_reqs_padded] for table in self.input_block_tables)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layers = {"sparse.layers.1.attn": object()}

    def forward_logits(self, input_ids, positions):
        return torch.stack((input_ids, positions, input_ids + positions), dim=-1).float(), [positions[:, None].float()]


class State:
    def __init__(self):
        self.device_metadata = object()
        self.attn_metadata = {"target": "untouched"}
        self.calls = []

    def prepare_attn(self, batch, mode, block_tables, slot_mappings, attn_groups, kv_cache_config):
        self.calls.append((batch, mode, block_tables, slot_mappings.clone()))
        self.attn_metadata = {"batch": batch, "slots": slot_mappings, "block_tables": block_tables}
        return self.attn_metadata


class GraphManager:
    """Interface stub, not an implementation of ACL Graph capture."""

    def __init__(self, vllm_config, device, cudagraph_mode, decode_query_len, model_runner, *, varlen_decode=False):
        self.mode = cudagraph_mode
        self.decode_query_len = decode_query_len
        self.varlen_decode = varlen_decode
        self.graphs = {}
        self.proxy = model_runner
        self.capture_args = None
        self.descriptor = SimpleNamespace(cg_mode=Mode.NONE, num_tokens=3, num_reqs=1, max_query_len=None)
        self.dispatch_calls = []
        self.replays = 0
        self.owner_at_init = ACTIVE_OWNER.get()

    def needs_capture(self):
        return self.mode is not Mode.NONE

    def capture(
        self,
        model,
        model_state,
        buffers,
        intermediate_tensors,
        block_tables,
        attn_groups,
        kv_cache_config,
        *,
        use_aux_hidden_state_outputs=False,
        progress_bar_desc="",
    ):
        assert ACTIVE_OWNER.get() is self.owner_at_init
        self.capture_args = (model, model_state, buffers, block_tables, use_aux_hidden_state_outputs)
        self.graphs["interface-stub"] = object()
        self.model = model
        self.buffers = buffers

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None):
        self.dispatch_calls.append((num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len))
        return self.descriptor

    def run_fullgraph(self, desc):
        assert ACTIVE_OWNER.get() is self.owner_at_init
        self.replays += 1
        return self.model(self.buffers.input_ids[: desc.num_tokens], self.buffers.positions[: desc.num_tokens])

    def run_pw_graph(self, model, model_inputs):
        return model(**model_inputs)


def load_runtime():
    dependencies = {}

    def module(name, **exports):
        value = types.ModuleType(name)
        value.__dict__.update(exports)
        dependencies[name] = value

    module("vllm.config.compilation", CUDAGraphMode=Mode)
    module("vllm.forward_context", BatchDescriptor=SimpleNamespace, set_forward_context=lambda *a, **k: nullcontext())
    module("vllm.logger", logger=MagicMock())
    module("vllm.v1.worker.gpu.cudagraph_utils", BatchExecutionDescriptor=Descriptor)
    module(
        "vllm.v1.worker.gpu.attn_utils",
        build_slot_mappings_by_layer=lambda slots, config: {"sparse": slots[0]},
        init_attn_backend=lambda *a, **k: ([[SimpleNamespace(backend="stub")]], object(), [4]),
    )
    module("vllm_ascend.attention.attention_v1", AscendAttentionState=SimpleNamespace(ChunkedPrefill="chunked"))
    module("vllm_ascend.worker.v2.aclgraph_utils", ModelAclGraphManager=GraphManager)
    module("vllm_ascend.compilation.acl_graph", GraphParamsOwner=SimpleNamespace, graph_params_scope=graph_scope)
    # Deliberately do not export init_attn_backend here: local attn_utils has
    # no such export. This catches accidental imports from the wrong module.
    module("vllm_ascend.worker.v2.attn_utils", build_attn_metadata_wrapper=nullcontext)
    module("vllm_ascend.worker.v2.input_batch", AscendInputBuffers=Buffers)
    source = Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify/runtime.py"
    spec = importlib.util.spec_from_file_location("_sparse_runtime_interface_test", source)
    runtime = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(runtime)
    return runtime


class TestSparseRuntimeInterfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = load_runtime()

    def runner(self, mode=Mode.FULL, max_model_len=64):
        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False, max_model_len=max_model_len),
            compilation_config=SimpleNamespace(cudagraph_mode=mode),
            speculative_config=object(),
        )
        metrics = SimpleNamespace(record_graph=MagicMock())
        speculator = SimpleNamespace(
            target_vllm_config=config,
            device=torch.device("cpu"),
            max_num_reqs=16,
            max_num_tokens=128,
            sparse_model=Model(),
            update_stream=object(),
            num_query_per_req=8,
            sparse_metrics=metrics,
            max_model_len=max_model_len,
        )
        target_state = State()
        tables = PointerBlockTables()
        runner = self.runtime.SparseForwardRunner(speculator, target_state, object(), tables)
        runner.init_graphs(mode)
        return runner, target_state, tables, speculator

    def test_private_gather_refreshes_destination_pointer_cache(self):
        original = PointerBlockTables()
        original.input_block_tables[0].fill_(31)
        private = self.runtime.private_block_tables(original)
        self.assertIs(private.block_tables, original.block_tables)
        self.assertNotEqual(private.input_block_tables[0].data_ptr(), original.input_block_tables[0].data_ptr())
        self.assertNotEqual(private.slot_mappings.data_ptr(), original.slot_mappings.data_ptr())
        private.gather_block_tables(torch.tensor([5]), 1)
        torch.testing.assert_close(private.input_block_tables[0][0], original.block_tables[0][5])
        self.assertTrue(bool((original.input_block_tables[0] == 31).all()))
        self.assertTrue(bool((original.slot_mappings == 77).all()))

    def test_single_request_stages_exact_identity_position_and_views(self):
        buffers = Buffers(16, 128, "cpu")
        template = Batch(input_ids=torch.tensor([90, 91, 92]), positions=torch.tensor([0, 1, 2]))
        batch = self.runtime.single_request_batch(
            template, buffers, torch.tensor([10, 11, 12]), torch.tensor([27, 28, 29]), 5, "request-5"
        )
        self.assertEqual(batch.req_ids, ["request-5"])
        self.assertEqual(batch.idx_mapping.tolist(), [5])
        self.assertEqual(batch.num_computed_tokens_np.tolist(), [27])
        self.assertEqual(batch.query_start_loc_np.tolist(), [0, 3])
        self.assertEqual(batch.seq_lens.tolist(), [30])
        self.assertEqual(batch.positions.tolist(), [27, 28, 29])
        self.assertEqual(batch.input_ids.tolist(), [10, 11, 12])
        self.assertEqual(batch.input_ids.data_ptr(), buffers.input_ids.data_ptr())
        self.assertEqual(batch.positions.data_ptr(), buffers.positions.data_ptr())
        self.assertEqual(batch.seq_lens_cpu_upper_bound.data_ptr(), buffers.seq_lens_cpu.data_ptr())
        self.assertEqual(template.req_ids, ["original-a", "original-b"])
        self.assertEqual(template.input_ids.tolist(), [90, 91, 92])

    def test_native_capture_signature_and_upstream_mode(self):
        runner, _, _, speculator = self.runner()
        manager = speculator.sparse_cudagraph_manager
        self.assertEqual(manager.mode, Mode.FULL)
        self.assertTrue(manager.varlen_decode)
        self.assertEqual(manager.decode_query_len, 8)
        self.assertIs(manager.owner_at_init, runner.graph_params_owner)
        self.assertIsNone(ACTIVE_OWNER.get())
        runner.capture()
        model, state, buffers, tables, aux = manager.capture_args
        self.assertIs(model, runner.graph_model)
        self.assertIs(state, runner.model_state)
        self.assertIs(buffers, runner.buffers)
        self.assertIs(tables, runner.block_tables)
        self.assertTrue(aux)
        speculator.sparse_metrics.record_graph.assert_called_with("sparse", "capture")

    def test_replay_stages_new_ids_positions_and_clones_persistent_outputs(self):
        runner, target_state, original_tables, speculator = self.runner()
        runner.capture()
        manager = speculator.sparse_cudagraph_manager
        manager.descriptor = SimpleNamespace(cg_mode=Mode.FULL, num_tokens=8, num_reqs=4, max_query_len=8)
        outputs, features = runner.forward(Batch(), torch.tensor([4, 5, 6]), torch.tensor([21, 22, 23]), 5, "r5")
        expected, expected_features = speculator.sparse_model.forward_logits(
            torch.tensor([4, 5, 6]), torch.tensor([21, 22, 23])
        )
        torch.testing.assert_close(outputs, expected)
        torch.testing.assert_close(features[0], expected_features[0])
        saved = outputs.clone()
        runner.forward(Batch(), torch.tensor([7, 8, 9]), torch.tensor([24, 25, 26]), 5, "r5")
        torch.testing.assert_close(outputs, saved)
        self.assertEqual(manager.replays, 2)
        self.assertEqual(manager.dispatch_calls[-1], (1, 3, None, 0, 3))
        self.assertEqual(target_state.attn_metadata, {"target": "untouched"})
        self.assertTrue(bool((original_tables.slot_mappings == 77).all()))

    def test_dummy_rows_cannot_extend_real_context_or_write_slots(self):
        runner, _, _, speculator = self.runner()
        runner.capture()
        manager = speculator.sparse_cudagraph_manager
        manager.descriptor = SimpleNamespace(cg_mode=Mode.FULL, num_tokens=16, num_reqs=16, max_query_len=8)
        runner.forward(Batch(), torch.tensor([4, 5, 6]), torch.tensor([61, 62, 63]), 2, "end")
        batch, _, tables, slots = runner.model_state.calls[-1]
        self.assertEqual(batch.seq_lens[0].item(), 64)
        self.assertEqual(batch.query_start_loc[1].item(), 3)
        self.assertEqual(batch.query_start_loc[-1].item(), 16)
        self.assertTrue(bool((slots[:, 3:] == -1).all()))
        self.assertTrue(bool((tables[0][1:] == 0).all()))
        # Native AscendDefaultModelState derives metadata max_query_len from
        # num_scheduled_tokens, so this must describe real row lengths.
        self.assertLessEqual(int(batch.num_scheduled_tokens.max()), 8)

    def test_prefill_dispatch_excludes_decode_only_descriptors(self):
        runner, _, _, speculator = self.runner(Mode.NONE)
        manager = speculator.sparse_cudagraph_manager
        manager.descriptor = SimpleNamespace(cg_mode=Mode.NONE, num_tokens=3, num_reqs=1, max_query_len=None)
        runner.forward(Batch(), torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]), 2, "prompt", prefill=True)
        self.assertIsNone(manager.dispatch_calls[-1][-1])
        self.assertEqual(manager.replays, 0)
        speculator.sparse_metrics.record_graph.assert_called_with("sparse", "eager")

    def test_single_request_graph_padding_at_context_end_uses_explicit_fallback(self):
        runner, _, _, speculator = self.runner()
        runner.capture()
        manager = speculator.sparse_cudagraph_manager
        manager.descriptor = Descriptor(Mode.FULL, 8, 1)
        logits, _ = runner.forward(Batch(), torch.tensor([4, 5, 6]), torch.tensor([61, 62, 63]), 2, "end")
        batch, effective_mode, _, _ = runner.model_state.calls[-1]
        self.assertEqual(effective_mode, Mode.NONE)
        self.assertEqual(batch.num_tokens_after_padding, 3)
        self.assertEqual(batch.seq_lens.tolist(), [64])
        self.assertEqual(logits.shape[0], 3)
        self.assertEqual(manager.replays, 0)
        speculator.sparse_metrics.record_graph.assert_called_with("sparse", "eager")

    def test_model_state_metadata_producer_is_not_shared(self):
        runner, target_state, _, _ = self.runner()
        self.assertIsNot(runner.model_state, target_state)
        self.assertIsNone(runner.model_state.device_metadata)
        self.assertIsNotNone(target_state.device_metadata)


if __name__ == "__main__":
    unittest.main()
