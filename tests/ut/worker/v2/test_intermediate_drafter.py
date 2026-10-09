# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Check the precomputed-context entry without importing vLLM or NPU kernels."""

from enum import Enum
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_intermediate_backend import backend as backend
from test_intermediate_backend import load_class


class GraphMode(Enum):
    NONE = "none"
    FULL = "full"
    PIECEWISE = "piecewise"


@pytest.fixture
def drafter():
    def make(mode, old_version=False, padded_reqs=4):
        events = []
        prepare = Mock(side_effect=lambda *args, **kwargs: events.append("prepare"))
        desc = NS(cg_mode=mode, num_reqs=padded_reqs, num_tokens=12)
        dp_tokens = object()
        dispatch = Mock(return_value=(desc, dp_tokens))
        slots = {"draft": object()}
        build_slots = Mock(return_value=slots)
        cls = load_class(
            "drafter.py",
            "IntermediateDFlashSpeculator",
            dict(
                AscendDFlashSpeculator=object,
                CUDAGraphMode=GraphMode,
                vllm_version_is=lambda version: old_version,
                prepare_dflash_inputs=prepare,
                dispatch_cg_and_sync_dp=dispatch,
                build_slot_mappings_by_layer=build_slots,
            ),
        )
        obj = cls()
        obj.num_query_per_req, obj.num_speculative_steps = 3, 2
        obj.max_num_reqs, obj.max_num_tokens, obj.max_model_len = 4, 12, 16
        obj.draft_kv_cache_group_ids = [2, 0]
        obj.block_tables = NS(
            slot_mappings=torch.full((3, 12), -1, dtype=torch.long),
            input_block_tables=[object(), object(), object()],
            kernel_block_sizes=[4, 8, 16],
            cp_rank=1,
            cp_size=2,
            cp_interleave=4,
        )
        obj.input_buffers = object()
        obj.context_positions = object()
        obj._context_slot_mappings = [object(), object()]
        obj.sample_indices, obj.sample_pos, obj.sample_idx_mapping = object(), object(), object()
        obj.temperature, obj.seeds = object(), object()
        obj.parallel_drafting_token_id, obj.sample_from_anchor = 99, True
        obj.dp_size, obj.dp_rank = 2, 1
        obj.kv_cache_config = object()
        obj.draft_tokens = torch.tensor([[7, 8], [9, 10], [-1, -1], [-1, -1]])
        # The fast entry must never read/project context hidden states again.
        obj.hidden_states = NS(copy_=Mock(side_effect=AssertionError("hidden copy")))
        obj.model = NS(
            precompute_and_store_context_kv=Mock(side_effect=AssertionError("context reprojection")),
            combine_hidden_states=Mock(side_effect=AssertionError("aux reprojection")),
        )
        metadata = {"draft": object()}
        obj.build_draft_attn_metadatas = Mock(side_effect=lambda *args: events.append("metadata") or (metadata, None))
        obj._prepare_eplb_forward = Mock(side_effect=lambda *args: events.append("eplb"))
        obj._generate_draft = Mock(side_effect=lambda *args: events.append("generate"))

        def replay(descriptor):
            events.append("replay")
            obj.build_draft_attn_metadatas(descriptor.num_reqs, obj.input_batch.seq_lens_cpu_upper_bound)

        obj.query_cudagraph_manager = NS(run_fullgraph=Mock(side_effect=replay))
        batch = NS(num_reqs=2, seq_lens_np=np.array([5, 15]), seq_lens_cpu_upper_bound=torch.tensor([5, 15]))
        inputs = (batch, torch.tensor([6, 16, 0, 0]), torch.ones(2), torch.zeros(2), object(), object())
        return NS(
            obj=obj,
            inputs=inputs,
            prepare=prepare,
            dispatch=dispatch,
            desc=desc,
            dp_tokens=dp_tokens,
            slots=slots,
            build_slots=build_slots,
            metadata=metadata,
            events=events,
        )

    return make


@pytest.mark.parametrize("old_version", [False, True])
@pytest.mark.parametrize("mode", [GraphMode.FULL, GraphMode.NONE, GraphMode.PIECEWISE])
def test_precomputed_entry_preserves_prepare_contract_and_real_batch(drafter, mode, old_version):
    case = drafter(mode, old_version)
    obj, batch = case.obj, case.inputs[0]
    result = obj.propose_precomputed(*case.inputs)
    assert result.tolist() == [[7, 8], [9, 10]]
    assert result.data_ptr() == obj.draft_tokens.data_ptr()
    assert obj.input_batch is batch
    assert obj.draft_max_seq_len == 16  # Clamp the longest context plus query.
    assert case.prepare.call_count == 2
    for i, (gid, call) in enumerate(zip([2, 0], case.prepare.call_args_list)):
        args, kwargs = call.args, call.kwargs
        assert args[0] is obj.input_buffers
        assert args[1].data_ptr() == obj.block_tables.slot_mappings[gid].data_ptr()
        assert args[2] is obj.context_positions and args[3] is obj._context_slot_mappings[i]
        assert args[4:9] == (obj.sample_indices, obj.sample_pos, obj.sample_idx_mapping, obj.temperature, obj.seeds)
        assert args[9] is batch
        assert args[10] is case.inputs[2] and args[11] is case.inputs[3]
        assert args[12] is args[13] is case.inputs[1]
        assert args[14] is case.inputs[4] and args[15] is case.inputs[5]
        assert args[16] is obj.block_tables.input_block_tables[gid]
        assert args[17] == obj.block_tables.kernel_block_sizes[gid]
        assert kwargs == dict(
            parallel_drafting_token_id=99,
            num_query_per_req=3,
            num_speculative_steps=2,
            max_num_reqs=4,
            max_num_tokens=12,
            max_model_len=16,
            sample_from_anchor=True,
            **({} if old_version else dict(cp_rank=1, cp_size=2, cp_interleave=4)),
        )
    case.dispatch.assert_called_once_with(
        obj.query_cudagraph_manager,
        2,
        6,
        uniform_token_count=3,
        dp_size=2,
        dp_rank=1,
        need_eager=False,
    )
    obj._prepare_eplb_forward.assert_called_once_with(6)
    obj.model.precompute_and_store_context_kv.assert_not_called()
    obj.model.combine_hidden_states.assert_not_called()
    obj.hidden_states.copy_.assert_not_called()
    obj.build_draft_attn_metadatas.assert_called_once_with(4, batch.seq_lens_cpu_upper_bound)
    if mode == GraphMode.FULL:
        assert case.events == ["prepare", "prepare", "eplb", "replay", "metadata"]
        obj._generate_draft.assert_not_called()
        case.build_slots.assert_not_called()
        obj.query_cudagraph_manager.run_fullgraph.assert_called_once_with(case.desc)
    else:
        assert case.events == ["prepare", "prepare", "eplb", "metadata", "generate"]
        obj.query_cudagraph_manager.run_fullgraph.assert_not_called()
        assert case.build_slots.call_args.args[0].shape == (3, 12)
        assert case.build_slots.call_args.args[1] is obj.kv_cache_config
        obj._generate_draft.assert_called_once_with(2, 12, case.metadata, case.slots, case.dp_tokens, mode)


def test_eager_descriptor_without_request_padding_uses_real_count(drafter):
    case = drafter(GraphMode.NONE, padded_reqs=0)
    case.obj.propose_precomputed(*case.inputs)
    case.obj.build_draft_attn_metadatas.assert_called_once_with(2, case.inputs[0].seq_lens_cpu_upper_bound)


def test_backend_cached_fast_proposal_skips_verifier_metadata_and_preserves_positions(backend):
    obj, metadata, _ = backend
    list(obj.verify([[1, 2], [7]], [[3], [8]], req_ids=["a", "b"]))
    obj.drafter.propose_precomputed = Mock(return_value=torch.tensor([[10, 11], [5, 6]]))
    obj.drafter.model.precompute_and_store_context_kv.reset_mock()
    metadata.reset_mock()
    # Reorder requests and reject a tail: each still uses its own valid pages.
    assert obj.propose([[7, 8, 9], [1, 2, 4]], req_ids=["b", "a"]) == [[10, 11], [5, 6]]
    metadata.assert_not_called()
    obj.drafter.propose.assert_not_called()
    obj.drafter.model.precompute_and_store_context_kv.assert_not_called()
    assert obj.model.call_count == 1
    batch, anchors, ones, zeros, temperature, seeds = obj.drafter.propose_precomputed.call_args.args
    assert batch.req_ids == ["b", "a"] and batch.num_reqs == 2
    assert batch.input_ids.tolist() == [8, 2]
    assert batch.positions.tolist() == [1, 1]
    assert batch.query_start_loc.tolist() == [0, 1, 2]
    assert batch.seq_lens.tolist() == [2, 2]
    assert anchors[:2].tolist() == [9, 4]
    assert ones.tolist() == [1, 1] and zeros.tolist() == [0, 0]
    assert temperature is obj._proposal_temperature and seeds is obj._proposal_seeds
    assert obj.block_tables.input_block_tables[0].tolist() == [[3, 4], [1, 2]]
    assert obj.cache.tokens[obj.cache.slots["a"]] == [1, 2]
    assert obj.cache.tokens[obj.cache.slots["b"]] == [7, 8]


def test_backend_fast_entry_refreshes_context_after_prefix_mismatch(backend):
    obj, metadata, _ = backend
    list(obj.verify([[1, 2]], [[3]], req_ids=["a"]))
    obj.drafter.propose_precomputed = Mock(return_value=torch.tensor([[10, 11]]))
    obj.drafter.model.precompute_and_store_context_kv.reset_mock()
    metadata.reset_mock()
    assert obj.propose([[1, 9, 8]], req_ids=["a"]) == [[10, 11]]
    metadata.assert_called_once()
    assert obj.model.call_count == 2
    assert obj.model.call_args.kwargs["input_ids"].tolist() == [9]
    assert obj.model.call_args.kwargs["positions"].tolist() == [1]
    hidden, positions, slots = obj.drafter.model.precompute_and_store_context_kv.call_args.args
    assert hidden.tolist() == [[9, 1]] and positions.tolist() == [1] and slots.tolist() == [5]
    assert obj.cache.tokens[obj.cache.slots["a"]] == [1, 9]
    obj.drafter.propose_precomputed.assert_called_once()
    obj.drafter.propose.assert_not_called()
