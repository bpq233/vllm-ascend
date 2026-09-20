# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise packed inputs and lifecycle with CPU stand-ins for NPU kernels."""

import ast
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage"


def load_class(filename, name, namespace):
    tree = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    exec(compile(tree, filename, "exec"), namespace)
    return namespace[name]


@pytest.fixture
def backend():
    metadata = Mock(side_effect=lambda **kw: kw)
    rope = NS(update_cos_sin=Mock(), _cos=object())
    namespace = dict(
        torch=torch,
        np=np,
        contextmanager=contextmanager,
        rope=rope,
        vllm_version_is=lambda version: False,
        set_current_vllm_config=lambda *a: nullcontext(),
        set_forward_context=lambda *a, **kw: nullcontext(),
        AscendInputBatch=NS,
        AscendAttentionState=NS(PrefillNoCache="prefill", ChunkedPrefill="extend"),
        CUDAGraphMode=NS(NONE=None),
        build_attn_metadata=metadata,
        build_slot_mappings_by_layer=lambda slots, config: {"layer": slots[0]},
    )
    cls = load_class("backend.py", "IntermediateBackend", namespace)
    obj = cls.__new__(cls)
    obj.config = NS(num_speculative_tokens=2)
    obj.device = torch.device("cpu")
    obj.max_num_reqs, obj.max_num_tokens, obj.max_model_len = 2, 12, 8
    cache_cls = load_class("backend.py", "IntermediateKVCache", dict(OrderedDict=OrderedDict))
    obj.cache = cache_cls(2)
    obj.graph_state = NS(context=nullcontext)
    obj.cudagraph_manager = None
    obj.graph_replays = 0
    obj.forward_tokens = obj.reused_tokens = 0
    obj.executed_tokens = 0
    obj.reused_hidden_tokens = 0
    obj._context_rows = [None] * obj.max_num_reqs
    obj._init_proposal_scratch()
    obj.vllm_config, obj.kv_cache_config, obj.attn_groups = NS(), NS(), []
    obj._rope_state = {"_cos": object()}
    obj.block_tables = NS(
        input_block_tables=[torch.tensor([[1, 2], [3, 4]])],
        kernel_block_sizes=[4],
        slot_mappings=torch.zeros(1, 12, dtype=torch.long),
    )
    obj.cache_block_tables = [t.clone() for t in obj.block_tables.input_block_tables]
    obj.input_buffers = NS(
        input_ids=torch.empty(12, dtype=torch.int32),
        positions=torch.empty(12, dtype=torch.int64),
        is_padding=torch.empty(12, dtype=torch.bool),
    )
    # Encode each input's position and ID, making logit alignment observable.
    obj.model = Mock(side_effect=lambda input_ids, positions: torch.stack((input_ids, positions), dim=-1))
    obj.model.compute_logits = lambda hidden: hidden.float()
    obj.drafter = NS(
        propose=Mock(return_value=torch.tensor([[9, 10], [11, 12]])),
        draft_kv_cache_group_ids=[0],
        _layer_group_idx=None,
        model=NS(precompute_and_store_context_kv=Mock()),
    )
    return obj, metadata, rope


def test_verify_alignment_and_request_isolation(backend):
    obj, metadata, _ = backend
    result = list(obj.verify([[1, 2], [5]], [[3, 4], [6]]))
    assert [x.tolist() for x in result] == [[[2, 1], [3, 2], [4, 3]], [[5, 0], [6, 1]]]
    assert metadata.call_args.kwargs["causal"] is True
    assert metadata.call_args.kwargs["query_start_loc_cpu"].tolist() == [0, 4, 6]
    assert obj.block_tables.slot_mappings[0, :6].tolist() == [4, 5, 6, 7, 12, 13]
    # A shorter, unrelated request reuses scratch slot 0 without a stale prefix.
    assert list(obj.verify([[7]], [[8]]))[0].tolist() == [[7, 0], [8, 1]]
    assert metadata.call_args.kwargs["seq_lens"].tolist() == [2]


def test_secondary_uses_last_token_as_anchor_and_excludes_it_from_context(backend):
    obj, _, _ = backend
    result = obj.propose([[1, 2, 3], [4, 5]])
    assert result == [[9, 10], [11, 12]]
    args = obj.drafter.propose.call_args.args
    assert args[0].input_ids.tolist() == [1, 2, 4]
    assert args[7].tolist() == [3, 5]
    assert args[5].tolist() == [1, 1]
    assert args[6].tolist() == [0, 0]


def test_boundary_falls_back_to_verifier_only(backend):
    obj, _, _ = backend
    obj.drafter.propose.return_value = torch.tensor([[9, 10]])
    assert obj.propose([[1] * 7, [2, 3], [4]]) == [[], [9, 10], []]
    assert obj.drafter.propose.call_args.args[7][0] == 3


def test_microbatch_limits_and_rope_restored_on_error(backend):
    obj, _, rope = backend
    assert [(i, len(rows)) for i, rows in obj._batches([[1] * 8, [2] * 6, [3], [4]])] == [(0, 1), (1, 2), (3, 1)]
    main_rope = rope._cos
    with pytest.raises(RuntimeError), obj._context():
        assert rope._cos is obj._rope_state["_cos"]
        raise RuntimeError("forward failed")
    assert rope._cos is main_rope


def test_adapter_replaces_only_candidate_buffer_and_publishes_real_lengths():
    primary = torch.tensor([[7, 8], [9, 10]])

    class OriginalDFlash:
        def propose(self, batch, *args, **kwargs):
            return primary

    cls = load_class(
        "adapter.py",
        "MultiStageDFlashSpeculator",
        dict(torch=torch, AscendDFlashSpeculator=OriginalDFlash, DraftTokenIds=lambda ids, tokens: (ids, tokens)),
    )
    obj = cls.__new__(cls)
    obj.final_capacity, obj.max_model_len, obj.device = 5, 16, torch.device("cpu")
    history = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]])
    obj.req_states = NS(
        total_len=NS(gpu=torch.tensor([4, 2])),
        all_token_ids=NS(gpu=history),
        max_seq_len=np.array([16, 16]),
        prefill_len=NS(np=np.array([3, 2])),
        req_id_to_index={"a": 0, "b": 1},
    )
    original_history = history.clone()
    obj.pipeline = NS(
        backend=NS(max_model_len=16, profile=Mock(), cache=NS(retain=Mock())),
        refine=Mock(return_value=[[7, 11, 12], []]),
    )
    batch = NS(num_reqs=2, idx_mapping=torch.tensor([0, 1]), idx_mapping_np=np.array([0, 1]), req_ids=["a", "b"])
    result = obj.propose(batch)
    assert result.tolist() == [[7, 11, 12, 0, 0], [0, 0, 0, 0, 0]]
    assert obj.pipeline.refine.call_args.args == ([[1, 2, 3, 4], [5, 6]], [[7, 8], [9, 10]], [5, 0])
    assert obj.pipeline.refine.call_args.kwargs == {"req_ids": ["a", "b"]}
    assert obj.get_draft_tokens() == (["a", "b"], [[7, 11, 12], []])
    assert torch.equal(history, original_history)
    assert primary.tolist() == [[7, 8], [9, 10]]
    obj.propose(batch, dummy_run=True, is_profile=True)
    obj.pipeline.backend.profile.assert_called_once()
    assert obj.pipeline.refine.call_count == 1


def test_cached_rounds_forward_only_suffix_and_hydrate_secondary(backend):
    obj, metadata, _ = backend
    list(obj.verify([[1, 2]], [[3, 4]], req_ids=["a"]))
    assert obj.forward_tokens == 4
    obj.drafter.propose.return_value = torch.tensor([[6, 7]])
    assert obj.propose([[1, 2, 3, 4, 5]], req_ids=["a"]) == [[6, 7]]
    assert obj.forward_tokens == 4  # Reuse the predictor: no extra verifier pass.
    assert obj.reused_hidden_tokens == 1
    assert obj.model.call_count == 1
    assert metadata.call_args.kwargs["attn_state"] == "extend"
    result = list(obj.verify([[1, 2, 3, 4, 5]], [[6, 7]], req_ids=["a"]))
    assert result[0].tolist() == [[5, 4], [6, 5], [7, 6]]
    assert obj.forward_tokens == 7  # Full-prefix implementation would execute 15.
    assert obj.reused_tokens == 8
    context_hidden, positions, slots = obj.drafter.model.precompute_and_store_context_kv.call_args.args
    assert context_hidden.tolist() == [[5, 4], [6, 5], [7, 6]]
    assert positions.tolist() == [4, 5, 6] and slots.tolist() == [8, 9, 10]
    # Final target rejected the old tail and sampled a different token.
    list(obj.verify([[1, 2, 9]], [[10]], req_ids=["a"]))
    assert obj.model.call_args.kwargs["input_ids"].tolist() == [9, 10]
    assert obj.model.call_args.kwargs["positions"].tolist() == [2, 3]
    assert metadata.call_args.kwargs["seq_lens"].tolist() == [4]


def test_cached_context_owns_storage_and_reuses_rejection_predictor(backend):
    obj, _, _ = backend
    output = torch.tensor([[1, 0], [2, 1], [3, 2], [4, 3]])
    obj.model.side_effect = lambda **kw: output
    list(obj.verify([[1, 2]], [[3, 4]], req_ids=["a"]))
    # Simulate another graph replay overwriting its output buffer.
    output.fill_(-99)
    obj.drafter.propose.return_value = torch.tensor([[6, 7]])
    obj.propose([[1, 2, 9]], req_ids=["a"])
    assert obj.model.call_count == 1
    assert obj.drafter.propose.call_args.args[3].tolist() == [[2, 1]]
    assert obj.reused_hidden_tokens == 1


def test_cached_context_projects_aux_once_and_respects_reordering(backend):
    obj, _, _ = backend
    obj.model.side_effect = lambda input_ids, positions: (input_ids[:, None], [positions[:, None].float()])
    obj.drafter.model.combine_hidden_states = Mock(side_effect=lambda states: states + 100)
    list(obj.verify([[1, 2], [7]], [[3], [8]], req_ids=["a", "b"]))
    obj.propose([[7, 8, 9], [1, 2, 3, 4]], req_ids=["b", "a"])
    assert obj.model.call_count == 1
    assert obj.drafter.model.combine_hidden_states.call_count == 1
    assert obj.drafter.propose.call_args.args[3].tolist() == [[101], [102]]
    assert obj.drafter.propose.call_args.args[4] is None


def test_changed_or_recycled_context_cannot_reuse_hidden(backend):
    obj, _, _ = backend
    list(obj.verify([[1, 2]], [[3]], req_ids=["a"]))
    obj.drafter.propose.return_value = torch.tensor([[6, 7]])
    obj.propose([[1, 8, 9]], req_ids=["a"])
    assert obj.model.call_count == 2
    obj.cache.retain(())
    obj.propose([[1, 8, 9]], req_ids=["a"])
    assert obj.model.call_count == 3


def test_cold_prompt_retains_only_prediction_context(backend):
    obj, _, _ = backend
    list(obj.verify([[1, 2, 3, 4, 5, 6]], [[7, 8]], req_ids=["a"]))
    saved = obj._context_rows[obj.cache.slots["a"]]
    assert saved[1:3] == (5, 8)
    assert saved[3].shape[0] == 3


def test_sparse_graph_gap_warms_context_without_large_padding(backend):
    obj, metadata, _ = backend
    obj.max_model_len = obj.max_num_tokens = 128
    obj.block_tables.input_block_tables = [torch.tensor([[1, 2, 3], [4, 5, 6]])]
    obj.cache_block_tables = [obj.block_tables.input_block_tables[0].clone()]
    obj.block_tables.slot_mappings = torch.zeros(1, 128, dtype=torch.long)
    obj.input_buffers = NS(
        input_ids=torch.empty(128, dtype=torch.int32),
        positions=torch.empty(128, dtype=torch.int64),
        is_padding=torch.empty(128, dtype=torch.bool),
    )
    obj._forward.__globals__["CUDAGraphMode"].PIECEWISE = "piecewise"
    obj._forward.__globals__["BatchDescriptor"] = NS
    obj.cudagraph_manager = NS(
        capture_sizes=[8, 128],
        dispatch=lambda n, total, *a: NS(cg_mode="piecewise", num_tokens=8 if total <= 8 else 128),
        run_pw_graph=lambda model, inputs: model(**inputs),
    )
    values = torch.zeros(28)

    def causal_forward(input_ids, positions):
        meta = metadata.call_args.kwargs
        actual = meta["num_actual_tokens"]
        values[meta["slot_mappings"][0, :actual]] = input_ids[:actual].float()
        output = torch.zeros(len(input_ids), 1)
        for i, position in enumerate(positions[:actual]):
            prefix = torch.arange(int(position) + 1)
            physical = meta["block_tables"][0][0, prefix // 4] * 4 + prefix % 4
            output[i, 0] = values[physical].sum()
        return output

    obj.model.side_effect = causal_forward
    result = list(obj.verify([list(range(1, 9))], [[9]], req_ids=["a"]))
    assert result[0].tolist() == [[36], [45]]
    assert obj.forward_tokens == 9
    assert obj.executed_tokens == 16  # Previously padded the single forward to 128.
    assert obj.model.call_count == 2
    assert list(obj.verify([list(range(1, 10))], [[10]], req_ids=["a"]))[0].tolist() == [[45], [55]]


def test_request_reordering_reuses_its_own_physical_pages(backend):
    obj, metadata, _ = backend
    list(obj.verify([[1, 2], [4]], [[3], [5]], req_ids=["a", "b"]))
    list(obj.verify([[4, 5]], [[6]], req_ids=["b"]))
    assert metadata.call_args.kwargs["num_computed_tokens_cpu"].tolist() == [1]
    assert metadata.call_args.kwargs["block_tables"][0].tolist() == [[3, 4]]
    assert obj.block_tables.slot_mappings[0, :2].tolist() == [13, 14]
    obj.cache.retain(["a"])
    assert list(obj.cache.slots) == ["a"]
    list(obj.verify([[7]], [[8]], req_ids=["c"]))
    assert metadata.call_args.kwargs["num_computed_tokens_cpu"].tolist() == [0]


def test_failed_cache_write_does_not_publish_partial_tail(backend):
    obj, _, _ = backend
    list(obj.verify([[1, 2]], [[3, 4]], req_ids=["a"]))
    obj.drafter.model.precompute_and_store_context_kv.side_effect = RuntimeError("cache write failed")
    with pytest.raises(RuntimeError, match="cache write failed"):
        list(obj.verify([[1, 2, 9]], [[10]], req_ids=["a"]))
    assert obj.cache.tokens[obj.cache.slots["a"]] == [1, 2]
    obj.drafter.model.precompute_and_store_context_kv.side_effect = None
    list(obj.verify([[1, 2, 9]], [[10]], req_ids=["a"]))
    assert obj.model.call_args.kwargs["input_ids"].tolist() == [9, 10]


def test_batch_budget_counts_new_queries_instead_of_cached_prefixes(backend):
    obj, _, _ = backend
    slots, _ = obj.cache.plan(["a", "b"], [[1] * 8, [2] * 8], [7, 7])
    obj.cache.commit(slots, [[1] * 8, [2] * 8])
    assert [(i, len(rows)) for i, rows in obj._batches([[1] * 8, [2] * 8], ["a", "b"], [7, 7])] == [(0, 2)]


def test_host_history_delta_reorder_growth_and_slot_reuse():
    cls = load_class("adapter.py", "MultiStageDFlashSpeculator", dict(torch=torch, AscendDFlashSpeculator=object))
    obj = cls.__new__(cls)
    obj.final_capacity = 2
    history = torch.arange(24).reshape(2, 12)
    lengths = torch.tensor([6, 4])
    obj.req_states = NS(total_len=NS(gpu=lengths), all_token_ids=NS(gpu=history), req_id_to_index={"a": 0, "b": 1})
    batch = NS(idx_mapping=torch.tensor([0, 1]), idx_mapping_np=np.array([0, 1]), req_ids=["a", "b"])
    drafts = torch.tensor([[7, 8], [9, 10]])
    assert obj._read_step(batch, drafts) == ([6, 4], [list(range(6)), list(range(12, 16))], drafts.tolist())
    # Poison the device prefix to prove the warm path does not transfer it.
    history[:, :2] = -99
    lengths[:] = torch.tensor([8, 5])
    batch.idx_mapping = torch.tensor([1, 0])
    batch.idx_mapping_np = np.array([1, 0])
    batch.req_ids = ["b", "a"]
    assert obj._read_step(batch, drafts)[1] == [list(range(12, 17)), list(range(8))]
    # A large prefill jump must read the full row rather than lose tokens.
    lengths[1] = 10
    assert obj._read_step(batch, drafts)[1][0] == history[1, :10].tolist()
    # Request IDs, not physical row indices, own the host history.
    obj.req_states.req_id_to_index = {"c": 0, "b": 1}
    batch.req_ids = ["b", "c"]
    lengths[0] = 3
    assert obj._read_step(batch, drafts)[1][1] == history[0, :3].tolist()
    assert "a" not in obj._host_histories


def test_cache_eviction_protects_other_rows_in_current_batch(backend):
    obj, _, _ = backend
    slots, _ = obj.cache.plan(["a", "b"], [[1, 2], [3, 4]], [1, 1])
    obj.cache.commit(slots, [[1, 2], [3, 4]])
    slots, starts = obj.cache.plan(["c", "a"], [[7, 8], [1, 2, 5]], [1, 2])
    assert slots == [1, 0] and starts == [0, 2]
    assert "b" not in obj.cache.slots


def test_cached_causal_output_matches_full_prefix_after_rejection(backend):
    obj, metadata, _ = backend
    values = torch.zeros(20)

    def causal_forward(input_ids, positions):
        meta = metadata.call_args.kwargs
        slots = meta["slot_mappings"][0]
        values[slots] = input_ids.float()
        starts = meta["query_start_loc_cpu"].tolist()
        tables = meta["block_tables"][0]
        output = []
        for row in range(len(starts) - 1):
            for j in range(starts[row], starts[row + 1]):
                prefix = torch.arange(int(positions[j]) + 1)
                physical = tables[row, prefix // 4] * 4 + prefix % 4
                output.append(values[physical].sum())
        return torch.stack(output).unsqueeze(1)

    obj.model.side_effect = causal_forward
    list(obj.verify([[1, 2], [7]], [[3, 4], [8]], req_ids=["a", "b"]))
    result = list(obj.verify([[7, 8], [1, 2, 9]], [[6], [5, 4]], req_ids=["b", "a"]))
    assert result[0].flatten().tolist() == [15, 21]
    assert result[1].flatten().tolist() == [12, 17, 21]


def test_intermediate_graph_padding_keeps_real_kv_and_logits(backend):
    obj, metadata, _ = backend
    # A captured 8-token gear serves a real 5-token query.
    obj._forward.__globals__["CUDAGraphMode"].PIECEWISE = "piecewise"
    obj._forward.__globals__["BatchDescriptor"] = NS
    obj.cudagraph_manager = NS(
        dispatch=Mock(return_value=NS(cg_mode="piecewise", num_tokens=8)),
        run_pw_graph=Mock(side_effect=lambda model, inputs: model(**inputs)),
    )
    result = list(obj.verify([[1, 2]], [[3, 4, 5]], req_ids=["a"]))
    assert result[0].tolist() == [[2, 1], [3, 2], [4, 3], [5, 4]]
    assert obj.model.call_args.kwargs["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, 0, 0]
    assert metadata.call_args.kwargs["num_actual_tokens"] == 5
    assert metadata.call_args.kwargs["num_input_tokens"] == 8
    assert obj.block_tables.slot_mappings[0, 5:8].tolist() == [-1, -1, -1]
    assert obj.drafter.model.precompute_and_store_context_kv.call_args.args[0].shape[0] == 5
    assert obj.graph_replays == 1
    assert obj.forward_tokens == 5
    assert obj.executed_tokens == 8


def test_prediction_hidden_uses_views_for_single_request_and_cached_batch(backend):
    obj, _, _ = backend
    forwards, logits_inputs = [], []

    def forward(input_ids, positions):
        output = torch.stack((input_ids, positions), dim=-1)
        forwards.append(output)
        return output

    obj.model.side_effect = forward
    obj.model.compute_logits = lambda hidden: logits_inputs.append(hidden) or hidden.float()
    # Cold single-request prefix is sliced without gathering.
    list(obj.verify([[1, 2, 3]], [[4]], req_ids=["a"]))
    assert logits_inputs[-1].untyped_storage().data_ptr() == forwards[-1].untyped_storage().data_ptr()
    assert logits_inputs[-1].tolist() == [[3, 2], [4, 3]]
    # Seed another request, then reorder a fully cached batch.
    list(obj.verify([[7]], [[8]], req_ids=["b"]))
    result = list(obj.verify([[7, 8], [1, 2, 3, 4]], [[9], [5]], req_ids=["b", "a"]))
    assert logits_inputs[-1].data_ptr() == forwards[-1].data_ptr()
    assert [row.tolist() for row in result] == [[[8, 1], [9, 2]], [[4, 3], [5, 4]]]


def test_proposal_scratch_addresses_stay_stable_across_batch_sizes(backend):
    obj, _, _ = backend
    obj.propose([[1, 2, 3], [4, 5]], req_ids=["a", "b"])
    first = obj.drafter.propose.call_args.args
    addresses = [first[i].data_ptr() for i in range(5, 11)]
    obj.drafter.propose.return_value = torch.tensor([[9, 10]])
    obj.propose([[4, 5, 6]], req_ids=["b"])
    second = obj.drafter.propose.call_args.args
    assert [second[i].data_ptr() for i in range(5, 11)] == addresses
    assert second[5].tolist() == [1]
    assert second[6].tolist() == [0]
    assert second[7][0].item() == 6
    assert second[9].tolist() == [0.0, 0.0]
    assert second[10].tolist() == [0, 0]


def test_missing_intermediate_graph_fails_instead_of_silent_eager(backend):
    obj, _, _ = backend
    obj._forward.__globals__["CUDAGraphMode"].PIECEWISE = "piecewise"
    obj.cudagraph_manager = NS(dispatch=Mock(return_value=NS(cg_mode=None)))
    with pytest.raises(RuntimeError, match="refusing silent eager"):
        list(obj.verify([[1]], [[2]], req_ids=["a"]))
    obj.model.assert_not_called()


@pytest.mark.parametrize("mismatch", [None, 0, 1023, 1024, 2047, 4095, 4096])
@pytest.mark.parametrize("required", [1, 1024, 4096])
def test_chunked_prefix_comparison_preserves_exact_boundary(mismatch, required):
    cache = load_class("backend.py", "IntermediateKVCache", dict(OrderedDict=OrderedDict))(1)
    sequence = list(range(4097))
    slots, _ = cache.plan(["a"], [sequence], [4096])
    cache.commit(slots, [sequence])
    if mismatch is not None:
        sequence[mismatch] = -1
    assert cache.query_start("a", sequence, required) == min(required, mismatch if mismatch is not None else required)
    assert cache.query_start("new", sequence, required) == 0


def test_commit_appends_only_after_plan_invalidates_rejected_tail():
    cache = load_class("backend.py", "IntermediateKVCache", dict(OrderedDict=OrderedDict))(1)
    sequence = list(range(4097))
    slots, _ = cache.plan(["a"], [sequence], [4096])
    cache.commit(slots, [sequence])
    storage = cache.tokens[0]
    changed = sequence[:4090] + [-1, -2]
    slots, starts = cache.plan(["a"], [changed], [4091])
    assert starts == [4090]
    assert storage == sequence[:4090]
    cache.commit(slots, [changed])
    assert cache.tokens[0] is storage
    assert storage == changed
    changed[-1] = 99
    assert storage[-1] == -2
