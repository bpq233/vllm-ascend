# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Secondary candidates stay on device until the verifier decision transfer."""

import sys
from time import perf_counter
from types import SimpleNamespace as NS

import pytest
import torch
from test_intermediate import modules as modules
from test_intermediate_backend import backend as backend
from test_intermediate_backend import load_class


@pytest.mark.parametrize("method", ["topk", "all", "prob_ratio"])
def test_device_pipeline_matches_host_and_merges_transfers(backend, modules, monkeypatch, method):
    obj, _, _ = backend
    config, Pipeline = modules
    acceptance = sys.modules["vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance"]
    policy = acceptance.AcceptancePolicy(method, 1)
    obj.model.compute_logits = lambda h: torch.nn.functional.one_hot(h[:, 0].long() + 1, 32).float() * 20
    obj.decision_runner = acceptance.IntermediateDecisionRunner(obj.model.compute_logits, policy, 3)
    obj.drafter.propose.return_value = torch.tensor([[4, 5], [14, 15]])
    options = config.IntermediateConfig("v", "d", num_rounds=2, verification=vars(policy))
    pipe = Pipeline(obj, options, capacity=6)
    transfers = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    result = pipe.refine([[1], [11]], [[2], [12]], [6, 6], ["a", "b"])
    assert result == [[2, 3, 4, 5, 6], [12, 13, 14, 15, 16]]
    assert transfers == [(2, 2), (8,)]  # decisions, then decisions + device drafts
    assert obj.cache.tokens[obj.cache.slots["a"]] == [1, 2, 3, 4, 5]
    assert obj.cache.tokens[obj.cache.slots["b"]] == [11, 12, 13, 14, 15]
    # A final-target rejection must overwrite the speculative tail on the next cycle.
    list(obj.verify([[1, 2, 9]], [[10]], req_ids=["a"]))
    assert obj.cache.tokens[obj.cache.slots["a"]] == [1, 2, 9, 10]


def test_device_proposals_own_storage_and_do_not_read_host(backend, monkeypatch):
    obj, _, _ = backend
    output = obj.drafter.propose.return_value
    original = torch.Tensor.tolist

    def tolist(tensor):
        if tensor.ndim == 2:
            pytest.fail("Secondary must not read device tokens")
        return original(tensor)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", tolist)
        result = obj.propose_device([[1, 2, 3], [4, 5]], ["a", "b"])
    output.fill_(0)  # Simulate a later replay overwriting DFlash output.
    assert [r.tolist() for r in result] == [[9, 10], [11, 12]]


def test_device_verifier_uploads_only_prefix_and_delays_cache_commit(backend):
    obj, metadata, _ = backend
    captured = []
    decision = NS(policy=NS(method="topk"))

    class Decision:
        policy = decision.policy

        def __call__(self, hidden, tokens, lengths):
            captured.append((hidden.clone(), tokens.clone(), lengths))
            return torch.zeros(len(lengths), 2, dtype=torch.int64)

    drafts = [torch.tensor([3, 4]), torch.empty(0, dtype=torch.int64)]
    list(obj.verify_batches([[1, 2], [7]], drafts, ["a", "b"], decision=Decision()))
    assert captured[0][0].tolist() == [[2, 1], [3, 2], [4, 3], [7, 0]]
    assert captured[0][2] == [3, 1]
    # Neither placeholder zeros nor unconfirmed device token IDs become valid KV.
    assert obj.cache.tokens == [[], []]
    obj.commit_verified_drafts([[1, 2], [7]], [[3, 4], []], ["a", "b"])
    assert obj.cache.tokens == [[1, 2, 3, 4], [7]]
    assert obj.input_buffers.input_ids[:5].tolist() == [1, 2, 3, 4, 7]
    # Packed metadata contains 3 host IDs, not all 5 input IDs; the 2 draft
    # IDs arrived through device copies. The metadata holds the H2D storage.
    assert metadata.call_args.kwargs["seq_lens"].untyped_storage().nbytes() == 144


def test_device_microbatches_preserve_reused_decision_output(backend, modules, monkeypatch):
    obj, _, _ = backend
    config, Pipeline = modules

    class Decision:
        graph_enabled = True
        policy = NS(method="all")
        output = torch.empty(1, 2, dtype=torch.int64)

        def __call__(self, hidden, tokens, lengths):
            self.output[0, 0] = lengths[0] - 1
            self.output[0, 1] = hidden[-1, 0] + 1
            return self.output

    obj.decision_runner = Decision()
    clones = []
    original_clone = torch.Tensor.clone

    def clone(tensor, *args, **kwargs):
        if tensor.data_ptr() == obj.decision_runner.output.data_ptr():
            clones.append(tensor.shape)
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    pipe = Pipeline(obj, config.IntermediateConfig("v", "d", num_rounds=1, verification={"method": "all"}), 3)
    prefixes = [list(range(1, 8)), list(range(11, 16))]
    output = torch.full((2, 3), -99, dtype=torch.int64)
    pointer = output.data_ptr()
    result = pipe.refine(prefixes, [torch.tensor([8]), torch.tensor([16])], [3, 3], ["a", "b"], output=output)
    assert obj.model.call_count == 2  # 14 tokens exceed the 12-token microbatch budget.
    assert result == [[8, 9], [16, 17]]
    assert len(clones) == 1  # Keep the first output; consume the last in place.
    assert obj.cache.tokens == [list(range(1, 9)), list(range(11, 17))]
    obj.decision_runner.output.fill_(-99)
    assert pipe.device_output_ready and output.data_ptr() == pointer
    assert output.tolist() == [[8, 9, 0], [16, 17, 0]]


@pytest.mark.parametrize("method", ["topk", "all", "prob_ratio"])
@pytest.mark.parametrize("eos", [None, 5, 6, 15])
@pytest.mark.parametrize("threshold", [None, 0, 3])
def test_device_publication_matches_scheduler_lists_across_rounds_and_limits(backend, modules, method, eos, threshold):
    obj, _, _ = backend
    config, Pipeline = modules
    acceptance = sys.modules["vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance"]
    policy = acceptance.AcceptancePolicy(method, 1)
    obj.model.compute_logits = lambda h: torch.nn.functional.one_hot(h[:, 0].long() + 1, 32).float() * 20
    obj.decision_runner = acceptance.IntermediateDecisionRunner(obj.model.compute_logits, policy, 3)
    obj.drafter.propose.side_effect = [torch.tensor([[4, 9], [14, 15]]), torch.tensor([[7, 0], [17, 18]])]
    options = config.IntermediateConfig(
        "v", "d", num_rounds=2, verification=vars(policy), max_generated_tokens=threshold
    )
    capacity = config.candidate_capacity(2, options)
    pipe = Pipeline(obj, options, capacity, eos_token_id=eos)
    output = torch.full((3, capacity), -99, dtype=torch.int64)
    # Third request is partial prefill/inactive; the second has less room.
    result = pipe.refine([[1, 2, 3], [11, 12, 13], [21]], None, [6, 4, 0], ["a", "b", "c"], output=output)
    assert pipe.device_output_ready
    assert output.tolist() == [row + [0] * (capacity - len(row)) for row in result]
    assert result[2] == []
    # Packet views must survive graph-output reuse, then release after publish.
    obj.drafter.propose.side_effect = None
    obj.drafter.propose.return_value.fill_(-99)
    assert output.tolist() == [row + [0] * (capacity - len(row)) for row in result]


def test_host_draft_publication_falls_back_without_touching_device_output(modules):
    from test_intermediate import Backend

    config, Pipeline = modules
    pipe = Pipeline(Backend([[[2, 3]]]), config.IntermediateConfig("v", "d", num_rounds=1), 3)
    output = torch.full((1, 3), -99, dtype=torch.int64)
    assert pipe.refine([[1]], [[2]], [3], ["a"], output=output) == [[2, 3]]
    assert not pipe.device_output_ready
    assert output.tolist() == [[-99, -99, -99]]


def test_device_publication_retains_original_row_order_across_resident_groups(backend, modules):
    obj, _, _ = backend
    config, Pipeline = modules
    acceptance = sys.modules["vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance"]
    obj.max_num_reqs = 1
    obj.model.compute_logits = lambda h: torch.nn.functional.one_hot(h[:, 0].long() + 1, 32).float() * 20
    obj.decision_runner = acceptance.IntermediateDecisionRunner(
        obj.model.compute_logits, acceptance.AcceptancePolicy("all"), 3
    )
    obj.drafter.propose_precomputed = lambda batch, anchors, *args: anchors[:1, None] + torch.tensor([[1, 2]])
    slots, _ = obj.cache.plan(["b"], [[11, 12]], [1])
    obj.cache.commit(slots, [[11, 12]])
    pipe = Pipeline(obj, config.IntermediateConfig("v", "d", num_rounds=2, verification={"method": "all"}), 6)
    output = torch.empty(3, 6, dtype=torch.int64)
    results = pipe.refine([[1, 2, 3], [11, 12, 13], [21, 22, 23]], None, [6, 6, 4], ["a", "b", "c"], output=output)
    assert results == [[4, 5, 6, 7, 8, 9], [14, 15, 16, 17, 18, 19], [24, 25, 26, 27]]
    assert obj.cache.capacity < len(results)  # Earlier group KV gets evicted.
    assert pipe.device_output_ready
    assert output.tolist() == [row + [0] * (6 - len(row)) for row in results]


@pytest.mark.parametrize("method", ["topk", "all", "prob_ratio"])
@pytest.mark.parametrize("primary_width", [1, 3])
@pytest.mark.parametrize("partial_prefill", [False, True])
def test_four_model_adapter_keeps_primary_on_device_and_publishes_without_h2d(
    backend, modules, monkeypatch, method, primary_width, partial_prefill
):
    back, _, _ = backend
    config, Pipeline = modules
    acceptance = sys.modules["vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance"]
    policy = acceptance.AcceptancePolicy(method, 1)
    primary = torch.tensor([list(range(2, 2 + primary_width)), list(range(12, 12 + primary_width))])
    if primary_width > 1:
        primary[0, 1] = 6  # First request rejects this token under topk/prob_ratio.

    class PrimaryDFlash:
        def propose(self, *args, **kwargs):
            return primary

    cls = load_class(
        "adapter.py",
        "MultiStageDFlashSpeculator",
        dict(
            torch=torch,
            AscendDFlashSpeculator=PrimaryDFlash,
            DraftTokenIds=lambda ids, tokens: (ids, tokens),
            perf_counter=perf_counter,
        ),
    )
    obj = cls.__new__(cls)
    options = config.IntermediateConfig("v", "d", num_rounds=2, verification=vars(policy))
    obj.final_capacity = config.candidate_capacity(primary_width, options)
    obj.max_model_len, obj.device = 8, torch.device("cpu")
    back.model.compute_logits = lambda h: torch.nn.functional.one_hot(h[:, 0].long() + 1, 32).float() * 20
    back.decision_runner = acceptance.IntermediateDecisionRunner(
        back.model.compute_logits, policy, max(primary_width, 2) + 1
    )
    back.drafter.propose.return_value = torch.tensor([[4, 5], [15, 16]])
    obj.pipeline = Pipeline(back, options, obj.final_capacity)
    history = torch.zeros(2, 8, dtype=torch.int64)
    history[:, 0] = torch.tensor([1, 11])
    committed = history.clone()
    obj.req_states = NS(
        total_len=NS(gpu=torch.tensor([1, 1])),
        num_computed_tokens=NS(gpu=torch.tensor([0, 0])),
        num_computed_tokens_cpu=torch.zeros(2, dtype=torch.int32),
        all_token_ids=NS(gpu=history),
        max_seq_len=torch.tensor([8, 8]).numpy(),
        prefill_len=NS(np=torch.tensor([0, int(partial_prefill)]).numpy()),
        req_id_to_index={"a": 0, "b": 1},
    )
    obj._host_histories = {"a": [1], "b": [11]}
    batch = NS(
        num_reqs=2, idx_mapping=torch.tensor([0, 1]), idx_mapping_np=torch.tensor([0, 1]).numpy(), req_ids=["a", "b"]
    )
    transfers = []
    original_cpu = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    output = obj.propose(batch)
    ids, candidates = obj.get_draft_tokens()
    assert ids == ["a", "b"]
    assert obj.pipeline.device_output_ready and not hasattr(obj, "_host_candidates")
    assert output.tolist() == [row + [0] * (obj.final_capacity - len(row)) for row in candidates]
    assert transfers[0] == (2, 2 + min(obj.final_capacity + 1, history.shape[1]))  # No Primary IDs in history D2H.
    assert len(transfers) == 3  # History/progress, first decision+Primary, second decision+Secondary.
    assert transfers[1] == ((2 + primary_width) * (1 if partial_prefill else 2),)
    if method != "all":
        assert candidates[0] == [2, 3, 4, 5, 6]
    if partial_prefill:
        assert candidates[1] == []
    torch.testing.assert_close(history, committed)
    primary.fill_(-99)  # Simulate the next Primary graph replay.
    assert output.tolist() == [row + [0] * (obj.final_capacity - len(row)) for row in candidates]


def test_failed_device_verification_keeps_only_valid_prefix(backend):
    obj, _, _ = backend
    list(obj.verify([[1, 2]], [[3]], req_ids=["a"]))
    obj.drafter.model.precompute_and_store_context_kv.side_effect = RuntimeError("KV write failed")
    with pytest.raises(RuntimeError, match="KV write failed"):
        obj._forward([[1, 2, 0, 0]], ["a"], [1], draft_tokens=[torch.tensor([9, 10])])
    assert obj.cache.tokens[obj.cache.slots["a"]] == [1]


@pytest.mark.parametrize("ids", [["a", "b", "c"], ["a", "a"], ["a"]])
def test_device_verification_rejects_nonresident_groups_before_forward(backend, ids):
    obj, _, _ = backend
    count = max(2, len(ids))
    with pytest.raises(ValueError, match="resident cache capacity"):
        list(obj.verify_batches([[1]] * count, [torch.tensor([2])] * count, ids, decision=object()))
    obj.model.assert_not_called()
    assert not obj.cache.slots
