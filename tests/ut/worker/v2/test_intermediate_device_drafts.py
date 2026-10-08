# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Secondary candidates stay on device until the verifier decision transfer."""

import sys
from types import SimpleNamespace as NS

import pytest
import torch
from test_intermediate import modules as modules
from test_intermediate_backend import backend as backend


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


def test_device_microbatches_preserve_reused_decision_output(backend, modules):
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
    pipe = Pipeline(obj, config.IntermediateConfig("v", "d", num_rounds=1, verification={"method": "all"}), 3)
    prefixes = [list(range(1, 8)), list(range(11, 16))]
    result = pipe.refine(prefixes, [torch.tensor([8]), torch.tensor([16])], [3, 3], ["a", "b"])
    assert obj.model.call_count == 2  # 14 tokens exceed the 12-token microbatch budget.
    assert result == [[8, 9], [16, 17]]
    assert obj.cache.tokens == [list(range(1, 9)), list(range(11, 17))]


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
