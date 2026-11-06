# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Device-logit access at the actual sequential proposal context."""

from unittest.mock import Mock

import pytest
import torch
from test_intermediate_backend import backend as backend


def test_next_token_logits_follow_the_actual_prefix_and_request_order(backend):
    obj, _, _ = backend
    logits = obj.proposal_logits([[1, 2, 3], [4, 5]], ["a", "b"])
    # The model stand-in encodes the input token and absolute position.
    torch.testing.assert_close(logits, torch.tensor([[3.0, 2.0], [5.0, 1.0]]), rtol=0, atol=0)
    assert logits.device == obj.device
    assert obj.cache.tokens[obj.cache.slots["a"]] == [1, 2, 3]
    assert obj.cache.tokens[obj.cache.slots["b"]] == [4, 5]
    assert all(row is None for row in obj._context_rows)

    # A residual correction changes the prefix, despite identical ID/length.
    corrected = obj.proposal_logits([[4, 7], [1, 2, 6]], ["b", "a"])
    torch.testing.assert_close(corrected, torch.tensor([[7.0, 1.0], [6.0, 2.0]]), rtol=0, atol=0)
    assert obj.cache.tokens[obj.cache.slots["a"]] == [1, 2, 6]
    assert obj.cache.tokens[obj.cache.slots["b"]] == [4, 7]
    # Earlier returned rows survive the subsequent forward/reorder.
    torch.testing.assert_close(logits, torch.tensor([[3.0, 2.0], [5.0, 1.0]]), rtol=0, atol=0)


def test_probability_rows_survive_microbatch_eviction_and_reused_output_buffers(backend):
    obj, _, _ = backend
    obj.max_num_reqs = 1
    logits = obj.proposal_logits([[1, 2], [3, 4, 5], [6]], ["a", "b", "c"])
    torch.testing.assert_close(logits, torch.tensor([[2.0, 1.0], [5.0, 2.0], [6.0, 0.0]]), rtol=0, atol=0)
    assert "a" not in obj.cache.slots  # Later groups can evict earlier rows.


def test_same_last_token_with_a_different_earlier_context_recomputes_the_suffix(backend):
    obj, _, _ = backend
    obj.proposal_logits([[1, 2, 3]], ["request"])
    obj.model.reset_mock()
    obj.proposal_logits([[1, 7, 3]], ["request"])
    assert obj.model.call_args.kwargs["input_ids"].tolist() == [7, 3]
    assert obj.model.call_args.kwargs["positions"].tolist() == [1, 2]
    assert obj.cache.tokens[obj.cache.slots["request"]] == [1, 7, 3]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_logit_accessor_owns_graph_rows_before_resuming_generator(backend, monkeypatch, dtype):
    obj, _, _ = backend
    storage = torch.empty((1, 3), dtype=dtype)

    def verify(prefixes, drafts, *, req_ids, retain_hidden):
        assert prefixes == [[1], [2]]
        assert drafts == [[], []]
        assert req_ids == ["first", "second"]
        assert retain_hidden is False
        storage.copy_(torch.tensor([[1, 2, 3]], dtype=dtype))
        yield storage, [1]
        storage.copy_(torch.tensor([[4, 5, 6]], dtype=dtype))
        yield storage, [1]
        storage.fill_(99)

    obj.verify_batches = verify
    monkeypatch.setattr(torch.Tensor, "cpu", Mock(side_effect=AssertionError("Full vocabulary D2H")))
    logits = obj.proposal_logits([[1], [2]], ["first", "second"])
    assert logits.dtype == dtype  # This API returns raw logits, not a disguised distribution.
    torch.testing.assert_close(logits, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype), rtol=0, atol=0)
    assert logits.data_ptr() != storage.data_ptr()


@pytest.mark.parametrize(
    "prefixes,req_ids",
    [([], []), ([[1], [2]], ["same", "same"]), ([[1]], []), ([[1], [2]], ["a"])],
)
def test_invalid_request_mappings_fail_before_forward(backend, prefixes, req_ids):
    obj, _, _ = backend
    obj.verify_batches = Mock(side_effect=AssertionError("Unexpected forward"))
    with pytest.raises(ValueError, match="one distinct request ID"):
        obj.proposal_logits(prefixes, req_ids)
    obj.verify_batches.assert_not_called()


@pytest.mark.parametrize("chunks", [[], [(torch.zeros(2, 3), [1])], [(torch.zeros(1, 3), [2])]])
def test_missing_or_non_next_token_rows_are_rejected(backend, chunks):
    obj, _, _ = backend
    obj.verify_batches = Mock(return_value=iter(chunks))
    with pytest.raises(RuntimeError, match="proposal logits"):
        obj.proposal_logits([[1]], ["request"])


def test_empty_context_is_rejected_by_verifier_contract(backend):
    obj, _, _ = backend
    with pytest.raises(ValueError, match="nonempty prefix"):
        obj.proposal_logits([[]], ["request"])


def test_block_rows_follow_teacher_forced_prefix_and_recompute_after_correction(backend):
    obj, _, _ = backend
    rows = obj.proposal_block_logits([[1, 2]], [[3, 4]], ["a"])[0]
    torch.testing.assert_close(rows, torch.tensor([[2.0, 1.0], [3.0, 2.0]]), rtol=0, atol=0)
    assert obj.cache.tokens[obj.cache.slots["a"]] == [1, 2, 3, 4]
    obj.model.reset_mock()
    corrected = obj.proposal_block_logits([[1, 2, 7]], [[6, 5]], ["a"])[0]
    torch.testing.assert_close(corrected, torch.tensor([[7.0, 2.0], [6.0, 3.0]]), rtol=0, atol=0)
    assert obj.model.call_args.kwargs["input_ids"].tolist() == [7, 6, 5]
    assert obj.cache.tokens[obj.cache.slots["a"]] == [1, 2, 7, 6, 5]
    torch.testing.assert_close(rows, torch.tensor([[2.0, 1.0], [3.0, 2.0]]), rtol=0, atol=0)


def test_block_rows_own_storage_and_omit_unrecorded_intermediate_bonus(backend, monkeypatch):
    obj, _, _ = backend
    storage = torch.empty(3, 2)

    def verify(prefixes, drafts, *, req_ids, retain_hidden):
        assert prefixes == [[1], [2]] and drafts == [[1, 2], [3]]
        assert req_ids == ["a", "b"] and retain_hidden is False
        storage.copy_(torch.tensor([[1, 2], [3, 4], [9, 9]]))
        yield storage, [3]
        storage[:2].copy_(torch.tensor([[5, 6], [9, 9]]))
        yield storage[:2], [2]
        storage.fill_(99)

    obj.verify_batches = verify
    monkeypatch.setattr(torch.Tensor, "cpu", Mock(side_effect=AssertionError("Full vocabulary D2H")))
    rows = obj.proposal_block_logits([[1], [2]], [[1, 2], [3]], ["a", "b"])
    torch.testing.assert_close(rows[0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]), rtol=0, atol=0)
    torch.testing.assert_close(rows[1], torch.tensor([[5.0, 6.0]]), rtol=0, atol=0)


@pytest.mark.parametrize(
    "prefixes,drafts,ids",
    [([], [], []), ([[1]], [[]], ["a"]), ([[1]], [], ["a"]), ([[1], [2]], [[3], [4]], ["a", "a"])],
)
def test_invalid_block_logit_mappings_fail_before_forward(backend, prefixes, drafts, ids):
    obj, _, _ = backend
    obj.verify_batches = Mock()
    with pytest.raises(ValueError, match="Block proposal logits"):
        obj.proposal_block_logits(prefixes, drafts, ids)
    obj.verify_batches.assert_not_called()


@pytest.mark.parametrize("chunks", [[], [(torch.zeros(3, 2), [2])], [(torch.zeros(1, 2), [1])]])
def test_invalid_block_prediction_lengths_fail_closed(backend, chunks):
    obj, _, _ = backend
    obj.verify_batches = Mock(return_value=iter(chunks))
    with pytest.raises(RuntimeError, match="Block proposal logits"):
        obj.proposal_block_logits([[1]], [[2]], ["a"])
