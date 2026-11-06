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
