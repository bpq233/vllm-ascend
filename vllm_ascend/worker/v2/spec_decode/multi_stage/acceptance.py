# SPDX-License-Identifier: Apache-2.0
"""Approximate acceptance policies, independent of model and scheduler state."""

from typing import Protocol

import torch
from torch import Tensor


class AcceptancePolicy(Protocol):
    def accept(self, logits: Tensor, draft_tokens: Tensor) -> Tensor:
        """Return a boolean mask for aligned logits [M,V] and tokens [M]."""
        ...


def _validate(logits: Tensor, draft_tokens: Tensor) -> None:
    if logits.ndim != 2 or draft_tokens.ndim != 1 or logits.shape[0] != draft_tokens.numel():
        raise ValueError("Expected aligned logits [M,V] and draft_tokens [M]")
    if logits.shape[1] == 0:
        raise ValueError("Vocabulary must be nonempty")
    if logits.device != draft_tokens.device:
        raise ValueError("Logits and draft tokens must use the same device")
    if draft_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Draft tokens must have an integer dtype")


class TopKPolicy:
    """Membership in the model's top k; this is not exact rejection sampling."""

    def __init__(self, top_k: int):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.top_k = top_k

    def accept(self, logits: Tensor, draft_tokens: Tensor) -> Tensor:
        _validate(logits, draft_tokens)
        indices = logits.topk(min(self.top_k, logits.shape[-1]), dim=-1).indices
        in_range = (draft_tokens >= 0) & (draft_tokens < logits.shape[-1])
        selected = logits.gather(1, draft_tokens.clamp(0, logits.shape[-1] - 1).long()[:, None]).squeeze(1)
        return in_range & torch.isfinite(selected) & (indices == draft_tokens[:, None]).any(dim=-1)


class AcceptAllPolicy:
    def accept(self, logits: Tensor, draft_tokens: Tensor) -> Tensor:
        _validate(logits, draft_tokens)
        # Full acceptance still respects hard masks and rejects invalid IDs.
        in_range = (draft_tokens >= 0) & (draft_tokens < logits.shape[-1])
        selected = logits.gather(1, draft_tokens.clamp(0, logits.shape[-1] - 1).long()[:, None]).squeeze(1)
        return in_range & torch.isfinite(selected)


def accepted_prefix_length(mask: Tensor) -> int:
    """Only the contiguous leading accepted positions can extend a context."""
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("Acceptance policy must return a one-dimensional boolean mask")
    return int(mask.to(torch.int32).cumprod(dim=0).sum().item())
