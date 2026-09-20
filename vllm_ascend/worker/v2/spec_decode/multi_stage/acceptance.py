# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AcceptancePolicy:
    """Approximate acceptance shared by intermediate and final verification."""

    method: str = "topk"
    top_k: int = 5

    def __post_init__(self):
        if self.method not in ("topk", "all"):
            raise ValueError("verification method must be 'topk' or 'all'")
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("verification top_k must be a positive integer")

    def accept(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if self.method == "all":
            return torch.ones_like(token_ids, dtype=torch.bool)
        top_ids = logits.topk(min(self.top_k, logits.shape[-1]), dim=-1).indices
        # Masked logits must never pass just because k exceeds the allowed set.
        allowed = torch.isfinite(logits.gather(-1, token_ids.long().unsqueeze(-1)).squeeze(-1))
        return (top_ids == token_ids.unsqueeze(-1)).any(dim=-1) & allowed


def assemble_verified_tokens(
    logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    target_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
    policy: AcceptancePolicy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack each accepted prefix and its target replacement/bonus, on device.

    MRV2 inputs contain one context token followed by draft tokens per request;
    each logit predicts the following token, with the final row for the bonus.
    """
    steps = torch.arange(num_speculative_steps + 1, device=logits.device)
    lengths = cu_num_logits[1:] - cu_num_logits[:-1]
    if policy.method == "all":
        # Every request terminates at its bonus row. No vocabulary scan,
        # acceptance mask or first-rejection reduction is needed.
        stop = lengths - 1
        rows = (cu_num_logits[:-1, None] + steps + 1).clamp(max=draft_sampled.shape[0] - 1).long()
        bonus = target_sampled[(cu_num_logits[1:] - 1).long()]
        sampled = torch.where(steps < stop[:, None], draft_sampled[rows], bonus[:, None])
        sampled.masked_fill_(steps > stop[:, None], -1)
        return sampled.long(), lengths.to(torch.int32)
    draft_next = draft_sampled.roll(-1)
    accepted = policy.accept(logits, draft_next)
    rows = (cu_num_logits[:-1, None] + steps).clamp(max=logits.shape[0] - 1).long()
    is_draft = steps < lengths[:, None] - 1
    # Bonus positions also terminate the prefix, including requests with no draft.
    stop = torch.where(is_draft & accepted[rows], num_speculative_steps + 1, steps).amin(dim=1)
    sampled = torch.where(steps < stop[:, None], draft_next[rows], target_sampled[rows])
    sampled = sampled.masked_fill(steps > stop[:, None], -1)
    return sampled.long(), (stop + 1).to(torch.int32)
