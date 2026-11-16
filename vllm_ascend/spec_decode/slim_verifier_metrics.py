# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import logging
from dataclasses import dataclass

import torch

COUNTER_NAMES = (
    "target_prefix",
    "slim_prefix",
    "target_matches",
    "slim_matches",
    "target_full",
    "slim_full",
    "prediction_agreement",
    "decision_agreement",
    "slim_only",
    "target_only",
)
SUM_NAMES = (
    "target_logprob",
    "slim_logprob",
    "target_confidence",
    "slim_confidence",
    "target_rank",
    "slim_rank",
    "top_k_overlap",
)


@dataclass
class VerificationResult:
    """Per-position diagnostics, never passed to a sampler or scheduler."""

    draft_tokens: torch.Tensor
    target_predictions: torch.Tensor
    slim_predictions: torch.Tensor
    target_accept_mask: torch.Tensor
    slim_accept_mask: torch.Tensor
    request_id: str | None = None
    prediction_positions: torch.Tensor | None = None


class SlimVerifierMetrics:
    """Device-side window/cumulative reductions; copy only at log boundaries."""

    def __init__(self, log_interval: int = 100, top_k: int = 5, logger=None):
        if log_interval <= 0 or top_k <= 0:
            raise ValueError("log_interval and top_k must be positive")
        self.log_interval = log_interval
        self.top_k = top_k
        self.logger = logger or logging.getLogger(__name__)
        self.blocks = self.tokens = self.window_blocks = self.window_tokens = 0
        self.counts = self.sums = self.window_counts = self.window_sums = None
        self.slim_ms = self.window_slim_ms = 0.0
        self.timed_blocks = self.window_timed_blocks = 0

    @torch.inference_mode()
    def record(self, draft_tokens, target_logits, slim_logits, slim_ms: float | None = None) -> VerificationResult:
        n = draft_tokens.numel()
        if n == 0 or draft_tokens.ndim != 1:
            raise ValueError("A verification block must have at least one valid draft token")
        if target_logits.shape != slim_logits.shape or target_logits.ndim != 2 or target_logits.shape[0] != n:
            raise ValueError("Logits must have matching [draft_tokens, global_vocab] shapes")
        target = target_logits.float()
        slim = slim_logits.float()
        target_pred, slim_pred = target.argmax(-1), slim.argmax(-1)
        target_mask, slim_mask = target_pred.eq(draft_tokens), slim_pred.eq(draft_tokens)
        # cumprod stops counting at the FIRST mismatch, unlike sum(mask).
        target_prefix = target_mask.long().cumprod(0).sum()
        slim_prefix = slim_mask.long().cumprod(0).sum()
        counts = torch.stack(
            (
                target_prefix,
                slim_prefix,
                target_mask.sum(),
                slim_mask.sum(),
                target_mask.all().long(),
                slim_mask.all().long(),
                target_pred.eq(slim_pred).sum(),
                target_mask.eq(slim_mask).sum(),
                (slim_mask & ~target_mask).sum(),
                (target_mask & ~slim_mask).sum(),
            )
        )
        target_draft = target.gather(1, draft_tokens.long().unsqueeze(1)).squeeze(1)
        slim_draft = slim.gather(1, draft_tokens.long().unsqueeze(1)).squeeze(1)
        target_lse, slim_lse = target.logsumexp(-1), slim.logsumexp(-1)
        k = min(self.top_k, target.shape[1])
        target_top = target.topk(k, dim=-1).indices
        slim_top = slim.topk(k, dim=-1).indices
        overlap = (target_top.unsqueeze(-1) == slim_top.unsqueeze(-2)).any(-1).float().sum() / k
        sums = torch.stack(
            (
                (target_draft - target_lse).sum(),
                (slim_draft - slim_lse).sum(),
                (target.max(-1).values - target_lse).exp().sum(),
                (slim.max(-1).values - slim_lse).exp().sum(),
                (1 + (target > target_draft.unsqueeze(1)).sum(-1)).float().sum(),
                (1 + (slim > slim_draft.unsqueeze(1)).sum(-1)).float().sum(),
                overlap,
            )
        )
        if self.counts is None:
            self.counts, self.sums = torch.zeros_like(counts), torch.zeros_like(sums)
            self.window_counts, self.window_sums = torch.zeros_like(counts), torch.zeros_like(sums)
        self.counts += counts
        self.sums += sums
        self.window_counts += counts
        self.window_sums += sums
        self.blocks += 1
        self.tokens += n
        self.window_blocks += 1
        self.window_tokens += n
        if slim_ms is not None:
            self.slim_ms += slim_ms
            self.window_slim_ms += slim_ms
            self.timed_blocks += 1
            self.window_timed_blocks += 1
        if self.window_blocks == self.log_interval:
            self.flush()
        return VerificationResult(draft_tokens, target_pred, slim_pred, target_mask, slim_mask)

    def snapshot(self, cumulative: bool = True) -> dict[str, float | int]:
        blocks = self.blocks if cumulative else self.window_blocks
        tokens = self.tokens if cumulative else self.window_tokens
        result = {"blocks": blocks, "draft_tokens": tokens}
        if blocks == 0:
            return result
        counts = self.counts if cumulative else self.window_counts
        sums = self.sums if cumulative else self.window_sums
        # A compact copy at the window boundary; no per-token scalar reads.
        c = dict(zip(COUNTER_NAMES, counts.cpu().tolist()))
        s = dict(zip(SUM_NAMES, sums.cpu().tolist()))
        for model in ("target", "slim"):
            result.update(
                {
                    f"{model}_avg_accepted_length": c[f"{model}_prefix"] / blocks,
                    f"{model}_prefix_acceptance_rate": c[f"{model}_prefix"] / tokens,
                    f"{model}_token_match_rate": c[f"{model}_matches"] / tokens,
                    f"{model}_full_block_acceptance_rate": c[f"{model}_full"] / blocks,
                    f"{model}_avg_draft_logprob": s[f"{model}_logprob"] / tokens,
                    f"{model}_avg_top1_confidence": s[f"{model}_confidence"] / tokens,
                    f"{model}_avg_draft_rank": s[f"{model}_rank"] / tokens,
                }
            )
        for key in ("prediction_agreement", "decision_agreement", "slim_only", "target_only"):
            result[f"{key}_rate"] = c[key] / tokens
        result["mean_logprob_gap"] = (s["target_logprob"] - s["slim_logprob"]) / tokens
        result["top_k_overlap"] = s["top_k_overlap"] / tokens
        timed = self.timed_blocks if cumulative else self.window_timed_blocks
        if timed:
            result["slim_replay_ms_per_block"] = (self.slim_ms if cumulative else self.window_slim_ms) / timed
        return result

    @torch.inference_mode()
    def flush(self) -> None:
        """Also callable by experiments to publish a final partial window."""
        if self.window_blocks == 0:
            return
        self.logger.info("[SlimVerifier] Top-1 Diagnostic (raw model logits; not sampling acceptance)")
        self.logger.info("[SlimVerifier] Window: %s", self.snapshot(cumulative=False))
        self.logger.info("[SlimVerifier] Cumulative: %s", self.snapshot())
        self.window_blocks = self.window_tokens = self.window_timed_blocks = 0
        self.window_slim_ms = 0.0
        self.window_counts.zero_()
        self.window_sums.zero_()
