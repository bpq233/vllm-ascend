# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import logging
import math
from dataclasses import dataclass

import torch

from vllm_ascend.spec_decode.slim_verifier_config import validate_acceptance_strategy

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
    "slim_accepted",
    "target_first",
    "slim_first",
)
SUM_NAMES = (
    "target_logprob",
    "slim_logprob",
    "target_confidence",
    "slim_confidence",
    "target_rank",
    "slim_rank",
    "top_k_overlap",
    "logits_mean_abs_diff",
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
    slim_top1_match_mask: torch.Tensor | None = None
    slim_probability_ratios: torch.Tensor | None = None
    slim_acceptance_strategy: str = "top1"


class SlimVerifierMetrics:
    """Device-side window/cumulative reductions; copy only at log boundaries."""

    def __init__(
        self,
        log_interval: int = 100,
        top_k: int = 5,
        logger=None,
        *,
        acceptance_strategy: str = "top1",
        probability_ratio_threshold: float = 0.5,
    ):
        if log_interval <= 0 or top_k <= 0:
            raise ValueError("log_interval and top_k must be positive")
        self.log_interval = log_interval
        self.top_k = top_k
        validate_acceptance_strategy(acceptance_strategy, probability_ratio_threshold)
        self.acceptance_strategy = acceptance_strategy
        self.probability_ratio_threshold = probability_ratio_threshold
        self.log_probability_ratio_threshold = (
            math.log(probability_ratio_threshold) if probability_ratio_threshold > 0 else -math.inf
        )
        self.logger = logger or logging.getLogger(__name__)
        self.blocks = self.tokens = self.window_blocks = self.window_tokens = 0
        self.counts = self.sums = self.window_counts = self.window_sums = None
        self.max_logits_diff = self.window_max_logits_diff = None
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
        target_mask, slim_top1_mask = target_pred.eq(draft_tokens), slim_pred.eq(draft_tokens)
        target_draft = target.gather(1, draft_tokens.long().unsqueeze(1)).squeeze(1)
        slim_draft = slim.gather(1, draft_tokens.long().unsqueeze(1)).squeeze(1)
        target_max, slim_max = target.max(-1).values, slim.max(-1).values
        slim_ratios = None
        if self.acceptance_strategy == "probability_ratio":
            # Softmax normalizers cancel: q(x)/q(max) = exp(z_x - z_max).
            # Compare in log space, with a STRICT boundary. Finite, very small
            # probabilities still pass threshold=0 even if exp underflows.
            log_ratios = slim_draft - slim_max
            slim_mask = torch.isfinite(log_ratios) & (log_ratios > self.log_probability_ratio_threshold)
            slim_ratios = log_ratios.exp()
        else:
            slim_mask = slim_top1_mask
        # cumprod stops counting at the FIRST mismatch, unlike sum(mask).
        target_prefix = target_mask.long().cumprod(0).sum()
        slim_prefix = slim_mask.long().cumprod(0).sum()
        counts = torch.stack(
            (
                target_prefix,
                slim_prefix,
                target_mask.sum(),
                slim_top1_mask.sum(),
                target_mask.all().long(),
                slim_mask.all().long(),
                target_pred.eq(slim_pred).sum(),
                target_mask.eq(slim_mask).sum(),
                (slim_mask & ~target_mask).sum(),
                (target_mask & ~slim_mask).sum(),
                slim_mask.sum(),
                target_mask[0].long(),
                slim_mask[0].long(),
            )
        )
        target_lse, slim_lse = target.logsumexp(-1), slim.logsumexp(-1)
        k = min(self.top_k, target.shape[1])
        target_top = target.topk(k, dim=-1).indices
        slim_top = slim.topk(k, dim=-1).indices
        overlap = (target_top.unsqueeze(-1) == slim_top.unsqueeze(-2)).any(-1).float().sum() / k
        # Identical masked entries (-inf, -inf) contribute zero, not NaN.
        logits_diff = torch.where(target == slim, 0.0, (target - slim).abs())
        max_logits_diff = logits_diff.max()
        sums = torch.stack(
            (
                (target_draft - target_lse).sum(),
                (slim_draft - slim_lse).sum(),
                (target_max - target_lse).exp().sum(),
                (slim_max - slim_lse).exp().sum(),
                (1 + (target > target_draft.unsqueeze(1)).sum(-1)).float().sum(),
                (1 + (slim > slim_draft.unsqueeze(1)).sum(-1)).float().sum(),
                overlap,
                logits_diff.mean(-1).sum(),
            )
        )
        if self.counts is None:
            self.counts, self.sums = torch.zeros_like(counts), torch.zeros_like(sums)
            self.window_counts, self.window_sums = torch.zeros_like(counts), torch.zeros_like(sums)
            self.max_logits_diff = torch.zeros_like(max_logits_diff)
            self.window_max_logits_diff = torch.zeros_like(max_logits_diff)
        self.max_logits_diff = torch.maximum(self.max_logits_diff, max_logits_diff)
        self.window_max_logits_diff = torch.maximum(self.window_max_logits_diff, max_logits_diff)
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
        return VerificationResult(
            draft_tokens,
            target_pred,
            slim_pred,
            target_mask,
            slim_mask,
            slim_top1_match_mask=slim_top1_mask,
            slim_probability_ratios=slim_ratios,
            slim_acceptance_strategy=self.acceptance_strategy,
        )

    def snapshot(self, cumulative: bool = True) -> dict[str, float | int | str]:
        blocks = self.blocks if cumulative else self.window_blocks
        tokens = self.tokens if cumulative else self.window_tokens
        result = {"blocks": blocks, "draft_tokens": tokens}
        if blocks == 0:
            return result
        result["slim_acceptance_strategy"] = self.acceptance_strategy
        if self.acceptance_strategy == "probability_ratio":
            result["probability_ratio_threshold"] = self.probability_ratio_threshold
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
                    f"{model}_first_token_acceptance_rate": c[f"{model}_first"] / blocks,
                    f"{model}_avg_draft_logprob": s[f"{model}_logprob"] / tokens,
                    f"{model}_avg_top1_confidence": s[f"{model}_confidence"] / tokens,
                    f"{model}_avg_draft_rank": s[f"{model}_rank"] / tokens,
                }
            )
        for key in ("prediction_agreement", "decision_agreement", "slim_only", "target_only"):
            result[f"{key}_rate"] = c[key] / tokens
        result["slim_token_acceptance_rate"] = c["slim_accepted"] / tokens
        result["mean_logprob_gap"] = (s["target_logprob"] - s["slim_logprob"]) / tokens
        result["top_k_overlap"] = s["top_k_overlap"] / tokens
        result["target_slim_logits_mean_abs_diff"] = s["logits_mean_abs_diff"] / tokens
        maximum = self.max_logits_diff if cumulative else self.window_max_logits_diff
        result["target_slim_logits_max_abs_diff"] = maximum.cpu().item()
        timed = self.timed_blocks if cumulative else self.window_timed_blocks
        if timed:
            result["slim_replay_ms_per_block"] = (self.slim_ms if cumulative else self.window_slim_ms) / timed
        return result

    @torch.inference_mode()
    def flush(self) -> None:
        """Also callable by experiments to publish a final partial window."""
        if self.window_blocks == 0:
            return
        if self.acceptance_strategy == "top1":
            self.logger.info("[SlimVerifier] Top-1 Diagnostic (raw model logits; not sampling acceptance)")
        else:
            self.logger.info(
                "[SlimVerifier] Shadow Diagnostic (Target: top1; Slim: probability_ratio; "
                "q(x)/q(max) > %s; raw model logits; not sampling acceptance)",
                self.probability_ratio_threshold,
            )
        self.logger.info("[SlimVerifier] Window: %s", self.snapshot(cumulative=False))
        self.logger.info("[SlimVerifier] Cumulative: %s", self.snapshot())
        self.window_blocks = self.window_tokens = self.window_timed_blocks = 0
        self.window_slim_ms = 0.0
        self.window_counts.zero_()
        self.window_sums.zero_()
        self.window_max_logits_diff.zero_()
