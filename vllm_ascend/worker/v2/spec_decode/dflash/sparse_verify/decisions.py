# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Continuous-prefix decisions shared by fixed and adaptive DFlash.

Only forward computations are captured. These tensor decisions run outside
the graphs; controllers may transfer their small results to the host in one
batch. No tensor scalar is read by the intermediate decision implementation.
"""

from dataclasses import dataclass

import torch

from .config import FinalVerificationConfig, SparseVerifyConfig


@dataclass(frozen=True)
class PrefixDecision:
    accepted_length: torch.Tensor
    first_rejection: torch.Tensor
    stop: torch.Tensor
    retry: torch.Tensor
    confidence: torch.Tensor
    accepted_mask: torch.Tensor


@dataclass(frozen=True)
class FinalDecision:
    accepted_length: torch.Tensor
    first_rejection: torch.Tensor
    next_token: torch.Tensor
    accepted_mask: torch.Tensor
    confidence: torch.Tensor


def rejection_logits(logits: torch.Tensor, candidate_token_id: int) -> dict[str, int | float]:
    """Small host diagnostic for one rejected row; call outside graph capture."""
    row = logits.detach().float()
    maximum, maximum_id = row.max(dim=-1)
    candidate = row[candidate_token_id]
    candidate_value, maximum_value, confidence = (
        torch.stack((candidate, maximum, torch.exp(candidate - maximum))).cpu().tolist()
    )
    return {
        "candidate_token_id": candidate_token_id,
        "candidate_logit": candidate_value,
        "max_token_id": int(maximum_id.item()),
        "max_logit": maximum_value,
        "logit_gap": maximum_value - candidate_value,
        "confidence": confidence,
    }


def _inputs(
    logits: torch.Tensor, candidates: torch.Tensor, lengths: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim < 2 or candidates.ndim != logits.ndim - 1 or logits.shape[:-1] != candidates.shape:
        raise ValueError("logits must have shape [..., candidates, vocabulary] matching candidate IDs")
    if logits.shape[-1] < 1 or not logits.is_floating_point():
        raise ValueError("logits must be floating point with a nonempty vocabulary")
    if candidates.dtype not in (torch.int32, torch.int64):
        raise TypeError("candidate IDs must be int32 or int64")
    if candidates.device != logits.device:
        raise ValueError("candidate IDs and logits must share a device")
    if lengths is None:
        lengths = torch.full(candidates.shape[:-1], candidates.shape[-1], dtype=torch.long, device=candidates.device)
    elif lengths.shape != candidates.shape[:-1] or lengths.device != candidates.device:
        raise ValueError("lengths must match candidate batch dimensions and device")
    elif lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("lengths must be int32 or int64")
    active = torch.arange(candidates.shape[-1], device=candidates.device) < lengths.unsqueeze(-1)
    return active, lengths


def _relative_confidence(logits: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Negative infinity is a legitimate masked logit. NaN, positive infinity
    # and rows without any finite entry have no well-defined distribution.
    logits_fp32 = logits.float()
    valid_rows = ~torch.isnan(logits_fp32).any(dim=-1) & ~torch.isposinf(logits_fp32).any(dim=-1)
    valid_rows &= torch.isfinite(logits_fp32).any(dim=-1)
    valid_ids = (candidates >= 0) & (candidates < logits.shape[-1])
    selected = logits_fp32.gather(-1, candidates.long().clamp(0, logits.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    confidence = torch.exp(selected - logits_fp32.amax(dim=-1))
    valid = valid_rows & valid_ids
    return torch.where(valid, confidence, 0.0), valid


def _prefix(keep: torch.Tensor, active: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, ...]:
    prefix = (keep | ~active).long().cumprod(dim=-1).bool() & active
    accepted = prefix.sum(dim=-1)
    rejected = accepted < lengths
    first = torch.where(rejected, accepted, -1)
    return accepted, first, rejected, prefix


class ConfidenceDecision:
    def __init__(self, mode: str = "fixed", threshold_high: float = 0.9, threshold_low: float = 0.5):
        validated = SparseVerifyConfig(mode=mode, threshold_high=threshold_high, threshold_low=threshold_low)
        self.mode = validated.mode
        self.threshold_high = validated.threshold_high
        self.threshold_low = validated.threshold_low

    def evaluate(
        self, logits: torch.Tensor, candidates: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> PrefixDecision:
        """Return only the contiguous accepted prefix of each request.

        ``first_rejection`` is -1 for an entirely accepted block. ``stop`` is
        true only when the *first* rejected token has low confidence or an
        invalid distribution. Later low-confidence positions cannot override
        an earlier middle-confidence retry.
        """
        active, lengths = _inputs(logits, candidates, lengths)
        confidence, valid = _relative_confidence(logits, candidates)
        if self.mode == "fixed":
            keep = valid & (logits.argmax(dim=-1) == candidates)
        else:
            keep = valid & (confidence >= self.threshold_high)
        accepted, first, rejected, prefix = _prefix(keep, active, lengths)
        if candidates.shape[-1] == 0:
            stop = torch.zeros_like(accepted, dtype=torch.bool)
        else:
            index = accepted.clamp_max(candidates.shape[-1] - 1).unsqueeze(-1)
            first_valid = valid.gather(-1, index).squeeze(-1)
            first_confidence = confidence.gather(-1, index).squeeze(-1)
            stop = rejected & ~first_valid
            if self.mode == "confidence":
                stop |= rejected & (first_confidence < self.threshold_low)
        return PrefixDecision(accepted, first, stop, rejected & ~stop, confidence, prefix)


class FinalTargetVerifier:
    """Four policies with one continuous-prefix and correction implementation.

    Strict uses PyTorch's first-index argmax tie rule. Confidence accepts all
    finite tied maxima when its threshold is one. Top-k uses torch.topk's
    membership (ties at the k boundary follow that operator's tie rule).
    Approximate policies intentionally do not preserve target greedy output.
    """

    def __init__(self, config: FinalVerificationConfig | None = None):
        self.config = config or FinalVerificationConfig()

    def verify(
        self, logits: torch.Tensor, candidates: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> FinalDecision:
        """Logits i predicts candidate i; the final row predicts the bonus.

        The caller has already performed the complete causal target forward.
        Accept-all may pass just the per-request bonus row, avoiding candidate
        LM-head work. A correction/bonus is always returned; generation/EOS
        limits are applied by the normal output commit path.
        """
        if logits.ndim != candidates.ndim + 1 or logits.shape[:-2] != candidates.shape[:-1]:
            raise ValueError("target logits and candidates must have matching batch dimensions")
        count = candidates.shape[-1]
        bonus_only = self.config.method == "accept_all" and logits.shape[-2] == 1
        if not bonus_only and logits.shape[-2] != count + 1:
            raise ValueError("target logits must include one row per candidate and a bonus row")
        # Reuse the shape/dtype checks without manufacturing any LM-head rows.
        if bonus_only:
            shape_logits = logits.expand(*candidates.shape, logits.shape[-1])
        else:
            shape_logits = logits[..., :count, :]
        active, lengths = _inputs(shape_logits, candidates, lengths)
        if self.config.method == "accept_all":
            accepted = lengths
            first = torch.full_like(lengths, -1)
            prefix = active
            confidence = torch.ones(candidates.shape, dtype=torch.float32, device=candidates.device)
        else:
            confidence, valid = _relative_confidence(shape_logits, candidates)
            # A target with invalid numerical output cannot supply a sound
            # correction. This explicit check is outside all captured forwards.
            if bool((active & ~valid).any()):
                invalid_ids = active & ((candidates < 0) | (candidates >= logits.shape[-1]))
                rows = shape_logits.float()
                issues = {
                    "invalid_id_positions": invalid_ids.nonzero().cpu().tolist()[:8],
                    "invalid_id_values": candidates[invalid_ids].cpu().tolist()[:8],
                    "nan_rows": (active & torch.isnan(rows).any(dim=-1)).nonzero().cpu().tolist()[:8],
                    "positive_inf_rows": (active & torch.isposinf(rows).any(dim=-1)).nonzero().cpu().tolist()[:8],
                    "all_negative_inf_rows": (active & torch.isneginf(rows).all(dim=-1)).nonzero().cpu().tolist()[:8],
                }
                raise ValueError(
                    "Target candidate logits contain NaN, +Inf, all -Inf or invalid candidate IDs: " + str(issues)
                )
            if self.config.method == "strict":
                keep = shape_logits.argmax(dim=-1) == candidates
            elif self.config.method == "topk":
                if self.config.topk > logits.shape[-1]:
                    raise ValueError("final_verification.topk exceeds the target vocabulary size")
                top_ids = shape_logits.topk(self.config.topk, dim=-1).indices
                keep = (top_ids == candidates.unsqueeze(-1)).any(dim=-1)
            else:
                keep = confidence >= self.config.target_threshold
            accepted, first, _, prefix = _prefix(keep, active, lengths)
        next_row_index = torch.zeros_like(accepted) if bonus_only else accepted
        next_row = logits.gather(
            -2, next_row_index[..., None, None].expand(*next_row_index.shape, 1, logits.shape[-1])
        ).squeeze(-2)
        _, valid_next = _relative_confidence(next_row.unsqueeze(-2), next_row.argmax(dim=-1).unsqueeze(-1))
        if not bool(valid_next.all()):
            raise ValueError("Target correction/bonus logits contain NaN, +Inf or all -Inf")
        next_token = next_row.argmax(dim=-1)
        return FinalDecision(accepted, first, next_token, prefix, confidence)
