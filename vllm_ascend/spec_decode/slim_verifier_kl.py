# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

KL_NAMES = ("kl_draft_to_slim", "kl_draft_to_target", "kl_slim_to_target")


def kl_per_token(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """KL(P || Q), summed over vocabulary, with exact zero-mass handling."""
    zero_mass = log_p.isneginf()
    terms = log_p.exp() * (log_p - log_q)
    # Finite log_p denotes positive mass, even if exp(log_p) underflows.
    terms = torch.where(log_p.isfinite() & log_q.isneginf(), torch.inf, terms)
    return torch.where(zero_mass, 0.0, terms).sum(-1).clamp_min(0)


class SlimVerifierKLMetrics:
    """Three directional, full-vocabulary KL means on the same draft rows.

    Sum vocabulary first, then average over actual draft tokens (no bonus or
    padding). Only snapshots synchronize device counters. Invalid alignment
    prevents publication; unavailable draft distributions are never faked.
    """

    def __init__(self, log_interval=100, logger=None):
        if type(log_interval) is not int or log_interval < 1:
            raise ValueError("log_interval must be a positive integer")
        self.log_interval, self.logger = log_interval, logger
        self.blocks = self.tokens = self.window_blocks = self.window_tokens = 0
        self.sums = self.window_sums = None

    @torch.inference_mode()
    def record(self, draft_logits, slim_logits, target_logits, aligned=None):
        if (
            draft_logits.ndim != 2
            or draft_logits.shape[0] == 0
            or draft_logits.shape != slim_logits.shape
            or draft_logits.shape != target_logits.shape
        ):
            raise ValueError("KL requires matching nonempty [draft_tokens, global_vocab] logits")
        draft, slim, target = (x.float().log_softmax(-1) for x in (draft_logits, slim_logits, target_logits))
        losses = torch.stack(
            (kl_per_token(draft, slim), kl_per_token(draft, target), kl_per_token(slim, target)), dim=1
        )
        if aligned is None:
            aligned = torch.ones_like(losses[:, 0], dtype=torch.bool)
        if aligned.shape != losses.shape[:1] or aligned.dtype != torch.bool:
            raise ValueError("Draft alignment must be a boolean per-token mask")
        values = torch.cat((losses.sum(0), (~aligned).sum().reshape(1).float()))
        if self.sums is None:
            self.sums = torch.zeros_like(values)
            self.window_sums = torch.zeros_like(values)
        self.sums += values
        self.window_sums += values
        self.blocks += 1
        self.window_blocks += 1
        self.tokens += losses.shape[0]
        self.window_tokens += losses.shape[0]
        if self.window_blocks == self.log_interval:
            self.flush()

    def snapshot(self, cumulative=True):
        blocks = self.blocks if cumulative else self.window_blocks
        tokens = self.tokens if cumulative else self.window_tokens
        result = {"blocks": blocks, "draft_tokens": tokens}
        if not blocks:
            return result
        values = (self.sums if cumulative else self.window_sums).cpu().tolist()
        if values[-1]:
            raise RuntimeError(
                "Draft KL alignment failed: missing/stale logits or mismatched proposal tokens/positions. "
                "The proposer must emit full logits through sample_draft/compute_draft_logits. "
                "No KL report was published."
            )
        if any(v != v for v in values[:-1]):
            raise RuntimeError("Nonfinite model distributions produced undefined KL; no report was published")
        result.update(zip(KL_NAMES, (v / tokens for v in values[:-1])))
        result["distribution_basis"] = "raw_logits_temperature_1"
        result["draft_distribution_source"] = "captured_proposal_logits"
        return result

    def flush(self):
        if not self.window_blocks:
            return
        window, cumulative = self.snapshot(False), self.snapshot(True)
        if self.logger is not None:
            self.logger.info("[SlimVerifier] KL Window: %s", window)
            self.logger.info("[SlimVerifier] KL Cumulative: %s", cumulative)
        self.window_blocks = self.window_tokens = 0
        self.window_sums.zero_()
