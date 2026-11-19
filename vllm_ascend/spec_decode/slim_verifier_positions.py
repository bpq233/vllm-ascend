# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

POSITION_COUNTERS = (
    "tokens",
    "target_accepts",
    "slim_accepts",
    "prediction_matches",
    "target_prefix",
    "slim_rejects_in_target_prefix",
    "joint_prefix",
    "target_boundaries",
    "slim_accepts_at_boundary",
)


class PositionMetrics:
    """Per-draft-position counts; transfer to CPU only at snapshot boundaries.

    Target acceptance here ALWAYS means raw-logit Top-1 matching, not sampler
    acceptance. A valid Target prefix ends before its first mismatch, inclusive
    of all successful tokens, and excludes every later teacher-forced position.
    """

    def __init__(self):
        self.counts = self.sums = None
        self.window_counts = self.window_sums = None

    def record(self, target_mask, slim_mask, predictions_equal, target_logprob, slim_logprob):
        prefix = target_mask.long().cumprod(0).bool()
        reachable = torch.cat((torch.ones_like(prefix[:1]), prefix[:-1]))
        boundary = reachable & ~target_mask
        counts = torch.stack(
            (
                torch.ones_like(target_mask),
                target_mask,
                slim_mask,
                predictions_equal,
                prefix,
                prefix & ~slim_mask,
                (target_mask & slim_mask).long().cumprod(0).bool(),
                boundary,
                boundary & slim_mask,
            ),
            dim=1,
        ).long()
        # Zero-probability Target entries, including common -inf masks, have
        # zero KL contribution. Do not hide genuine nonfinite model outputs.
        probability = target_logprob.exp()
        delta = torch.where(probability > 0, target_logprob - slim_logprob, 0.0)
        kl = (probability * delta).sum(-1).clamp_min(0)
        sums = torch.stack((kl, torch.where(prefix, kl, 0.0)), dim=1)
        n = counts.shape[0]
        if self.counts is None:
            self.counts = torch.zeros_like(counts)
            self.sums = torch.zeros_like(sums)
            self.window_counts = torch.zeros_like(counts)
            self.window_sums = torch.zeros_like(sums)
        elif n > self.counts.shape[0]:
            extra = n - self.counts.shape[0]
            for name in ("counts", "sums", "window_counts", "window_sums"):
                old = getattr(self, name)
                setattr(self, name, torch.cat((old, old.new_zeros((extra, old.shape[1])))))
        self.counts[:n] += counts
        self.window_counts[:n] += counts
        self.sums[:n] += sums
        self.window_sums[:n] += sums

    def snapshot(self, blocks, cumulative=True):
        counts = (self.counts if cumulative else self.window_counts).cpu().tolist()
        sums = (self.sums if cumulative else self.window_sums).cpu().tolist()
        rows = []
        totals = {name: sum(row[i] for row in counts) for i, name in enumerate(POSITION_COUNTERS)}
        for i, (row, s) in enumerate(zip(counts, sums)):
            c = dict(zip(POSITION_COUNTERS, row))
            if not c["tokens"]:
                continue
            rows.append(
                {
                    "draft_position": i + 1,
                    "tokens": c["tokens"],
                    "target_top1_acceptance_rate": c["target_accepts"] / c["tokens"],
                    "slim_acceptance_rate": c["slim_accepts"] / c["tokens"],
                    "prediction_agreement_rate": c["prediction_matches"] / c["tokens"],
                    "mean_kl_target_to_slim": s[0] / c["tokens"],
                    "target_prefix_tokens": c["target_prefix"],
                    "slim_rejection_rate_in_target_prefix": (
                        c["slim_rejects_in_target_prefix"] / c["target_prefix"] if c["target_prefix"] else None
                    ),
                }
            )
        prefix = totals["target_prefix"]
        boundaries = totals["target_boundaries"]
        return {
            "target_acceptance_basis": "raw_top1_diagnostic",
            "per_position": rows,
            "mean_kl_target_to_slim": sum(s[0] for s in sums) / totals["tokens"],
            "target_prefix_tokens": prefix,
            "slim_rejection_rate_in_target_prefix": totals["slim_rejects_in_target_prefix"] / prefix
            if prefix
            else None,
            "joint_avg_accepted_length": totals["joint_prefix"] / blocks,
            "target_prefix_preservation_rate": totals["joint_prefix"] / prefix if prefix else None,
            "mean_kl_in_target_prefix": sum(s[1] for s in sums) / prefix if prefix else None,
            "target_rejection_boundaries": boundaries,
            "slim_acceptance_at_target_rejection_rate": totals["slim_accepts_at_boundary"] / boundaries
            if boundaries
            else None,
        }

    def reset_window(self):
        self.window_counts.zero_()
        self.window_sums.zero_()
