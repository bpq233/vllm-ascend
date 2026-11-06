# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Past-Target-only mixture selection and small device feedback reductions.

Vocabulary tensors stay on their original device. Callers transfer only feature
or overlap packets, and commit feedback after the Target verification finishes.
The shared table learns across requests; it never stores request sampling RNGs.
"""

import json
import logging
import math
from bisect import bisect_right
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

logger = logging.getLogger(__name__)

DEFAULT_LAMBDA_CANDIDATES = tuple(index / 10 for index in range(11))
ENTROPY_BUCKET_EDGES = (0.3, 0.7)
MARGIN_BUCKET_EDGES = (0.1, 0.5)
DISAGREEMENT_BUCKET_EDGES = (0.25, 0.5, 0.75)
MAX_ROUND_BUCKET = 4
MAX_POSITION_BUCKET = 7
POSITIONS_PER_BUCKET = 4
LOG_FIELDS = (
    "request_id",
    "spec_round",
    "token_position",
    "draft_source",
    "lambda",
    "draft_token",
    "intermediate_accept",
    "target_accept",
    "intermediate_accepted_length",
    "target_accepted_length",
    "pM_draft_token",
    "qD_draft_token",
    "qMix_draft_token",
    "pT_draft_token",
    "tv_target_intermediate",
    "tv_target_dflash",
    "tv_target_mixture",
    "draft_time_ms",
    "intermediate_verify_time_ms",
    "target_verify_time_ms",
    "sampling_overhead_ms",
    "total_time_ms",
)


def _option(config, name, default):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _unit_interval(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return float(value)


def _probability_rows(probabilities):
    if probabilities.ndim != 2 or probabilities.shape[-1] == 0:
        raise ValueError("Probability inputs must be nonempty [positions, vocabulary] tensors.")
    return probabilities.float()


def _draft_probabilities(intermediate, draft):
    if draft.device != intermediate.device:
        raise ValueError("Draft and intermediate probability inputs must share a device.")
    if draft.ndim == 1:
        if draft.shape[0] != intermediate.shape[0] or draft.dtype not in (torch.int32, torch.int64):
            raise ValueError("Greedy draft IDs must be integer [positions] tensors.")
        return draft.long()
    if draft.shape != intermediate.shape:
        raise ValueError("Draft probabilities must have the same shape as intermediate probabilities.")
    return draft.float()


def _entropy(probabilities):
    # Use the neutral log argument one only at zero-mass entries; probabilities
    # are unchanged. Basic log/where also avoid a separate special-op dependency.
    log_argument = torch.where(probabilities > 0, probabilities, torch.ones_like(probabilities))
    return -(probabilities * log_argument.log()).sum(-1)


def _margin(probabilities):
    values = probabilities.topk(min(2, probabilities.shape[-1]), dim=-1).values
    return values[:, 0] if values.shape[-1] == 1 else values[:, 0] - values[:, 1]


def device_features(intermediate: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
    """Return [entropyM, entropyD, TV(M,D), marginM, marginD] per token.

    Entropies are normalized by log(vocabulary size). A one-dimensional draft
    tensor contains greedy IDs and represents the actual delta proposal, saving
    an otherwise unnecessary full-vocabulary allocation.
    """
    intermediate = _probability_rows(intermediate)
    draft = _draft_probabilities(intermediate, draft)
    entropy_scale = math.log(intermediate.shape[-1]) if intermediate.shape[-1] > 1 else 1.0
    entropy_m = _entropy(intermediate) / entropy_scale
    margin_m = _margin(intermediate)
    if draft.ndim == 1:
        entropy_d = torch.zeros_like(entropy_m)
        disagreement = 1 - intermediate.gather(-1, draft[:, None]).squeeze(-1)
        margin_d = torch.ones_like(margin_m)
    else:
        entropy_d = _entropy(draft) / entropy_scale
        disagreement = (intermediate - draft).abs().sum(-1) * 0.5
        margin_d = _margin(draft)
    return torch.stack((entropy_m, entropy_d, disagreement, margin_m, margin_d), dim=-1)


def make_state(features: Sequence[float], spec_round: int, token_position: int, draft_source: str) -> tuple:
    """Bucket an already-copied five-scalar feature row; never copy tensors."""
    if isinstance(features, torch.Tensor):
        raise TypeError("Copy the small feature packet explicitly before creating feedback states.")
    if len(features) != 5 or any(not math.isfinite(float(value)) for value in features):
        raise ValueError("Feedback features must contain five finite scalars.")
    if type(spec_round) is not int or spec_round < 0 or type(token_position) is not int or token_position < 0:
        raise ValueError("Round and token position must be nonnegative integers.")
    return (
        str(draft_source),
        bisect_right(ENTROPY_BUCKET_EDGES, features[0]),
        bisect_right(ENTROPY_BUCKET_EDGES, features[1]),
        bisect_right(DISAGREEMENT_BUCKET_EDGES, features[2]),
        bisect_right(MARGIN_BUCKET_EDGES, features[3]),
        bisect_right(MARGIN_BUCKET_EDGES, features[4]),
        min(spec_round, MAX_ROUND_BUCKET),
        min(token_position // POSITIONS_PER_BUCKET, MAX_POSITION_BUCKET),
    )


def overlap_scores(target, intermediate, draft, lambda_candidates=DEFAULT_LAMBDA_CANDIDATES):
    """Reduce full device distributions to [AM, AD, AMix(lambda_0), ...].

    Only one candidate-mixture vocabulary tensor is materialized at a time.
    Inputs must already be valid normalized probability distributions from the
    sampling layer; this function does not insert host synchronization checks.
    """
    intermediate = _probability_rows(intermediate)
    if target.shape != intermediate.shape or target.device != intermediate.device:
        raise ValueError("Target and intermediate distributions must have matching shapes and devices.")
    target = target.float()
    draft = _draft_probabilities(intermediate, draft)
    candidates = tuple(_unit_interval(value, "lambda candidate") for value in lambda_candidates)
    if not candidates:
        raise ValueError("At least one lambda candidate is required.")
    overlap_m = 1 - (target - intermediate).abs().sum(-1) * 0.5
    if draft.ndim == 1:
        overlap_d = target.gather(-1, draft[:, None]).squeeze(-1)
    else:
        overlap_d = 1 - (target - draft).abs().sum(-1) * 0.5
    scores = [overlap_m, overlap_d]
    for weight in candidates:
        if weight == 0:
            score = overlap_m
        elif weight == 1:
            score = overlap_d
        else:
            mixture = intermediate * (1 - weight)
            if draft.ndim == 1:
                mixture.scatter_add_(1, draft[:, None], mixture.new_full((mixture.shape[0], 1), weight))
            else:
                mixture.add_(draft, alpha=weight)
            score = 1 - (target - mixture).abs().sum(-1) * 0.5
        scores.append(score)
    # Roundoff in normalized FP32 inputs can move an overlap a few ulps outside
    # its mathematical range; clamp metrics only, never sampling probabilities.
    return torch.stack(scores, dim=-1).clamp_(0, 1)


def offline_oracle_lambda(scores: Sequence[float], candidates=DEFAULT_LAMBDA_CANDIDATES) -> float:
    """Post-hoc experiment only; this function is not an online selection mode."""
    if isinstance(scores, torch.Tensor):
        raise TypeError("Oracle analysis accepts an explicit scalar reduction packet, not device tensors.")
    if not scores or len(scores) != len(candidates) or any(not math.isfinite(float(value)) for value in scores):
        raise ValueError("Oracle scores must be finite and match the candidate count.")
    return float(candidates[max(range(len(scores)), key=lambda index: scores[index])])


@dataclass(frozen=True)
class BucketStats:
    count: int
    overlaps: tuple[float, ...]


@dataclass(frozen=True)
class FeedbackSnapshot:
    version: int
    buckets: Mapping[Hashable, BucketStats]


class ProposalFeedback:
    """Interpretable EMA policy updated only by completed Target feedback."""

    def __init__(self, config):
        self.mode = _option(config, "mode", "fixed")
        if self.mode not in ("fixed", "adaptive"):
            raise ValueError("Online proposal mode must be 'fixed' or 'adaptive'; oracle is offline only.")
        self.default = _unit_interval(_option(config, "mixture_lambda", 0.5), "mixture_lambda")
        self.candidates = tuple(
            _unit_interval(value, "lambda candidate")
            for value in _option(config, "lambda_candidates", DEFAULT_LAMBDA_CANDIDATES)
        )
        if not self.candidates or len(set(self.candidates)) != len(self.candidates):
            raise ValueError("lambda_candidates must be nonempty and contain no duplicates.")
        self.decay = _unit_interval(_option(config, "ema_decay", 0.9), "ema_decay")
        if self.decay == 1:
            raise ValueError("ema_decay must be less than one so new feedback can contribute.")
        self.min_observations = _option(config, "min_observations", 8)
        if type(self.min_observations) is not int or self.min_observations < 1:
            raise ValueError("min_observations must be a positive integer.")
        self.enabled = _option(config, "feedback_enabled", True)
        self.debug = _option(config, "debug", False)
        if type(self.enabled) is not bool or type(self.debug) is not bool:
            raise ValueError("feedback_enabled and debug must be booleans.")
        self.version = 0
        self.buckets = {}

    def snapshot(self) -> FeedbackSnapshot:
        """Freeze all lambda choices for an in-flight proposal cycle."""
        return FeedbackSnapshot(self.version, MappingProxyType(dict(self.buckets)))

    def select(self, state: Hashable, snapshot: FeedbackSnapshot | None = None) -> float:
        if self.mode == "fixed":
            return self.default
        bucket = (self.buckets if snapshot is None else snapshot.buckets).get(state)
        if bucket is None or bucket.count < self.min_observations:
            return self.default
        # Near-default ties make cold/warm policy transitions deterministic.
        index = max(
            range(len(self.candidates)),
            key=lambda candidate: (bucket.overlaps[candidate], -abs(self.candidates[candidate] - self.default)),
        )
        return self.candidates[index]

    def update(self, state: Hashable, overlaps: Sequence[float], *, target_verified: bool):
        """Consume only the K scalar AMix scores after Target verification.

        Callers can keep AM/AD separately for logging. No probability tensor is
        accepted here, preventing an accidental vocabulary D2H in this method.
        """
        if target_verified is not True:
            raise ValueError("Mixture feedback may only be committed after completed Target verification.")
        if isinstance(overlaps, torch.Tensor):
            raise TypeError("Feedback requires an explicitly copied scalar reduction packet.")
        if len(overlaps) != len(self.candidates):
            raise ValueError("Overlap feedback must match lambda_candidates.")
        values = tuple(_unit_interval(value, "overlap") for value in overlaps)
        if not self.enabled:
            return
        old = self.buckets.get(state)
        smoothed = (
            values
            if old is None
            else tuple(
                self.decay * previous + (1 - self.decay) * current for previous, current in zip(old.overlaps, values)
            )
        )
        self.buckets[state] = BucketStats(1 if old is None else old.count + 1, smoothed)
        self.version += 1

    def emit_record(self, values: Mapping):
        """Emit a JSON record only when debug was explicitly requested.

        Unavailable timing/probability fields remain null, never fabricated.
        Device tensors are deliberately not serialized by an implicit .item().
        """
        if not self.debug:
            return
        record = dict.fromkeys(LOG_FIELDS)
        record.update(values)
        logger.info("proposal_sampling %s", json.dumps(record, allow_nan=False, sort_keys=True))
