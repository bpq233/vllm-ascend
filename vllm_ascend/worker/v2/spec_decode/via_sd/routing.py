"""Routing primitives for hierarchical VIA-SD verification.

The runtime verifier produces logits, while this module owns the small piece
of policy that decides whether a draft token is accepted by q', rewritten by
q', or escalated to the target model.  Keeping that policy device agnostic is
intentional: it makes threshold boundaries and mixed-batch compaction
testable without an Ascend device.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Hashable


class ViaSdRoute(str, Enum):
    """The three outcomes of one q' confidence check."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    # Readable aliases used by callers that describe the action rather than
    # the confidence band.
    ACCEPT = "high"
    REWRITE = "medium"
    FALLBACK = "low"


def validate_thresholds(
    accept_ratio: float = 0.7,
    escalate_ratio: float = 0.5,
) -> tuple[float, float]:
    """Validate and return the q' routing thresholds.

    The comparison is deliberately strict at the lower boundary and inclusive
    at both configured boundaries: ``score == accept`` is HIGH and
    ``score == escalate`` is MEDIUM.
    """

    values = (accept_ratio, escalate_ratio)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("VIA-SD thresholds must be numeric")
    accept = float(accept_ratio)
    escalate = float(escalate_ratio)
    if not math.isfinite(accept) or not math.isfinite(escalate):
        raise ValueError("VIA-SD thresholds must be finite")
    if not 0.0 <= escalate < accept <= 1.0:
        raise ValueError("VIA-SD thresholds must satisfy 0 <= escalate_ratio < accept_ratio <= 1")
    return accept, escalate


def route_from_score(
    score: float,
    accept_ratio: float = 0.7,
    escalate_ratio: float = 0.5,
) -> ViaSdRoute:
    """Map one relative confidence score to a route.

    This is the central threshold function.  It intentionally does not infer
    thresholds from paper-specific alpha/beta symbols.
    """

    accept, escalate = validate_thresholds(accept_ratio, escalate_ratio)
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"VIA-SD confidence must be finite, got {score!r}")
    if value < 0.0 or value > 1.0 + 1e-6:
        raise ValueError(f"VIA-SD confidence must be in [0, 1], got {score!r}")
    value = min(1.0, max(0.0, value))
    if value >= accept:
        return ViaSdRoute.HIGH
    if value >= escalate:
        return ViaSdRoute.MEDIUM
    return ViaSdRoute.LOW


# Common names make the primitive easy to discover and preserve compatibility
# with early experiments that called it ``route_confidence``.
route_confidence = route_from_score
classify_score = route_from_score


def _python_value(value: Any) -> Any:
    """Detach tensor/array-like inputs without importing either dependency."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _as_float_rows(logits: Any) -> list[list[list[float]]]:
    value = _python_value(logits)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("q' logits must be a rank-2 or rank-3 sequence")
    value = list(value)
    if not value:
        return []
    # A single [steps, vocab] row is accepted for convenience.
    first = _python_value(value[0])
    if isinstance(first, Sequence) and first and not isinstance(_python_value(first[0]), Sequence):
        value = [value]
    rows: list[list[list[float]]] = []
    for batch_row in value:
        positions = _python_value(batch_row)
        if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
            raise ValueError("q' logits must be rectangular over positions")
        position_rows: list[list[float]] = []
        for row in positions:
            tokens = _python_value(row)
            if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)) or not tokens:
                raise ValueError("q' logits rows must be non-empty")
            try:
                position_rows.append([float(token) for token in tokens])
            except (TypeError, ValueError) as exc:
                raise ValueError("q' logits contain a non-numeric value") from exc
        rows.append(position_rows)
    return rows


def _as_draft_rows(draft_token_ids: Any) -> list[list[int]]:
    value = _python_value(draft_token_ids)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("draft token IDs must be a rank-1 or rank-2 sequence")
    value = list(value)
    if not value:
        return []
    first = _python_value(value[0])
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        value = [value]
    rows: list[list[int]] = []
    for row in value:
        row_value = _python_value(row)
        if not isinstance(row_value, Sequence) or isinstance(row_value, (str, bytes)):
            raise ValueError("draft token IDs must be rectangular over requests")
        rows.append([int(token) for token in row_value])
    return rows


def relative_confidence(logits: Sequence[float], token_id: int) -> float:
    """Return ``q'(token) / max(q')`` from logits or probabilities.

    For model logits the softmax normalizer cancels, so the stable expression
    is ``exp(logit[token] - max(logit))``.  A row that is clearly a probability
    vector is also accepted; this is useful for deterministic CPU tests and
    calibration tools.  NaN, positive infinity, and an all ``-inf`` row are
    rejected instead of silently selecting a route.
    """

    values = [float(value) for value in _python_value(logits)]
    if not values:
        raise ValueError("q' logits row is empty")
    if token_id < 0 or token_id >= len(values):
        raise ValueError(f"draft token {token_id} is outside q' vocabulary {len(values)}")
    if any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError("q' logits contain NaN or positive infinity")
    finite = [value for value in values if value != -math.inf]
    if not finite:
        raise ValueError("q' logits row is all negative infinity")

    # Do not use max probability as confidence.  This branch only handles an
    # explicitly probability-shaped row (non-negative and normalized).
    if all(value >= 0.0 for value in values):
        total = sum(values)
        maximum = max(values)
        if maximum > 0.0 and abs(total - 1.0) <= 1e-4:
            return min(1.0, max(0.0, values[token_id] / maximum))

    maximum = max(finite)
    selected = values[token_id]
    if selected == -math.inf:
        return 0.0
    gap = min(0.0, selected - maximum)
    return min(1.0, max(0.0, math.exp(gap)))


def relative_confidences(
    logits: Any,
    draft_token_ids: Any,
    valid_lengths: Sequence[int] | None = None,
) -> list[list[float]]:
    """Compute relative confidence for every non-padding draft position."""

    logit_rows = _as_float_rows(logits)
    draft_rows = _as_draft_rows(draft_token_ids)
    if len(logit_rows) != len(draft_rows):
        raise ValueError(f"q' batch mismatch: logits={len(logit_rows)}, drafts={len(draft_rows)}")
    if valid_lengths is not None and len(valid_lengths) != len(draft_rows):
        raise ValueError("VIA-SD valid_lengths must match the batch")
    result: list[list[float]] = []
    for batch_index, (position_rows, draft_row) in enumerate(zip(logit_rows, draft_rows)):
        limit = len(draft_row)
        if valid_lengths is not None:
            limit = min(limit, max(0, int(valid_lengths[batch_index])))
        if limit > len(position_rows):
            raise ValueError(f"q' logits row {batch_index} has {len(position_rows)} positions, need {limit}")
        result.append([relative_confidence(position_rows[pos], draft_row[pos]) for pos in range(limit)])
    return result


@dataclass(frozen=True)
class ViaSdDecision:
    """One position's immutable q' routing decision."""

    position: int
    draft_token: int
    score: float
    route: ViaSdRoute

    @property
    def is_high(self) -> bool:
        return self.route is ViaSdRoute.HIGH

    @property
    def is_medium(self) -> bool:
        return self.route is ViaSdRoute.MEDIUM

    @property
    def is_low(self) -> bool:
        return self.route is ViaSdRoute.LOW


@dataclass(frozen=True)
class ViaSdFallback:
    """A compact target-fallback item with its original batch coordinates."""

    batch_row: int
    request_id: Hashable
    position: int
    draft_token: int
    score: float

    @property
    def row(self) -> int:
        return self.batch_row


@dataclass(frozen=True)
class ViaSdRoutePlan:
    """Ragged route decisions plus the compact LOW mapping."""

    request_ids: tuple[Hashable, ...]
    draft_tokens: tuple[tuple[int, ...], ...]
    scores: tuple[tuple[float, ...], ...]
    decisions: tuple[tuple[ViaSdDecision, ...], ...]
    fallbacks: tuple[ViaSdFallback, ...]
    logits: Any = None
    batch_rows: tuple[int, ...] = ()

    @property
    def routes(self) -> tuple[tuple[ViaSdRoute, ...], ...]:
        return tuple(tuple(decision.route for decision in row) for row in self.decisions)

    @property
    def route_names(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(route.value for route in row) for row in self.routes)

    @property
    def fallback_rows(self) -> tuple[int, ...]:
        return tuple(item.batch_row for item in self.fallbacks)

    @property
    def compact_fallback_rows(self) -> tuple[int, ...]:
        return self.fallback_rows

    @property
    def requires_target(self) -> bool:
        return bool(self.fallbacks)

    @property
    def all_high(self) -> bool:
        return bool(self.decisions) and all(
            decision.route is ViaSdRoute.HIGH for row in self.decisions for decision in row
        )

    @property
    def all_low(self) -> bool:
        return bool(self.decisions) and all(
            decision.route is ViaSdRoute.LOW for row in self.decisions for decision in row
        )

    @property
    def empty(self) -> bool:
        return not any(self.decisions)

    def first_non_high(self, row: int) -> ViaSdDecision | None:
        for decision in self.decisions[row]:
            if decision.route is not ViaSdRoute.HIGH:
                return decision
        return None

    def logits_for(self, row: int, position: int) -> Any:
        if self.logits is None:
            return None
        return self.logits[row][position]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_ids": self.request_ids,
            "routes": self.route_names,
            "scores": self.scores,
            "fallback_rows": self.fallback_rows,
            "fallback_positions": tuple(item.position for item in self.fallbacks),
        }


def build_route_plan(
    qprime_logits: Any,
    draft_token_ids: Any,
    accept_ratio: float = 0.7,
    escalate_ratio: float = 0.5,
    *,
    request_ids: Sequence[Hashable] | None = None,
    valid_lengths: Sequence[int] | None = None,
    batch_rows: Sequence[int] | None = None,
) -> ViaSdRoutePlan:
    """Build a ragged route plan and compact LOW fallback list."""

    accept, escalate = validate_thresholds(accept_ratio, escalate_ratio)
    # Keep vocabulary-sized logits on device; copy only routing scores to CPU.
    tensor_scores = None
    if hasattr(qprime_logits, "gather") and hasattr(qprime_logits, "detach"):
        import torch

        logit_rows = qprime_logits.detach()
        if logit_rows.ndim == 2:
            logit_rows = logit_rows.unsqueeze(0)
        if logit_rows.ndim != 3 or logit_rows.shape[-1] == 0:
            raise ValueError("qprime logits must be a non-empty rank-2 or rank-3 tensor")
    else:
        logit_rows = _as_float_rows(qprime_logits)
    draft_rows = _as_draft_rows(draft_token_ids)
    if len(logit_rows) != len(draft_rows):
        raise ValueError(f"q' batch mismatch: logits={len(logit_rows)}, drafts={len(draft_rows)}")
    batch_size = len(draft_rows)
    ids = tuple(range(batch_size)) if request_ids is None else tuple(request_ids)
    if len(ids) != batch_size:
        raise ValueError("VIA-SD request IDs must match the q' batch")
    rows = tuple(range(batch_size)) if batch_rows is None else tuple(int(row) for row in batch_rows)
    if len(rows) != batch_size:
        raise ValueError("VIA-SD batch_rows must match the q' batch")
    if valid_lengths is not None and len(valid_lengths) != batch_size:
        raise ValueError("VIA-SD valid_lengths must match the q' batch")

    if hasattr(logit_rows, "gather"):
        lengths = [
            len(row) if valid_lengths is None else min(len(row), max(0, int(valid_lengths[i])))
            for i, row in enumerate(draft_rows)
        ]
        if any(length > logit_rows.shape[1] for length in lengths):
            raise ValueError("qprime logits have fewer positions than the draft block")
        tokens = torch.zeros(logit_rows.shape[:2], dtype=torch.long, device=logit_rows.device)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, length in enumerate(lengths):
            row = draft_rows[i][:length]
            if any(token < 0 or token >= logit_rows.shape[-1] for token in row):
                raise ValueError("draft token is outside qprime vocabulary")
            tokens[i, :length] = torch.as_tensor(row, device=tokens.device)
            mask[i, :length] = True
        probabilities = torch.softmax(logit_rows.float(), dim=-1)
        maximum = probabilities.amax(dim=-1)
        selected = probabilities.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        invalid = (
            torch.isnan(logit_rows).any(dim=-1) | torch.isposinf(logit_rows).any(dim=-1) | torch.isneginf(maximum)
        ) & mask
        if invalid.any().item():
            raise ValueError("qprime logits contain NaN, positive infinity, or an all -inf row")
        tensor_scores = (selected / maximum.clamp_min(torch.finfo(probabilities.dtype).tiny)).cpu().tolist()

    all_scores: list[tuple[float, ...]] = []
    all_decisions: list[tuple[ViaSdDecision, ...]] = []
    all_drafts: list[tuple[int, ...]] = []
    all_logits: list[Any] = []
    fallbacks: list[ViaSdFallback] = []
    for index, (logits_row, draft_row) in enumerate(zip(logit_rows, draft_rows)):
        limit = len(draft_row)
        if valid_lengths is not None:
            limit = min(limit, max(0, int(valid_lengths[index])))
        if limit > len(logits_row):
            raise ValueError(f"q' logits row {index} has {len(logits_row)} positions, need {limit}")
        draft = tuple(draft_row[:limit])
        if tensor_scores is not None:
            position_logits = logits_row[:limit]
            scores = tuple(tensor_scores[index][:limit])
        else:
            position_logits = tuple(tuple(row) for row in logits_row[:limit])
            scores = tuple(relative_confidence(position_logits[position], draft[position]) for position in range(limit))
        decisions = tuple(
            ViaSdDecision(
                position=position,
                draft_token=draft[position],
                score=score,
                route=route_from_score(score, accept, escalate),
            )
            for position, score in enumerate(scores)
        )
        all_drafts.append(draft)
        all_scores.append(scores)
        all_decisions.append(decisions)
        all_logits.append(position_logits)
        for decision in decisions:
            if decision.route is ViaSdRoute.LOW:
                fallbacks.append(
                    ViaSdFallback(
                        batch_row=rows[index],
                        request_id=ids[index],
                        position=decision.position,
                        draft_token=decision.draft_token,
                        score=decision.score,
                    )
                )
    return ViaSdRoutePlan(
        request_ids=ids,
        draft_tokens=tuple(all_drafts),
        scores=tuple(all_scores),
        decisions=tuple(all_decisions),
        fallbacks=tuple(fallbacks),
        logits=tuple(all_logits),
        batch_rows=rows,
    )


def sample_from_logits(
    logits: Any,
    sampling_params: Any | None = None,
    *,
    generator: Any | None = None,
) -> int:
    """Sample one q' rewrite using the caller's sampling parameters.

    Temperature zero uses greedy argmax.  For non-greedy sampling, torch is
    used when available so a vLLM generator can be passed through; the small
    Python fallback is useful for CPU-only tests and never affects the NPU
    path.
    """

    values = [float(value) for value in _python_value(logits)]
    if not values or any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError("Cannot sample from invalid q' logits")
    temperature = float(getattr(sampling_params, "temperature", 0.0) if sampling_params is not None else 0.0)
    if temperature < 0.0 or not math.isfinite(temperature):
        raise ValueError("sampling temperature must be finite and non-negative")

    if temperature == 0.0:
        maximum = max(values)
        return next(index for index, value in enumerate(values) if value == maximum)

    top_k = int(getattr(sampling_params, "top_k", -1)) if sampling_params is not None else -1
    top_p = float(getattr(sampling_params, "top_p", 1.0)) if sampling_params is not None else 1.0
    min_p = float(getattr(sampling_params, "min_p", 0.0)) if sampling_params is not None else 0.0
    if not 0.0 < top_p <= 1.0 or min_p < 0.0:
        raise ValueError("invalid top_p/min_p sampling parameter")

    # Prefer torch's numerically stable multinomial and generator semantics.
    try:
        import torch

        tensor = torch.tensor(values, dtype=torch.float32)
        scaled = tensor / temperature
        if top_k > 0 and top_k < scaled.numel():
            threshold = torch.topk(scaled, top_k).values[-1]
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
        probabilities = torch.softmax(scaled, dim=-1)
        if min_p > 0.0:
            probabilities = probabilities.masked_fill(probabilities < probabilities.max() * min_p, 0.0)
            total = probabilities.sum()
            if total <= 0:
                probabilities = torch.softmax(scaled, dim=-1)
            else:
                probabilities = probabilities / total
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative - sorted_probs >= top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            probabilities = torch.zeros_like(probabilities).scatter(0, sorted_indices, sorted_probs)
            probabilities = probabilities / probabilities.sum()
        return int(torch.multinomial(probabilities, 1, generator=generator).item())
    except ImportError:
        pass

    maximum = max(values)
    weights = [math.exp((value - maximum) / temperature) for value in values]
    if top_k > 0 and top_k < len(weights):
        allowed = set(sorted(range(len(weights)), key=weights.__getitem__, reverse=True)[:top_k])
        weights = [weight if index in allowed else 0.0 for index, weight in enumerate(weights)]
    total = sum(weights)
    if total <= 0.0:
        return values.index(maximum)
    chooser = generator if callable(generator) else random.random
    target = float(chooser()) * total
    running = 0.0
    for index, weight in enumerate(weights):
        running += weight
        if target <= running:
            return index
    return len(weights) - 1


__all__ = [
    "ViaSdDecision",
    "ViaSdFallback",
    "ViaSdRoute",
    "ViaSdRoutePlan",
    "build_route_plan",
    "classify_score",
    "relative_confidence",
    "relative_confidences",
    "route_confidence",
    "route_from_score",
    "sample_from_logits",
    "validate_thresholds",
]
