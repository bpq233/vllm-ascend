# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Request-local control for speculative, never externally committed prefixes.

The runner owns physical KV pages and graph replay. This module owns only the
logical temporary prefix. Neither a successful intermediate decision nor the
returned proposals constitutes a Target acceptance. The existing scheduler must
submit the complete returned sequence to one final logical Target verification.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import torch

from .config import SparseVerifyConfig
from .decisions import ConfidenceDecision


@dataclass(frozen=True)
class FeatureBundle:
    """Checkpoint-ordered features for a contiguous interval of token positions.

    ``start_position`` is inclusive; ``end_position`` is exclusive. Features are
    of shape ``[tokens, hidden_size]`` separately for each checkpoint layer. A
    drafter consumes context strictly before its anchor token, so the last
    feature position is ``anchor_position - 1``. The current anchor itself is
    supplied separately in ``DraftRequest``.
    """

    values: tuple[torch.Tensor, ...]
    layer_ids: tuple[int, ...]
    start_position: int
    source: Literal["target", "sparse"]
    token_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.source not in ("target", "sparse"):
            raise ValueError("feature source must be target or sparse")
        if not self.values or len(self.values) != len(self.layer_ids):
            raise ValueError("feature count must match checkpoint layer order")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("feature layers must not repeat")
        if self.start_position < 0:
            raise ValueError("feature positions must be nonnegative")
        shape = self.values[0].shape
        if len(shape) != 2 or shape[1] <= 0:
            raise ValueError("features must have shape [tokens, hidden_size]")
        if any(value.shape != shape for value in self.values):
            raise ValueError("all checkpoint feature layers must have the same shape")
        if any(value.device != self.values[0].device or value.dtype != self.values[0].dtype for value in self.values):
            raise ValueError("all checkpoint feature layers must have the same device and dtype")
        if self.token_ids is not None and (
            self.token_ids.ndim != 1
            or self.token_ids.numel() != shape[0]
            or self.token_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("feature token IDs must be an integer vector with one token per row")

    @property
    def end_position(self) -> int:
        return self.start_position + self.values[0].shape[0]

    @property
    def positions(self) -> torch.Tensor:
        return torch.arange(self.start_position, self.end_position, device=self.values[0].device)

    @property
    def states(self) -> tuple[torch.Tensor, ...]:
        return self.values

    def require_anchor(self, anchor_position: int) -> None:
        if self.end_position != anchor_position or self.end_position <= self.start_position:
            raise ValueError(
                f"{self.source} features cover [{self.start_position}, {self.end_position}), "
                f"but DFlash anchor is at {anchor_position}; stale or rejected features cannot be reused"
            )

    def before(self, position: int) -> "FeatureBundle":
        if not self.start_position <= position <= self.end_position:
            raise ValueError("requested feature prefix is outside its valid position interval")
        length = position - self.start_position
        token_ids = self.token_ids[:length] if self.token_ids is not None else None
        return FeatureBundle(
            tuple(value[:length] for value in self.values),
            self.layer_ids,
            self.start_position,
            self.source,
            token_ids,
        )


@dataclass(frozen=True)
class DraftRequest:
    request_id: str
    round_index: int
    anchor_token: int
    anchor_position: int
    features: FeatureBundle
    max_tokens: int
    accumulated_tokens: tuple[int, ...]


@dataclass(frozen=True)
class SparseForwardRequest:
    request_id: str
    round_index: int
    # Includes the anchor. Logits row zero predicts candidates row zero.
    input_ids: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class SparseForwardOutput:
    # [candidate_count, vocab_size], already shifted by the runner: output of
    # input row i predicts candidate i at position input_position[i] + 1.
    candidate_logits: torch.Tensor
    features: FeatureBundle
    # Retain cache positions strictly below this absolute endpoint. Physical
    # pages need not be zeroed, but later attention may not see their suffix.
    # The endpoint deliberately excludes the next anchor, which is recomputed.
    retain_cache_prefix: Callable[[int], None]


@dataclass(frozen=True)
class RoundRecord:
    round_index: int
    proposed_length: int
    temporarily_accepted_length: int
    first_rejection: int
    low_confidence_stop: bool
    retry: bool
    confidence: tuple[float, ...]
    feature_source: str
    feature_shape: tuple[int, ...]
    feature_positions: tuple[int, int]
    dflash_seconds: float
    sparse_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class ProposalResult:
    # All candidates, excluding the original already-sampled anchor. These are
    # returned to the upstream scheduler, never emitted directly to the user.
    tokens: torch.Tensor
    rounds: tuple[RoundRecord, ...]
    stop_reason: str
    temporary_accepted_length: int
    original_anchor_position: int


class MultiRoundDFlashController:
    """Run a bounded multi-round proposal without mutating Target state.

    Callbacks operate on independent sparse/draft caches and may replay their
    existing graphs. Decisions and bounded Python control stay outside capture.
    Scalar device-to-host reads here are intentional control dependencies; they
    are never performed from a captured forward.
    """

    def __init__(self, config: SparseVerifyConfig, draft_length: int, max_candidate_tokens: int):
        if isinstance(draft_length, bool) or not isinstance(draft_length, int) or draft_length <= 0:
            raise ValueError("draft_length must be a positive integer")
        if (
            isinstance(max_candidate_tokens, bool)
            or not isinstance(max_candidate_tokens, int)
            or max_candidate_tokens < draft_length * config.num_rounds
        ):
            raise ValueError("Target candidate capacity must cover draft_length * num_rounds in one verification")
        self.config = config
        self.draft_length = draft_length
        self.max_candidate_tokens = max_candidate_tokens
        self.decision = ConfidenceDecision(config.mode, config.threshold_high, config.threshold_low)

    def run_request(
        self,
        request_id: str,
        anchor_token: int,
        anchor_position: int,
        initial_features: FeatureBundle,
        draft: Callable[[DraftRequest], torch.Tensor],
        verify: Callable[[SparseForwardRequest], SparseForwardOutput],
        *,
        token_budget: int,
        eos_token_ids: Sequence[int] = (),
        target_features: FeatureBundle | None = None,
    ) -> ProposalResult:
        """Construct candidates for one eventual Target submission.

        ``token_budget`` already accounts for generation/context capacity and
        excludes the anchor. A smaller request-specific budget shortens the last
        block; it never splits Target verification into multiple submissions.
        ``target`` feature mode stops safely if real Target features do not cover
        a new context. It never relabels sparse approximations as Target output.
        """
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 0:
            raise ValueError("token_budget must be a nonnegative integer")
        if isinstance(anchor_position, bool) or not isinstance(anchor_position, int) or anchor_position < 1:
            raise ValueError("anchor_position must follow at least one computed context token")
        if initial_features.source != "target":
            raise ValueError("the initial DFlash round requires true Target features")
        initial_features.require_anchor(anchor_position)
        features = initial_features
        original_anchor_position = anchor_position
        original_anchor_token = anchor_token
        capacity = min(token_budget, self.max_candidate_tokens)
        candidates: list[int] = []
        records: list[RoundRecord] = []
        eos = set(eos_token_ids)
        stop_reason = "round_limit"
        temporary_accepted = 0
        device = initial_features.values[0].device

        for round_index in range(self.config.num_rounds):
            remaining = capacity - len(candidates)
            if remaining <= 0:
                stop_reason = "token_budget"
                break
            if anchor_token in eos:
                stop_reason = "eos"
                break
            features.require_anchor(anchor_position)
            round_start = perf_counter()
            request = DraftRequest(
                request_id,
                round_index,
                anchor_token,
                anchor_position,
                features,
                min(self.draft_length, remaining),
                tuple(candidates),
            )
            proposed = draft(request)
            draft_end = perf_counter()
            if proposed.ndim != 1 or proposed.numel() > request.max_tokens:
                raise ValueError("drafter must return a one-dimensional block within the requested token budget")
            if proposed.dtype not in (torch.int32, torch.int64):
                raise ValueError("drafter tokens must use an integer dtype")
            proposed_ids = proposed.detach().cpu().tolist()
            if any(token < 0 for token in proposed_ids):
                raise ValueError("drafter returned invalid negative token IDs")
            for position, token in enumerate(proposed_ids):
                if token in eos:
                    proposed_ids = proposed_ids[: position + 1]
                    proposed = proposed[: position + 1]
                    break
            if not proposed_ids:
                stop_reason = "empty_draft"
                break

            proposed = proposed.to(device=device, dtype=torch.long)
            inputs = torch.cat((proposed.new_tensor([anchor_token]), proposed))
            positions = torch.arange(anchor_position, anchor_position + inputs.numel(), device=device)
            output = verify(SparseForwardRequest(request_id, round_index, inputs, positions))
            sparse_end = perf_counter()
            try:
                if output.candidate_logits.ndim != 2 or output.candidate_logits.shape[0] != len(proposed_ids):
                    raise ValueError("sparse candidate logits must match the current proposed block")
                if output.features.source != "sparse":
                    raise ValueError("sparse forward must label approximate features as sparse")
                if output.features.layer_ids != initial_features.layer_ids:
                    raise ValueError("sparse feature layer order differs from the DFlash checkpoint")
                if output.features.values[0].shape[1] != initial_features.values[0].shape[1]:
                    raise ValueError("sparse feature width differs from the DFlash checkpoint")
                if (
                    output.features.start_position != anchor_position
                    or output.features.end_position != anchor_position + inputs.numel()
                ):
                    raise ValueError("sparse features do not match actual input token positions")
                decision = self.decision.evaluate(output.candidate_logits, proposed)
                accepted = int(decision.accepted_length.item())
                rejected_at = int(decision.first_rejection.item())
                low_stop = bool(decision.stop.item())
                retry = bool(decision.retry.item())
                if not 0 <= accepted <= len(proposed_ids):
                    raise RuntimeError("sparse decision returned an invalid accepted prefix")
                # Exclude the next anchor from retained KV; any rejected suffix
                # stays physically allocated but becomes logically invisible.
                output.retain_cache_prefix(anchor_position + accepted)
            except BaseException:
                output.retain_cache_prefix(anchor_position)
                raise

            confidence = (
                tuple(float(value) for value in decision.confidence.detach().cpu().tolist())
                if self.config.enable_metrics
                else ()
            )
            records.append(
                RoundRecord(
                    round_index,
                    len(proposed_ids),
                    accepted,
                    rejected_at,
                    low_stop,
                    retry,
                    confidence,
                    features.source,
                    tuple(features.values[0].shape),
                    (features.start_position, features.end_position),
                    draft_end - round_start,
                    sparse_end - draft_end,
                    perf_counter() - round_start,
                )
            )
            temporary_accepted += accepted
            if low_stop:
                # No later candidate is temporarily accepted. Nevertheless the
                # full current pending block is preserved for the final Target,
                # exactly as required by the low-threshold handoff semantics.
                candidates.extend(proposed_ids)
                stop_reason = "low_confidence"
                break
            candidates.extend(proposed_ids[:accepted])
            if accepted and proposed_ids[accepted - 1] in eos:
                stop_reason = "eos"
                break
            if len(candidates) >= capacity:
                stop_reason = "token_budget"
                break
            if round_index + 1 == self.config.num_rounds:
                break
            if accepted:
                anchor_position += accepted
                anchor_token = proposed_ids[accepted - 1]
                if self.config.feature_source == "target":
                    # A real Target bundle can be supplied only when its token
                    # identity/coverage is known by the caller. Absent coverage,
                    # hand off to Target rather than reusing stale positions.
                    if (
                        target_features is None
                        or target_features.source != "target"
                        or target_features.layer_ids != initial_features.layer_ids
                        or target_features.token_ids is None
                        or target_features.start_position > anchor_position - accepted
                        or not target_features.start_position < anchor_position <= target_features.end_position
                    ):
                        stop_reason = "target_features_unavailable"
                        break
                    known_start = target_features.start_position
                    check_start = max(original_anchor_position, known_start)
                    # For subsequent rounds, the original anchor plus the
                    # complete accumulated draft prefix defines token identity.
                    # Supplied Target features may cover a suffix, but that
                    # suffix must correspond to these same actual tokens.
                    actual_prefix = [original_anchor_token, *candidates]
                    expected = actual_prefix[
                        check_start - original_anchor_position : anchor_position - original_anchor_position
                    ]
                    actual = (
                        target_features.token_ids[check_start - known_start : anchor_position - known_start]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    if actual != expected:
                        stop_reason = "target_features_unavailable"
                        break
                    features = target_features.before(anchor_position)
                else:
                    features = output.features.before(anchor_position)
            # Zero-accept retries intentionally keep the same anchor/features,
            # consume a round, and cannot loop forever.

        return ProposalResult(
            torch.tensor(candidates, dtype=torch.long, device=device),
            tuple(records),
            stop_reason,
            temporary_accepted,
            original_anchor_position,
        )
