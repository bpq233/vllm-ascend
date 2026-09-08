"""Device-independent execution coordinator for hierarchical VIA-SD.

The Ascend runner owns model execution and scheduler buffers.  This module
owns the causal state transitions around those forwards.  Callers provide
small callbacks for q' logits, target catch-up, and target verification; the
coordinator then guarantees that a rewrite invalidates every later suffix and
that only LOW positions are presented to the target callback.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Hashable

from .routing import (
    ViaSdDecision,
    ViaSdFallback,
    ViaSdRoute,
    ViaSdRoutePlan,
    build_route_plan,
    sample_from_logits,
    validate_thresholds,
)


@dataclass
class ViaSdRequestState:
    """Logical state for one request and its two independent KV stages.

    ``committed_tokens`` is the only sequence visible to the scheduler.
    ``qprime_computed_len`` and ``target_computed_len`` count positions that
    were actually forwarded by the corresponding model.  They are never
    advanced merely because a token was accepted by another stage.
    """

    request_id: Hashable
    committed_tokens: list[int] | tuple[int, ...] = field(default_factory=list)
    committed_len: int | None = None
    qprime_computed_len: int = 0
    target_computed_len: int = 0
    active: bool = True

    def __post_init__(self) -> None:
        self.committed_tokens = [int(token) for token in self.committed_tokens]
        if self.committed_len is None:
            self.committed_len = len(self.committed_tokens)
        self.committed_len = max(0, min(int(self.committed_len), len(self.committed_tokens)))
        if len(self.committed_tokens) > self.committed_len:
            del self.committed_tokens[self.committed_len :]
        self.qprime_computed_len = max(0, int(self.qprime_computed_len))
        self.target_computed_len = max(0, int(self.target_computed_len))

    @property
    def tokens(self) -> list[int]:
        """Compatibility alias used by stage-oriented callers."""

        return self.committed_tokens

    @property
    def logical_lengths(self) -> dict[str, int]:
        return {
            "committed_len": int(self.committed_len or 0),
            "qprime_computed_len": self.qprime_computed_len,
            "target_computed_len": self.target_computed_len,
        }

    def reconcile(self, prefix: Sequence[int]) -> None:
        """Reconcile scheduler state and invalidate divergent suffixes."""

        desired = [int(token) for token in prefix]
        common = 0
        for old, new in zip(self.committed_tokens, desired):
            if old != new:
                break
            common += 1
        if common != len(self.committed_tokens) or len(desired) < len(self.committed_tokens):
            self.truncate(common)
        if len(desired) > len(self.committed_tokens):
            self.committed_tokens.extend(desired[len(self.committed_tokens) :])
        self.committed_len = len(self.committed_tokens)
        self.qprime_computed_len = min(self.qprime_computed_len, self.committed_len)
        self.target_computed_len = min(self.target_computed_len, self.committed_len)

    def record_qprime_forward(self, computed_len: int) -> None:
        """Record a q' length produced by a real forward."""

        value = int(computed_len)
        if value < 0:
            raise ValueError("q' computed length cannot be negative")
        self.qprime_computed_len = value

    def record_target_forward(self, computed_len: int) -> None:
        """Record a target length produced by a real forward."""

        value = int(computed_len)
        if value < 0:
            raise ValueError("target computed length cannot be negative")
        self.target_computed_len = value

    def truncate(self, valid_prefix_len: int) -> None:
        """Invalidate logical suffixes while retaining physical pages.

        q' has a one-token causal offset: after a committed prefix of length
        ``L``, at most ``L`` q' positions may be considered valid.  The
        coordinator applies a stricter ``L - 1`` cap when it knows the last
        token was not forwarded; this method intentionally keeps the public
        operation unsurprising and never increases a stage length.
        """

        length = int(valid_prefix_len)
        if length < 0:
            raise ValueError("valid_prefix_len cannot be negative")
        length = min(length, len(self.committed_tokens))
        del self.committed_tokens[length:]
        self.committed_len = length
        self.qprime_computed_len = min(self.qprime_computed_len, length)
        self.target_computed_len = min(self.target_computed_len, length)

    def commit(
        self,
        tokens: Sequence[int],
        *,
        qprime_computed_len: int | None = None,
        target_computed_len: int | None = None,
    ) -> None:
        self.committed_tokens = [int(token) for token in tokens]
        self.committed_len = len(self.committed_tokens)
        if qprime_computed_len is None:
            self.qprime_computed_len = min(self.qprime_computed_len, max(0, self.committed_len - 1))
        else:
            self.qprime_computed_len = min(int(qprime_computed_len), self.committed_len)
        if target_computed_len is not None:
            self.target_computed_len = min(int(target_computed_len), self.committed_len)
        else:
            self.target_computed_len = min(self.target_computed_len, self.committed_len)

    def snapshot(self) -> "ViaSdRequestState":
        return ViaSdRequestState(
            request_id=self.request_id,
            committed_tokens=list(self.committed_tokens),
            committed_len=self.committed_len,
            qprime_computed_len=self.qprime_computed_len,
            target_computed_len=self.target_computed_len,
            active=self.active,
        )


@dataclass(frozen=True)
class ViaSdTargetRequest:
    """One compact target fallback item passed to a callback."""

    batch_row: int
    request_id: Hashable
    position: int
    draft_token: int
    prefix_tokens: tuple[int, ...]
    committed_len: int
    target_computed_len: int

    @property
    def row(self) -> int:
        return self.batch_row


@dataclass(frozen=True)
class ViaSdTargetDecision:
    """Normalized result of one target verification step."""

    accepted: bool
    token: int
    computed_len: int | None = None

    @property
    def rewritten(self) -> bool:
        return not self.accepted


@dataclass(frozen=True)
class ViaSdRequestResult:
    request_id: Hashable
    batch_row: int
    committed_tokens: tuple[int, ...]
    source: str
    accepted_draft_tokens: tuple[int, ...] = ()
    qprime_rewrite: tuple[int, int] | None = None
    target_fallback_positions: tuple[int, ...] = ()
    target_accepted_positions: tuple[int, ...] = ()
    target_rewrite: tuple[int, int] | None = None
    stop_position: int | None = None
    all_high: bool = False
    bonus_token: int | None = None
    scheduler_tokens: tuple[int, ...] = ()
    target_catchup_tokens: int = 0
    qprime_rollback_len: int = 0
    committed_len: int = 0
    qprime_computed_len: int = 0
    target_computed_len: int = 0

    @property
    def kept(self) -> int:
        return len(self.accepted_draft_tokens)

    @property
    def rewrite(self) -> tuple[int, int] | None:
        return self.qprime_rewrite or self.target_rewrite

    @property
    def requires_target(self) -> bool:
        return bool(self.target_fallback_positions)

    def rollback_slots(self, scheduled_drafts: int | None = None) -> int:
        """Return the vLLM scheduler rollback count for this result."""

        draft_count = len(self.accepted_draft_tokens) if scheduled_drafts is None else int(scheduled_drafts)
        result = draft_count + 1 - len(self.scheduler_tokens or self.committed_tokens)
        if result < 0:
            raise ValueError("VIA-SD output exceeds scheduled draft capacity")
        return result


@dataclass(frozen=True)
class ViaSdBatchResult:
    plan: ViaSdRoutePlan
    results: tuple[ViaSdRequestResult, ...]
    compact_fallbacks: tuple[ViaSdTargetRequest, ...] = ()
    target_calls: int = 0
    qprime_calls: int = 1
    observed_only: bool = False

    @property
    def all_high(self) -> bool:
        return bool(self.results) and all(result.all_high for result in self.results)

    @property
    def requires_target(self) -> bool:
        return bool(self.compact_fallbacks)

    @property
    def fallback_rows(self) -> tuple[int, ...]:
        return tuple(item.batch_row for item in self.compact_fallbacks)

    @property
    def by_request(self) -> dict[Hashable, ViaSdRequestResult]:
        return {result.request_id: result for result in self.results}


def _invoke(callback: Callable[..., Any], event: Any, *, kind: str) -> Any:
    """Invoke callbacks with a small, explicit compatibility surface."""

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(event)
    parameters = list(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        if kind == "target":
            return callback(
                request=event,
                request_id=event.request_id,
                position=event.position,
                draft_token=event.draft_token,
                prefix_tokens=event.prefix_tokens,
            )
        return callback(event=event)
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    count = len(positional)
    if count <= 1:
        return callback(event)
    if kind == "catchup":
        values = (event.request_id, event.target_computed_len, event.committed_len, event.prefix_tokens)
    else:
        values = (event.request_id, event.position, event.draft_token, event.prefix_tokens)
    return callback(*values[:count])


def _normalize_target_decision(value: Any, draft_token: int) -> ViaSdTargetDecision:
    if isinstance(value, ViaSdTargetDecision):
        return value
    if isinstance(value, bool):
        return ViaSdTargetDecision(value, draft_token)
    if isinstance(value, int):
        token = int(value)
        return ViaSdTargetDecision(token == draft_token, token)
    if isinstance(value, Mapping):
        token = value.get("token", value.get("replacement", value.get("rewrite", draft_token)))
        accepted = value.get("accepted", value.get("accept", int(token) == draft_token))
        computed = value.get("computed_len", value.get("target_computed_len"))
        return ViaSdTargetDecision(bool(accepted), int(token), None if computed is None else int(computed))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("empty target decision")
        if len(value) == 1:
            return _normalize_target_decision(value[0], draft_token)
        first, second = value[0], value[1]
        if isinstance(first, bool):
            return ViaSdTargetDecision(bool(first), int(draft_token if second is None else second))
        return _normalize_target_decision(first, draft_token)
    accepted = getattr(value, "accepted", None)
    token = getattr(value, "token", getattr(value, "replacement", None))
    if accepted is not None or token is not None:
        token = draft_token if token is None else int(token)
        return ViaSdTargetDecision(bool(accepted if accepted is not None else token == draft_token), token)
    raise ValueError(f"unsupported target decision type: {type(value).__name__}")


class ViaSdExecutionCoordinator:
    """Apply q' routes and compact only LOW positions to the target."""

    def __init__(
        self,
        accept_ratio: float = 0.7,
        escalate_ratio: float = 0.5,
        *,
        mode: str = "hierarchical",
        qprime_sampler: Callable[..., int] | None = None,
    ) -> None:
        self.accept_ratio, self.escalate_ratio = validate_thresholds(
            accept_ratio, escalate_ratio
        )
        if mode not in {"disabled", "observe", "hierarchical", "shadow", "via"}:
            raise ValueError(f"unknown VIA-SD mode: {mode}")
        self.mode = mode
        self.qprime_sampler = qprime_sampler
        self.states: dict[Hashable, ViaSdRequestState] = {}
        self.iteration = 0

    def state_for(
        self,
        request_id: Hashable,
        prefix_tokens: Sequence[int] | None = None,
        *,
        qprime_computed_len: int | None = None,
        target_computed_len: int | None = None,
    ) -> ViaSdRequestState:
        state = self.states.get(request_id)
        if state is None:
            state = ViaSdRequestState(
                request_id=request_id,
                committed_tokens=list(prefix_tokens or ()),
                qprime_computed_len=0 if qprime_computed_len is None else qprime_computed_len,
                target_computed_len=0 if target_computed_len is None else target_computed_len,
            )
            self.states[request_id] = state
        elif prefix_tokens is not None:
            state.reconcile(prefix_tokens)
        if qprime_computed_len is not None:
            state.record_qprime_forward(qprime_computed_len)
        if target_computed_len is not None:
            state.record_target_forward(target_computed_len)
        return state

    def truncate(self, request_id: Hashable, valid_prefix_len: int) -> None:
        state = self.states.get(request_id)
        if state is not None:
            state.truncate(valid_prefix_len)

    def discard(self, request_ids: Sequence[Hashable] | None = None) -> None:
        if request_ids is None:
            self.states.clear()
            return
        for request_id in request_ids:
            self.states.pop(request_id, None)

    def clear(self) -> None:
        self.discard()

    def _sample_qprime(
        self,
        request_id: Hashable,
        decision: ViaSdDecision,
        logits: Sequence[float] | None,
        sampling_params: Any | None,
    ) -> int:
        if logits is None:
            raise ValueError("q' logits are required for a MEDIUM rewrite")
        if self.qprime_sampler is not None:
            callback = self.qprime_sampler
            try:
                signature = inspect.signature(callback)
                count = len(
                    [
                        parameter
                        for parameter in signature.parameters.values()
                        if parameter.kind
                        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    ]
                )
            except (TypeError, ValueError):
                count = 1
            values = (request_id, decision.position, logits, sampling_params)
            return int(callback(*values[: max(1, count)]))
        return sample_from_logits(logits, sampling_params)

    def _catch_up(
        self,
        state: ViaSdRequestState,
        prefix_tokens: Sequence[int],
        target_catchup: Callable[..., Any] | None,
    ) -> int:
        needed = len(prefix_tokens)
        current = state.target_computed_len
        if current >= needed:
            return 0
        if target_catchup is None:
            raise RuntimeError(
                f"target KV catch-up is required for {state.request_id!r}: "
                f"computed={current}, committed={needed}"
            )
        event = ViaSdTargetRequest(
            batch_row=-1,
            request_id=state.request_id,
            position=max(0, needed - 1),
            draft_token=-1,
            prefix_tokens=tuple(int(token) for token in prefix_tokens),
            committed_len=needed,
            target_computed_len=current,
        )
        returned = _invoke(target_catchup, event, kind="catchup")
        if isinstance(returned, Mapping):
            returned = returned.get("computed_len", returned.get("target_computed_len"))
        if returned is None:
            raise RuntimeError("target catch-up callback must return the actual computed length")
        computed = int(returned)
        if computed < needed:
            raise RuntimeError(
                f"target catch-up ended before committed prefix: got={computed}, need={needed}"
            )
        if computed < current:
            raise RuntimeError("target catch-up cannot move target KV backwards")
        state.record_target_forward(computed)
        return computed - current

    def _target_batch_decisions(
        self,
        requests: Sequence[ViaSdTargetRequest],
        callback: Callable[..., Any] | None,
    ) -> dict[tuple[Hashable, int], ViaSdTargetDecision]:
        if not requests or callback is None:
            return {}
        returned = callback(tuple(requests))
        if isinstance(returned, Mapping):
            result: dict[tuple[Hashable, int], ViaSdTargetDecision] = {}
            for request in requests:
                value = returned.get(
                    (request.request_id, request.position),
                    returned.get(request.request_id),
                )
                if value is not None:
                    result[(request.request_id, request.position)] = _normalize_target_decision(
                        value, request.draft_token
                    )
            return result
        if isinstance(returned, Sequence) and not isinstance(returned, (str, bytes)):
            if len(returned) != len(requests):
                raise ValueError("compact target callback returned the wrong number of decisions")
            return {
                (request.request_id, request.position): _normalize_target_decision(value, request.draft_token)
                for request, value in zip(requests, returned)
            }
        raise ValueError("compact target callback must return a mapping or sequence")

    def run(
        self,
        request_ids: Sequence[Hashable],
        prefix_token_ids: Sequence[Sequence[int]],
        draft_token_ids: Any,
        qprime_logits: Any | None = None,
        *,
        valid_lengths: Sequence[int] | None = None,
        batch_rows: Sequence[int] | None = None,
        target_verify: Callable[..., Any] | None = None,
        target_verify_batch: Callable[..., Any] | None = None,
        target_catchup: Callable[..., Any] | None = None,
        qprime_forward: Callable[..., Any] | None = None,
        qprime_sampler: Callable[..., int] | None = None,
        sampling_params: Any | Sequence[Any] | Mapping[Hashable, Any] | None = None,
        qprime_computed_lengths: Sequence[int] | None = None,
        target_computed_lengths: Sequence[int] | None = None,
        mode: str | None = None,
    ) -> ViaSdBatchResult:
        """Run one hierarchical verification round.

        ``target_verify_batch`` receives exactly the compact LOW events and is
        useful when the runner can build a compact attention batch.  When it is
        absent, ``target_verify`` is invoked one LOW position at a time.  Both
        forms preserve original request rows in every event.
        """

        active_mode = self.mode if mode is None else mode
        if active_mode in {"via", "shadow"}:
            active_mode = "hierarchical" if active_mode == "via" else "observe"
        if active_mode not in {"disabled", "observe", "hierarchical"}:
            raise ValueError(f"unknown VIA-SD mode: {active_mode}")
        ids = tuple(request_ids)
        prefixes = tuple(tuple(int(token) for token in prefix) for prefix in prefix_token_ids)
        if len(ids) != len(prefixes):
            raise ValueError("request IDs and prefixes must have equal length")
        drafts = draft_token_ids
        if qprime_logits is None and qprime_forward is not None:
            qprime_logits = qprime_forward(ids, prefixes, drafts)
        if qprime_logits is None:
            # Empty drafts are legal only for a disabled/observe no-op.
            if active_mode in {"disabled", "observe"}:
                plan = ViaSdRoutePlan(ids, tuple(() for _ in ids), tuple(() for _ in ids), tuple(() for _ in ids), ())
                return ViaSdBatchResult(plan, tuple(), observed_only=active_mode == "observe", qprime_calls=0)
            raise ValueError("q' logits or qprime_forward callback is required")

        plan = build_route_plan(
            qprime_logits,
            drafts,
            self.accept_ratio,
            self.escalate_ratio,
            request_ids=ids,
            valid_lengths=valid_lengths,
            batch_rows=batch_rows,
        )
        for row, request_id in enumerate(ids):
            self.state_for(
                request_id,
                prefixes[row],
                qprime_computed_len=(
                    None
                    if qprime_computed_lengths is None
                    else int(qprime_computed_lengths[row])
                ),
                target_computed_len=(
                    None
                    if target_computed_lengths is None
                    else int(target_computed_lengths[row])
                ),
            )

        if active_mode == "disabled":
            return ViaSdBatchResult(plan, tuple(), observed_only=False)
        if active_mode == "observe":
            return ViaSdBatchResult(plan, tuple(), observed_only=True)

        # Build compact fallback events before mutating any request.  The
        # prefix for a LOW position includes HIGH draft tokens before it; this
        # is the causal prefix the target actually has to catch up to.  LOW
        # positions after a MEDIUM rewrite are not candidates at all because
        # that rewrite terminates the request's block.
        compact_events: list[ViaSdTargetRequest] = []
        for row, request_id in enumerate(ids):
            working_prefix = list(prefixes[row])
            for decision in plan.decisions[row]:
                if decision.route is ViaSdRoute.HIGH:
                    working_prefix.append(decision.draft_token)
                    continue
                if decision.route is ViaSdRoute.MEDIUM:
                    break
                compact_events.append(
                    ViaSdTargetRequest(
                        batch_row=plan.batch_rows[row] if plan.batch_rows else row,
                        request_id=request_id,
                        position=decision.position,
                        draft_token=decision.draft_token,
                        prefix_tokens=tuple(working_prefix),
                        committed_len=len(working_prefix),
                        target_computed_len=self.states[request_id].target_computed_len,
                    )
                )
                # LOW ends this block even when target accepts the token.
                break
        compact_requests = tuple(compact_events)
        batch_decisions = self._target_batch_decisions(compact_requests, target_verify_batch)
        results: list[ViaSdRequestResult] = []
        target_calls = 0
        for row, request_id in enumerate(ids):
            state = self.states[request_id]
            prefix = list(prefixes[row])
            working = list(prefix)
            decisions = plan.decisions[row]
            accepted_drafts: list[int] = []
            fallback_positions: list[int] = []
            target_accepted_positions: list[int] = []
            qprime_rewrite: tuple[int, int] | None = None
            target_rewrite: tuple[int, int] | None = None
            source = "qprime_high"
            stop_position: int | None = None
            catchup_tokens = 0
            qprime_valid = len(prefix)
            target_valid = state.target_computed_len
            sampled_params = (
                sampling_params.get(request_id)
                if isinstance(sampling_params, Mapping)
                else sampling_params[row]
                if isinstance(sampling_params, Sequence) and not isinstance(sampling_params, (str, bytes))
                and len(sampling_params) == len(ids)
                else sampling_params
            )
            for decision in decisions:
                if decision.route is ViaSdRoute.HIGH:
                    working.append(decision.draft_token)
                    accepted_drafts.append(decision.draft_token)
                    qprime_valid = len(working) - 1
                    source = "qprime_high"
                    continue
                if decision.route is ViaSdRoute.MEDIUM:
                    logits = plan.logits_for(row, decision.position)
                    sampler = self.qprime_sampler if qprime_sampler is None else qprime_sampler
                    if sampler is None:
                        replacement = sample_from_logits(logits, sampled_params)
                    else:
                        try:
                            signature = inspect.signature(sampler)
                            count = len(
                                [
                                    parameter
                                    for parameter in signature.parameters.values()
                                    if parameter.kind
                                    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                ]
                            )
                        except (TypeError, ValueError):
                            count = 1
                        replacement = int(
                            sampler(
                                *((request_id, decision.position, logits, sampled_params)[: max(1, count)])
                            )
                        )
                    working.append(replacement)
                    qprime_rewrite = (decision.draft_token, replacement)
                    source = "qprime_rewrite"
                    stop_position = decision.position
                    qprime_valid = len(working) - 1
                    break

                # LOW: the target must own this position.  Catch up from the
                # target's actual computed length before asking for logits.
                fallback_positions.append(decision.position)
                catchup_tokens += self._catch_up(state, working, target_catchup)
                target_valid = state.target_computed_len
                request = ViaSdTargetRequest(
                    batch_row=plan.batch_rows[row] if plan.batch_rows else row,
                    request_id=request_id,
                    position=decision.position,
                    draft_token=decision.draft_token,
                    prefix_tokens=tuple(working),
                    committed_len=len(working),
                    target_computed_len=target_valid,
                )
                key = (request_id, decision.position)
                if key in batch_decisions:
                    target_decision = batch_decisions[key]
                elif target_verify is not None:
                    target_decision = _normalize_target_decision(
                        _invoke(target_verify, request, kind="target"), decision.draft_token
                    )
                else:
                    raise RuntimeError(
                        f"LOW q' route requires target_verify for request {request_id!r}"
                    )
                target_calls += 1
                if target_decision.computed_len is not None:
                    if target_decision.computed_len < state.target_computed_len:
                        raise RuntimeError("target verification moved computed length backwards")
                    state.record_target_forward(target_decision.computed_len)
                else:
                    # A target callback represents an actual verification
                    # forward for this position.  Advance by one only after it
                    # has returned; catch-up above is never inferred.
                    state.record_target_forward(max(state.target_computed_len, len(working) + 1))
                target_valid = state.target_computed_len
                if target_decision.accepted:
                    working.append(decision.draft_token)
                    accepted_drafts.append(decision.draft_token)
                    target_accepted_positions.append(decision.position)
                    qprime_valid = len(working) - 1
                    source = "target_accept"
                    stop_position = decision.position
                    break
                working.append(target_decision.token)
                target_rewrite = (decision.draft_token, target_decision.token)
                source = "target_rewrite"
                stop_position = decision.position
                qprime_valid = len(working) - 1
                break

            else:
                if decisions:
                    source = "qprime_high"
                stop_position = None
                qprime_valid = min(qprime_valid, len(working) - 1)

            # Any suffix after a rewrite was computed from the old draft and
            # is logically stale.  Keep physical pages untouched, but cap both
            # stage lengths before the next round can request reuse.
            state.commit(
                working,
                qprime_computed_len=qprime_valid,
                target_computed_len=target_valid,
            )
            bonus_token = None
            scheduler_tokens = tuple(working[len(prefix) :])
            all_high = bool(decisions) and all(
                decision.route is ViaSdRoute.HIGH for decision in decisions
            )
            results.append(
                ViaSdRequestResult(
                    request_id=request_id,
                    batch_row=plan.batch_rows[row] if plan.batch_rows else row,
                    committed_tokens=tuple(working[len(prefix) :]),
                    source=source,
                    accepted_draft_tokens=tuple(accepted_drafts),
                    qprime_rewrite=qprime_rewrite,
                    target_fallback_positions=tuple(fallback_positions),
                    target_accepted_positions=tuple(target_accepted_positions),
                    target_rewrite=target_rewrite,
                    stop_position=stop_position,
                    all_high=all_high,
                    bonus_token=bonus_token,
                    scheduler_tokens=scheduler_tokens,
                    target_catchup_tokens=catchup_tokens,
                    qprime_rollback_len=max(0, len(prefix) + len(decisions) - qprime_valid),
                    committed_len=state.committed_len or 0,
                    qprime_computed_len=state.qprime_computed_len,
                    target_computed_len=state.target_computed_len,
                )
            )
        self.iteration += 1
        return ViaSdBatchResult(
            plan=plan,
            results=tuple(results),
            compact_fallbacks=compact_requests,
            target_calls=target_calls,
        )

    execute = run
    verify = run


__all__ = [
    "ViaSdBatchResult",
    "ViaSdExecutionCoordinator",
    "ViaSdRequestResult",
    "ViaSdRequestState",
    "ViaSdTargetDecision",
    "ViaSdTargetRequest",
]
