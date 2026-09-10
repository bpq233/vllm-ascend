# SPDX-License-Identifier: Apache-2.0
"""Bounded speculative expansion with independent per-request stopping."""

import logging
import sys
from time import perf_counter

from .interfaces import Drafter, Verifier
from .metrics import PipelineMetrics
from .state import SpeculativeState

logger = logging.getLogger(__name__)


class SpeculativePipeline:
    def __init__(
        self,
        verifier: Verifier,
        secondary_drafter: Drafter,
        num_rounds: int,
        debug_logging: bool = False,
        metrics_enabled: bool = False,
    ):
        if isinstance(num_rounds, bool) or not isinstance(num_rounds, int) or num_rounds < 1:
            raise ValueError("num_rounds must be a positive integer")
        self.verifier = verifier
        self.secondary_drafter = secondary_drafter
        self.num_rounds = num_rounds
        self.debug_logging = debug_logging
        self.metrics = PipelineMetrics(metrics_enabled)

    def run(self, states: list[SpeculativeState], primary_drafts: dict[str, list[int]]) -> dict[str, list[int]]:
        if len({s.request_id for s in states}) != len(states):
            raise ValueError("Request IDs must be unique")
        if any(s.intermediate_accepted or s.current_draft or s.stop_reason for s in states):
            raise ValueError("Each pipeline run requires fresh speculative state")
        try:
            for state in states:
                if state.should_continue():
                    state.current_draft = list(primary_drafts[state.request_id][: state.remaining])
                if self.debug_logging:
                    logger.debug("request_id=%s primary_draft_tokens=%s", state.request_id, state.current_draft)
            for round_id in range(self.num_rounds):
                active = []
                for state in states:
                    if not state.should_continue():
                        continue
                    if not state.current_draft:
                        state.stop_reason = "empty_draft"
                        continue
                    state.round_id = round_id
                    active.append(state)
                if not active:
                    break
                if self.debug_logging:
                    for state in active:
                        logger.debug(
                            "request_id=%s round_id=%s intermediate_input_tokens=%s",
                            state.request_id,
                            round_id,
                            state.context + tuple(state.current_draft),
                        )
                input_count = sum(len(s.current_draft) for s in active)
                start = perf_counter()
                results = self.verifier.verify(active)
                elapsed = (perf_counter() - start) * 1000
                accepted_count = 0
                for state in active:
                    result = results[state.request_id]
                    before = len(state.intermediate_accepted)
                    state.accept_prefix(list(result.accepted_tokens), result.should_stop)
                    accepted_count += len(state.intermediate_accepted) - before
                    if self.debug_logging:
                        logger.debug(
                            "request_id=%s round_id=%s intermediate_topk=%s "
                            "intermediate_accepted_tokens=%s intermediate_accepted_length=%s stop_reason=%s",
                            state.request_id,
                            round_id,
                            result.topk,
                            state.intermediate_accepted[before:],
                            len(state.intermediate_accepted) - before,
                            state.stop_reason,
                        )
                self.metrics.record("intermediate_verifier", input_count, elapsed, accepted_count)
                if round_id + 1 == self.num_rounds:
                    break
                continuing = [s for s in active if s.should_continue()]
                if not continuing:
                    break
                start = perf_counter()
                drafts = self.secondary_drafter.propose(continuing)
                elapsed = (perf_counter() - start) * 1000
                for state in continuing:
                    state.current_draft = list(drafts[state.request_id][: state.remaining])
                    if self.debug_logging:
                        logger.debug(
                            "request_id=%s round_id=%s secondary_draft_tokens=%s",
                            state.request_id,
                            round_id,
                            state.current_draft,
                        )
                self.metrics.record("secondary_drafter", sum(len(s.current_draft) for s in continuing), elapsed)
            for state in states:
                state.stop_reason = state.stop_reason or "num_rounds"
                if self.debug_logging:
                    logger.debug(
                        "request_id=%s total_candidate_tokens=%s committed_tokens=%s",
                        state.request_id,
                        state.intermediate_accepted,
                        state.committed_tokens,
                    )
            return {s.request_id: list(s.intermediate_accepted) for s in states}
        finally:
            # Do not let one backend's failure prevent cleanup of the others.
            # Roll back before final target verification, so target rejection
            # cannot leave speculative intermediate KV live into the next run.
            original_error = sys.exc_info()[0] is not None
            cleanup_error = None
            for state in states:
                state.current_draft = []
                for backend in (self.verifier, self.secondary_drafter):
                    try:
                        backend.rollback(state.request_id, len(state.committed_tokens))
                    except Exception as exc:
                        cleanup_error = cleanup_error or exc
                        logger.exception("Speculative KV rollback failed for request %s", state.request_id)
            if cleanup_error is not None and not original_error:
                raise cleanup_error
