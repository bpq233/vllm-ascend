# SPDX-License-Identifier: Apache-2.0
"""Request-local speculative state. Nothing here commits scheduler history."""

from dataclasses import dataclass, field


@dataclass
class SpeculativeState:
    request_id: str
    committed_tokens: tuple[int, ...]
    max_candidates: int
    eos_token_ids: frozenset[int] = frozenset()
    finished: bool = False
    intermediate_accepted: list[int] = field(default_factory=list)
    current_draft: list[int] = field(default_factory=list)
    round_id: int = 0
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        self.committed_tokens = tuple(self.committed_tokens)
        self.eos_token_ids = frozenset(self.eos_token_ids)
        if self.max_candidates < 0:
            raise ValueError("max_candidates must be nonnegative")

    @property
    def context(self) -> tuple[int, ...]:
        return self.committed_tokens + tuple(self.intermediate_accepted)

    @property
    def remaining(self) -> int:
        return max(0, self.max_candidates - len(self.intermediate_accepted))

    def should_continue(self) -> bool:
        if self.finished:
            self.stop_reason = "finished"
        elif self.remaining == 0:
            self.stop_reason = self.stop_reason or "max_candidates"
        return self.stop_reason is None

    def accept_prefix(self, tokens: list[int], should_stop: bool = False) -> None:
        if tokens != self.current_draft[: len(tokens)]:
            raise ValueError("Verifier acceptance must be a prefix of current_draft")
        if not tokens:
            self.stop_reason = "empty_acceptance"
        for token in tokens[: self.remaining]:
            self.intermediate_accepted.append(token)
            if token in self.eos_token_ids:
                self.stop_reason = "eos"
                break
        self.current_draft = []
        if should_stop:
            self.stop_reason = self.stop_reason or "verifier_stop"
        self.should_continue()
