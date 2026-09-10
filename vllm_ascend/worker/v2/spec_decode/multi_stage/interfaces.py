# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Protocol

from .state import SpeculativeState


@dataclass
class VerificationResult:
    accepted_tokens: list[int]
    should_stop: bool = False
    topk: list[list[int]] | None = None


class Verifier(Protocol):
    def verify(self, states: list[SpeculativeState]) -> dict[str, VerificationResult]: ...

    def rollback(self, request_id: str, committed_length: int) -> None:
        """Idempotently discard speculative KV, preserving committed KV."""
        ...


class Drafter(Protocol):
    def propose(self, states: list[SpeculativeState]) -> dict[str, list[int]]: ...

    def rollback(self, request_id: str, committed_length: int) -> None:
        """Idempotently discard speculative KV, preserving committed KV."""
        ...
