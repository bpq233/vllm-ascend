# SPDX-License-Identifier: Apache-2.0
"""Optional host-side stage counters. Durations are wall time, not NPU events."""

from dataclasses import dataclass


@dataclass
class StageMetrics:
    calls: int = 0
    tokens: int = 0
    milliseconds: float = 0.0
    accepted_tokens: int = 0


class PipelineMetrics:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.stages: dict[str, StageMetrics] = {}

    def record(self, stage: str, tokens: int, milliseconds: float, accepted_tokens: int = 0) -> None:
        if not self.enabled:
            return
        item = self.stages.setdefault(stage, StageMetrics())
        item.calls += 1
        item.tokens += tokens
        item.milliseconds += milliseconds
        item.accepted_tokens += accepted_tokens
