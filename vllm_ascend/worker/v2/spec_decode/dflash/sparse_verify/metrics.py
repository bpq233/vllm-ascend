# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in request-local records; callers supply synchronized timing evidence."""

from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any

MAX_DETAIL_RECORDS = 2048


@dataclass(frozen=True)
class RoundMetrics:
    request_id: str
    round_index: int
    proposed: int
    temporary_accepted: int
    first_rejection: int
    confidences: tuple[float, ...]
    outcome: str
    feature_source: str
    feature_shape: tuple[int, ...]
    position_start: int
    position_end: int
    dflash_seconds: float = 0.0
    sparse_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class FinalMetrics:
    request_id: str
    method: str
    proposed: int
    accepted: int
    first_rejection: int
    target_seconds: float
    verification_seconds: float
    emitted: int
    # Optional same-forward reference for approximate policies. Accept-all
    # leaves these unknown rather than computing unused candidate LM-head rows.
    strict_accepted_length: int | None = None
    additional_accepted: int | None = None


class SparseVerificationMetrics:
    """Instance-scoped metrics with no global state or work when disabled.

    Durations must come from an explicitly synchronized benchmark or device
    event. Python submission time is not presented as NPU execution time.
    Graph counters describe observed capture/replay calls, never requested
    configuration. The controller owns retention and logging of snapshots.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.rounds: deque[RoundMetrics] = deque(maxlen=MAX_DETAIL_RECORDS)
        self.finals: deque[FinalMetrics] = deque(maxlen=MAX_DETAIL_RECORDS)
        self.graph_calls: Counter[tuple[str, str]] = Counter()
        # Fixed-key aggregate counters retain the lifetime totals without
        # retaining unbounded request IDs, confidence vectors, or tensor shapes.
        self.totals: Counter[str] = Counter()

    def record_round(self, record: RoundMetrics) -> None:
        if self.enabled:
            self.rounds.append(record)
            self.totals["rounds"] += 1
            self.totals["temporary_accepted"] += record.temporary_accepted
            self.totals["middle_threshold_retries"] += record.outcome == "retry"
            self.totals["low_threshold_stops"] += record.outcome == "low_confidence"

    def record_final(self, record: FinalMetrics) -> None:
        if self.enabled:
            self.finals.append(record)
            self.totals["finals"] += 1
            self.totals["emitted"] += record.emitted
            self.totals["final_accepted"] += record.accepted
            if record.method == "accept_all":
                self.totals["accept_all_passed"] += record.accepted
            if record.strict_accepted_length is not None:
                self.totals["strict_reference_records"] += 1
                self.totals["strict_reference_accepted"] += record.strict_accepted_length
            if record.additional_accepted is not None:
                self.totals["approx_reference_records"] += 1
                self.totals["approx_extra_accepted"] += record.additional_accepted

    def record_graph(self, model: str, action: str) -> None:
        if self.enabled:
            if model not in ("dflash", "sparse", "target") or action not in ("capture", "replay", "eager"):
                raise ValueError("graph evidence must name a model and an observed capture/replay/eager action")
            self.graph_calls[(model, action)] += 1

    def snapshot(self, elapsed_seconds: float | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {}
        emitted = self.totals["emitted"]
        accepted = self.totals["final_accepted"]
        final_count = self.totals["finals"]
        per_request_rounds = Counter(record.request_id for record in self.rounds)
        result: dict[str, Any] = {
            "aggregate_scope": "since_last_clear",
            "detail_scope": "most_recent_records",
            "detail_limit_per_kind": MAX_DETAIL_RECORDS,
            "total_round_records": self.totals["rounds"],
            "total_final_records": final_count,
            "retained_round_records": len(self.rounds),
            "retained_final_records": len(self.finals),
            "rounds": [asdict(record) for record in self.rounds],
            "finals": [asdict(record) for record in self.finals],
            # This map describes only the retained detail window. Lifetime
            # request identifiers are deliberately not accumulated.
            "rounds_per_request": dict(per_request_rounds),
            "rounds_per_request_scope": "retained_round_records",
            "temporary_accepted": self.totals["temporary_accepted"],
            "final_accepted": accepted,
            "mean_target_accepted_length": accepted / final_count if final_count else 0.0,
            "high_threshold_accepts": self.totals["temporary_accepted"],
            "middle_threshold_retries": self.totals["middle_threshold_retries"],
            "low_threshold_stops": self.totals["low_threshold_stops"],
            "accept_all_passed": self.totals["accept_all_passed"],
            "strict_reference_records": self.totals["strict_reference_records"],
            "strict_reference_accepted": self.totals["strict_reference_accepted"],
            "approx_reference_records": self.totals["approx_reference_records"],
            "approx_extra_accepted": self.totals["approx_extra_accepted"],
            "graph_calls": {f"{model}.{action}": count for (model, action), count in self.graph_calls.items()},
        }
        if elapsed_seconds is not None:
            if elapsed_seconds <= 0:
                raise ValueError("elapsed_seconds must be positive")
            result["emitted_tokens_per_second"] = emitted / elapsed_seconds
            result["seconds_per_emitted_token"] = elapsed_seconds / emitted if emitted else None
        return result

    def clear(self) -> None:
        self.rounds.clear()
        self.finals.clear()
        self.graph_calls.clear()
        self.totals.clear()
