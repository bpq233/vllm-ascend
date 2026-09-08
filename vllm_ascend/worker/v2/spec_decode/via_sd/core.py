"""Small device-independent VIA-SD state machine.

This module mirrors the causal contract used by the NPU runner.  A ``Stage``
backend only needs an ``append(ids, start, row_start)`` method returning logits
for the requested input rows, which makes the implementation useful for unit
tests and for backend bring-up before an Ascend device is available.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .routing import ViaSdRoute, relative_confidence, route_from_score


@dataclass(frozen=True)
class Commit:
    token_ids: tuple[int, ...]
    kept: int
    source: str
    reached: int
    all_keep: bool
    stop_reason: str | None

    def rollback_slots(self, scheduled_drafts: int) -> int:
        result = int(scheduled_drafts) + 1 - len(self.token_ids)
        if result < 0:
            raise ValueError("commit exceeds scheduled output capacity")
        return result


class Stage:
    """Logical KV stage with explicit causal truncation."""

    def __init__(self, backend: Any, chunk_size: int = 256):
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive")
        self.backend = backend
        self.chunk_size = int(chunk_size)
        self.tokens: list[int] = []
        self.next_logits: Any | None = None
        self.calls = 0
        self.processed_tokens = 0

    @property
    def computed_len(self) -> int:
        return len(self.tokens)

    @property
    def valid_len(self) -> int:
        return len(self.tokens)

    def truncate(self, length: int) -> None:
        length = int(length)
        if length < 0:
            raise ValueError("negative cache prefix")
        if length < len(self.tokens):
            del self.tokens[length:]
            # A cached last-row distribution predicts a token after the old
            # suffix and therefore cannot survive a logical rollback.
            self.next_logits = None

    def reconcile(self, prefix: list[int] | tuple[int, ...]) -> None:
        common = 0
        for old, new in zip(self.tokens, prefix):
            if old != int(new):
                break
            common += 1
        self.truncate(common)

    def _append(self, ids: list[int] | tuple[int, ...], row_start: int):
        ids = [int(token) for token in ids]
        if not ids or not 0 <= int(row_start) < len(ids):
            raise ValueError("empty append or invalid output row")
        rows = self.backend.append(ids, len(self.tokens), int(row_start))
        shape = getattr(rows, "shape", None)
        if shape is None or int(shape[0]) != len(ids) - int(row_start):
            raise ValueError("backend returned wrong number of logits rows")
        self.tokens.extend(ids)
        self.next_logits = rows[-1]
        self.calls += 1
        self.processed_tokens += len(ids)
        return rows

    def next(self, prefix: list[int] | tuple[int, ...]):
        if not prefix:
            raise ValueError("supply a nonempty prefix")
        prefix = [int(token) for token in prefix]
        self.reconcile(prefix)
        if len(self.tokens) == len(prefix):
            if self.next_logits is not None:
                return self.next_logits
            # The final token is present but its output row was not retained.
            self.truncate(len(prefix) - 1)
        while len(self.tokens) < len(prefix):
            end = min(len(prefix), len(self.tokens) + self.chunk_size)
            missing = prefix[len(self.tokens) : end]
            self._append(missing, len(missing) - 1)
        return self.next_logits

    def score(
        self,
        prefix: list[int] | tuple[int, ...],
        draft: list[int] | tuple[int, ...],
        *,
        bonus: bool = False,
    ):
        """Return rows predicting each draft token with causal alignment."""

        if not prefix or not draft:
            raise ValueError("score requires a nonempty prefix and draft")
        prefix = [int(token) for token in prefix]
        draft = [int(token) for token in draft]
        self.reconcile(prefix)
        # Keep the last committed input pending so its row predicts draft[0].
        self.truncate(len(prefix) - 1)
        history_end = len(prefix) - 1
        while len(self.tokens) < history_end:
            end = min(history_end, len(self.tokens) + self.chunk_size)
            self._append(prefix[len(self.tokens) : end], end - len(self.tokens) - 1)
        inputs = [prefix[-1]] + (draft if bonus else draft[:-1])
        return self._append(inputs, 0)


class Session:
    """Greedy reference implementation of exact/observe/hierarchical modes."""

    def __init__(
        self,
        prefix,
        prompt_len: int,
        max_tokens: int,
        stages: dict[str, Stage | None],
        numerics: Any,
        config: Any,
        eos_ids=(),
    ):
        if not prefix or not 0 < int(prompt_len) <= len(prefix) or int(max_tokens) < 1:
            raise ValueError("invalid request length")
        if set(stages) != {"p", "u", "q"}:
            raise ValueError("stages must contain p, u and q")
        self.tokens = [int(token) for token in prefix]
        self.prompt_len = int(prompt_len)
        self.max_tokens = int(max_tokens)
        self.stages = stages
        self.ops = numerics
        self.config = config
        self.eos_ids = {int(token) for token in eos_ids}
        self.stats = Counter()
        self.last_draft: list[int] = []
        self.stopped = False

    @property
    def committed_len(self) -> int:
        return len(self.tokens)

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - (len(self.tokens) - self.prompt_len))

    def _mode(self) -> str:
        mode = getattr(self.config, "mode", "hierarchical")
        aliases = {"via": "hierarchical", "shadow": "observe", "exact": "disabled"}
        return aliases.get(mode, mode)

    def _route_rows(self, rows, draft):
        routes = getattr(self.ops, "routes", None)
        if callable(routes):
            return routes(rows, draft, self.config)
        return [
            route_from_score(
                relative_confidence(row, token),
                getattr(self.config, "accept_ratio", 0.7),
                getattr(self.config, "escalate_ratio", 0.5),
            )
            for row, token in zip(rows, draft)
        ]

    def _argmax(self, row):
        argmax = getattr(self.ops, "argmax", None)
        if callable(argmax):
            return int(argmax(row))
        return max(range(len(row)), key=lambda index: float(row[index]))

    def _commit(self, ids, kept, source, reached, all_keep):
        output: list[int] = []
        stop_reason = None
        for token in ids[: self.remaining]:
            token = int(token)
            output.append(token)
            if token in self.eos_ids:
                stop_reason = "eos"
                break
        if len(output) == self.remaining and stop_reason is None:
            stop_reason = "length"
        if not output:
            raise ValueError("active decode must make progress")
        kept = min(int(kept), len(output))
        if len(output) <= kept:
            source = "keep"
        self.tokens.extend(output)
        self.stopped = stop_reason is not None
        # Reconcile every stage.  Physical backend memory may retain stale
        # pages, but no logical stage can reuse a divergent suffix.
        for stage in self.stages.values():
            if stage is not None:
                stage.reconcile(self.tokens)
                stage.truncate(max(0, len(self.tokens) - 1))
        result = Commit(
            tuple(output),
            kept,
            source,
            min(int(reached), len(output)),
            bool(all_keep and len(output) == len(ids)),
            stop_reason,
        )
        self.stats["committed_tokens"] += len(output)
        self.stats["kept_draft_tokens"] += result.kept
        self.stats["reached_decisions"] += result.reached
        self.stats["all_keep_rounds"] += int(result.all_keep)
        if source in {"qprime_rewrite", "slim"}:
            self.stats["qprime_rewrite_rounds"] += 1
        if source in {"target_rewrite", "full"}:
            self.stats["target_rewrite_rounds"] += 1
        return result

    def first_token(self):
        if self.stopped or not self.remaining:
            raise ValueError("request has no output capacity")
        row = self.stages["q"].next(self.tokens)
        return self._commit([self._argmax(row)], 0, "prefill", 0, False)

    def propose(self, count: int):
        if self.stopped:
            self.last_draft = []
            return []
        count = min(int(count), self.remaining)
        candidate_prefix = list(self.tokens)
        draft: list[int] = []
        for _ in range(count):
            row = self.stages["p"].next(candidate_prefix)
            token = self._argmax(row)
            draft.append(token)
            candidate_prefix.append(token)
            if token in self.eos_ids:
                break
        self.last_draft = draft
        self.stats["drafted_tokens"] += len(draft)
        return list(draft)

    def verify(self, draft):
        if self.stopped or not self.remaining:
            raise ValueError("no decode allowed after stop")
        draft = [int(token) for token in draft]
        if draft != self.last_draft[: len(draft)]:
            raise ValueError("scheduler drafts differ from proposed prefix")
        if len(draft) > self.remaining:
            raise ValueError("scheduler exceeds remaining output budget")
        self.stats["decode_rounds"] += 1
        mode = self._mode()
        if not draft:
            return self._commit([self._argmax(self.stages["q"].next(self.tokens))], 0, "full", 0, False)
        if mode == "observe":
            if self.stages["u"] is not None:
                self.stages["u"].score(self.tokens, draft)
            # Observation mode must not alter generated output; exact target
            # verification remains the source of truth.
            rows = self.stages["q"].score(self.tokens, draft, bonus=True)
            target = [self._argmax(row) for row in rows]
            for index, token in enumerate(draft):
                if token != target[index]:
                    return self._commit(draft[: index] + [target[index]], index, "full", index + 1, False)
                if token in self.eos_ids or index + 1 == self.remaining:
                    return self._commit(draft[: index + 1], index + 1, "keep", index + 1, index + 1 == len(draft))
            return self._commit(draft + [target[-1]], len(draft), "bonus", len(draft), True)
        if mode == "disabled":
            rows = self.stages["q"].score(self.tokens, draft, bonus=True)
            target = [self._argmax(row) for row in rows]
            for index, token in enumerate(draft):
                if token != target[index]:
                    return self._commit(draft[:index] + [target[index]], index, "full", index + 1, False)
                if token in self.eos_ids or index + 1 == self.remaining:
                    return self._commit(draft[: index + 1], index + 1, "keep", index + 1, index + 1 == len(draft))
            return self._commit(draft + [target[-1]], len(draft), "bonus", len(draft), True)

        if self.stages["u"] is None:
            raise ValueError("hierarchical mode requires a q' stage")
        rows = self.stages["u"].score(self.tokens, draft)
        routes = self._route_rows(rows, draft)
        for index, (token, route) in enumerate(zip(draft, routes)):
            route_value = route.value if isinstance(route, ViaSdRoute) else str(route).lower()
            if route_value in {"high", "0", "accept"}:
                if token in self.eos_ids or index + 1 == self.remaining:
                    return self._commit(draft[: index + 1], index + 1, "keep", index + 1, index + 1 == len(draft))
                continue
            if route_value in {"medium", "1", "rewrite", "slim", "qprime_rewrite"}:
                replacement = self._argmax(rows[index])
                return self._commit(draft[:index] + [replacement], index, "qprime_rewrite", index + 1, False)
            # LOW is the sole target fallback.  ``Stage.next`` catches the
            # target KV up to the committed prefix using an actual forward.
            target_row = self.stages["q"].next(self.tokens + draft[:index])
            replacement = self._argmax(target_row)
            if replacement == token:
                if token in self.eos_ids or index + 1 == self.remaining:
                    return self._commit(draft[: index + 1], index + 1, "target_accept", index + 1, False)
                continue
            return self._commit(draft[:index] + [replacement], index, "target_rewrite", index + 1, False)
        return self._commit(draft, len(draft), "keep", len(draft), True)

    def metrics(self) -> dict[str, int]:
        result = dict(self.stats)
        result["committed_len"] = len(self.tokens)
        for name, stage in self.stages.items():
            if stage is not None:
                result[f"{name}_calls"] = stage.calls
                result[f"{name}_processed_tokens"] = stage.processed_tokens
                result[f"{name}_valid_len"] = stage.valid_len
        return result


__all__ = ["Commit", "Session", "Stage"]
