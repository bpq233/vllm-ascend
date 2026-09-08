"""Logical q' KV-cache ownership and prefix reuse bookkeeping."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Hashable


@dataclass
class _CacheEntry:
    request_index: int
    tokens: tuple[int, ...]
    block_signature: tuple[tuple[int, ...], ...] | None = None
    # Logical lengths are separate from the physical token tuple.  The tuple
    # may intentionally retain stale pages after a rewrite; only these
    # lengths participate in reuse decisions.
    committed_len: int = 0
    qprime_computed_len: int = 0
    target_computed_len: int = 0


@dataclass(frozen=True)
class ViaSdKVCacheState:
    """Public snapshot of one request's logical q'/target cache state."""

    request_id: Hashable
    request_index: int
    tokens: tuple[int, ...]
    committed_len: int
    qprime_computed_len: int
    target_computed_len: int
    block_signature: tuple[tuple[int, ...], ...] | None = None

    @property
    def valid_prefix_len(self) -> int:
        return self.committed_len


class ViaSdKVCacheManager:
    """Track which logical q' prefix is present in the physical q' pages.

    Physical pages are allocated by the normal MRv2 KV allocator.  This class
    owns only request-to-prefix validity, allowing q' to reuse the same block
    table while keeping its cache tensors separate from target attention.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._entries: dict[Hashable, _CacheEntry] = {}

    @staticmethod
    def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
        length = min(len(left), len(right))
        index = 0
        while index < length and left[index] == right[index]:
            index += 1
        return index

    @staticmethod
    def _signature_covers_prefix(
        stored: tuple[tuple[int, ...], ...] | None,
        prefix: tuple[tuple[int, ...], ...],
    ) -> bool:
        if stored is None or len(stored) != len(prefix):
            return False
        return all(
            len(stored_pages) >= len(prefix_pages)
            and stored_pages[: len(prefix_pages)] == prefix_pages
            for stored_pages, prefix_pages in zip(stored, prefix)
        )

    def reusable_prefix(
        self,
        request_id: Hashable,
        request_index: int,
        input_tokens: Sequence[int],
        max_reusable: int,
        block_signature: tuple[tuple[int, ...], ...] | None = None,
    ) -> int:
        if not self.enabled or max_reusable <= 0:
            return 0
        entry = self._entries.get(request_id)
        if entry is None or entry.request_index != request_index:
            return 0
        # A growing request normally appends block IDs. Requiring equality
        # between the old full table and the new full table turns every block
        # boundary into a false miss. Only the pages covering the candidate
        # reusable prefix must still be backed by the same physical blocks.
        if block_signature is not None and not self._signature_covers_prefix(
            entry.block_signature,
            block_signature,
        ):
            return 0
        common = self._common_prefix(entry.tokens, input_tokens)
        # A physical suffix can survive a rewrite, but it is never logically
        # reusable past the last q' forward recorded for this request.
        logical_limit = max(0, min(entry.qprime_computed_len, len(entry.tokens)))
        return min(common, max_reusable, logical_limit)

    def commit(
        self,
        request_id: Hashable,
        request_index: int,
        input_tokens: Sequence[int],
        block_signature: tuple[tuple[int, ...], ...] | None = None,
        *,
        committed_len: int | None = None,
        qprime_computed_len: int | None = None,
        target_computed_len: int | None = None,
    ) -> None:
        if self.enabled:
            token_tuple = tuple(int(token) for token in input_tokens)
            physical_len = len(token_tuple)
            committed = physical_len if committed_len is None else int(committed_len)
            qprime = physical_len if qprime_computed_len is None else int(qprime_computed_len)
            target = 0 if target_computed_len is None else int(target_computed_len)
            if min(committed, qprime, target) < 0:
                raise ValueError("VIA-SD cache lengths cannot be negative")
            self._entries[request_id] = _CacheEntry(
                request_index=request_index,
                tokens=token_tuple,
                block_signature=block_signature,
                committed_len=min(committed, physical_len),
                qprime_computed_len=min(qprime, physical_len),
                target_computed_len=min(target, physical_len),
            )

    def record_forward(
        self,
        request_id: Hashable,
        request_index: int,
        computed_len: int,
        *,
        stage: str = "qprime",
        tokens: Sequence[int] | None = None,
        block_signature: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        """Record an actual stage forward without fabricating KV validity."""

        if not self.enabled:
            return
        stage = str(stage).lower()
        if stage not in {"qprime", "target"}:
            raise ValueError("stage must be 'qprime' or 'target'")
        entry = self._entries.get(request_id)
        if entry is None or entry.request_index != int(request_index):
            base_tokens = tuple(int(token) for token in (tokens or ()))
            entry = _CacheEntry(
                request_index=int(request_index),
                tokens=base_tokens,
                block_signature=block_signature,
                committed_len=len(base_tokens),
                qprime_computed_len=0,
                target_computed_len=0,
            )
            self._entries[request_id] = entry
        elif tokens is not None:
            entry.tokens = tuple(int(token) for token in tokens)
        value = int(computed_len)
        if value < 0:
            raise ValueError("computed_len cannot be negative")
        value = min(value, len(entry.tokens))
        if stage == "qprime":
            entry.qprime_computed_len = value
        else:
            entry.target_computed_len = value
        if block_signature is not None:
            entry.block_signature = block_signature

    def update_lengths(
        self,
        request_id: Hashable,
        *,
        committed_len: int | None = None,
        qprime_computed_len: int | None = None,
        target_computed_len: int | None = None,
    ) -> None:
        """Update logical lengths monotonically downward or to real values."""

        entry = self._entries.get(request_id)
        if entry is None:
            return
        physical_len = len(entry.tokens)
        if committed_len is not None:
            entry.committed_len = max(0, min(int(committed_len), physical_len))
        if qprime_computed_len is not None:
            entry.qprime_computed_len = max(0, min(int(qprime_computed_len), physical_len))
        if target_computed_len is not None:
            entry.target_computed_len = max(0, min(int(target_computed_len), physical_len))

    def truncate(self, request_id: Hashable, valid_prefix_len: int) -> None:
        """Logically invalidate a suffix; physical page storage is untouched."""

        entry = self._entries.get(request_id)
        if entry is None:
            return
        length = int(valid_prefix_len)
        if length < 0:
            raise ValueError("valid_prefix_len cannot be negative")
        length = min(length, len(entry.tokens))
        entry.committed_len = min(entry.committed_len, length)
        entry.qprime_computed_len = min(entry.qprime_computed_len, length)
        entry.target_computed_len = min(entry.target_computed_len, length)

    # More explicit aliases are useful at request lifecycle call sites.
    truncate_request = truncate

    def state(self, request_id: Hashable) -> ViaSdKVCacheState | None:
        entry = self._entries.get(request_id)
        if entry is None:
            return None
        return ViaSdKVCacheState(
            request_id=request_id,
            request_index=entry.request_index,
            tokens=entry.tokens,
            committed_len=entry.committed_len,
            qprime_computed_len=entry.qprime_computed_len,
            target_computed_len=entry.target_computed_len,
            block_signature=entry.block_signature,
        )

    get_state = state

    def lengths(self, request_id: Hashable) -> dict[str, int] | None:
        snapshot = self.state(request_id)
        return None if snapshot is None else {
            "committed_len": snapshot.committed_len,
            "qprime_computed_len": snapshot.qprime_computed_len,
            "target_computed_len": snapshot.target_computed_len,
        }

    def discard(self, request_ids: Iterable[Hashable]) -> None:
        if not self.enabled:
            return
        for request_id in request_ids:
            self._entries.pop(request_id, None)

    def clear(self) -> None:
        self._entries.clear()


__all__ = ["ViaSdKVCacheManager", "ViaSdKVCacheState"]
