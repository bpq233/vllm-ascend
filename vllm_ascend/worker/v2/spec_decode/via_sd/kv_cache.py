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
        return min(common, max_reusable)

    def commit(
        self,
        request_id: Hashable,
        request_index: int,
        input_tokens: Sequence[int],
        block_signature: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        if self.enabled:
            self._entries[request_id] = _CacheEntry(
                request_index=request_index,
                tokens=tuple(int(token) for token in input_tokens),
                block_signature=block_signature,
            )

    def discard(self, request_ids: Iterable[Hashable]) -> None:
        if not self.enabled:
            return
        for request_id in request_ids:
            self._entries.pop(request_id, None)

    def clear(self) -> None:
        self._entries.clear()


__all__ = ["ViaSdKVCacheManager"]
