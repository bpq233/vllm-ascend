"""Draft-token alignment and q' logits collection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Hashable

import torch

from .backend import ViaSdPagedBackend
from .kv_cache import ViaSdKVCacheManager, ViaSdKVCacheState
from .routing import ViaSdRoutePlan, build_route_plan


def _as_token_list(tokens: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(tokens, torch.Tensor):
        return [int(token) for token in tokens.detach().cpu().flatten().tolist()]
    return [int(token) for token in tokens]


def normalize_draft_tokens(tokens: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    """Return the real draft prefix before the ``-1`` padding sentinel."""

    normalized: list[int] = []
    for token in _as_token_list(tokens):
        if token < 0:
            break
        normalized.append(token)
    return tuple(normalized)


@dataclass
class ViaSdValidationStats:
    """Work counters for one q' validation call."""

    cache_enabled: bool
    request_count: int = 0
    scored_request_count: int = 0
    draft_tokens: int = 0
    cached_prefix_tokens: int = 0
    recomputed_prefix_tokens: int = 0
    model_input_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    request_ids: tuple[Hashable, ...] = ()
    draft_token_ids: tuple[tuple[int, ...], ...] = ()
    positions: tuple[tuple[int, ...], ...] = ()


class ViaSdVerifier:
    """Compute q' logits for each proposed draft position.

    The input alignment is causal: for a committed prefix ``p`` and draft
    block ``d``, q' receives ``p + d[:-1]`` and the rows beginning at
    ``len(p) - 1`` predict ``d[0:]``.  No result is sent to target sampling.
    """

    def __init__(
        self,
        runner: Any,
        model: Any,
        config: Any,
        backend: Any | None = None,
    ) -> None:
        self.runner = runner
        self.model = model
        self.config = config
        self.cache_enabled = bool(getattr(config, "kv_cache_enabled", True))
        self.accept_ratio = float(getattr(config, "accept_ratio", 0.7))
        self.escalate_ratio = float(getattr(config, "escalate_ratio", 0.5))
        self.max_num_tokens = int(runner.max_num_tokens)
        self.cache = ViaSdKVCacheManager(self.cache_enabled)
        self.backend = backend or ViaSdPagedBackend(runner, model, self.cache_enabled)
        self.last_stats = ViaSdValidationStats(cache_enabled=self.cache_enabled)

    def build_route_plan(
        self,
        logits: torch.Tensor,
        draft_token_ids: torch.Tensor,
        *,
        request_ids: Sequence[Hashable] | None = None,
        valid_lengths: Sequence[int] | None = None,
        batch_rows: Sequence[int] | None = None,
    ) -> ViaSdRoutePlan:
        """Classify q' rows using the configured direct thresholds."""

        return build_route_plan(
            logits,
            draft_token_ids,
            self.accept_ratio,
            self.escalate_ratio,
            request_ids=request_ids,
            valid_lengths=valid_lengths,
            batch_rows=batch_rows,
        )

    @staticmethod
    def build_scoring_input(
        prefix_tokens: Sequence[int], draft_tokens: Sequence[int]
    ) -> tuple[list[int], int]:
        prefix = _as_token_list(prefix_tokens)
        draft = _as_token_list(draft_tokens)
        if not prefix:
            raise ValueError("q' requires at least one committed prefix token")
        return prefix + draft[:-1], len(prefix) - 1

    @staticmethod
    def _valid_length(tokens: Sequence[int]) -> int:
        return len(normalize_draft_tokens(tokens))

    def _score_request(
        self,
        request_id: Hashable,
        request_index: int,
        prefix_tokens: Sequence[int],
        draft_tokens: Sequence[int],
        valid_length: int | None = None,
        table_index: int | None = None,
        stats: ViaSdValidationStats | None = None,
    ) -> torch.Tensor | None:
        sentinel_length = self._valid_length(draft_tokens)
        valid_length = (
            sentinel_length
            if valid_length is None
            else min(sentinel_length, max(0, min(int(valid_length), len(draft_tokens))))
        )
        if valid_length <= 0:
            return None
        draft = list(draft_tokens[:valid_length])
        input_tokens, score_start = self.build_scoring_input(prefix_tokens, draft)
        table_index = request_index if table_index is None else int(table_index)
        if not self.cache_enabled:
            logits = self.backend.forward(input_tokens, 0, table_index, return_logits=True)
            if logits is None:
                raise RuntimeError("q' backend returned no logits")
            if stats is not None:
                stats.recomputed_prefix_tokens += score_start
                stats.model_input_tokens += len(input_tokens)
            return logits[score_start : score_start + valid_length]

        signature_fn = getattr(self.backend, "block_signature", None)
        # Find the logical token match first, then validate only the physical
        # pages that cover that match. Comparing full page tables makes an
        # append at a block boundary look like a cache invalidation.
        cached = self.cache.reusable_prefix(
            request_id,
            request_index,
            input_tokens,
            score_start,
        )
        if cached > 0 and signature_fn is not None:
            prefix_signature = signature_fn(table_index, cached)
            if prefix_signature is None:
                cached = 0
            else:
                cached = self.cache.reusable_prefix(
                    request_id,
                    request_index,
                    input_tokens,
                    cached,
                    block_signature=prefix_signature,
                )
        cursor = cached
        while cursor < score_start:
            chunk_end = min(score_start, cursor + self.max_num_tokens)
            self.backend.forward(
                input_tokens[cursor:chunk_end],
                cursor,
                table_index,
                return_logits=False,
            )
            cursor = chunk_end

        logits = self.backend.forward(
            input_tokens[score_start:],
            score_start,
            table_index,
            return_logits=True,
        )
        if logits is None:
            raise RuntimeError("q' backend returned no logits")
        block_signature = (
            signature_fn(table_index, len(input_tokens)) if signature_fn is not None else None
        )
        self.cache.commit(request_id, request_index, input_tokens, block_signature=block_signature)
        if stats is not None:
            stats.cached_prefix_tokens += cached
            stats.recomputed_prefix_tokens += score_start - cached
            stats.model_input_tokens += len(input_tokens) - cached
            if cached > 0:
                stats.cache_hits += 1
            else:
                stats.cache_misses += 1
        return logits[:valid_length]

    def _score_batch(
        self,
        request_ids: Sequence[Hashable],
        request_indices: Sequence[int],
        prefix_token_ids: Sequence[Sequence[int]],
        draft_cpu: Sequence[Sequence[int]],
        resolved_valid_lengths: Sequence[int],
        table_indices: Sequence[int],
        stats: ViaSdValidationStats,
    ) -> list[torch.Tensor | None]:
        """Score all non-empty rows with one packed backend invocation."""

        signature_fn = getattr(self.backend, "block_signature", None)
        prepared: list[dict[str, Any]] = []
        rows: list[torch.Tensor | None] = [None] * len(request_ids)
        for row, valid_length in enumerate(resolved_valid_lengths):
            if valid_length <= 0:
                continue
            input_tokens, score_start = self.build_scoring_input(
                prefix_token_ids[row], draft_cpu[row][:valid_length]
            )
            table_index = int(table_indices[row])
            if self.cache_enabled:
                cached = self.cache.reusable_prefix(
                    request_ids[row],
                    int(request_indices[row]),
                    input_tokens,
                    score_start,
                )
                if cached > 0 and signature_fn is not None:
                    prefix_signature = signature_fn(table_index, cached)
                    if prefix_signature is None:
                        cached = 0
                    else:
                        cached = self.cache.reusable_prefix(
                            request_ids[row],
                            int(request_indices[row]),
                            input_tokens,
                            cached,
                            block_signature=prefix_signature,
                        )
                start = cached
            else:
                cached = 0
                start = 0
            prepared.append(
                {
                    "row": row,
                    "input_tokens": input_tokens,
                    "score_start": score_start,
                    "start": start,
                    "cached": cached,
                    "table_index": table_index,
                }
            )

        if not prepared:
            return rows
        # A cold long-context request may need several warm-up chunks. Keep
        # that established path instead of exceeding MRv2's per-forward or
        # total batched-token budget; normal decode batches still take the
        # single ragged forward below.
        batch_token_limit = self.max_num_tokens
        batch_token_count = sum(
            len(entry["input_tokens"]) - entry["start"] for entry in prepared
        )
        if (
            any(
                len(entry["input_tokens"]) - entry["start"] > batch_token_limit
                for entry in prepared
            )
            or batch_token_count > batch_token_limit
        ):
            for entry in prepared:
                row = entry["row"]
                rows[row] = self._score_request(
                    request_ids[row],
                    int(request_indices[row]),
                    prefix_token_ids[row],
                    draft_cpu[row],
                    resolved_valid_lengths[row],
                    entry["table_index"],
                    stats=stats,
                )
            return rows
        outputs = self.backend.forward_batch(
            [entry["input_tokens"][entry["start"] :] for entry in prepared],
            [entry["start"] for entry in prepared],
            [entry["table_index"] for entry in prepared],
        )
        if len(outputs) != len(prepared):
            raise RuntimeError(
                "q' backend returned an unexpected number of ragged batch outputs: "
                f"expected={len(prepared)}, got={len(outputs)}"
            )
        for entry, logits in zip(prepared, outputs):
            if logits is None:
                raise RuntimeError("q' backend returned no logits for a ragged batch row")
            local_score_start = entry["score_start"] - entry["start"]
            valid_length = resolved_valid_lengths[entry["row"]]
            row_logits = logits[local_score_start : local_score_start + valid_length]
            if row_logits.shape[0] < valid_length:
                raise RuntimeError(
                    "q' backend returned too few logits for a ragged batch row: "
                    f"expected={valid_length}, got={row_logits.shape[0]}"
                )
            rows[entry["row"]] = row_logits

            if self.cache_enabled:
                block_signature = (
                    signature_fn(entry["table_index"], len(entry["input_tokens"]))
                    if signature_fn is not None
                    else None
                )
                self.cache.commit(
                    request_ids[entry["row"]],
                    int(request_indices[entry["row"]]),
                    entry["input_tokens"],
                    block_signature=block_signature,
                )
                stats.cached_prefix_tokens += entry["cached"]
                stats.recomputed_prefix_tokens += (
                    entry["score_start"] - entry["cached"]
                )
                stats.model_input_tokens += len(entry["input_tokens"]) - entry["cached"]
                if entry["cached"] > 0:
                    stats.cache_hits += 1
                else:
                    stats.cache_misses += 1
            else:
                stats.recomputed_prefix_tokens += entry["score_start"]
                stats.model_input_tokens += len(entry["input_tokens"])
        return rows

    @torch.inference_mode()
    def verify(
        self,
        draft_token_ids: torch.Tensor,
        request_ids: Sequence[Hashable],
        request_indices: Sequence[int],
        prefix_token_ids: Sequence[Sequence[int]],
        valid_lengths: Sequence[int] | None = None,
        table_indices: Sequence[int] | None = None,
        committed_lengths: Sequence[int] | None = None,
        target_computed_lengths: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if draft_token_ids.ndim != 2:
            raise ValueError(f"q' draft tokens must be rank 2, got {draft_token_ids.shape}")
        batch, draft_steps = draft_token_ids.shape
        if not (len(request_ids) == len(request_indices) == len(prefix_token_ids) == batch):
            raise ValueError("q' request metadata does not match draft-token batch")
        if valid_lengths is not None and len(valid_lengths) != batch:
            raise ValueError("q' valid-length metadata does not match draft-token batch")
        if table_indices is not None and len(table_indices) != batch:
            raise ValueError("q' table-index metadata does not match draft-token batch")

        draft_cpu = draft_token_ids.detach().cpu().tolist()
        sentinel_lengths = [self._valid_length(tokens) for tokens in draft_cpu]
        resolved_valid_lengths = [
            sentinel_length
            if valid_lengths is None
            else min(
                sentinel_length,
                max(0, min(int(valid_lengths[row]), len(draft_cpu[row]))),
            )
            for row, sentinel_length in enumerate(sentinel_lengths)
        ]
        stats = ViaSdValidationStats(
            cache_enabled=self.cache_enabled,
            request_count=batch,
            scored_request_count=sum(length > 0 for length in resolved_valid_lengths),
            draft_tokens=sum(resolved_valid_lengths),
            request_ids=tuple(request_ids),
            draft_token_ids=tuple(
                tuple(int(token) for token in tokens[: resolved_valid_lengths[row]])
                for row, tokens in enumerate(draft_cpu)
            ),
            positions=tuple(
                tuple(range(len(prefix_token_ids[row]), len(prefix_token_ids[row]) + length))
                if length > 0 and len(prefix_token_ids[row]) > 0
                else ()
                for row, length in enumerate(resolved_valid_lengths)
            ),
        )
        self.last_stats = stats
        resolved_table_indices = (
            [int(index) for index in table_indices]
            if table_indices is not None
            else [int(index) for index in request_indices]
        )
        if hasattr(self.backend, "forward_batch"):
            rows = self._score_batch(
                request_ids,
                request_indices,
                prefix_token_ids,
                draft_cpu,
                resolved_valid_lengths,
                resolved_table_indices,
                stats,
            )
        else:
            rows = []
            for row in range(batch):
                rows.append(
                    self._score_request(
                        request_ids[row],
                        int(request_indices[row]),
                        prefix_token_ids[row],
                        draft_cpu[row],
                        resolved_valid_lengths[row],
                        resolved_table_indices[row],
                        stats=stats,
                    )
                )

        vocab_size = int(self.runner.vocab_size)
        dtype = next((row.dtype for row in rows if row is not None), torch.float32)
        device = self.runner.device
        output = torch.full(
            (batch, draft_steps, vocab_size),
            float("-inf"),
            dtype=dtype,
            device=device,
        )
        for row, logits in enumerate(rows):
            if logits is not None:
                output[row, : logits.shape[0]] = logits
        if committed_lengths is not None and len(committed_lengths) != batch:
            raise ValueError("q' committed-length metadata does not match the batch")
        if target_computed_lengths is not None and len(target_computed_lengths) != batch:
            raise ValueError("target computed-length metadata does not match the batch")
        # The default observation path intentionally keeps the historical
        # physical-token bookkeeping.  Hierarchical callers may provide the
        # logical committed/target lengths explicitly after the real forward.
        if committed_lengths is not None or target_computed_lengths is not None:
            for row, request_id in enumerate(request_ids):
                self.cache.update_lengths(
                    request_id,
                    committed_len=(None if committed_lengths is None else int(committed_lengths[row])),
                    target_computed_len=(
                        None
                        if target_computed_lengths is None
                        else int(target_computed_lengths[row])
                    ),
                )
        return output

    @torch.inference_mode()
    def verify_and_route(
        self,
        draft_token_ids: torch.Tensor,
        request_ids: Sequence[Hashable],
        request_indices: Sequence[int],
        prefix_token_ids: Sequence[Sequence[int]],
        valid_lengths: Sequence[int] | None = None,
        table_indices: Sequence[int] | None = None,
        *,
        batch_rows: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, ViaSdRoutePlan]:
        """Run q' once and return logits together with its route plan."""

        logits = self.verify(
            draft_token_ids,
            request_ids=request_ids,
            request_indices=request_indices,
            prefix_token_ids=prefix_token_ids,
            valid_lengths=valid_lengths,
            table_indices=table_indices,
        )
        plan = self.build_route_plan(
            logits,
            draft_token_ids,
            request_ids=request_ids,
            valid_lengths=valid_lengths,
            batch_rows=batch_rows,
        )
        return logits, plan

    def truncate(self, request_id: Hashable, valid_prefix_len: int) -> None:
        """Invalidate a q' suffix after a rewrite without freeing pages."""

        self.cache.truncate(request_id, valid_prefix_len)

    def cache_state(self, request_id: Hashable) -> ViaSdKVCacheState | None:
        return self.cache.state(request_id)

    get_cache_state = cache_state

    def clear(self) -> None:
        self.cache.clear()

    def discard(self, request_ids: Sequence[Hashable]) -> None:
        self.cache.discard(request_ids)


__all__ = ["ViaSdValidationStats", "ViaSdVerifier", "normalize_draft_tokens"]
