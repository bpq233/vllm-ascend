"""Draft-token alignment and q' logits collection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Hashable

import torch

from .backend import ViaSdPagedBackend
from .kv_cache import ViaSdKVCacheManager


def _as_token_list(tokens: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(tokens, torch.Tensor):
        return [int(token) for token in tokens.detach().cpu().flatten().tolist()]
    return [int(token) for token in tokens]


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
        self.max_num_tokens = int(runner.max_num_tokens)
        self.cache = ViaSdKVCacheManager(self.cache_enabled)
        self.backend = backend or ViaSdPagedBackend(runner, model, self.cache_enabled)

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
        length = 0
        for token in tokens:
            if token < 0:
                break
            length += 1
        return length

    def _score_request(
        self,
        request_id: Hashable,
        request_index: int,
        prefix_tokens: Sequence[int],
        draft_tokens: Sequence[int],
        valid_length: int | None = None,
        table_index: int | None = None,
    ) -> torch.Tensor | None:
        valid_length = (
            self._valid_length(draft_tokens)
            if valid_length is None
            else max(0, min(int(valid_length), len(draft_tokens)))
        )
        if valid_length <= 0:
            return None
        draft = list(draft_tokens[:valid_length])
        input_tokens, score_start = self.build_scoring_input(prefix_tokens, draft)
        table_index = request_index if table_index is None else int(table_index)
        signature_fn = getattr(self.backend, "block_signature", None)
        block_signature = (
            signature_fn(table_index, len(input_tokens)) if signature_fn is not None else None
        )
        if not self.cache_enabled:
            logits = self.backend.forward(input_tokens, 0, table_index, return_logits=True)
            if logits is None:
                raise RuntimeError("q' backend returned no logits")
            return logits[score_start : score_start + valid_length]

        cached = self.cache.reusable_prefix(
            request_id,
            request_index,
            input_tokens,
            score_start,
            block_signature=block_signature,
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
        self.cache.commit(request_id, request_index, input_tokens, block_signature=block_signature)
        return logits[:valid_length]

    @torch.inference_mode()
    def verify(
        self,
        draft_token_ids: torch.Tensor,
        request_ids: Sequence[Hashable],
        request_indices: Sequence[int],
        prefix_token_ids: Sequence[Sequence[int]],
        valid_lengths: Sequence[int] | None = None,
        table_indices: Sequence[int] | None = None,
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
        rows: list[torch.Tensor | None] = []
        active_ids = set(request_ids)
        for row in range(batch):
            rows.append(
                self._score_request(
                    request_ids[row],
                    int(request_indices[row]),
                    prefix_token_ids[row],
                    draft_cpu[row],
                    None if valid_lengths is None else int(valid_lengths[row]),
                    None if table_indices is None else int(table_indices[row]),
                )
            )
        self.cache.retain(active_ids)

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
        return output

    def clear(self) -> None:
        self.cache.clear()


__all__ = ["ViaSdVerifier"]
