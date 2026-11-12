# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Represent long causal queries as bounded virtual paged-attention rows.

No tokens, cache slots or output rows are removed or reordered. Each virtual
request references the logical request's block table and ends its KV visibility
at its own chunk boundary. With bottom-right causal attention, query row i in
that chunk sees exactly the keys visible to row i in the original long query.

This is a metadata transformation, not independent speculative verification:
all outputs are still consumed by one final continuous-prefix decision. It
requires the attention backend to write all query KV entries before its paged
attention call, as the Ascend causal paged path does.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QueryChunk:
    source_request: int
    query_begin: int
    query_end: int
    logical_begin: int
    logical_end: int
    seq_len: int

    @property
    def query_len(self) -> int:
        return self.query_end - self.query_begin


@dataclass(frozen=True)
class LongQueryPlan:
    query_start_loc: tuple[int, ...]
    seq_lens: tuple[int, ...]
    source_rows: tuple[int, ...]
    seq_len_offsets: tuple[int, ...]
    chunks: tuple[QueryChunk, ...]
    logical_last_chunk: tuple[int, ...]
    num_logical_requests: int
    num_tokens: int

    @property
    def num_virtual_requests(self) -> int:
        return len(self.source_rows)

    @property
    def query_lens(self) -> tuple[int, ...]:
        return tuple(chunk.query_len for chunk in self.chunks)

    @property
    def is_split(self) -> bool:
        return any(chunk.logical_begin > 0 for chunk in self.chunks)

    def source_indices(self, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(self.source_rows, dtype=torch.long, device=device)

    def query_start_tensor(self, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(self.query_start_loc, dtype=torch.int32, device=device)

    def seq_lens_tensor(self, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(self.seq_lens, dtype=torch.int32, device=device)

    def expand_block_table(self, block_table: torch.Tensor) -> torch.Tensor:
        """Repeat table rows without modifying the underlying KV allocation."""
        if block_table.ndim != 2 or block_table.shape[0] < self.num_logical_requests:
            raise ValueError("block_table must contain a row for each logical request")
        return block_table.index_select(0, self.source_indices(block_table.device))

    def expand_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        """Preserve device-corrected lengths instead of uploading a CPU mirror."""
        if seq_lens.ndim != 1 or seq_lens.shape[0] < self.num_logical_requests:
            raise ValueError("seq_lens must contain a value for each logical request")
        if seq_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError("seq_lens must be int32 or int64")
        offsets = torch.tensor(self.seq_len_offsets, dtype=seq_lens.dtype, device=seq_lens.device)
        return seq_lens.index_select(0, self.source_indices(seq_lens.device)) + offsets


def build_long_query_plan(
    query_start_loc: Sequence[int], seq_lens: Sequence[int], max_query_len: int = 16
) -> LongQueryPlan:
    """Split logical query rows with exact causal KV visibility per chunk.

    ``seq_lens[r]`` includes the request's entire new query, not just its
    historical prefix. Zero-query requests are omitted; no padding tokens are
    invented. A caller that pads before this function retains all its actual
    padding query rows as ordinary virtual requests.

    This host planning function runs when constructing metadata, outside graph
    capture. The same plan format is used for eager execution, capture and
    replay updates. Only metadata changes; each flattened token remains at its
    original query, position and slot index.
    """
    starts = tuple(query_start_loc)
    lengths = tuple(seq_lens)
    if type(max_query_len) is not int or max_query_len < 1:
        raise ValueError("max_query_len must be a positive integer")
    if any(type(value) is not int for value in (*starts, *lengths)):
        raise TypeError("query_start_loc and seq_lens must contain Python integers")
    if len(starts) != len(lengths) + 1 or not starts or starts[0] != 0:
        raise ValueError("query_start_loc must start at zero and have len(seq_lens) + 1 entries")
    if any(right < left for left, right in zip(starts, starts[1:])):
        raise ValueError("query_start_loc must be nondecreasing")
    virtual_starts = [0]
    virtual_lengths = []
    source_rows = []
    offsets = []
    chunks = []
    logical_last_chunk = []
    for request, (begin, end, seq_len) in enumerate(zip(starts, starts[1:], lengths)):
        query_len = end - begin
        if seq_len < query_len:
            raise ValueError("seq_lens must include the whole logical query and a nonnegative historical prefix")
        historical_prefix = seq_len - query_len
        last_chunk = -1
        for local_begin in range(0, query_len, max_query_len):
            local_end = min(local_begin + max_query_len, query_len)
            chunk = QueryChunk(
                source_request=request,
                query_begin=begin + local_begin,
                query_end=begin + local_end,
                logical_begin=local_begin,
                logical_end=local_end,
                seq_len=historical_prefix + local_end,
            )
            chunks.append(chunk)
            virtual_starts.append(begin + local_end)
            virtual_lengths.append(chunk.seq_len)
            source_rows.append(request)
            offsets.append(local_end - query_len)
            last_chunk = len(chunks) - 1
        logical_last_chunk.append(last_chunk)
    return LongQueryPlan(
        query_start_loc=tuple(virtual_starts),
        seq_lens=tuple(virtual_lengths),
        source_rows=tuple(source_rows),
        seq_len_offsets=tuple(offsets),
        chunks=tuple(chunks),
        logical_last_chunk=tuple(logical_last_chunk),
        num_logical_requests=len(lengths),
        num_tokens=starts[-1],
    )
