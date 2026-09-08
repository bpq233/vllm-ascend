"""Eager Ascend MRv2 backend used by the q' side pass."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


class ViaSdPagedBackend:
    """Run q' requests with MRv2's existing block-table allocation.

    q' attention modules have distinct names, so enabled caching is backed by
    distinct physical pages allocated through the regular MRv2 KV allocator.
    The request's target block table and slot positions can nevertheless be
    reused because both passes have the same token layout.
    """

    def __init__(self, runner: Any, model: Any, cache_enabled: bool) -> None:
        self.runner = runner
        self.model = model
        self.cache_enabled = bool(cache_enabled)
        self.vllm_config = runner.vllm_config
        self._request_block_tables: Sequence[torch.Tensor] | None = None
        # ``input_block_tables`` is gathered in batch order, while
        # ``num_blocks`` remains indexed by the request-state slot.  Keep the
        # latter mapping explicitly so a batch reorder cannot make the page
        # capacity check inspect another request's allocation.
        self._request_state_indices: Sequence[int] | None = None
        self.groups = self._build_groups()

    def set_request_block_tables(
        self,
        block_tables: Sequence[torch.Tensor] | None,
        request_indices: Sequence[int] | None = None,
    ) -> None:
        """Set the batch-local block tables produced by MRv2's target pass."""

        self._request_block_tables = block_tables
        self._request_state_indices = (
            tuple(int(index) for index in request_indices)
            if request_indices is not None
            else None
        )

    def _state_index_for_table(self, table_index: int) -> int:
        if self._request_state_indices is None:
            return table_index
        if table_index < 0 or table_index >= len(self._request_state_indices):
            raise RuntimeError(f"invalid q' batch table index: {table_index}")
        return int(self._request_state_indices[table_index])

    def block_signature(
        self, table_index: int, end: int
    ) -> tuple[tuple[int, ...], ...] | None:
        """Return the page IDs backing the prefix used by q'."""

        if not self.cache_enabled:
            return None
        signature: list[tuple[int, ...]] = []
        for group_index, block_size in enumerate(self.runner.kernel_block_sizes):
            needed = (end + block_size - 1) // block_size
            state_index = self._state_index_for_table(table_index)
            if self._request_block_tables is not None:
                table_source = self._request_block_tables[group_index]
                row_index = table_index
            else:
                table_source = self.runner.block_tables.block_tables[group_index].gpu
                row_index = state_index
            if row_index < 0 or row_index >= table_source.shape[0] or needed > table_source.shape[1]:
                # A stale/malformed gathered table must invalidate reuse, not
                # crash the verifier or accidentally reuse an unrelated page.
                return None
            signature.append(
                tuple(
                    int(value)
                    for value in table_source[row_index, :needed].detach().cpu().tolist()
                )
            )
        return tuple(signature)

    def _build_groups(self):
        qprime_names = set(self.model.attention_layer_names)
        source_to_qprime = {
            source: qprime for qprime, source in self.model.source_attention_layer_names.items()
        }
        groups_by_cache_group = []
        discovered: set[str] = set()
        for original_groups in self.runner.attn_groups:
            selected_groups = []
            for original in original_groups:
                if self.cache_enabled:
                    names = sorted(set(original.layer_names) & qprime_names)
                else:
                    names = sorted(
                        source_to_qprime[source]
                        for source in original.layer_names
                        if source in source_to_qprime
                    )
                if not names:
                    continue
                group_id = int(original.kv_cache_group_id)
                if group_id < 0 or group_id >= len(self.runner.kernel_block_sizes):
                    raise RuntimeError(
                        f"invalid q' KV cache group id: {group_id}"
                    )
                group = copy.copy(original)
                group.layer_names = names
                # ``AttentionGroup``'s builders retain the layer names passed
                # at construction time.  A shallow group copy would therefore
                # make q' use target metadata (and, for DSA, target-owned
                # buffers).  Recreate them against the q' registrations while
                # preserving the original ubatch-builder count.
                group.create_metadata_builders(
                    vllm_config=self.vllm_config,
                    device=self.runner.device,
                    kernel_block_size=self.runner.kernel_block_sizes[group_id],
                    num_metadata_builders=max(1, len(original.metadata_builders)),
                )
                selected_groups.append(group)
                discovered.update(names)
            groups_by_cache_group.append(selected_groups)
        if discovered != qprime_names:
            missing = sorted(qprime_names - discovered)
            raise RuntimeError(f"q' attention layers are missing from MRv2 KV groups: {missing}")
        return groups_by_cache_group

    def _page_layout(
        self,
        table_index: int,
        end: int,
        positions: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        runner = self.runner
        tables: list[torch.Tensor] = []
        slots: list[torch.Tensor] = []
        for group_index, block_size in enumerate(runner.kernel_block_sizes):
            needed = (end + block_size - 1) // block_size
            state_index = self._state_index_for_table(table_index)
            allocated = int(runner.block_tables.num_blocks.np[group_index, state_index])
            if needed > allocated:
                raise RuntimeError(
                    "q' would write beyond the request's allocated KV pages: "
                    f"group={group_index}, request={state_index}, needed={needed}, "
                    f"allocated={allocated}"
                )
            if self._request_block_tables is not None:
                table_source = self._request_block_tables[group_index]
                if table_index < 0 or table_index >= table_source.shape[0] or needed > table_source.shape[1]:
                    raise RuntimeError(
                        "q' request block table does not cover the requested positions: "
                        f"group={group_index}, table_index={table_index}, needed={needed}"
                    )
                table = table_source[table_index : table_index + 1]
            else:
                # The gathered batch-local tables are preferred.  This
                # fallback is useful for callers that invoke the backend
                # directly before MRv2 has populated input_block_tables.
                table = runner.block_tables.block_tables[group_index].gpu[
                    state_index : state_index + 1
                ]
            page_ids = table[0, positions // block_size].to(torch.int64)
            slots.append((page_ids * block_size + positions % block_size).to(torch.int32))
            tables.append(table)
        return tables, torch.stack(slots)

    def _batch_page_layout(
        self,
        table_indices: Sequence[int],
        ends: Sequence[int],
        positions: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Build block tables and packed slot mappings for a ragged batch."""

        runner = self.runner
        tables: list[torch.Tensor] = []
        slots: list[torch.Tensor] = []
        for group_index, block_size in enumerate(runner.kernel_block_sizes):
            group_tables: list[torch.Tensor] = []
            group_slots: list[torch.Tensor] = []
            max_needed = max(
                1, max((int(end) + block_size - 1) // block_size for end in ends)
            )
            for table_index, end, row_positions in zip(table_indices, ends, positions):
                state_index = self._state_index_for_table(int(table_index))
                needed = (int(end) + block_size - 1) // block_size
                allocated = int(runner.block_tables.num_blocks.np[group_index, state_index])
                if needed > allocated:
                    raise RuntimeError(
                        "q' would write beyond the request's allocated KV pages: "
                        f"group={group_index}, request={state_index}, needed={needed}, "
                        f"allocated={allocated}"
                    )
                if self._request_block_tables is not None:
                    table_source = self._request_block_tables[group_index]
                    row_index = int(table_index)
                else:
                    table_source = runner.block_tables.block_tables[group_index].gpu
                    row_index = state_index
                if (
                    row_index < 0
                    or row_index >= table_source.shape[0]
                    or needed > table_source.shape[1]
                ):
                    raise RuntimeError(
                        "q' request block table does not cover the requested positions: "
                        f"group={group_index}, table_index={table_index}, needed={needed}"
                    )
                table = table_source[row_index : row_index + 1, :max_needed]
                page_ids = table[0, row_positions // block_size].to(torch.int64)
                group_tables.append(table)
                group_slots.append(
                    (page_ids * block_size + row_positions % block_size).to(torch.int32)
                )
            tables.append(torch.cat(group_tables, dim=0))
            slots.append(torch.cat(group_slots, dim=0))
        return tables, torch.stack(slots)

    @torch.inference_mode()
    def forward_batch(
        self,
        token_ids: Sequence[Sequence[int]],
        starts: Sequence[int],
        table_indices: Sequence[int],
        *,
        return_features: bool = False,
    ) -> list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run all q' requests in one ragged MRv2 forward.

        Each row may start at a different cached prefix length.  The model sees
        the concatenated query tokens while ``query_start_loc`` and
        ``num_computed_tokens`` preserve the per-request boundaries and KV
        reuse semantics.
        """

        if not token_ids:
            if return_features:
                raise ValueError('Feature forward requires at least one request')
            return []
        if not (len(token_ids) == len(starts) == len(table_indices)):
            raise ValueError("q' batch token, start, and table metadata must have equal length")
        runner = self.runner
        counts = [len(row) for row in token_ids]
        if any(count <= 0 for count in counts):
            raise ValueError("q' batch rows must contain at least one token")
        starts = [int(start) for start in starts]
        ends = [start + count for start, count in zip(starts, counts)]
        token_limit = runner.max_num_tokens if self.cache_enabled else runner.max_model_len
        if any(
            count > token_limit or start < 0 or end > runner.max_model_len
            for count, start, end in zip(counts, starts, ends)
        ):
            raise RuntimeError(
                "q' batched forward exceeds MRv2 limits: "
                f"counts={counts}, starts={starts}, max_model_len={runner.max_model_len}, "
                f"token_limit={token_limit}"
            )
        if not self.cache_enabled and any(start != 0 for start in starts):
            raise RuntimeError("q' cache-disabled mode only supports complete from-scratch forwards")

        device = runner.device
        inputs = torch.as_tensor(
            [token for row in token_ids for token in row],
            dtype=torch.int32,
            device=device,
        )
        positions_by_row = [
            torch.arange(start, end, dtype=torch.int64, device=device)
            for start, end in zip(starts, ends)
        ]
        positions = torch.cat(positions_by_row, dim=0)
        query_start_cpu = torch.tensor(
            [0] + list(np.cumsum(counts, dtype=np.int32)), dtype=torch.int32
        )
        query_start_gpu = query_start_cpu.to(device)
        if self.cache_enabled:
            block_tables, slot_mappings = self._batch_page_layout(
                table_indices, ends, positions_by_row
            )
        else:
            max_blocks_by_group = [
                max(
                    1,
                    max((end + block_size - 1) // block_size for end in ends),
                )
                for block_size in runner.kernel_block_sizes
            ]
            block_tables = [
                torch.zeros(
                    (len(token_ids), max_blocks), dtype=torch.int32, device=device
                )
                for max_blocks in max_blocks_by_group
            ]
            slot_mappings = torch.full(
                (len(block_tables), int(inputs.numel())),
                -1,
                dtype=torch.int32,
                device=device,
            )

        seq_lens = torch.tensor(ends, dtype=torch.int32, device=device)
        is_prefilling = torch.tensor(
            [start == 0 for start in starts], dtype=torch.bool, device=device
        )
        metadata = build_attn_metadata(
            attn_groups=self.groups,
            num_reqs=len(token_ids),
            num_tokens=int(inputs.numel()),
            query_start_loc_gpu=query_start_gpu,
            query_start_loc_cpu=query_start_cpu,
            max_query_len=max(counts),
            seq_lens=seq_lens,
            max_seq_len=max(ends),
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=runner.kv_cache_config,
            seq_lens_np=np.asarray(ends, dtype=np.int32),
            seq_lens_cpu_upper_bound=torch.tensor(ends, dtype=torch.int32),
            num_computed_tokens_cpu=torch.tensor(starts, dtype=torch.int32),
            positions=positions,
            attn_state=self._batch_attention_state(starts, counts),
            is_prefilling=is_prefilling,
            num_actual_tokens=int(inputs.numel()),
            num_input_tokens=int(inputs.numel()),
        )
        slots_by_layer = (
            {
                name: slot_mappings[group_index]
                for group_index, groups in enumerate(self.groups)
                for group in groups
                for name in group.layer_names
            }
            if self.cache_enabled
            else {}
        )
        num_tokens = int(inputs.numel())
        with set_forward_context(
            metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
            slot_mapping=slots_by_layer,
            skip_compiled=True,
            is_padding=torch.zeros(num_tokens, dtype=torch.bool, device=device),
        ):
            hidden_states = self.model(input_ids=inputs, positions=positions)
        self.last_forward_metadata = metadata
        self.last_slot_mappings = slots_by_layer
        if return_features:
            return hidden_states, list(self.model.last_aux_hidden_states)
        logits = self.model.compute_logits(hidden_states)
        return [
            logits[start:end]
            for start, end in zip(
                query_start_cpu[:-1].tolist(), query_start_cpu[1:].tolist()
            )
        ]

    def forward_features(self, token_ids, starts, table_indices):
        previous = self.model.capture_draft_features
        self.model.capture_draft_features = True
        try:
            return self.forward_batch(token_ids, starts, table_indices, return_features=True)
        finally:
            self.model.capture_draft_features = previous

    @torch.inference_mode()
    def forward(
        self,
        token_ids: Sequence[int],
        start: int,
        request_index: int,
        return_logits: bool = True,
    ) -> torch.Tensor | None:
        count = len(token_ids)
        end = start + count
        runner = self.runner
        if count <= 0:
            return None
        token_limit = runner.max_num_tokens if self.cache_enabled else runner.max_model_len
        if count > token_limit or end > runner.max_model_len:
            raise RuntimeError(
                f"q' forward exceeds MRv2 limits: count={count}, end={end}, "
                f"token_limit={token_limit}, max_model_len={runner.max_model_len}"
            )
        if not self.cache_enabled and start != 0:
            raise RuntimeError("q' cache-disabled mode only supports a complete from-scratch forward")

        device = runner.device
        inputs = torch.as_tensor(token_ids, dtype=torch.int32, device=device)
        positions = torch.arange(start, end, dtype=torch.int64, device=device)
        if self.cache_enabled:
            block_tables, slot_mappings = self._page_layout(request_index, end, positions)
        else:
            # PrefillNoCache bypasses KV allocation.  Keep shape-compatible
            # zero tables for metadata builders (MLA/DSA inspect the table
            # while constructing metadata), but never expose target pages.
            block_tables = [
                torch.zeros(
                    (
                        1,
                        max(
                            1,
                            (end + runner.kernel_block_sizes[group_index] - 1)
                            // runner.kernel_block_sizes[group_index],
                        ),
                    ),
                    dtype=torch.int32,
                    device=device,
                )
                for group_index, _ in enumerate(runner.kv_cache_config.kv_cache_groups)
            ]
            slot_mappings = torch.full(
                (len(block_tables), count),
                -1,
                dtype=torch.int32,
                device=device,
            )
        query_start = torch.tensor([0, count], dtype=torch.int32)
        metadata = build_attn_metadata(
            attn_groups=self.groups,
            num_reqs=1,
            num_tokens=count,
            query_start_loc_gpu=query_start.to(device),
            query_start_loc_cpu=query_start,
            max_query_len=count,
            seq_lens=torch.tensor([end], dtype=torch.int32, device=device),
            max_seq_len=end,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=runner.kv_cache_config,
            seq_lens_np=np.asarray([end], dtype=np.int32),
            seq_lens_cpu_upper_bound=query_start.new_tensor([end]),
            num_computed_tokens_cpu=torch.tensor([start], dtype=torch.int32),
            positions=positions,
            attn_state=self._attention_state(start, count),
            is_prefilling=torch.tensor([start == 0], dtype=torch.bool, device=device),
            num_actual_tokens=count,
            num_input_tokens=count,
        )
        slots_by_layer = (
            {
                name: slot_mappings[group_index]
                for group_index, groups in enumerate(self.groups)
                for group in groups
                for name in group.layer_names
            }
            if self.cache_enabled
            else {}
        )
        with set_forward_context(
            metadata,
            self.vllm_config,
            num_tokens=count,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=count),
            slot_mapping=slots_by_layer,
            skip_compiled=True,
            is_padding=torch.zeros(count, dtype=torch.bool, device=device),
        ):
            hidden_states = self.model(input_ids=inputs, positions=positions)
        if not return_logits:
            return None
        return self.model.compute_logits(hidden_states)

    @staticmethod
    def _attention_state(start: int, count: int) -> AscendAttentionState:
        if start == 0:
            return AscendAttentionState.PrefillNoCache
        if count == 1:
            return AscendAttentionState.DecodeOnly
        return AscendAttentionState.ChunkedPrefill

    @staticmethod
    def _batch_attention_state(
        starts: Sequence[int], counts: Sequence[int]
    ) -> AscendAttentionState:
        if all(start == 0 for start in starts):
            return AscendAttentionState.PrefillNoCache
        if all(start > 0 and count == 1 for start, count in zip(starts, counts)):
            return AscendAttentionState.DecodeOnly
        return AscendAttentionState.ChunkedPrefill


__all__ = ["ViaSdPagedBackend"]
