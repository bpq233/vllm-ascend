# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Sparse forwards through the existing MRV2 metadata and ACL graph managers."""

from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import logger
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer, init_attn_backend
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.compilation.acl_graph import GraphParamsOwner, graph_params_scope
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers


def private_block_tables(block_tables):
    """Share read-only request page assignments, isolate gathered input buffers."""
    result = copy(block_tables)
    result.input_block_tables = [torch.zeros_like(table) for table in block_tables.input_block_tables]
    # The gather kernel receives cached device pointers, not the Python list.
    # Refresh them or it would continue writing the Target's gathered buffers.
    result.input_block_table_ptrs = result._make_ptr_tensor(result.input_block_tables)
    result._num_dirty_rows = 0
    result.slot_mappings = torch.full_like(block_tables.slot_mappings, -1)
    return result


def single_request_batch(template, buffers, token_ids, positions, state_index, req_id, *, padded_tokens=None):
    count = token_ids.numel()
    padded = padded_tokens or count
    end = int(positions[-1].item()) + 1
    buffers.input_ids.zero_()
    buffers.positions.zero_()
    buffers.is_padding.fill_(True)
    buffers.input_ids[:count].copy_(token_ids)
    buffers.positions[:count].copy_(positions)
    buffers.is_padding[:count] = False
    # Padding occupies the tail of the same causal request. It cannot influence
    # an earlier real token and its slot mapping is always PAD_SLOT_ID.
    buffers.query_start_loc.fill_(padded)
    buffers.query_start_loc[0] = 0
    buffers.seq_lens.zero_()
    buffers.seq_lens[0] = end + padded - count
    buffers.seq_lens_cpu.zero_()
    buffers.seq_lens_cpu[0] = end + padded - count
    idx = torch.tensor([state_index], dtype=torch.long, device=token_ids.device)
    return replace(
        template,
        req_ids=[req_id],
        num_reqs=1,
        num_reqs_after_padding=1,
        idx_mapping=idx,
        idx_mapping_np=np.array([state_index]),
        num_scheduled_tokens=np.array([padded], dtype=np.int32),
        num_tokens=count,
        num_tokens_after_padding=padded,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=buffers.query_start_loc[:2],
        query_start_loc_np=np.array([0, padded], dtype=np.int32),
        seq_lens=buffers.seq_lens[:1],
        seq_lens_cpu_upper_bound=buffers.seq_lens_cpu[:1],
        num_computed_tokens_np=np.array([end - count]),
        is_prefilling_np=np.array([False]),
        has_prefill=False,
        decode_graph_eligible=True,
        input_ids=buffers.input_ids[:padded],
        positions=buffers.positions[:padded],
        is_padding=buffers.is_padding[:padded],
        seq_lens_np=buffers.seq_lens_np[:1],
        attn_state=AscendAttentionState.ChunkedPrefill,
        max_query_len=padded,
        fast_prefill=None,
        pcp=None,
        prompt_lens=None,
        dcp_local_seq_lens=None,
    )


class SparseGraphModel(nn.Module):
    """LM head and checkpoint-ordered auxiliary states are part of capture."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, positions, **kwargs):
        return self.model.forward_logits(input_ids, positions)


class SparseForwardRunner:
    def __init__(self, speculator, model_state, kv_cache_config, block_tables):
        self.speculator = speculator
        self.vllm_config = speculator.target_vllm_config
        self.device = speculator.device
        self.buffers = AscendInputBuffers(speculator.max_num_reqs, speculator.max_num_tokens, self.device)
        self.block_tables = private_block_tables(block_tables)
        self.kv_cache_config = kv_cache_config
        self.model_state = copy(model_state)
        # TargetDeviceMetadata owns producer events and shared staging buffers;
        # a nested sparse forward must not retire the Target's submission.
        self.model_state.device_metadata = None
        self.model_state.kvpp_runtime = None
        self.model_state.decoder_replay = None
        self.model_state.pcp_manager = None
        self.attn_groups, self.attn_cg_support, _ = init_attn_backend(
            kv_cache_config,
            self.vllm_config,
            self.device,
            active_layer_names=set(speculator.sparse_model.attention_layers),
        )
        self.graph_model = SparseGraphModel(speculator.sparse_model)
        self.graph_params_owner = GraphParamsOwner()
        self.proxy = SimpleNamespace(
            update_stream=speculator.update_stream,
            dp_size=1,
            attn_groups=self.attn_groups,
            model_state=self.model_state,
            speculative_config=self.vllm_config.speculative_config,
        )

    def init_graphs(self, mode):
        # PIECEWISE needs a compilable model or the native breakable wrapper;
        # the sparse view is intentionally not torch-compiled. Use the upstream
        # breakable manager when available, otherwise report its native error.
        if self.vllm_config.model_config.enforce_eager:
            mode = CUDAGraphMode.NONE
        self.graph_params_owner = GraphParamsOwner()
        with graph_params_scope(self.graph_params_owner):
            self.speculator.sparse_cudagraph_manager = ModelAclGraphManager(
                self.vllm_config,
                self.device,
                mode,
                decode_query_len=self.speculator.num_query_per_req,
                model_runner=self.proxy,
                varlen_decode=True,
            )

    def capture(self):
        manager = self.speculator.sparse_cudagraph_manager
        if not manager.needs_capture():
            return
        with graph_params_scope(self.graph_params_owner), build_attn_metadata_wrapper():
            manager.capture(
                self.graph_model,
                self.model_state,
                self.buffers,
                None,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                use_aux_hidden_state_outputs=True,
                progress_bar_desc="Capturing sparse Target ACL graphs",
            )
        if manager.graphs:
            self.speculator.sparse_metrics.record_graph("sparse", "capture")
            logger.info("Sparse Target ACL capture completed: %s shapes", len(manager.graphs))

    def forward(self, template, ids, positions, state_index, req_id, *, prefill=False):
        with graph_params_scope(self.graph_params_owner):
            return self._forward(template, ids, positions, state_index, req_id, prefill=prefill)

    def _forward(self, template, ids, positions, state_index, req_id, *, prefill=False):
        count = ids.numel()
        manager = self.speculator.sparse_cudagraph_manager
        desc = manager.dispatch(
            1,
            count,
            uniform_token_count=None,
            num_active_loras=0,
            max_query_len=None if prefill else count,
        )
        end_position = int(positions[-1].item()) + 1
        if (desc.num_reqs or 1) == 1 and end_position + desc.num_tokens - count > self.speculator.max_model_len:
            # Never describe an out-of-context KV endpoint just to pad a graph.
            # At this context boundary use the upstream eager fallback explicitly.
            desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, count, 1)
        # Native FULL_DECODE_ONLY descriptors may pad requests rather than only
        # tokens. Keep their exact static request split, including inert rows.
        padded = desc.num_tokens
        batch = single_request_batch(template, self.buffers, ids, positions, state_index, req_id, padded_tokens=padded)
        self.block_tables.gather_block_tables(batch.idx_mapping, num_reqs_padded=1)
        for group_id, table in enumerate(self.block_tables.input_block_tables):
            block_size = self.block_tables.kernel_block_sizes[group_id]
            block_indices = positions // block_size
            physical = table[0].gather(0, block_indices.long())
            slots = physical.long() * block_size + positions % block_size
            self.block_tables.slot_mappings[group_id].fill_(-1)
            self.block_tables.slot_mappings[group_id, :count].copy_(torch.where(physical != 0, slots, -1))
        if desc.cg_mode == CUDAGraphMode.FULL and (desc.num_reqs or 1) > 1:
            # Distribute only the padding among dummy requests. No dummy row
            # owns a physical cache page or can alter the real prefix.
            requests = desc.num_reqs
            extra = padded - count
            padding_lens = np.full(requests - 1, extra // (requests - 1), dtype=np.int32)
            padding_lens[: extra % (requests - 1)] += 1
            lens = np.concatenate(([count], padding_lens)).astype(np.int32)
            boundaries = np.concatenate(([0], np.cumsum(lens))).astype(np.int32)
            self.buffers.query_start_loc[: requests + 1].copy_(torch.from_numpy(boundaries).to(self.device))
            self.buffers.seq_lens[0] = int(positions[-1].item()) + 1
            self.buffers.seq_lens[1:requests] = torch.from_numpy(padding_lens).to(self.device)
            self.buffers.seq_lens_cpu[:requests].copy_(self.buffers.seq_lens[:requests].cpu())
            for table in self.block_tables.input_block_tables:
                table[1:requests].zero_()
            batch = replace(
                batch,
                num_reqs_after_padding=requests,
                query_start_loc_np=boundaries,
                query_start_loc=self.buffers.query_start_loc[: requests + 1],
                seq_lens=self.buffers.seq_lens[:requests],
                seq_lens_cpu_upper_bound=self.buffers.seq_lens_cpu[:requests],
                seq_lens_np=self.buffers.seq_lens_np[:requests],
                num_scheduled_tokens=lens,
                max_query_len=desc.max_query_len or int(lens.max()),
            )
        with build_attn_metadata_wrapper():
            metadata = self.model_state.prepare_attn(
                batch,
                desc.cg_mode,
                tuple(self.block_tables.input_block_tables),
                self.block_tables.slot_mappings,
                self.attn_groups,
                self.kv_cache_config,
            )
        self.model_state.attn_metadata = metadata
        if desc.cg_mode == CUDAGraphMode.FULL:
            logits, features = manager.run_fullgraph(desc)
            self.speculator.sparse_metrics.record_graph("sparse", "replay")
            logger.info_once("Sparse Target ACL graph replay is active (observed).")
        else:
            slots = build_slot_mappings_by_layer(self.block_tables.slot_mappings[:, :padded], self.kv_cache_config)
            with set_forward_context(
                metadata,
                self.vllm_config,
                num_tokens=padded,
                cudagraph_runtime_mode=desc.cg_mode,
                slot_mapping=slots,
                batch_descriptor=BatchDescriptor(num_tokens=padded),
                is_padding=self.buffers.is_padding[:padded],
            ):
                if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                    logits, features = manager.run_pw_graph(
                        self.graph_model,
                        {
                            "input_ids": self.buffers.input_ids[:padded],
                            "positions": self.buffers.positions[:padded],
                        },
                    )
                else:
                    logits, features = self.graph_model(
                        self.buffers.input_ids[:padded], self.buffers.positions[:padded]
                    )
            self.speculator.sparse_metrics.record_graph("sparse", "eager")
            if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
                logger.warning_once(
                    "Sparse Target forward used %s: no compatible captured shape for %s tokens (prefill=%s)",
                    desc.cg_mode,
                    count,
                    prefill,
                )
        # Graph outputs are persistent and overwritten by the next forward.
        return logits[:count].clone(), [feature[:count].clone() for feature in features]
