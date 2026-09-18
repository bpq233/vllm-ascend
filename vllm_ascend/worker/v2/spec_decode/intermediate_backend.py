# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Worker-local verifier and DFlash with independent, persistent paged KV."""

import logging
from contextlib import contextmanager
from copy import copy, deepcopy
from math import ceil

import numpy as np
import torch
from vllm.config import CompilationConfig, ModelConfig, SpeculativeConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.model_loader import get_model_loader
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer, init_attn_backend, init_kv_cache
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import set_eagle3_aux_hidden_state_layers

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops import rotary_embedding as rope
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, get_kv_cache_spec
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_states import init_asecnd_model_state
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.intermediate_cache import IntermediateKVCache
from vllm_ascend.worker.v2.spec_decode.intermediate_graph import (
    IntermediateGraphState,
    capture_secondary_graphs,
    init_secondary_graphs,
)
from vllm_ascend.worker.v2.spec_decode.multi_stage_config import intermediate_capture_sizes, primary_draft_width

logger = logging.getLogger(__name__)


class IntermediateBackend:
    def __init__(self, parent_config, config, device):
        self.config = config
        self.device = device
        self.parent_config = parent_config
        self.graph_enabled = (
            not parent_config.model_config.enforce_eager
            and parent_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        )
        self.graph_state = IntermediateGraphState()
        self.cudagraph_manager = None
        self.graph_replays = 0
        self.max_num_reqs = config.max_num_seqs
        self.cache = IntermediateKVCache(self.max_num_reqs)
        self.forward_tokens = 0
        self.reused_tokens = 0
        self.max_model_len = config.max_model_len or parent_config.model_config.max_model_len
        # Full prefixes may exceed the main runner's chunked-prefill budget.
        self.max_num_tokens = min(
            self.max_num_reqs * self.max_model_len,
            max(self.max_model_len, parent_config.scheduler_config.max_num_batched_tokens),
        )
        self.max_num_tokens = max(self.max_num_tokens, self.max_num_reqs * (config.num_speculative_tokens + 1))
        self.vllm_config = copy(parent_config)
        self.vllm_config.additional_config = deepcopy(parent_config.additional_config or {})
        self.vllm_config.additional_config.pop("multi_stage_speculative", None)
        self.vllm_config.compilation_config = CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE if self.graph_enabled else CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE if self.graph_enabled else CUDAGraphMode.NONE,
            custom_ops=list(parent_config.compilation_config.custom_ops),
        )
        self.vllm_config.scheduler_config = copy(parent_config.scheduler_config)
        self.vllm_config.scheduler_config.max_num_seqs = self.max_num_reqs
        self.vllm_config.scheduler_config.max_num_batched_tokens = self.max_num_tokens
        self.vllm_config.cache_config = copy(parent_config.cache_config)
        self.vllm_config.cache_config.cache_dtype = "auto"
        self.vllm_config.cache_config.enable_prefix_caching = False
        # Ascend RoPE uses process-global buffers; keep the main graph's tensor
        # identities and caches intact when switching to the smaller model.
        self._rope_state = dict.fromkeys(
            (
                "_cos_mla",
                "_sin_mla",
                "_cos_cache",
                "_sin_cache",
                "_cos_sin_cache",
                "_cos",
                "_sin",
                "_cos_slice",
                "_sin_slice",
            )
        )

    @contextmanager
    def _context(self):
        saved = {name: getattr(rope, name) for name in self._rope_state}
        for name, value in self._rope_state.items():
            setattr(rope, name, value)
        try:
            with self.graph_state.context(), set_current_vllm_config(self.vllm_config):
                yield
        finally:
            self._rope_state = {name: getattr(rope, name) for name in saved}
            for name, value in saved.items():
                setattr(rope, name, value)

    def load_model(self):
        parent = self.parent_config.model_config
        cfg = self.vllm_config
        cfg.model_config = ModelConfig(
            model=self.config.verifier_model,
            tokenizer=self.config.verifier_model,
            dtype=parent.dtype,
            seed=parent.seed,
            trust_remote_code=parent.trust_remote_code,
            max_model_len=self.max_model_len,
            enforce_eager=not self.graph_enabled,
        )
        # A copied target config may carry target-specific quantization state.
        cfg.quant_config = None
        if cfg.model_config.quantization is not None:
            raise ValueError("Intermediate verification currently requires unquantized weights.")
        if cfg.model_config.is_multimodal_model or cfg.model_config.use_mla:
            raise ValueError("Intermediate verification requires a text-only full-attention model.")
        if cfg.model_config.get_vocab_size() != parent.get_vocab_size():
            raise ValueError("Intermediate verifier and final target must use the same vocabulary.")
        if (
            cached_tokenizer_from_config(cfg.model_config).get_vocab()
            != cached_tokenizer_from_config(parent).get_vocab()
        ):
            raise ValueError("Intermediate verifier and final target must use identical token IDs.")
        cfg.speculative_config = SpeculativeConfig(
            model=self.config.drafter_model,
            method="dflash",
            num_speculative_tokens=self.config.num_speculative_tokens,
            target_model_config=cfg.model_config,
            target_parallel_config=cfg.parallel_config,
            draft_tensor_parallel_size=cfg.parallel_config.tensor_parallel_size,
            draft_sample_method="greedy",
        )
        if cfg.speculative_config.draft_model_config.max_model_len < self.max_model_len:
            raise ValueError("Set intermediate.max_model_len within the secondary drafter's context limit.")
        if self.max_model_len <= self.config.num_speculative_tokens + 1:
            raise ValueError("Intermediate max_model_len must leave room for DFlash's anchor and draft.")
        if self.graph_enabled:
            from vllm_ascend.platform import _setup_compile_backend

            # Every bucket captures pieces in every verifier layer and holds
            # runtime/TP stream resources. Dense 1..32 gears are too costly
            # with four resident models; use sparse padding-compatible gears.
            cfg.compilation_config.cudagraph_capture_sizes = intermediate_capture_sizes(
                self.max_num_tokens,
                self.max_num_reqs,
                self.config.num_speculative_tokens,
                self.config.cudagraph_capture_sizes,
            )
            cfg.compilation_config.max_cudagraph_capture_size = self.max_num_tokens
            _setup_compile_backend(cfg, self.parent_config.compilation_config.oot_compiler)
        with self._context():
            self.model = get_model_loader(cfg.load_config).load_model(vllm_config=cfg, model_config=cfg.model_config)
            set_eagle3_aux_hidden_state_layers(self.model, cfg.speculative_config)
            self.drafter = AscendDFlashSpeculator(cfg, self.device)
            self.drafter.load_model(self.model)
            rope.set_cos_and_sin(cfg, self.max_num_reqs, self.drafter.num_query_per_req, parent.dtype, self.device)
            self.input_buffers = AscendInputBuffers(self.max_num_reqs, self.max_num_tokens, self.device)
            self._init_scratch()
            if self.graph_enabled:
                self._capture_graphs()

    @torch.inference_mode()
    def _capture_graphs(self):
        self.update_stream = self.drafter.update_stream
        self.cudagraph_manager = ModelAclGraphManager(
            self.vllm_config,
            self.device,
            CUDAGraphMode.PIECEWISE,
            1,
            self,
        )
        logger.info(
            "multi_stage_graph_plan verifier=%s verifier_mode=PIECEWISE verifier_sizes=%s "
            "secondary_mode=FULL_DECODE_ONLY secondary_sizes=%s max_model_len=%d max_num_tokens=%d",
            self.config.verifier_model,
            self.cudagraph_manager.capture_sizes,
            sorted(
                {
                    desc.num_tokens
                    for descs in self.drafter.query_cudagraph_manager._capture_descs.values()
                    for desc in descs
                }
            ),
            self.max_model_len,
            self.max_num_tokens,
        )
        self.cudagraph_manager.capture(
            self.model,
            self.model_state,
            self.input_buffers,
            None,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            use_aux_hidden_state_outputs=True,
            progress_bar_desc="Capturing intermediate verifier graphs",
        )
        capture_secondary_graphs(self.drafter)
        # Dummy capture inputs must not publish a valid token prefix.
        self.cache.retain(())

    def _init_scratch(self):
        cfg = self.vllm_config
        specs = get_kv_cache_spec(cfg)
        groups = []
        for name, spec in specs.items():
            if not isinstance(spec, FullAttentionSpec):
                raise ValueError("Intermediate scratch KV currently supports full attention only.")
            is_draft = name in self.drafter.draft_attn_layer_names
            group = next((g for g in groups if g.kv_cache_spec == spec and g.is_eagle_group == is_draft), None)
            if group is None:
                group = KVCacheGroupSpec([], spec, is_eagle_group=is_draft)
                groups.append(group)
            group.layer_names.append(name)
        # Block zero stays reserved. Each persistent request slot owns disjoint
        # ranges in the verifier and drafter groups, including speculative KV.
        num_blocks = 1 + self.max_num_reqs * max(ceil(self.max_model_len / g.kv_cache_spec.block_size) for g in groups)
        self.kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[KVCacheTensor(num_blocks * spec.page_size_bytes, [name]) for name, spec in specs.items()],
            kv_cache_groups=groups,
        )
        all_groups, _, kernel_sizes = init_attn_backend(self.kv_cache_config, cfg, self.device)
        self.attn_groups = [
            [g for g in group if not set(g.layer_names).intersection(self.drafter.draft_attn_layer_names)]
            for group in all_groups
        ]
        self.block_tables = BlockTables(
            block_sizes=[g.kv_cache_spec.block_size for g in groups],
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=[ceil(self.max_model_len / g.kv_cache_spec.block_size) for g in groups],
            device=self.device,
            kernel_block_sizes=kernel_sizes,
        )
        for table, group, size in zip(self.block_tables.input_block_tables, groups, kernel_sizes):
            factor = group.kv_cache_spec.block_size // size
            # Account for backends splitting one allocator block into kernels.
            table.copy_(
                torch.arange(factor, factor + table.numel(), device=self.device, dtype=torch.int32).view_as(table)
            )
        self.cache_block_tables = [table.clone() for table in self.block_tables.input_block_tables]
        self.model_state = init_asecnd_model_state(cfg, self.model, None, self.device)
        self.drafter.set_attn(
            self.model_state, self.kv_cache_config, self.block_tables, self.input_buffers, self.attn_groups
        )
        init_secondary_graphs(self.drafter, cfg.compilation_config.cudagraph_mode, self.device)
        self.kv_caches = []
        init_kv_cache(
            self.kv_caches,
            cfg.compilation_config.static_forward_context,
            self.kv_cache_config,
            all_groups,
            self.device,
            cfg.cache_config.cache_dtype,
            kernel_sizes,
            cfg,
        )

    def _batches(self, sequences, req_ids=None, required_starts=None):
        start, num_tokens = 0, 0
        for i, sequence in enumerate(sequences):
            if not sequence or len(sequence) > self.max_model_len:
                raise ValueError("Intermediate prefix must be nonempty and fit max_model_len.")
            required = required_starts[i] if required_starts is not None else len(sequence) - 1
            cached = self.cache.query_start(req_ids[i], sequence, required) if req_ids is not None else 0
            if i - start == self.max_num_reqs or num_tokens + len(sequence) - cached > self.max_num_tokens:
                yield start, sequences[start:i]
                start, num_tokens = i, 0
                # The previous microbatch may have evicted this request.
                cached = self.cache.query_start(req_ids[i], sequence, required) if req_ids is not None else 0
            num_tokens += len(sequence) - cached
        if start < len(sequences):
            yield start, sequences[start:]

    def _forward(self, sequences, req_ids=None, required_starts=None):
        n = len(sequences)
        if req_ids is None:
            req_ids = [str(i) for i in range(n)]
        if required_starts is None:
            required_starts = [len(s) - 1 for s in sequences]
        cache_slots, computed = self.cache.plan(req_ids, sequences, required_starts)
        lengths = np.array([len(s) for s in sequences], dtype=np.int32)
        computed = np.asarray(computed, dtype=np.int32)
        query_lens = lengths - computed
        starts = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(query_lens, dtype=np.int32)))
        total = int(starts[-1])
        if total > self.max_num_tokens:
            raise ValueError("Intermediate incremental query exceeds the token buffer.")
        manager = self.cudagraph_manager
        desc = manager.dispatch(n, total, None, 0) if manager is not None else None
        if desc is not None and desc.cg_mode != CUDAGraphMode.PIECEWISE:
            raise RuntimeError("Intermediate query has no captured piecewise graph; refusing silent eager fallback.")
        padded_total = desc.num_tokens if desc is not None else total
        # Pack CPU-produced inputs/metadata into one pinned H2D transfer. Views
        # stay alive through the queued work; no reusable host buffer can race
        # with an asynchronous copy from a previous microbatch.
        fields = [
            cache_slots,
            lengths.tolist(),
            starts.tolist(),
            np.repeat(np.arange(n, dtype=np.int64), query_lens).tolist(),
            [p for s, c in zip(sequences, computed) for p in range(c, len(s))],
            [t for s, c in zip(sequences, computed) for t in s[c:]],
        ]
        host_inputs = torch.tensor(
            [value for field in fields for value in field],
            dtype=torch.int64,
            pin_memory=self.device.type != "cpu",
        )
        cache_slots_gpu, lengths_gpu, starts_gpu, token_rows, positions_gpu, ids_gpu = host_inputs.to(
            self.device, non_blocking=True
        ).split([len(field) for field in fields])
        for table, cached in zip(self.block_tables.input_block_tables, self.cache_block_tables):
            table[:n].copy_(cached.index_select(0, cache_slots_gpu))
        positions = self.input_buffers.positions[:total]
        positions.copy_(positions_gpu)
        ids = self.input_buffers.input_ids[:total]
        ids.copy_(ids_gpu)
        self.input_buffers.input_ids[total:padded_total].zero_()
        self.input_buffers.positions[total:padded_total].zero_()
        mapping = torch.arange(n, device=self.device, dtype=torch.int32)
        lengths_cpu = torch.from_numpy(lengths)
        seq_lens = lengths_gpu.int()
        starts_gpu = starts_gpu.int()
        is_prefilling = computed == 0
        attn_state = (
            AscendAttentionState.PrefillNoCache if np.all(is_prefilling) else AscendAttentionState.ChunkedPrefill
        )
        # Writes use absolute positions in each request's persistent pages.
        for gid, (table, size) in enumerate(
            zip(self.block_tables.input_block_tables, self.block_tables.kernel_block_sizes)
        ):
            slots = table[token_rows, positions // size].long() * size + positions % size
            self.block_tables.slot_mappings[gid, :total] = slots
        self.block_tables.slot_mappings[:, total:padded_total].fill_(-1)
        batch = AscendInputBatch(
            req_ids=req_ids,
            num_reqs=n,
            num_reqs_after_padding=n,
            idx_mapping=mapping,
            idx_mapping_np=np.arange(n, dtype=np.int32),
            expanded_idx_mapping=mapping,
            expanded_local_pos=torch.zeros_like(mapping),
            num_scheduled_tokens=query_lens,
            num_tokens=total,
            num_tokens_after_padding=total,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=starts_gpu,
            query_start_loc_np=starts,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=lengths_cpu,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=computed,
            prefill_len_np=lengths,
            num_computed_prefill_tokens_np=computed,
            is_prefilling_np=is_prefilling,
            max_seq_len_np=None,
            input_ids=ids,
            positions=positions,
            is_padding=torch.zeros(total, device=self.device, dtype=torch.bool),
            logits_indices=starts_gpu[1:].long() - 1,
            cu_num_logits=torch.arange(n + 1, device=self.device, dtype=torch.int32),
            cu_num_logits_np=np.arange(n + 1, dtype=np.int32),
            has_structured_output_reqs=False,
            prompt_lens=None,
            seq_lens_np=lengths,
            attn_state=attn_state,
            **({} if vllm_version_is("0.27.1") else {"has_prefill": bool(np.any(is_prefilling))}),
        )
        metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=n,
            num_tokens=total,
            query_start_loc_gpu=starts_gpu,
            query_start_loc_cpu=torch.from_numpy(starts),
            max_query_len=int(query_lens.max()),
            seq_lens=seq_lens,
            max_seq_len=int(lengths.max()),
            block_tables=[table[:n] for table in self.block_tables.input_block_tables],
            slot_mappings=self.block_tables.slot_mappings[:, :padded_total],
            kv_cache_config=self.kv_cache_config,
            seq_lens_np=lengths,
            seq_lens_cpu_upper_bound=lengths_cpu,
            num_computed_tokens_cpu=torch.from_numpy(computed),
            positions=positions,
            attn_state=attn_state,
            is_prefilling=seq_lens == starts_gpu[1:] - starts_gpu[:-1],
            causal=True,
            num_actual_tokens=total,
            num_input_tokens=padded_total,
        )
        slots = build_slot_mappings_by_layer(self.block_tables.slot_mappings[:, :padded_total], self.kv_cache_config)
        model_positions = self.input_buffers.positions[:padded_total]
        padding = self.input_buffers.is_padding[:padded_total]
        padding[:total].fill_(False)
        padding[total:].fill_(True)
        rope.update_cos_sin(model_positions)
        with set_forward_context(
            metadata,
            self.vllm_config,
            num_tokens=padded_total,
            slot_mapping=slots,
            cudagraph_runtime_mode=desc.cg_mode if desc is not None else CUDAGraphMode.NONE,
            batch_descriptor=BatchDescriptor(num_tokens=padded_total) if desc is not None else None,
            is_padding=padding,
        ):
            model_inputs = dict(input_ids=self.input_buffers.input_ids[:padded_total], positions=model_positions)
            if manager is not None:
                output = manager.run_pw_graph(self.model, model_inputs)
                self.graph_replays += 1
            else:
                output = self.model(**model_inputs)
        hidden, aux = output if isinstance(output, tuple) else (output, None)
        hidden = hidden[:total]
        aux = [state[:total] for state in aux] if aux else aux
        # DFlash's context K/V depends on verifier hidden states, not its own
        # temporary query K/V. Populate it for the new suffix while those
        # hidden states are available, so later proposals need no prefix pass.
        context_hidden = self.drafter.model.combine_hidden_states(torch.cat(aux, dim=-1)) if aux else hidden
        gids = self.drafter.draft_kv_cache_group_ids
        layer_groups = self.drafter._layer_group_idx
        context_slots = (
            [self.block_tables.slot_mappings[gids[i], :total] for i in layer_groups]
            if layer_groups is not None
            else self.block_tables.slot_mappings[gids[0], :total]
        )
        self.drafter.model.precompute_and_store_context_kv(context_hidden, positions, context_slots)
        self.cache.commit(cache_slots, sequences)
        self.forward_tokens += total
        self.reused_tokens += int(computed.sum())
        return batch, metadata, slots, hidden, aux

    @torch.inference_mode()
    def verify(self, prefixes, drafts, req_ids=None):
        for logits, lengths in self.verify_batches(prefixes, drafts, req_ids=req_ids):
            yield from logits.split(lengths)

    @torch.inference_mode()
    def verify_batches(self, prefixes, drafts, req_ids=None):
        if len(prefixes) != len(drafts) or any(not p for p in prefixes):
            raise ValueError("Intermediate verification requires one nonempty prefix per draft.")
        sequences = [p + d for p, d in zip(prefixes, drafts)]
        req_ids = req_ids if req_ids is not None else [str(i) for i in range(len(sequences))]
        required_starts = [len(p) - 1 for p in prefixes]
        with self._context():
            for offset, rows in self._batches(sequences, req_ids, required_starts):
                required = required_starts[offset : offset + len(rows)]
                batch, _, _, hidden, _ = self._forward(rows, req_ids[offset : offset + len(rows)], required)
                indices = [
                    j
                    for i in range(len(rows))
                    for j in range(
                        int(batch.query_start_loc_np[i]) + required[i] - int(batch.num_computed_tokens_np[i]),
                        int(batch.query_start_loc_np[i + 1]),
                    )
                ]
                logits = self.model.compute_logits(hidden[torch.tensor(indices, device=self.device, dtype=torch.int64)])
                # Keep microbatches packed through acceptance, without retaining
                # batch * draft * vocab storage across forwards.
                yield logits, [len(drafts[offset + i]) + 1 for i in range(len(rows))]

    @torch.inference_mode()
    def propose(self, prefixes, req_ids=None):
        # DFlash has a fixed query shape. Near the model boundary use an empty
        # draft: the next verifier round can still provide one bonus token.
        active = [
            i for i, p in enumerate(prefixes) if 2 <= len(p) <= self.max_model_len - self.config.num_speculative_tokens
        ]
        result = [[] for _ in prefixes]
        req_ids = req_ids if req_ids is not None else [str(i) for i in range(len(prefixes))]
        active_ids = [req_ids[i] for i in active]
        with self._context():
            for offset, rows in self._batches([prefixes[i][:-1] for i in active], active_ids):
                n = len(rows)
                batch_ids = active_ids[offset : offset + n]
                batch, metadata, slots, hidden, aux = self._forward(rows, batch_ids)
                anchors = torch.zeros(self.max_num_reqs, device=self.device, dtype=torch.int64)
                anchors[:n] = torch.tensor([prefixes[i][-1] for i in active[offset : offset + n]], device=self.device)
                tokens = self.drafter.propose(
                    batch,
                    metadata,
                    slots,
                    hidden,
                    aux,
                    torch.ones(n, device=self.device, dtype=torch.int32),
                    torch.zeros(n, device=self.device, dtype=torch.int32),
                    anchors,
                    anchors,
                    torch.zeros(self.max_num_reqs, device=self.device),
                    torch.zeros(self.max_num_reqs, device=self.device, dtype=torch.int64),
                )
                for i, token_ids in zip(active[offset : offset + n], tokens.tolist()):
                    result[i] = token_ids
        return result

    def profile(self):
        # Include full-prefix activations in the main worker's peak measurement.
        length = self.max_model_len - self.config.num_speculative_tokens
        rows, remaining = [], self.max_num_tokens
        while remaining >= 2 and len(rows) < self.max_num_reqs:
            size = min(length, remaining)
            rows.append([0] * size)
            remaining -= size
        self.propose(rows)
        width = max(
            self.config.num_speculative_tokens,
            primary_draft_width(
                self.parent_config.additional_config["multi_stage_speculative"],
                self.parent_config.speculative_config.num_speculative_tokens,
            ),
        )
        width = min(width, self.max_model_len - 1)
        for logits in self.verify([row[: max(1, len(row) - width)] for row in rows], [[0] * width for _ in rows]):
            logits.topk(min(self.config.verification.get("top_k", 5), logits.shape[-1]), dim=-1)
        self.cache.retain(())
