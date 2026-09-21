# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Worker-local verifier and DFlash with independent, persistent paged KV."""

import logging
from collections import OrderedDict
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
from vllm_ascend.compilation import acl_graph
from vllm_ascend.ops import rotary_embedding as rope
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, get_kv_cache_spec
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_states import init_asecnd_model_state
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.multi_stage.config import intermediate_capture_sizes, primary_draft_width

logger = logging.getLogger(__name__)


class IntermediateKVCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.slots = OrderedDict()
        self.tokens = [[] for _ in range(capacity)]

    def retain(self, live_request_ids):
        live = set(live_request_ids)
        for req_id in list(self.slots):
            if req_id not in live:
                self.tokens[self.slots.pop(req_id)] = []

    def query_start(self, req_id, sequence, required):
        if req_id not in self.slots:
            return 0
        cached = self.tokens[self.slots[req_id]]
        common = min(len(cached), required)
        # Most of a long prefix is unchanged. Compare bounded slices in C
        # rather than dispatching a Python iteration for every cached token.
        # Only the first divergent block needs a token-level scan.
        for start in range(0, common, 1024):
            end = min(start + 1024, common)
            if cached[start:end] != sequence[start:end]:
                for i in range(start, end):
                    if cached[i] != sequence[i]:
                        return i
        return common

    def plan(self, req_ids, sequences, required_starts):
        """Reserve disjoint slots and invalidate a divergent/unneeded tail.

        required_starts retains the predictor row needed for logits or the
        drafter's anchor. No caller may read a cached hidden state: that row
        is recomputed, while all preceding KV is reused.
        """
        if (
            len(set(req_ids)) != len(req_ids)
            or len(req_ids) > self.capacity
            or len(sequences) != len(req_ids)
            or len(required_starts) != len(req_ids)
        ):
            raise ValueError("Intermediate cache needs distinct request IDs within its capacity.")
        protected = set(req_ids)
        slots, starts = [], []
        for req_id, sequence, required in zip(req_ids, sequences, required_starts):
            if not sequence or not 0 <= required < len(sequence):
                raise ValueError("A nonempty query and a valid predictor position are required.")
            if req_id not in self.slots:
                free = set(range(self.capacity)) - set(self.slots.values())
                if free:
                    slot = min(free)
                else:
                    victim = next(key for key in self.slots if key not in protected)
                    slot = self.slots.pop(victim)
                self.tokens[slot] = []
                self.slots[req_id] = slot
            slot = self.slots[req_id]
            self.slots.move_to_end(req_id)
            cached = self.tokens[slot]
            common = self.query_start(req_id, sequence, required)
            # Token equality also handles final-target rejection, preemption,
            # and request-ID reuse without trusting speculative lengths.
            del cached[common:]
            slots.append(slot)
            starts.append(common)
        return slots, starts

    def commit(self, slots, sequences):
        # Called only after BOTH verifier KV and DFlash context KV are written.
        for slot, sequence in zip(slots, sequences):
            # plan() already retained exactly the equal, valid prefix. Only
            # append newly computed IDs instead of copying a long prefix again.
            cached = self.tokens[slot]
            cached.extend(sequence[len(cached) :])


class IntermediateGraphState:
    """Own graph parameter buckets without replacing the main runner's buckets.

    ACL attention stores capture handles in module globals. The intermediate
    verifier and secondary draft have independent weights and KV, so sharing
    those handles with the target/primary draft would update the wrong graphs.
    Like the worker's forward context, this scope is used on its execution
    thread. Reentrant entry must retain any buckets initialized by the caller.
    """

    _names = ("_graph_params", "_draft_graph_params", "_draft_graph_prefill_params")

    def __init__(self):
        self._params = dict.fromkeys(self._names)
        self._depth = 0

    @contextmanager
    def context(self):
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        saved = {name: getattr(acl_graph, name) for name in self._names}
        for name, value in self._params.items():
            setattr(acl_graph, name, value)
        self._depth = 1
        try:
            yield
        finally:
            self._params = {name: getattr(acl_graph, name) for name in self._names}
            for name, value in saved.items():
                setattr(acl_graph, name, value)
            self._depth = 0


def init_secondary_graphs(drafter, mode, device):
    """Initialize after set_attn, inside the intermediate graph-state scope."""
    enabled = mode != CUDAGraphMode.NONE
    drafter.update_stream = torch.npu.Stream(device=device) if enabled else None
    # DFlash's parallel query is uniform, including when verifier queries are
    # ragged. Request its full decode graph separately from verifier piecewise.
    drafter.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY if enabled else CUDAGraphMode.NONE)
    if enabled and not drafter.query_cudagraph_manager.needs_capture():
        raise ValueError("Secondary DFlash requires full graph attention support and nonempty capture sizes.")


def capture_secondary_graphs(drafter):
    """Capture only after both models' KV tensors have been initialized."""
    if drafter.query_cudagraph_manager.needs_capture():
        drafter.capture()


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
        self.executed_tokens = 0
        self.reused_tokens = 0
        self.reused_hidden_tokens = 0
        self._context_rows = [None] * self.max_num_reqs
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
        self._init_proposal_scratch()

    def _init_proposal_scratch(self):
        # Stable device addresses also avoid tiny allocations/kernels on every
        # intermediate round. Only anchors vary; the other inputs are constants.
        self._proposal_anchors = torch.zeros(self.max_num_reqs, device=self.device, dtype=torch.int64)
        self._proposal_ones = torch.ones(self.max_num_reqs, device=self.device, dtype=torch.int32)
        self._proposal_zeros = torch.zeros(self.max_num_reqs, device=self.device, dtype=torch.int32)
        self._proposal_temperature = torch.zeros(self.max_num_reqs, device=self.device)
        self._proposal_seeds = torch.zeros(self.max_num_reqs, device=self.device, dtype=torch.int64)
        self._request_offsets = torch.arange(self.max_num_reqs + 1, device=self.device, dtype=torch.int32)
        self._query_padding = torch.zeros(self.max_num_tokens, device=self.device, dtype=torch.bool)

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

    def _cached_context_rows(self, sequences, req_ids):
        """Use owned hidden rows only while the corresponding prefix KV is valid."""
        rows = []
        for req_id, sequence in zip(req_ids, sequences):
            slot = self.cache.slots.get(req_id)
            if slot is None or self.cache.query_start(req_id, sequence, len(sequence)) != len(sequence):
                return None
            saved = self._context_rows[slot]
            position = len(sequence) - 1
            if saved is None or saved[0] != req_id or not saved[1] <= position < saved[2]:
                return None
            offset = position - saved[1]
            rows.append(saved[3][offset : offset + 1])
        return torch.cat(rows, dim=0) if rows else None

    def _forward(self, sequences, req_ids=None, required_starts=None, reuse_context=False):
        n = len(sequences)
        if req_ids is None:
            req_ids = [str(i) for i in range(n)]
        if required_starts is None:
            required_starts = [len(s) - 1 for s in sequences]
        cached_context = self._cached_context_rows(sequences, req_ids) if reuse_context else None
        cache_slots, computed = self.cache.plan(req_ids, sequences, required_starts)
        lengths = np.array([len(s) for s in sequences], dtype=np.int32)
        computed = np.asarray(computed, dtype=np.int32)
        query_lens = lengths - computed
        starts = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(query_lens, dtype=np.int32)))
        total = int(starts[-1])
        if total > self.max_num_tokens:
            raise ValueError("Intermediate incremental query exceeds the token buffer.")
        manager = self.cudagraph_manager if cached_context is None else None
        desc = manager.dispatch(n, total, None, 0) if manager is not None else None
        if desc is not None and desc.cg_mode != CUDAGraphMode.PIECEWISE:
            raise RuntimeError("Intermediate query has no captured piecewise graph; refusing silent eager fallback.")
        padded_total = desc.num_tokens if desc is not None else total
        # Pack CPU-produced inputs/metadata into one pinned H2D transfer. Views
        # stay alive through the queued work; no reusable host buffer can race
        # with an asynchronous copy from a previous microbatch.
        field_sizes = [n, n, n + 1, total, total, total]
        host_inputs = torch.empty(sum(field_sizes), dtype=torch.int64, pin_memory=self.device.type != "cpu")
        # numpy() is a view of CPU pinned storage, not a device read. Fill it
        # in bulk instead of materializing multiple Python integers per token.
        host_fields = np.split(host_inputs.numpy(), np.cumsum(field_sizes)[:-1])
        host_fields[0][:] = cache_slots
        host_fields[1][:] = lengths
        host_fields[2][:] = starts
        token_rows_cpu = np.repeat(np.arange(n), query_lens)
        host_fields[3][:] = token_rows_cpu
        host_fields[4][:] = np.arange(total) + computed[token_rows_cpu] - starts[token_rows_cpu]
        for row, (sequence, count) in enumerate(zip(sequences, computed)):
            host_fields[5][starts[row] : starts[row + 1]] = sequence[count:]
        cache_slots_gpu, lengths_gpu, starts_gpu, token_rows, positions_gpu, ids_gpu = host_inputs.to(
            self.device, non_blocking=True
        ).split(field_sizes)
        for table, cached in zip(self.block_tables.input_block_tables, self.cache_block_tables):
            table[:n].copy_(cached.index_select(0, cache_slots_gpu))
        positions = self.input_buffers.positions[:total]
        positions.copy_(positions_gpu)
        ids = self.input_buffers.input_ids[:total]
        ids.copy_(ids_gpu)
        self.input_buffers.input_ids[total:padded_total].zero_()
        self.input_buffers.positions[total:padded_total].zero_()
        mapping = self._request_offsets[:n]
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
            expanded_local_pos=self._proposal_zeros[:n],
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
            is_padding=self._query_padding[:total],
            logits_indices=starts_gpu[1:].long() - 1,
            cu_num_logits=self._request_offsets[: n + 1],
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
        if cached_context is not None:
            # KV and projected auxiliary rows already exist from verification.
            # DFlash accepts this combined context directly with aux=None.
            # Avoid a verifier forward AND a second auxiliary projection here.
            self.cache.commit(cache_slots, sequences)
            self.reused_hidden_tokens += total
            self.reused_tokens += sum(lengths)
            return batch, metadata, slots, cached_context, None
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
        for row, (req_id, slot, sequence, required) in enumerate(zip(req_ids, cache_slots, sequences, required_starts)):
            # Only retain prediction rows, never the full prompt activations.
            # clone owns storage: future graph replay overwrites model outputs.
            begin = int(starts[row]) + required - int(computed[row])
            end = int(starts[row + 1])
            self._context_rows[slot] = (req_id, required, len(sequence), context_hidden[begin:end].clone())
        self.forward_tokens += total
        self.executed_tokens += padded_total
        self.reused_tokens += int(computed.sum())
        return batch, metadata, slots, hidden, aux

    @torch.inference_mode()
    def verify(self, prefixes, drafts, req_ids=None):
        for logits, lengths in self.verify_batches(prefixes, drafts, req_ids=req_ids):
            yield from logits.split(lengths)

    def _warm_prefixes(self, sequences, req_ids, required):
        """Avoid pathological padding across a large gap in sparse graph gears."""
        sizes = getattr(self.cudagraph_manager, "capture_sizes", ())
        computed = [self.cache.query_start(r, s, p) for r, s, p in zip(req_ids, sequences, required)]
        total = sum(len(s) - c for s, c in zip(sequences, computed))
        padded = next((size for size in sizes if size >= total), total)
        smaller = [size for size in sizes if size < total]
        if padded <= 8 * total or not smaller or not any(c < p for c, p in zip(computed, required)):
            return
        # Reserve the whole group first, so per-request warming cannot evict
        # another member. Only context rows are split; prediction stays packed.
        self.cache.plan(req_ids, sequences, required)
        chunk = max(smaller)
        for req_id, sequence, end, start in zip(req_ids, sequences, required, computed):
            while start < end:
                start = min(start + chunk, end)
                self._forward([sequence[:start]], [req_id], [start - 1])

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
                batch_ids = req_ids[offset : offset + len(rows)]
                self._warm_prefixes(rows, batch_ids, required)
                batch, _, _, hidden, _ = self._forward(rows, batch_ids, required)
                # On cache hits every forwarded row predicts a candidate/bonus.
                # Avoid a CPU index list, H2D index copy and device gather there.
                if np.array_equal(batch.num_computed_tokens_np, required):
                    prediction_hidden = hidden
                    prediction_tokens = batch.input_ids
                elif len(rows) == 1:
                    begin = required[0] - int(batch.num_computed_tokens_np[0])
                    prediction_hidden = hidden[begin:]
                    prediction_tokens = batch.input_ids[begin:]
                else:
                    indices = [
                        j
                        for i in range(len(rows))
                        for j in range(
                            int(batch.query_start_loc_np[i]) + required[i] - int(batch.num_computed_tokens_np[i]),
                            int(batch.query_start_loc_np[i + 1]),
                        )
                    ]
                    host_indices = torch.tensor(indices, dtype=torch.int64, pin_memory=self.device.type != "cpu")
                    device_indices = host_indices.to(self.device, non_blocking=True)
                    prediction_hidden = hidden.index_select(0, device_indices)
                    prediction_tokens = batch.input_ids.index_select(0, device_indices)
                logits = self.model.compute_logits(prediction_hidden)
                # Keep microbatches packed through acceptance, without retaining
                # batch * draft * vocab storage across forwards.
                # Predictor inputs are [anchor, draft...]. Shift on device;
                # the value at each bonus row is ignored by the acceptance mask.
                # This view is valid only until the generator resumes: the next
                # model replay may overwrite input buffers.
                self.verification_tokens = (
                    prediction_tokens.roll(-1) if self.config.verification.get("method", "topk") != "all" else None
                )
                try:
                    yield logits, [len(drafts[offset + i]) + 1 for i in range(len(rows))]
                finally:
                    self.verification_tokens = None

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
                batch, metadata, slots, hidden, aux = self._forward(rows, batch_ids, reuse_context=True)
                anchors = self._proposal_anchors
                # Fresh pinned host storage avoids an asynchronous H2D reuse
                # race while keeping the captured device pointer stable.
                host_anchors = torch.tensor(
                    [prefixes[i][-1] for i in active[offset : offset + n]],
                    dtype=torch.int64,
                    pin_memory=self.device.type != "cpu",
                )
                anchors[:n].copy_(host_anchors, non_blocking=True)
                tokens = self.drafter.propose(
                    batch,
                    metadata,
                    slots,
                    hidden,
                    aux,
                    self._proposal_ones[:n],
                    self._proposal_zeros[:n],
                    anchors,
                    anchors,
                    self._proposal_temperature,
                    self._proposal_seeds,
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
        self._context_rows = [None] * self.max_num_reqs
