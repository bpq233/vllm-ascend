# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Eager, worker-local verifier and DFlash with disposable prefix KV."""

from contextlib import contextmanager
from copy import copy
from math import ceil

import numpy as np
import torch
from vllm.config import CompilationConfig, ModelConfig, SpeculativeConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model_loader
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer, init_attn_backend, init_kv_cache
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import set_eagle3_aux_hidden_state_layers

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops import rotary_embedding as rope
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, get_kv_cache_spec
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_states import init_asecnd_model_state
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator


class IntermediateBackend:
    def __init__(self, parent_config, config, device):
        self.config = config
        self.device = device
        self.parent_config = parent_config
        self.max_num_reqs = config.max_num_seqs
        self.max_model_len = config.max_model_len or parent_config.model_config.max_model_len
        # Full prefixes may exceed the main runner's chunked-prefill budget.
        self.max_num_tokens = min(
            self.max_num_reqs * self.max_model_len,
            max(self.max_model_len, parent_config.scheduler_config.max_num_batched_tokens),
        )
        self.max_num_tokens = max(self.max_num_tokens, self.max_num_reqs * (config.num_speculative_tokens + 1))
        self.vllm_config = copy(parent_config)
        self.vllm_config.additional_config = dict(parent_config.additional_config or {})
        self.vllm_config.additional_config.pop("multi_stage_speculative", None)
        self.vllm_config.compilation_config = CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
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
            with set_current_vllm_config(self.vllm_config):
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
            enforce_eager=True,
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
        with self._context():
            self.model = get_model_loader(cfg.load_config).load_model(vllm_config=cfg, model_config=cfg.model_config)
            set_eagle3_aux_hidden_state_layers(self.model, cfg.speculative_config)
            self.drafter = AscendDFlashSpeculator(cfg, self.device)
            self.drafter.update_stream = None  # This backend never captures graphs.
            self.drafter.load_model(self.model)
            rope.set_cos_and_sin(cfg, self.max_num_reqs, self.drafter.num_query_per_req, parent.dtype, self.device)
            self.input_buffers = AscendInputBuffers(self.max_num_reqs, self.max_num_tokens, self.device)
            self._init_scratch()

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
        # Block zero stays reserved. Each microbatch row owns a disjoint range;
        # every call rewrites all visible prefix positions before reading them.
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
        state = init_asecnd_model_state(cfg, self.model, None, self.device)
        self.drafter.set_attn(state, self.kv_cache_config, self.block_tables, self.input_buffers, self.attn_groups)
        self.drafter.init_cudagraph_manager(CUDAGraphMode.NONE)
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

    def _batches(self, sequences):
        start, num_tokens = 0, 0
        for i, sequence in enumerate(sequences):
            if not sequence or len(sequence) > self.max_model_len:
                raise ValueError("Intermediate prefix must be nonempty and fit max_model_len.")
            if i - start == self.max_num_reqs or num_tokens + len(sequence) > self.max_num_tokens:
                yield start, sequences[start:i]
                start, num_tokens = i, 0
            num_tokens += len(sequence)
        if start < len(sequences):
            yield start, sequences[start:]

    def _forward(self, sequences):
        n = len(sequences)
        lengths = np.array([len(s) for s in sequences], dtype=np.int32)
        starts = np.concatenate((np.zeros(1, dtype=np.int32), np.cumsum(lengths, dtype=np.int32)))
        total = int(starts[-1])
        positions = torch.tensor([p for s in sequences for p in range(len(s))], device=self.device, dtype=torch.int64)
        ids = torch.tensor([t for s in sequences for t in s], device=self.device, dtype=torch.int32)
        mapping = torch.arange(n, device=self.device, dtype=torch.int32)
        lengths_cpu = torch.from_numpy(lengths)
        seq_lens = lengths_cpu.to(self.device)
        starts_gpu = torch.from_numpy(starts).to(self.device)
        # Every packed request starts at position zero and owns disjoint pages.
        for gid, (table, size) in enumerate(
            zip(self.block_tables.input_block_tables, self.block_tables.kernel_block_sizes)
        ):
            slots = torch.cat(
                [
                    table[i, positions[int(starts[i]) : int(starts[i + 1])] // size].long() * size
                    + positions[int(starts[i]) : int(starts[i + 1])] % size
                    for i in range(n)
                ]
            )
            self.block_tables.slot_mappings[gid, :total] = slots
        batch = AscendInputBatch(
            req_ids=[str(i) for i in range(n)],
            num_reqs=n,
            num_reqs_after_padding=n,
            idx_mapping=mapping,
            idx_mapping_np=np.arange(n, dtype=np.int32),
            expanded_idx_mapping=mapping,
            expanded_local_pos=torch.zeros_like(mapping),
            num_scheduled_tokens=lengths,
            num_tokens=total,
            num_tokens_after_padding=total,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
            query_start_loc=starts_gpu,
            query_start_loc_np=starts,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=lengths_cpu,
            dcp_local_seq_lens=None,
            num_computed_tokens_np=np.zeros_like(lengths),
            prefill_len_np=lengths,
            num_computed_prefill_tokens_np=np.zeros_like(lengths),
            is_prefilling_np=np.ones(n, dtype=bool),
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
            attn_state=AscendAttentionState.PrefillNoCache,
            **({} if vllm_version_is("0.27.1") else {"has_prefill": True}),
        )
        metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=n,
            num_tokens=total,
            query_start_loc_gpu=starts_gpu,
            query_start_loc_cpu=torch.from_numpy(starts),
            max_query_len=int(lengths.max()),
            seq_lens=seq_lens,
            max_seq_len=int(lengths.max()),
            block_tables=[table[:n] for table in self.block_tables.input_block_tables],
            slot_mappings=self.block_tables.slot_mappings[:, :total],
            kv_cache_config=self.kv_cache_config,
            seq_lens_np=lengths,
            seq_lens_cpu_upper_bound=lengths_cpu,
            num_computed_tokens_cpu=torch.zeros(n, dtype=torch.int32),
            positions=positions,
            attn_state=AscendAttentionState.PrefillNoCache,
            is_prefilling=torch.ones(n, device=self.device, dtype=torch.bool),
            causal=True,
        )
        slots = build_slot_mappings_by_layer(self.block_tables.slot_mappings[:, :total], self.kv_cache_config)
        rope.update_cos_sin(positions)
        with set_forward_context(
            metadata, self.vllm_config, num_tokens=total, slot_mapping=slots, cudagraph_runtime_mode=CUDAGraphMode.NONE
        ):
            output = self.model(input_ids=ids, positions=positions)
        hidden, aux = output if isinstance(output, tuple) else (output, None)
        return batch, metadata, slots, hidden, aux

    @torch.inference_mode()
    def verify(self, prefixes, drafts):
        if len(prefixes) != len(drafts) or any(not p for p in prefixes):
            raise ValueError("Intermediate verification requires one nonempty prefix per draft.")
        sequences = [p + d for p, d in zip(prefixes, drafts)]
        with self._context():
            for offset, rows in self._batches(sequences):
                batch, _, _, hidden, _ = self._forward(rows)
                indices = [
                    j
                    for i in range(len(rows))
                    for j in range(
                        int(batch.query_start_loc_np[i]) + len(prefixes[offset + i]) - 1,
                        int(batch.query_start_loc_np[i + 1]),
                    )
                ]
                logits = self.model.compute_logits(hidden[torch.tensor(indices, device=self.device, dtype=torch.int64)])
                # Stream microbatch logits so the caller can discard them after
                # acceptance, instead of retaining batch * draft * vocab storage.
                yield from logits.split([len(drafts[offset + i]) + 1 for i in range(len(rows))])

    @torch.inference_mode()
    def propose(self, prefixes):
        # DFlash has a fixed query shape. Near the model boundary use an empty
        # draft: the next verifier round can still provide one bonus token.
        active = [
            i for i, p in enumerate(prefixes) if 2 <= len(p) <= self.max_model_len - self.config.num_speculative_tokens
        ]
        result = [[] for _ in prefixes]
        with self._context():
            for offset, rows in self._batches([prefixes[i][:-1] for i in active]):
                batch, metadata, slots, hidden, aux = self._forward(rows)
                n = len(rows)
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
            self.parent_config.additional_config["multi_stage_speculative"].get(
                "primary_num_speculative_tokens", self.parent_config.speculative_config.num_speculative_tokens
            ),
        )
        width = min(width, self.max_model_len - 1)
        for logits in self.verify([row[: max(1, len(row) - width)] for row in rows], [[0] * width for _ in rows]):
            logits.topk(min(self.config.verification.get("top_k", 5), logits.shape[-1]), dim=-1)
