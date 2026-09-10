# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private eager MRV2 runner for intermediate verification and DFlash expansion.

Requests execute serially here, each with a whole candidate forward. The final
target retains its normal packed batch forward. No target KV tensors are passed
to this backend. Metadata is rebound on each call; cached prefix KV is retained.
"""

import copy
from dataclasses import dataclass, field
from time import perf_counter

import torch
from vllm.config import CompilationConfig, ModelConfig, SpeculativeConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec

from .acceptance import accepted_prefix_length
from .interfaces import VerificationResult


@dataclass
class PrivateCache:
    tokens: tuple[int, ...] = ()
    blocks: list[list[int]] = field(default_factory=list)


class IntermediateBackend:
    def __init__(self, vllm_config, device, config, policy):
        # These imports are deliberately lazy: constructing a nested runner from
        # a speculator otherwise creates a module import cycle.
        from vllm.tokenizers import get_tokenizer

        from vllm_ascend.worker.v2.model_runner import NPUModelRunner

        self.config = config
        self.policy = policy
        self.device = device
        private = copy.copy(vllm_config)
        private.model_config = ModelConfig(
            model=config.intermediate_model,
            revision=config.intermediate_revision,
            dtype=vllm_config.model_config.dtype,
            max_model_len=vllm_config.model_config.max_model_len,
            enforce_eager=True,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )
        private.load_config = copy.copy(vllm_config.load_config)
        private.compilation_config = CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            custom_ops=list(vllm_config.compilation_config.custom_ops),
        )
        if private.model_config.is_moe:
            raise ValueError("Multi-stage intermediate runner currently supports dense models only")
        # Model-specific quantization must not leak from the final target.
        private.quant_config = type(private)._get_quantization_config(private.model_config, private.load_config)
        private.parallel_config = copy.copy(vllm_config.parallel_config)
        private.scheduler_config = copy.copy(vllm_config.scheduler_config)
        private.scheduler_config.async_scheduling = False
        private.cache_config = copy.copy(vllm_config.cache_config)
        private.cache_config.enable_prefix_caching = False
        private.cache_config.num_gpu_blocks_override = None
        private.cache_config.kv_cache_memory_bytes = config.kv_cache_memory_bytes
        private.additional_config = dict(vllm_config.additional_config or {})
        private.additional_config.pop("multi_stage_spec_config", None)
        private.attention_config = copy.copy(vllm_config.attention_config)
        private.speculative_config = SpeculativeConfig(
            model=config.secondary_model,
            method="dflash",
            revision=config.secondary_revision,
            num_speculative_tokens=config.secondary_num_speculative_tokens,
            target_model_config=private.model_config,
            target_parallel_config=private.parallel_config,
        )
        self.vllm_config = private
        target_tokenizer = get_tokenizer(
            vllm_config.model_config.tokenizer,
            revision=vllm_config.model_config.tokenizer_revision,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )
        intermediate_tokenizer = get_tokenizer(
            private.model_config.tokenizer,
            revision=private.model_config.tokenizer_revision,
            trust_remote_code=private.model_config.trust_remote_code,
        )
        if target_tokenizer.get_vocab() != intermediate_tokenizer.get_vocab():
            raise ValueError("Multi-stage models must use identical token-to-ID mappings")
        with set_current_vllm_config(private):
            self.runner = NPUModelRunner(private, device)
            self.runner.load_model()
            specs = self.runner.get_kv_cache_spec()
            if not specs or any(
                type(spec) is not FullAttentionSpec
                or spec.sliding_window is not None
                or spec.attention_chunk_size is not None
                for spec in specs.values()
            ):
                raise ValueError("Intermediate multi-stage runner currently supports full attention KV only")
            kv_config = get_kv_cache_configs(private, [specs], [config.kv_cache_memory_bytes])[0]
            self.runner.initialize_kv_cache(kv_config)
            self.runner._init_kv_zero_meta()
            self.runner.kv_block_zeroer.zero_block_ids([0])
        self.block_sizes = [g.kv_cache_spec.block_size for g in kv_config.kv_cache_groups]
        # A shared pool avoids assuming cache groups use distinct storage.
        # Block zero is reserved for padding, as in the scheduler block pool.
        self.free_blocks = list(range(1, kv_config.num_blocks))
        self.caches: dict[str, PrivateCache] = {}
        self.next_drafts: dict[str, list[int]] = {}
        self.secondary_calls = 0
        self.secondary_tokens = 0
        self.secondary_ms = 0.0
        self.verifier_calls = 0
        self.verifier_tokens = 0
        self.verifier_ms = 0.0

    def _reserve(self, request_id, capacity):
        cache = self.caches.setdefault(request_id, PrivateCache(blocks=[[] for _ in self.block_sizes]))
        counts = [(capacity + size - 1) // size for size in self.block_sizes]
        needed = sum(max(0, n - len(blocks)) for n, blocks in zip(counts, cache.blocks))
        if needed > len(self.free_blocks):
            raise RuntimeError(
                "Private multi-stage KV budget exhausted; increase kv_cache_memory_bytes or reduce batch/length"
            )
        allocated = []
        for n, blocks in zip(counts, cache.blocks):
            while len(blocks) < n:
                block = self.free_blocks.pop()
                blocks.append(block)
                allocated.append(block)
        if allocated:
            self.runner.kv_block_zeroer.zero_block_ids(allocated)
        return cache

    def _store_draft_context(self, execution, cache):
        draft = self.runner.speculator
        hidden = execution.hidden_states
        if execution.aux_hidden_states:
            hidden = draft.model.combine_hidden_states(torch.cat(execution.aux_hidden_states, dim=-1))
        batch = execution.input_batch
        positions = batch.positions[: batch.num_tokens]
        group_slots = []
        for gid in draft.draft_kv_cache_group_ids:
            blocks = torch.tensor(cache.blocks[gid], dtype=torch.int64, device=self.device)
            size = self.block_sizes[gid]
            group_slots.append((blocks[positions.long() // size] * size + positions % size).to(torch.int32))
        slots = (
            [group_slots[i] for i in draft._layer_group_idx] if draft._layer_group_idx is not None else group_slots[0]
        )
        draft.model.precompute_and_store_context_kv(hidden[: batch.num_tokens], positions, slots)

    def _forward(self, request_id, tokens, context_len):
        """Incrementally forward a prefix, overlapping its last token for logits."""
        capacity = min(self.runner.max_model_len, len(tokens) + self.config.secondary_num_speculative_tokens)
        cache = self._reserve(request_id, capacity)
        common = 0
        for a, b in zip(cache.tokens, tokens):
            if a != b:
                break
            common += 1
        # The last context token must yield the logits predicting draft[0].
        start = min(common, context_len - 1)
        if start < 0:
            raise ValueError("Intermediate verification requires a nonempty context")
        runner = self.runner
        execution = None
        logits_parts = []
        while start < len(tokens):
            end = min(len(tokens), start + runner.max_num_tokens)
            scheduled = SchedulerOutput.make_empty()
            scheduled.scheduled_new_reqs = [
                NewRequestData(
                    req_id=request_id,
                    prompt_token_ids=list(tokens),
                    mm_features=[],
                    sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
                    pooling_params=None,
                    block_ids=tuple(cache.blocks),
                    num_computed_tokens=start,
                    lora_request=None,
                    prefill_token_ids=list(tokens),
                )
            ]
            scheduled.num_scheduled_tokens = {request_id: end - start}
            scheduled.total_num_scheduled_tokens = end - start
            scheduled.num_common_prefix_blocks = [0] * len(cache.blocks)
            if self.config.metrics_enabled:
                torch.npu.synchronize()
            forward_start = perf_counter()
            runner.execute_model(scheduled)
            if self.config.metrics_enabled:
                torch.npu.synchronize()
                self.verifier_calls += 1
                self.verifier_tokens += end - start
                self.verifier_ms += (perf_counter() - forward_start) * 1000
            execution = runner.execute_model_state
            if execution is None:
                raise RuntimeError("Intermediate MRV2 forward returned no execution state")
            self._store_draft_context(execution, cache)
            # Hidden rows correspond exactly to [start:end], including chunked
            # initial prefill. Only materialize logits for the candidate span.
            lo = max(context_len - 1, start)
            hi = min(len(tokens) - 1, end)
            if lo < hi:
                logits_parts.append(runner.model.compute_logits(execution.hidden_states[lo - start : hi - start]))
            cache.tokens = tuple(tokens[:end])
            start = end
        assert execution is not None
        return execution, torch.cat(logits_parts), cache

    @torch.inference_mode()
    def verify(self, states):
        results = {}
        with set_current_vllm_config(self.vllm_config):
            for state in states:
                self.next_drafts.pop(state.request_id, None)
                context = state.context
                tokens = context + tuple(state.current_draft)
                execution, logits, cache = self._forward(state.request_id, tokens, len(context))
                drafts = torch.tensor(state.current_draft, dtype=torch.int64, device=self.device)
                accepted = accepted_prefix_length(self.policy.accept(logits, drafts))
                accepted_tokens = state.current_draft[:accepted]
                for i, token in enumerate(accepted_tokens):
                    if token in state.eos_token_ids:
                        accepted_tokens = accepted_tokens[: i + 1]
                        break
                accepted_end = len(context) + len(accepted_tokens)
                topk = None
                if self.config.debug_logging:
                    k = min(self.config.intermediate_verification.top_k, logits.shape[-1])
                    topk = logits.topk(k, dim=-1).indices.cpu().tolist()
                results[state.request_id] = VerificationResult(accepted_tokens, accepted == 0, topk)
                can_draft = (
                    accepted_tokens
                    and state.round_id + 1 < self.config.num_intermediate_rounds
                    and len(accepted_tokens) < state.remaining
                    and accepted_tokens[-1] not in state.eos_token_ids
                    and accepted_end + self.config.secondary_num_speculative_tokens <= self.runner.max_model_len
                )
                if can_draft:
                    # Draft while this request's input buffers are still bound.
                    # Pipeline.propose consumes the cached result after updating
                    # its logical accepted prefix. No extra target forward.
                    self._draft(state.request_id, execution, accepted_end, accepted_tokens[-1])
                cache.tokens = tuple(tokens[:accepted_end])
                self.runner.execute_model_state = None
        return results

    def _draft(self, request_id, execution, accepted_end, anchor):
        runner = self.runner
        batch = execution.input_batch
        # DFlash's query anchor lives at last_valid_position + 1. Exclude the
        # accepted tail token from its context even though verifier computed it.
        last_pos = int(batch.positions[batch.num_tokens - 1].cpu())
        rejected = last_pos + 2 - accepted_end
        if rejected >= batch.num_tokens:
            # Very large candidate spans can straddle prefill chunks. Context
            # was already stored by _store_draft_context; omit expansion rather
            # than let the kernel read before the current input batch.
            return
        idx = runner.req_states.req_id_to_index[request_id]
        runner.req_states.last_sampled_tokens[idx, 0] = anchor
        num_sampled = torch.ones(1, dtype=torch.int32, device=self.device)
        num_rejected = torch.tensor([rejected], dtype=torch.int32, device=self.device)
        if self.config.metrics_enabled:
            torch.npu.synchronize()
        start = perf_counter()
        proposed = runner.speculator.propose(
            batch,
            execution.attn_metadata,
            execution.slot_mappings_by_layer,
            execution.hidden_states,
            execution.aux_hidden_states,
            num_sampled,
            num_rejected,
            runner.req_states.last_sampled_tokens,
            runner.req_states.next_prefill_tokens,
            runner.sampler.sampling_states.temperature.gpu,
            runner.sampler.sampling_states.seeds.gpu,
        )
        self.next_drafts[request_id] = proposed[0].cpu().tolist()
        if self.config.metrics_enabled:
            self.secondary_calls += 1
            self.secondary_tokens += proposed.shape[1]
            self.secondary_ms += (perf_counter() - start) * 1000

    def propose(self, states):
        return {s.request_id: self.next_drafts.pop(s.request_id, []) for s in states}

    def rollback(self, request_id, committed_length):
        cache = self.caches.get(request_id)
        if cache is None:
            return
        cache.tokens = cache.tokens[:committed_length]
        # Rebinding NewRequestData at the next forward synchronizes both CPU
        # and GPU computed counters. Stale physical tails are unreachable.
        for size, blocks in zip(self.block_sizes, cache.blocks):
            keep = (len(cache.tokens) + size - 1) // size
            self.free_blocks.extend(blocks[keep:])
            del blocks[keep:]
        self.next_drafts.pop(request_id, None)

    def release(self, request_id):
        self.rollback(request_id, 0)
        self.caches.pop(request_id, None)
        with set_current_vllm_config(self.vllm_config):
            self.runner._remove_request(request_id)

    def shutdown(self):
        self.caches.clear()
        self.next_drafts.clear()
        with set_current_vllm_config(self.vllm_config):
            self.runner.shutdown()
