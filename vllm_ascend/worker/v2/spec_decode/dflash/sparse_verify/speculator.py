# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Opt-in multi-round DFlash; ordinary DFlash stays on its original class."""

from dataclasses import asdict

import torch
from vllm.logger import logger

from vllm_ascend.utils import lmhead_tp_enable
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator

from .config import SparseVerifyConfig
from .controller import FeatureBundle, MultiRoundDFlashController, SparseForwardOutput
from .integration import draft_block_config
from .metrics import RoundMetrics, SparseVerificationMetrics
from .model import SparseTargetVerifier
from .runtime import SparseForwardRunner, private_block_tables, single_request_batch
from .selection import required_layers_from_aux_capture, select_sparse_layers


class SparseDFlashSpeculator(AscendDFlashSpeculator):
    def __init__(self, vllm_config, device, req_states):
        if lmhead_tp_enable():
            raise ValueError("dflash_sparse_verify does not support the specialized LM-head TP path")
        self.target_vllm_config = vllm_config
        self.sparse_config = SparseVerifyConfig.from_additional_config(vllm_config.additional_config)
        self.req_states = req_states
        self.total_candidates = vllm_config.speculative_config.num_speculative_tokens
        super().__init__(draft_block_config(vllm_config), device)
        self.sparse_metrics = SparseVerificationMetrics(self.sparse_config.enable_metrics)
        self.controller = MultiRoundDFlashController(
            self.sparse_config, self.num_speculative_steps, self.total_candidates
        )
        self.sparse_cudagraph_manager = None
        self.request_sampling = {}

    def load_model(self, target_model):
        target_inner = target_model.model
        # Upstream has already configured the target's actual capture boundaries.
        # Preserve its output order (Qwen's forward walks original layer order).
        aux_ids = tuple(sorted(target_inner.aux_hidden_state_layers))
        if not aux_ids:
            raise ValueError("DFlash checkpoint has no configured Target auxiliary hidden-state layers")
        total_layers = len(target_inner.layers)
        selected = select_sparse_layers(
            total_layers,
            self.sparse_config.num_intermediate_layers,
            required_layers_from_aux_capture(total_layers, aux_ids),
        )
        self.sparse_model = SparseTargetVerifier(
            target_model, selected, aux_ids, self.vllm_config.compilation_config.static_forward_context
        )
        # Register sparse cache owners BEFORE the superclass snapshots target
        # names, so they cannot be classified as DFlash's non-causal layers.
        super().load_model(target_model)
        logger.info(
            "DFlash sparse verification: layers=%s aux_capture_ids=%s K=%s capacity=%s mode=%s final=%s",
            selected,
            aux_ids,
            self.num_speculative_steps,
            self.total_candidates,
            self.sparse_config.mode,
            self.sparse_config.final_verification.method,
        )
        if self.sparse_config.final_verification.method != "strict":
            logger.warning(
                "DFlash final policy %s is approximate and does not preserve Target greedy output",
                self.sparse_config.final_verification.method,
            )

    def set_attn(self, model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups):
        self.sparse_runner = SparseForwardRunner(self, model_state, kv_cache_config, block_tables)
        # DFlash input preparation and replay see their own staging buffers.
        # Neither an inner round nor its query slot mapping can mutate Target inputs.
        self.draft_context_buffers = AscendInputBuffers(self.max_num_reqs, self.max_num_tokens, self.device)
        super().set_attn(
            model_state,
            kv_cache_config,
            private_block_tables(block_tables),
            self.draft_context_buffers,
            target_attn_groups,
        )

    def init_cudagraph_manager(self, cudagraph_mode):
        super().init_cudagraph_manager(cudagraph_mode)
        self.sparse_runner.init_graphs(cudagraph_mode)

    def capture(self):
        super().capture()
        if self.query_cudagraph_manager.graphs:
            self.sparse_metrics.record_graph("dflash", "capture")
            logger.info("DFlash ACL capture completed: %s shapes", len(self.query_cudagraph_manager.graphs))
        self.sparse_runner.capture()

    def _generate_draft(self, *args, **kwargs):
        # This runs during eager/capture, never during replay. Replay evidence
        # is recorded at run_fullgraph itself by the existing graph manager.
        return super()._generate_draft(*args, **kwargs)

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync_state=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        num_speculative_tokens=None,
    ):
        if dummy_run:
            result = super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=min(
                    num_speculative_tokens or self.num_speculative_steps, self.num_speculative_steps
                ),
            )
            padded = result.new_full((input_batch.num_reqs, self.total_candidates), -1)
            padded[:, : result.shape[1]] = result
            return padded
        if aux_hidden_states is None or len(aux_hidden_states) != len(self.sparse_model.aux_layer_ids):
            raise ValueError("Target auxiliary features do not match the loaded DFlash checkpoint")
        count = input_batch.num_reqs
        # Explicit control dependencies outside all captured forwards. The first
        # version prioritizes correct prefix/cache state over host submission rate.
        rejected = num_rejected[:count].cpu().tolist()
        sampled = num_sampled[:count].cpu().tolist()
        anchors = last_sampled[input_batch.idx_mapping].cpu().tolist()
        result = torch.full((count, self.total_candidates), -1, dtype=torch.int64, device=self.device)
        requested_budget = self.total_candidates if num_speculative_tokens is None else num_speculative_tokens
        # Gather all private tables before the first inner draft overwrites its
        # own gathered rows; sparse and Target mappings have distinct addresses.
        for row, req_id in enumerate(input_batch.req_ids):
            state_index = int(input_batch.idx_mapping_np[row])
            begin = int(input_batch.query_start_loc_np[row])
            end = int(input_batch.query_start_loc_np[row + 1]) - int(rejected[row])
            if end <= begin:
                continue
            ids = input_batch.input_ids[begin:end]
            positions = input_batch.positions[begin:end]
            first_position = int(positions[0].item())
            anchor_position = int(positions[-1].item()) + 1
            # Prime sparse history for every target prefill chunk, even when
            # that request is not yet allowed to emit a draft. This is what makes
            # its independent cache valid over the entire prompt.
            prime_chunk = (
                self.target_vllm_config.compilation_config.max_cudagraph_capture_size or self.num_query_per_req
            )
            for offset in range(0, ids.numel(), prime_chunk):
                self.sparse_runner.forward(
                    input_batch,
                    ids[offset : offset + prime_chunk],
                    positions[offset : offset + prime_chunk],
                    state_index,
                    req_id,
                    prefill=bool(input_batch.is_prefilling_np[row]),
                )
            if sampled[row] == 0 or requested_budget == 0:
                continue
            features = FeatureBundle(
                tuple(h[begin:end].clone() for h in aux_hidden_states),
                self.sparse_model.aux_layer_ids,
                first_position,
                "target",
                ids.clone(),
            )
            # Reserve room for the anchor; the pending final correction/bonus
            # follows the normal vLLM delayed-KV lifecycle.
            budget = min(requested_budget, self.max_model_len - anchor_position - 1)
            params = self.request_sampling.get(req_id)
            if params is not None:
                prompt_len = int(self.req_states.prefill_len.np[state_index])
                generated = anchor_position + 1 - prompt_len
                budget = min(budget, max(0, params.max_tokens - generated))
            if budget <= 0:
                continue
            local_last = last_sampled.clone()
            local_rejected = torch.zeros(1, dtype=num_rejected.dtype, device=self.device)
            local_sampled = torch.ones(1, dtype=num_sampled.dtype, device=self.device)
            round_events = {}

            def draft(
                request,
                state_index=state_index,
                req_id=req_id,
                local_last=local_last,
                local_sampled=local_sampled,
                local_rejected=local_rejected,
                round_events=round_events,
            ):
                if request.anchor_position + self.num_speculative_steps >= self.max_model_len:
                    # Native DFlash always forwards the checkpoint's full mask
                    # block, even if its returned candidates are sliced. Stop
                    # before launching an out-of-range query at context end.
                    return torch.empty(0, dtype=torch.long, device=self.device)
                context_ids = request.features.token_ids
                if context_ids is None:
                    raise ValueError("DFlash context features must carry their exact token positions and identities")
                batch = single_request_batch(
                    input_batch,
                    self.draft_context_buffers,
                    context_ids,
                    request.features.positions,
                    state_index,
                    req_id,
                )
                self.block_tables.gather_block_tables(batch.idx_mapping, num_reqs_padded=1)
                local_last[state_index] = request.anchor_token
                if self.sparse_metrics.enabled:
                    events = tuple(torch.npu.Event(enable_timing=True) for _ in range(4))
                    round_events[request.round_index] = events
                    events[0].record()
                draft_tokens = super(SparseDFlashSpeculator, self).propose(
                    batch,
                    {},
                    {},
                    request.features.values[-1],
                    list(request.features.values),
                    local_sampled,
                    local_rejected,
                    local_last,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                    num_speculative_tokens=min(request.max_tokens, self.num_speculative_steps),
                )
                if self.sparse_metrics.enabled:
                    events[1].record()
                return draft_tokens[0, : request.max_tokens].clone()

            def verify(request, state_index=state_index, req_id=req_id, round_events=round_events):
                if self.sparse_metrics.enabled:
                    round_events[request.round_index][2].record()
                logits, hidden = self.sparse_runner.forward(
                    input_batch, request.input_ids, request.positions, state_index, req_id
                )
                if self.sparse_metrics.enabled:
                    round_events[request.round_index][3].record()
                bundle = FeatureBundle(
                    tuple(hidden),
                    self.sparse_model.aux_layer_ids,
                    int(request.positions[0].item()),
                    "sparse",
                    request.input_ids.clone(),
                )
                # Cache suffix invalidation is logical: every subsequent call
                # rebuilds seq_lens from its actual prefix endpoint, and writes
                # from its anchor. Old suffix pages are never visible as history.
                return SparseForwardOutput(logits[:-1], bundle, lambda endpoint: None)

            eos_token_id = self.target_vllm_config.model_config.hf_text_config.eos_token_id
            eos_ids = (
                ()
                if eos_token_id is None
                else (tuple(eos_token_id) if isinstance(eos_token_id, list) else (eos_token_id,))
            )
            if params is not None:
                eos_ids = () if params.ignore_eos else eos_ids
                eos_ids = (*eos_ids, *params.stop_token_ids)
            proposal = self.controller.run_request(
                req_id,
                int(anchors[row]),
                anchor_position,
                features,
                draft,
                verify,
                token_budget=budget,
                eos_token_ids=eos_ids,
            )
            result[row, : proposal.tokens.numel()] = proposal.tokens
            if self.sparse_config.enable_metrics:
                for record in proposal.rounds:
                    events = round_events[record.round_index]
                    events[3].synchronize()
                    dflash_seconds = events[0].elapsed_time(events[1]) / 1000
                    sparse_seconds = events[2].elapsed_time(events[3]) / 1000
                    self.sparse_metrics.record_round(
                        RoundMetrics(
                            req_id,
                            record.round_index,
                            record.proposed_length,
                            record.temporarily_accepted_length,
                            record.first_rejection,
                            record.confidence,
                            "low_confidence" if record.low_confidence_stop else "retry" if record.retry else "accept",
                            record.feature_source,
                            record.feature_shape,
                            *record.feature_positions,
                            dflash_seconds,
                            sparse_seconds,
                            events[0].elapsed_time(events[3]) / 1000,
                        )
                    )
                # These are host submission durations, labelled explicitly; use
                # NPU events or the synchronized benchmark for device latencies.
                logger.info(
                    "DFlash sparse proposal req=%s stop=%s temporary=%s proposed=%s rounds=%s host_records=%s",
                    req_id,
                    proposal.stop_reason,
                    proposal.temporary_accepted_length,
                    proposal.tokens.numel(),
                    len(proposal.rounds),
                    [asdict(r) for r in proposal.rounds],
                )
        return result
