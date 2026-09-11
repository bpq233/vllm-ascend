# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small MRV2 lifecycle adapter; scheduler history stays owned by MRV2."""

import logging
from time import perf_counter

import torch
from vllm.v1.outputs import DraftTokenIds

from .backend import IntermediateBackend
from .config import make_policy
from .metrics import PipelineMetrics
from .pipeline import SpeculativePipeline
from .state import SpeculativeState

logger = logging.getLogger(__name__)


class RaggedDraftTokensHandler:
    def __init__(self, runtime):
        self.runtime = runtime
        self.output = None

    def set_draft_tokens(self, input_batch, draft_tokens):
        ids = list(input_batch.req_ids)
        self.output = DraftTokenIds(ids, [list(self.runtime.drafts.get(r, [])) for r in ids])

    def get_draft_tokens(self):
        return self.output


class MultiStageRuntime:
    def __init__(self, runner, config):
        self.runner = runner
        self.config = config
        self.backend = None
        self.pipeline = None
        self.drafts = {}
        self.stop_ids = {}
        self.context_lengths = {}
        self.max_lengths = {}
        self.metrics = PipelineMetrics(config.metrics_enabled or config.summary_logging)
        self.last_primary_ms = 0.0
        self.last_target_forward_ms = 0.0
        self.current_target_candidate_counts = {}

    def load_backend(self):
        self.backend = IntermediateBackend(
            self.runner.vllm_config,
            self.runner.device,
            self.config,
            make_policy(self.config.intermediate_verification),
        )
        self.pipeline = SpeculativePipeline(
            self.backend,
            self.backend,
            self.config.num_intermediate_rounds,
            self.config.debug_logging,
            self.config.metrics_enabled,
            self.config.summary_logging,
        )

    def observe(self, scheduled):
        target_drafts = getattr(scheduled, "scheduled_spec_decode_tokens", None) or {}
        self.current_target_candidate_counts = {request_id: len(tokens) for request_id, tokens in target_drafts.items()}
        for request_id in scheduled.finished_req_ids | (scheduled.preempted_req_ids or set()):
            if self.backend is not None:
                self.backend.release(request_id)
            self.drafts.pop(request_id, None)
            if request_id in scheduled.finished_req_ids:
                for mapping in (self.stop_ids, self.context_lengths, self.max_lengths):
                    mapping.pop(request_id, None)
        for req in scheduled.scheduled_new_reqs:
            params = req.sampling_params
            if req.mm_features or req.prompt_embeds is not None or req.lora_request is not None:
                raise ValueError("Multi-stage decoding currently supports text token-ID requests without LoRA")
            if params is None:
                raise ValueError("Multi-stage decoding requires a generation request")
            if getattr(params, "structured_outputs", None) is not None:
                raise ValueError("Multi-stage decoding does not yet support structured outputs")
            if getattr(params, "min_tokens", 0):
                raise ValueError("Multi-stage decoding currently requires min_tokens=0")
            if getattr(params, "logprob_token_ids", None):
                raise ValueError("Multi-stage decoding does not yet support logprob_token_ids")
            stops = set(params.stop_token_ids or [])
            if not params.ignore_eos:
                stops.update(getattr(params, "_all_stop_token_ids", set()))
                eos = self.runner.model_config.hf_config.eos_token_id
                if eos is not None:
                    stops.update(eos if isinstance(eos, list) else [eos])
            self.stop_ids[req.req_id] = frozenset(stops)
            self.context_lengths[req.req_id] = len(req.prefill_token_ids)
            self.max_lengths[req.req_id] = min(
                self.runner.max_model_len,
                req.prompt_len + params.max_tokens,
            )

    def expand(self, input_batch, primary, num_sampled):
        runner = self.runner
        indices = input_batch.idx_mapping.long()
        # Batch transfers: no per-token .item() or per-token target forward.
        lengths = runner.req_states.total_len.gpu[indices].cpu().tolist()
        sampled_counts = num_sampled.cpu().tolist()
        histories = runner.req_states.all_token_ids.gpu[indices, : max(lengths)].cpu().tolist()
        primary_ids = primary.cpu().tolist()
        states = []
        for req_id, length, history, count in zip(input_batch.req_ids, lengths, histories, sampled_counts):
            committed = tuple(history[:length])
            self.context_lengths[req_id] = length
            stops = self.stop_ids.get(req_id, frozenset())
            max_length = self.max_lengths.get(req_id, runner.max_model_len)
            states.append(
                SpeculativeState(
                    request_id=req_id,
                    committed_tokens=committed,
                    max_candidates=max(0, min(runner.num_speculative_steps, max_length - length - 1)),
                    eos_token_ids=stops,
                    finished=count == 0 or (bool(committed) and committed[-1] in stops),
                )
            )
        assert self.pipeline is not None
        self.drafts = self.pipeline.run(states, dict(zip(input_batch.req_ids, primary_ids)))
        output = torch.zeros(
            (input_batch.num_reqs, runner.num_speculative_steps), dtype=torch.int64, device=runner.device
        )
        for row, req_id in enumerate(input_batch.req_ids):
            tokens = self.drafts[req_id]
            if tokens:
                output[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.int64, device=runner.device)
        if self.config.summary_logging:
            candidate_counts = {request_id: len(tokens) for request_id, tokens in self.drafts.items()}
            run = self.pipeline.last_run
            logger.info(
                "multi_stage_candidates_ready candidates_by_request=%s total_candidates=%s "
                "primary_model_ms=%.3f intermediate_model_ms=%.3f "
                "secondary_model_ms=%.3f",
                candidate_counts,
                sum(candidate_counts.values()),
                self.last_primary_ms,
                run["intermediate_verifier_ms"],
                run["secondary_drafter_ms"],
            )
        return output

    def before_forward(self):
        if self.config.metrics_enabled or self.config.summary_logging:
            torch.npu.synchronize()
        return perf_counter()

    def after_forward(self, start, num_tokens):
        if self.config.metrics_enabled or self.config.summary_logging:
            torch.npu.synchronize()
            elapsed = (perf_counter() - start) * 1000
            self.last_target_forward_ms = elapsed
            self.metrics.record("target_forward", num_tokens, elapsed)
            if self.config.summary_logging:
                logger.info(
                    "multi_stage_target target_candidates_by_request=%s "
                    "total_target_candidates=%s scheduled_tokens=%s "
                    "target_model_forward_ms=%.3f",
                    self.current_target_candidate_counts,
                    sum(self.current_target_candidate_counts.values()),
                    num_tokens,
                    elapsed,
                )

    def shutdown(self):
        if self.backend is not None:
            self.backend.shutdown()
            self.backend = None
