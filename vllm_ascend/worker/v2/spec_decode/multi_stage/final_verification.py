# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import logging
from time import perf_counter

import numpy as np
import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance import AcceptancePolicy, assemble_verified_tokens

logger = logging.getLogger(__name__)


class FinalVerificationSampler(RejectionSampler):
    """Optional top-k/all verification; retain MRV2 output and logprob handling."""

    def __init__(self, sampler, spec_config, device: torch.device, policy: AcceptancePolicy):
        super().__init__(sampler, spec_config, device)
        self.policy = policy
        self._verification_steps = torch.arange(self.num_speculative_steps + 1, device=device)
        self.forward_events = None
        self.trace_forward = False
        self.verification_path = "decode"
        self.defer_trace = False
        self._pending_traces = []

    def begin_forward(self, scheduler_output, dummy_run, proposal_timing=None, update_stream=None):
        self.trace_forward = bool(
            not dummy_run and scheduler_output.scheduled_spec_decode_tokens and logger.isEnabledFor(logging.DEBUG)
        )
        if self.trace_forward:
            if proposal_timing is not None:
                refined_at, published_at, intermediate_stream = proposal_timing
                started_at = perf_counter()
                logger.debug(
                    "multi_stage_target_handoff intermediate_to_target_ms=%.3f candidate_publish_ms=%.3f "
                    "scheduler_handoff_ms=%.3f timing=host_wall intermediate_compute_stream=%s "
                    "target_compute_stream=%s attention_update_stream=%s",
                    (started_at - refined_at) * 1000,
                    (published_at - refined_at) * 1000,
                    (started_at - published_at) * 1000,
                    intermediate_stream,
                    torch.npu.current_stream(),
                    update_stream,
                )
            self.verification_path = "prefill" if max(scheduler_output.num_scheduled_tokens.values()) > 16 else "decode"
            if self.forward_events is None:
                self.forward_events = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            self.forward_events[0].record()

    def end_forward(self):
        if self.trace_forward:
            self.forward_events[1].record()

    def flush_trace(self):
        """Called after the adapter's existing D2H completes this stream."""
        for counts, path in self._pending_traces:
            lengths, accepted = counts.tolist()
            self._log_trace(lengths, accepted, path)
        self._pending_traces.clear()

    def _log_trace(self, lengths, accepted, path):
        logger.debug(
            "multi_stage_final candidate_lengths=%s accepted_lengths=%s verification_path=%s target_ms=%.3f "
            "timing=npu_forward",
            lengths,
            accepted,
            path,
            self.forward_events[0].elapsed_time(self.forward_events[1]),
        )

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample only the target distribution. Primary draft logits no longer
        # describe candidates replaced by the intermediate pipeline.
        target_sampled, processed_logits = self.sampler.sample(
            logits=logits,
            expanded_idx_mapping=expanded_idx_mapping,
            idx_mapping_np=idx_mapping_np,
            pos=pos,
            input_ids=draft_sampled,
            expanded_local_pos=expanded_local_pos,
            return_logprobs=True,
            **({} if vllm_version_is("0.27.1") else {"idx_mapping": idx_mapping}),
        )
        sampled, num_sampled = assemble_verified_tokens(
            processed_logits,
            draft_sampled,
            target_sampled,
            cu_num_logits,
            self.num_speculative_steps,
            self.policy,
            self._verification_steps,
        )
        if self.trace_forward:
            counts = torch.stack((cu_num_logits[1:] - cu_num_logits[:-1] - 1, num_sampled - 1))
            if self.defer_trace:
                # The multi-stage adapter already blocks for primary tokens and
                # committed progress. Queue this tiny copy on the SAME stream;
                # log after that existing boundary, with no new stream/event or
                # synchronization. Each chunk owns its pinned destination.
                host_counts = torch.empty_like(counts, device="cpu", pin_memory=True)
                host_counts.copy_(counts, non_blocking=True)
                self._pending_traces.append((host_counts, self.verification_path))
            else:
                # Final-policy-only runs have no synchronous multi-stage adapter.
                lengths, accepted = counts.cpu().tolist()
                self._log_trace(lengths, accepted, self.verification_path)
        return processed_logits, sampled, num_sampled
