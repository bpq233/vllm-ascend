# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import logging

import numpy as np
import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.acceptance import AcceptancePolicy, assemble_verified_tokens

logger = logging.getLogger(__name__)


class FinalVerificationSampler(RejectionSampler):
    """Optional top-k/all verification; retain MRV2 output and logprob handling."""

    def __init__(self, sampler, spec_config, device: torch.device, policy: AcceptancePolicy):
        super().__init__(sampler, spec_config, device)
        self.policy = policy
        self.forward_events = None
        self.trace_forward = False
        self.verification_path = "decode"

    def begin_forward(self, scheduler_output, dummy_run):
        self.trace_forward = bool(
            not dummy_run and scheduler_output.scheduled_spec_decode_tokens and logger.isEnabledFor(logging.DEBUG)
        )
        if self.trace_forward:
            self.verification_path = "prefill" if max(scheduler_output.num_scheduled_tokens.values()) > 16 else "decode"
            if self.forward_events is None:
                self.forward_events = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            self.forward_events[0].record()

    def end_forward(self):
        if self.trace_forward:
            self.forward_events[1].record()

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
        )
        if self.trace_forward:
            # Debug-only synchronization; normal execution stays on device.
            self.forward_events[1].synchronize()
            lengths = (cu_num_logits[1:] - cu_num_logits[:-1] - 1).cpu().tolist()
            logger.debug(
                "multi_stage_final candidate_lengths=%s accepted_lengths=%s verification_path=%s target_ms=%.3f "
                "timing=npu_forward",
                lengths,
                (num_sampled - 1).cpu().tolist(),
                self.verification_path,
                self.forward_events[0].elapsed_time(self.forward_events[1]),
            )
        return processed_logits, sampled, num_sampled
