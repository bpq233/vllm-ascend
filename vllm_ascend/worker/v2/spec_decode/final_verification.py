# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np
import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.acceptance import AcceptancePolicy, assemble_verified_tokens


class FinalVerificationSampler(RejectionSampler):
    """Optional top-k/all verification; retain MRV2 output and logprob handling."""

    def __init__(self, sampler, spec_config, device: torch.device, policy: AcceptancePolicy):
        super().__init__(sampler, spec_config, device)
        self.policy = policy

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
        return processed_logits, sampled, num_sampled
