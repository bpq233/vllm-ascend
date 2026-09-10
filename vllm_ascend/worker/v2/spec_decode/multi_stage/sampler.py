# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replace acceptance only; preserve MRV2 penalties, sampling and KV accounting."""

import logging

import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from .config import make_policy

logger = logging.getLogger(__name__)


def pack_accepted_prefixes(policy, logits, draft_sampled, target_sampled, cu_num_logits, width):
    """Packed inputs contain an anchor followed by drafts for each request.

    Unlike a rectangular verification, no padded vocabulary rows are created.
    Only the small output/acceptance matrices are padded to the output width.
    """
    positions = torch.arange(width, device=logits.device)
    lengths = cu_num_logits[1:] - cu_num_logits[:-1] - 1
    rows = cu_num_logits[:-1, None] + positions[None, :]
    rows = rows.clamp(max=logits.shape[0] - 1).long()
    next_tokens = torch.roll(draft_sampled, shifts=-1, dims=0)
    accepted = policy.accept(logits, next_tokens)
    valid = positions[None, :] < lengths[:, None]
    first_rejected = torch.where(valid & ~accepted[rows], positions[None, :], lengths[:, None]).min(dim=1).values
    output = torch.where(positions[None, :] < first_rejected[:, None], next_tokens[rows], target_sampled[rows])
    output = output.masked_fill(positions[None, :] > first_rejected[:, None], -1)
    return output, (first_rejected + 1).to(torch.int32)


class PolicyRejectionSampler(RejectionSampler):
    def __init__(self, sampler, speculative_config, device, config, runtime):
        super().__init__(sampler, speculative_config, device)
        self.policy = make_policy(config.final_verification)
        self.config = config
        self.runtime = runtime

    def _verify(
        self,
        logits,
        draft_logits,
        draft_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        idx_mapping_np,
        expanded_idx_mapping,
        expanded_local_pos,
    ):
        # Reuse target sampling (including seed/position and penalties) for the
        # replacement/bonus. There is deliberately no residual distribution.
        target_sampled, processed = self.sampler.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            return_logprobs=True,
        )
        output, counts = pack_accepted_prefixes(
            self.policy,
            processed,
            draft_sampled,
            target_sampled,
            cu_num_logits,
            self.num_speculative_steps + 1,
        )
        accepted_lengths = counts - 1
        # Stopping is applied before post_update, including an accepted EOS.
        # This prevents the bonus after EOS from entering worker-side history.
        output, counts = self.runtime.limit_final_output(output, counts, idx_mapping_np)
        if self.config.debug_logging or self.config.metrics_enabled:
            self.runtime.record_final(
                processed,
                draft_sampled,
                cu_num_logits,
                output,
                counts,
                idx_mapping_np,
                accepted_lengths,
            )
        return processed, output, counts
