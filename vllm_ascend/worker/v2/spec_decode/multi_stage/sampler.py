# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configurable acceptance at the native MRV2 target verification boundary."""

from time import perf_counter

import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from .config import make_policy


def pack_accepted_prefixes(policy, logits, draft_sampled, target_sampled, cu_num_logits, width):
    """Pack a policy-approved candidate prefix and one target-sampled token."""
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
    return output, (first_rejected + 1).to(torch.int32), first_rejected.to(torch.int32)


class PolicyRejectionSampler(RejectionSampler):
    """Keep native MRV2 orchestration while replacing its acceptance rule."""

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
        del draft_logits
        measure = self.config.summary_logging or self.config.metrics_enabled
        if measure:
            torch.npu.synchronize()
        start = perf_counter()
        # Reuse the target sampler so penalties, temperature, top-k/top-p,
        # seeds, replacement tokens, and all-accepted bonus tokens retain the
        # target model's normal behavior.
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
        output, counts, accepted_lengths = pack_accepted_prefixes(
            self.policy,
            processed,
            draft_sampled,
            target_sampled,
            cu_num_logits,
            self.num_speculative_steps + 1,
        )
        # Clamp an accepted EOS and per-request max_tokens before MRV2 writes
        # the sampled row into request history.
        output, counts = self.runtime.limit_final_output(output, counts, idx_mapping_np)
        elapsed = 0.0
        if measure:
            torch.npu.synchronize()
            elapsed = (perf_counter() - start) * 1000
        if self.config.debug_logging or self.config.summary_logging or self.config.metrics_enabled:
            self.runtime.record_final(
                processed,
                draft_sampled,
                cu_num_logits,
                output,
                counts,
                idx_mapping_np,
                accepted_lengths,
                elapsed,
            )
        return processed, output, counts
