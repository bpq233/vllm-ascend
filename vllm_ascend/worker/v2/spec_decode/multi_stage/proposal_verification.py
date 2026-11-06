# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""MRV2 final rejection sampling against the recorded conditional mixture."""

from time import perf_counter

import torch

from vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance import AcceptancePolicy
from vllm_ascend.worker.v2.spec_decode.multi_stage.final_verification import FinalVerificationSampler
from vllm_ascend.worker.v2.spec_decode.multi_stage.proposal_math import verify_sequence


class ProposalVerificationSampler(FinalVerificationSampler):
    def __init__(self, sampler, spec_config, device, context):
        # Exact verification must retain the final pT bonus, including on the
        # independent three-model path. Approximate policies are never called.
        super().__init__(sampler, spec_config, device, AcceptancePolicy(top_k=1), emit_bonus_token=True)
        self.proposal_context = context

    def begin_forward(self, scheduler_output, dummy_run, proposal_timing=None, update_stream=None):
        super().begin_forward(scheduler_output, dummy_run, proposal_timing, update_stream)
        if not dummy_run:
            self.proposal_context.target_started_at = perf_counter()
            self.proposal_context.metrics["target_calls"] += 1

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
        seq_lens_upper_bound_np,
        verify_draft_sampled=None,
    ):
        context = self.proposal_context
        if getattr(self.sampler, "trace_replay_state", None) is not None:
            raise ValueError("Proposal sampling cannot force trace-replay output tokens.")
        # This is the same Target processing path as standard rejection,
        # including temperature/min_p/top_k/top_p. Do not apply temperature twice.
        processed = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        p_target = processed.float().softmax(-1)
        tokens = draft_sampled if verify_draft_sampled is None else verify_draft_sampled
        boundaries = cu_num_logits.cpu().tolist()
        metadata = torch.stack((pos.long(), tokens.long()), dim=1).cpu().tolist()
        sampled = torch.full(
            (len(idx_mapping_np), self.num_speculative_steps + 1), -1, dtype=torch.int64, device=logits.device
        )
        counts = torch.empty(len(idx_mapping_np), dtype=torch.int32, device=logits.device)
        for row, index in enumerate(idx_mapping_np):
            start, end = boundaries[row : row + 2]
            req_id = context.request_id(int(index))
            candidate_tokens = [item[1] for item in metadata[start + 1 : end]]
            absolute_start = metadata[start][0] + 1
            entries = (
                context.entries_for(req_id, absolute_start, candidate_tokens, metadata[start][1])
                if candidate_tokens
                else []
            )
            q_mix = (
                torch.stack([entry.mixture for entry in entries])
                if entries
                else p_target.new_empty((0, logits.shape[-1]))
            )
            candidate_ids = tokens[start + 1 : end].long()
            n = len(entries)
            uniforms = context.uniforms(req_id, "target", 2 * n + 1)
            result = verify_sequence(
                p_target[start:end],
                q_mix,
                candidate_ids,
                accept_uniforms=uniforms[:n],
                sample_uniforms=uniforms[n : 2 * n],
                bonus_uniform=uniforms[-1],
            )
            sampled[row, : n + 1] = result.sampled_tokens
            counts[row] = result.num_sampled
            accepted = int(result.accepted_length.cpu())
            context.target_feedback(
                req_id,
                entries,
                p_target[start : end - 1],
                accepted,
                context.target_started_at or perf_counter(),
                proposal_start=absolute_start,
                submitted_tokens=accepted + 1,
            )
        return processed, sampled, counts
