# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Strict-round mixture proposals with one teacher-forced verifier pass per block.

Every reached position samples its recorded mixture exactly once. A correction
is retained and ends its round; subsequent positions use a fresh draft block.
Stopping affects only future positions, never filters an already sampled token.
"""

from time import perf_counter

import torch

from vllm_ascend.worker.v2.spec_decode.multi_stage.proposal_math import (
    delta_probabilities,
    mix_probabilities,
    rejection_step,
)
from vllm_ascend.worker.v2.spec_decode.multi_stage.proposal_sampling import ProposalToken, intermediate_probabilities


class ProposalPipeline:
    def __init__(self, backend, config, capacity, eos_token_id, context):
        self.backend = backend
        self.config = config
        self.capacity = capacity
        self.context = context
        self.eos = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        self.device_output_ready = False
        self.last_refine_finished_at = None
        self.last_compute_stream = None

    def refine(self, prefixes, primary, limits, req_ids, output=None):
        if len(req_ids) != len(set(req_ids)) or not len(req_ids) == len(prefixes) == len(limits):
            raise ValueError("Proposal batches require distinct request IDs and aligned inputs.")
        if primary is not None and len(primary) != len(prefixes):
            raise ValueError("Primary proposal rows must match the request count.")
        self.device_output_ready = False
        if output is not None:
            output.zero_()
        snapshot = self.context.feedback.snapshot()
        candidates = []
        for row, (req_id, prefix, limit) in enumerate(zip(req_ids, prefixes, limits)):
            if req_id not in self.context.requests:
                raise ValueError(f"Missing sampling parameters for proposal request {req_id}.")
            params = self.context.requests[req_id]
            budget = min(self.capacity, max(0, limit))
            candidate_cap = getattr(self.config, "max_generated_tokens", None)
            if candidate_cap is not None:
                budget = min(budget, candidate_cap)
            entries, emitted = [], []
            for spec_round in range(self.config.num_rounds):
                if len(entries) >= budget:
                    break
                current = list(prefix) + emitted
                draft_started = perf_counter()
                if spec_round == 0 and primary is not None and len(primary[row]):
                    tokens = primary[row]
                    block = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
                    source = "primary_dflash"
                else:
                    block = list(self.backend.propose([current], req_ids=[req_id])[0])
                    source = "intermediate_dflash"
                draft_ms = (perf_counter() - draft_started) * 1000
                block = block[: budget - len(entries)]
                if any(token < 0 for token in block):
                    block = block[: next(i for i, token in enumerate(block) if token < 0)]
                if not block:
                    break
                round_entries = self._sample_round(
                    req_id,
                    current,
                    block,
                    params,
                    spec_round,
                    source,
                    len(entries),
                    snapshot,
                    draft_ms,
                    output[row, len(entries) :] if output is not None else None,
                )
                entries.extend(round_entries)
                emitted.extend(entry.token for entry in round_entries)
                if emitted[-1] in self.eos and not params.ignore_eos:
                    break
            self.context.record(req_id, prefix, entries)
            candidates.append(emitted)
        self.device_output_ready = output is not None
        self.last_refine_finished_at = perf_counter()
        self.last_compute_stream = (
            torch.npu.current_stream() if output is not None and output.device.type == "npu" else None
        )
        return candidates

    def _sample_round(self, req_id, prefix, block, params, spec_round, source, offset, snapshot, draft_ms, output):
        verify_started = perf_counter()
        logits = self.backend.proposal_block_logits([prefix], [block], [req_id])[0]
        if logits.ndim != 2 or logits.shape[0] != len(block):
            raise ValueError("Intermediate block verification must return one probability row per draft token.")
        probabilities = intermediate_probabilities(logits, params)
        verify_ms = (perf_counter() - verify_started) * 1000
        entries = []
        for local_position, (draft_token, p_m) in enumerate(zip(block, probabilities)):
            sampling_started = perf_counter()
            state, weight = self.context.state_and_weight(
                p_m, draft_token, spec_round, offset + local_position, source, snapshot
            )
            draft_id = torch.tensor(draft_token, device=p_m.device)
            q_d = delta_probabilities(draft_id, p_m.numel())
            q_mix = mix_probabilities(p_m, q_d, weight)
            uniforms = self.context.uniforms(req_id, "intermediate", 2)
            result = rejection_step(q_mix, q_d, draft_id, accept_uniform=uniforms[0], sample_uniform=uniforms[1])
            token, accepted = torch.stack((result.tokens, result.accepted.long())).cpu().tolist()
            token, accepted = int(token), bool(accepted)
            sampling_ms = (perf_counter() - sampling_started) * 1000
            # Charge the shared round forward/draft exactly once in scalar logs.
            round_verify_ms = verify_ms if local_position == 0 else 0.0
            round_draft_ms = draft_ms if local_position == 0 else 0.0
            entries.append(
                ProposalToken(
                    token,
                    int(draft_token),
                    p_m.detach().clone(),
                    q_mix.detach().clone(),
                    weight,
                    state,
                    source,
                    spec_round,
                    accepted,
                    local_position + int(accepted),
                    round_draft_ms,
                    round_verify_ms,
                    sampling_ms,
                    round_draft_ms + round_verify_ms + sampling_ms,
                )
            )
            if output is not None:
                output[local_position] = result.tokens
            # Teacher-forced rows remain valid only while every earlier draft
            # matched the actual output. Keep the correction, never its tail.
            if not accepted or (token in self.eos and not params.ignore_eos):
                break
        return entries
