# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Correctness-first sequential conditional mixture proposal generation.

The token budget is chosen before drawing any token. A correction invalidates
the old parallel block, but does not shorten the preselected budget. This avoids
conditioning the final proposal on acceptance-dependent length or filtering.
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
        self.device_output_ready = False
        if output is not None:
            output.zero_()
        snapshot = self.context.feedback.snapshot()
        candidates = []
        for row, (req_id, prefix, limit) in enumerate(zip(req_ids, prefixes, limits)):
            if req_id not in self.context.requests:
                raise ValueError(f"Missing sampling parameters for proposal request {req_id}.")
            params = self.context.requests[req_id]
            # Each position is reached using only prior token/acceptance draws.
            # No previously generated token is filtered, reranked or reverified
            # into another mixture distribution during this generation cycle.
            budget = min(self.capacity, max(0, limit))
            block = []
            source = "intermediate_dflash"
            if primary is not None:
                tokens = primary[row]
                block = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
                source = "primary_dflash"
            entries, emitted = [], []
            intermediate_accepted = 0
            spec_round = 0 if block else -1
            for position in range(budget):
                started = perf_counter()
                current = list(prefix) + emitted
                draft_started = perf_counter()
                if not block:
                    spec_round += 1
                    intermediate_accepted = 0
                    block = list(self.backend.propose([current], req_ids=[req_id])[0])
                    source = "intermediate_dflash"
                draft_ms = (perf_counter() - draft_started) * 1000
                # This decision is made before the next token is drawn. If the
                # actual drafter cannot propose at this context boundary, end
                # the proposal and leave the next token to the Target sampler.
                if not block or block[0] < 0:
                    break
                draft_token = int(block[0])
                verify_started = perf_counter()
                logits = self.backend.proposal_logits([current], [req_id])[0]
                p_m = intermediate_probabilities(logits, params)
                verify_ms = (perf_counter() - verify_started) * 1000
                sampling_started = perf_counter()
                state, weight = self.context.state_and_weight(p_m, draft_token, spec_round, position, source, snapshot)
                q_d = delta_probabilities(torch.tensor(draft_token, device=p_m.device), p_m.numel())
                q_mix = mix_probabilities(p_m, q_d, weight)
                uniforms = self.context.uniforms(req_id, "intermediate", 2)
                result = rejection_step(
                    q_mix,
                    q_d,
                    torch.tensor(draft_token, device=p_m.device),
                    accept_uniform=uniforms[0],
                    sample_uniform=uniforms[1],
                )
                # One token/decision packet crosses this CPU-controlled boundary.
                token, accepted = torch.stack((result.tokens, result.accepted.long())).cpu().tolist()
                token, accepted = int(token), bool(accepted)
                intermediate_accepted += int(accepted)
                sampling_ms = (perf_counter() - sampling_started) * 1000
                entries.append(
                    ProposalToken(
                        token,
                        draft_token,
                        p_m.detach().clone(),
                        q_mix.detach().clone(),
                        weight,
                        state,
                        source,
                        spec_round,
                        accepted,
                        intermediate_accepted,
                        draft_ms,
                        verify_ms,
                        sampling_ms,
                        (perf_counter() - started) * 1000,
                    )
                )
                emitted.append(token)
                # A residual correction changes the prefix seen by all later
                # positions of this parallel DFlash block. None of its old
                # suffix probabilities/tokens can be reused after that point.
                block = block[1:] if accepted else []
                if output is not None:
                    output[row, position] = result.tokens
                if token in self.eos and not params.ignore_eos:
                    break
            self.context.record(req_id, prefix, entries)
            candidates.append(emitted)
        self.device_output_ready = output is not None
        self.last_refine_finished_at = perf_counter()
        self.last_compute_stream = (
            torch.npu.current_stream() if output is not None and output.device.type == "npu" else None
        )
        return candidates
