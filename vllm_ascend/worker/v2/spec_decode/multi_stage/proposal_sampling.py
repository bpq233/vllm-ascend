# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Request-local RNGs and device-resident conditional proposal provenance."""

import hashlib
import math
from dataclasses import dataclass
from time import perf_counter

import torch

from vllm_ascend.worker.v2.spec_decode.multi_stage.proposal_config import ProposalSamplingConfig
from vllm_ascend.worker.v2.spec_decode.multi_stage.proposal_feedback import (
    ProposalFeedback,
    device_features,
    make_state,
    overlap_scores,
)
from vllm_ascend.worker.v2.spec_decode.multi_stage.proposal_math import normalize_probabilities


@dataclass(frozen=True)
class RequestSampling:
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    ignore_eos: bool
    seed: int | None = None


@dataclass
class ProposalToken:
    token: int
    draft_token: int
    intermediate: torch.Tensor
    mixture: torch.Tensor
    weight: float
    state: tuple
    source: str
    spec_round: int
    intermediate_accept: bool
    intermediate_accepted_length: int
    draft_time_ms: float
    intermediate_verify_time_ms: float
    sampling_overhead_ms: float
    total_time_ms: float


@dataclass
class ProposalLedger:
    start: int
    context: tuple[int, ...]
    entries: list[ProposalToken]


def intermediate_probabilities(logits, params: RequestSampling):
    """Apply the declared context-independent sampling transforms in FP32."""
    scores = logits.float() / params.temperature
    if not bool((~torch.isnan(scores) & (scores != torch.inf)).all()):
        raise ValueError("Intermediate logits contain NaN or positive infinity.")
    if params.min_p > 0:
        scores = scores.masked_fill(scores < scores.amax(-1, keepdim=True) + math.log(params.min_p), -torch.inf)
    if params.top_k > 0 and params.top_k < scores.shape[-1]:
        threshold = scores.topk(params.top_k, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    probabilities = scores.softmax(-1)
    if params.top_p < 1:
        values, order = probabilities.sort(dim=-1, descending=True)
        # Keep the first crossing token as well as the maximum-probability one.
        remove = values.cumsum(-1) - values >= params.top_p
        values = values.masked_fill(remove, 0)
        probabilities = torch.zeros_like(probabilities).scatter(-1, order, values)
    return normalize_probabilities(probabilities)


class ProposalSamplingContext:
    """Own provenance until final verification; never reconstruct it from logits.

    Uniforms use independent CPU generators for each request and stage. Only
    these tiny random packets move H2D; all categorical/residual arithmetic stays
    on the probability device. This avoids relying on unverified NPU Generator
    behavior and makes TP ranks and request reorderings use the same draws.
    """

    def __init__(self, options, sampler, req_states, device, default_seed=0):
        self.config = ProposalSamplingConfig.from_dict(options)
        self.sampler = sampler
        self.req_states = req_states
        self.device = device
        self.default_seed = default_seed
        self.feedback = ProposalFeedback(self.config)
        self.requests = {}
        self.generators = {}
        self.ledgers = {}
        self._pending_requests = []
        self._preempted_requests = set()
        self.target_started_at = None
        self.metrics = {
            "target_calls": 0,
            "request_verifications": 0,
            "target_accepted": 0,
            "target_proposed": 0,
            "ledger_bytes": 0,
        }

    def register(self, req_id, params):
        for name, default in (
            ("presence_penalty", 0.0),
            ("frequency_penalty", 0.0),
            ("repetition_penalty", 1.0),
            ("min_tokens", 0),
        ):
            if getattr(params, name, default) != default:
                raise ValueError(f"Proposal sampling does not support request parameter {name}.")
        for name in (
            "bad_words",
            "bad_words_token_ids",
            "logit_bias",
            "allowed_token_ids",
            "structured_outputs",
            "guided_decoding",
            "logits_processors",
            "extra_args",
        ):
            if getattr(params, name, None):
                raise ValueError(f"Proposal sampling does not support request parameter {name}.")
        temperature = getattr(params, "temperature", 1.0)
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "Proposal sampling requires temperature > 0; use the existing Top-1 path for strict greedy."
            )
        top_k = getattr(params, "top_k", -1)
        top_p, min_p = getattr(params, "top_p", 1.0), getattr(params, "min_p", 0.0)
        seed = getattr(params, "seed", None)
        if type(top_k) is not int or any(
            type(value) not in (float, int) or not math.isfinite(value) for value in (top_p, min_p)
        ):
            raise ValueError("Proposal sampling requires integer top_k and finite numeric top_p/min_p.")
        if seed is not None and type(seed) is not int:
            raise ValueError("Proposal sampling requires an integer seed or None.")
        request = RequestSampling(
            float(temperature),
            top_k,
            float(top_p),
            float(min_p),
            bool(getattr(params, "ignore_eos", False)),
            seed,
        )
        if not 0 < request.top_p <= 1 or not 0 <= request.min_p <= 1 or request.top_k == 0 or request.top_k < -1:
            raise ValueError("Unsupported proposal sampling top_k/top_p/min_p.")
        if req_id in self.requests:
            if self.requests[req_id] != request:
                raise ValueError("Cannot change sampling parameters while a proposal request is live.")
            return
        identity = req_id if seed is None else "seeded"
        seed = self.default_seed if seed is None else seed
        generators = {}
        for stage in ("intermediate", "target"):
            digest = hashlib.blake2b(f"{seed}:{identity}:{stage}".encode(), digest_size=8).digest()
            generators[stage] = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest, "little") % (2**63))
        self.requests[req_id] = request
        self.generators[req_id] = generators

    def discard(self, req_id):
        self.ledgers.pop(req_id, None)
        self._update_ledger_bytes()
        if req_id in self._preempted_requests:
            return
        self.requests.pop(req_id, None)
        self.generators.pop(req_id, None)

    def begin_step(self, scheduler_output):
        # Slot eviction must not rewind a live request's random streams. The
        # scheduler re-adds preempted requests as new slots after recomputation.
        self._preempted_requests.update(getattr(scheduler_output, "preempted_req_ids", ()))
        self._preempted_requests.difference_update(getattr(scheduler_output, "finished_req_ids", ()))
        for req_id in getattr(scheduler_output, "preempted_req_ids", ()):
            self.ledgers.pop(req_id, None)
        self._update_ledger_bytes()
        # MRV2 add_requests first removes each new request's old slot, invoking
        # our discard callback even when no previous request occupied that ID.
        # Register only after the parent has completed those slot updates.
        self._pending_requests = list(getattr(scheduler_output, "scheduled_new_reqs", ()))

    def finish_step_registration(self):
        for request in self._pending_requests:
            self.register(request.req_id, request.sampling_params)
            self._preempted_requests.discard(request.req_id)
        self._pending_requests.clear()

    def uniforms(self, req_id, stage, count):
        if req_id not in self.generators:
            raise ValueError(f"Missing proposal sampling state for request {req_id}.")
        return torch.rand(count, generator=self.generators[req_id][stage], dtype=torch.float32).to(self.device)

    def record(self, req_id, prefix, entries):
        self.ledgers[req_id] = ProposalLedger(len(prefix), tuple(prefix), entries)
        self._update_ledger_bytes()
        if self.config.debug:
            self.feedback.emit_record(
                {
                    "event": "proposal_generated",
                    "request_id": req_id,
                    "target_cycle_id": f"{req_id}:{len(prefix)}",
                    "proposal_start": len(prefix),
                    "candidate_length": len(entries),
                    "intermediate_accepted_tokens": sum(entry.intermediate_accept for entry in entries),
                    "intermediate_blocks": len({entry.spec_round for entry in entries}),
                    "sampling_overhead_ms": sum(entry.sampling_overhead_ms for entry in entries),
                    "intermediate_verify_time_ms": sum(entry.intermediate_verify_time_ms for entry in entries),
                    "draft_time_ms": sum(entry.draft_time_ms for entry in entries),
                    "total_time_ms": sum(entry.total_time_ms for entry in entries),
                    "retained_probability_bytes": self.metrics["ledger_bytes"],
                    "timing": "host_wall",
                }
            )

    def _update_ledger_bytes(self):
        self.metrics["ledger_bytes"] = sum(
            (entry.intermediate.numel() + entry.mixture.numel()) * 4
            for ledger in self.ledgers.values()
            for entry in ledger.entries
        )

    def entries_for(self, req_id, start, tokens, anchor=None):
        ledger = self.ledgers.get(req_id)
        if ledger is None:
            raise ValueError(f"Missing conditional proposal distributions for request {req_id}.")
        offset = start - ledger.start
        entries = ledger.entries[offset : offset + len(tokens)] if offset >= 0 else []
        if len(entries) != len(tokens) or [entry.token for entry in entries] != tokens:
            raise ValueError(f"Candidate/context provenance mismatch for request {req_id} at position {start}.")
        expected_anchor = ledger.context[-1] if offset == 0 else ledger.entries[offset - 1].token
        if anchor is not None and anchor != expected_anchor:
            raise ValueError(f"Candidate anchor/context mismatch for request {req_id}.")
        return entries

    def request_id(self, index):
        reverse = {index: req_id for req_id, index in self.req_states.req_id_to_index.items()}
        if index not in reverse:
            raise ValueError("Final verification referred to a recycled or unknown request slot.")
        return reverse[index]

    def target_feedback(
        self, req_id, entries, probabilities, accepted_length, started_at, *, proposal_start, submitted_tokens
    ):
        if not entries:
            return
        self.metrics["request_verifications"] += 1
        self.metrics["target_accepted"] += accepted_length
        self.metrics["target_proposed"] += len(entries)
        if not self.config.feedback_enabled and not self.config.debug:
            return
        candidate_length = len(entries)
        count = min(candidate_length, accepted_length + 1)
        entries, probabilities = entries[:count], probabilities[:count]
        p_m = torch.stack([entry.intermediate for entry in entries])
        q_d_ids = torch.tensor([entry.draft_token for entry in entries], device=self.device, dtype=torch.int64)
        candidates = torch.tensor([entry.token for entry in entries], device=self.device, dtype=torch.int64)
        q_mix = torch.stack([entry.mixture for entry in entries])
        scores = overlap_scores(probabilities, p_m, q_d_ids, self.config.lambda_candidates)
        chosen_overlap = 1 - (probabilities - q_mix).abs().sum(-1) * 0.5
        scalar_probs = torch.stack(
            (
                p_m.gather(1, candidates[:, None]).squeeze(1),
                (candidates == q_d_ids).float(),
                q_mix.gather(1, candidates[:, None]).squeeze(1),
                probabilities.gather(1, candidates[:, None]).squeeze(1),
                chosen_overlap,
            ),
            dim=1,
        )
        # This is one scalar packet per verified request, never [tokens,vocab].
        packet = torch.cat((scores, scalar_probs), dim=1).cpu().tolist()
        verify_ms = (perf_counter() - started_at) * 1000
        for position, (entry, row) in enumerate(zip(entries[:count], packet[:count])):
            if self.config.feedback_enabled:
                self.feedback.update(entry.state, row[2 : 2 + len(self.config.lambda_candidates)], target_verified=True)
            if self.config.debug:
                p_m_x, q_d_x, q_mix_x, p_t_x, overlap = row[-5:]
                self.feedback.emit_record(
                    {
                        "event": "target_verified",
                        "request_id": req_id,
                        "spec_round": entry.spec_round,
                        "target_cycle_id": f"{req_id}:{proposal_start}",
                        "proposal_start": proposal_start,
                        "candidate_length": candidate_length,
                        "lambda_candidates": self.config.lambda_candidates,
                        "lambda_candidate_overlaps": row[2 : 2 + len(self.config.lambda_candidates)],
                        "submitted_tokens": submitted_tokens,
                        "token_position": position,
                        "draft_source": entry.source,
                        "lambda": entry.weight,
                        "draft_token": entry.token,
                        "original_draft_token": entry.draft_token,
                        "intermediate_accept": entry.intermediate_accept,
                        "target_accept": position < accepted_length,
                        "intermediate_accepted_length": entry.intermediate_accepted_length,
                        "target_accepted_length": accepted_length,
                        "pM_draft_token": p_m_x,
                        "qD_draft_token": q_d_x,
                        "qMix_draft_token": q_mix_x,
                        "pT_draft_token": p_t_x,
                        "tv_target_intermediate": 1 - row[0],
                        "tv_target_dflash": 1 - row[1],
                        "tv_target_mixture": 1 - overlap,
                        "draft_time_ms": entry.draft_time_ms,
                        "intermediate_verify_time_ms": entry.intermediate_verify_time_ms,
                        "target_verify_time_ms": verify_ms,
                        "sampling_overhead_ms": entry.sampling_overhead_ms,
                        "total_time_ms": entry.total_time_ms + verify_ms,
                        "timing": "host_wall",
                    }
                )

    def state_and_weight(self, probabilities, draft_token, spec_round, position, source, snapshot):
        if self.config.mode == "fixed" and not self.config.feedback_enabled and not self.config.debug:
            # No consumer needs distribution features in this experiment mode.
            # Avoid a full-vocabulary feature pass and its scalar D2H sync.
            return ("fixed", spec_round, position, source), self.config.mixture_lambda
        draft_id = torch.tensor([draft_token], device=self.device, dtype=torch.int64)
        features = device_features(probabilities[None], draft_id).cpu()[0].tolist()
        state = make_state(features, spec_round, position, source)
        return state, self.feedback.select(state, snapshot=snapshot)
