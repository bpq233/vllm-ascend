# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""One final decision on the scheduled accumulated candidate prefix."""

from time import perf_counter

import torch
from vllm.logger import logger
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from .decisions import FinalTargetVerifier, rejection_logits
from .metrics import FinalMetrics


def sample_sparse_target(runner, hidden_states, input_batch, grammar_output):
    speculator = runner.speculator
    if grammar_output is not None:
        raise ValueError("dflash_sparse_verify does not support structured outputs")
    config = speculator.sparse_config
    verifier = FinalTargetVerifier(config.final_verification)
    accept_all = config.final_verification.method == "accept_all"
    boundaries = input_batch.cu_num_logits_np
    active_rows = [i for i in range(input_batch.num_reqs) if boundaries[i + 1] > boundaries[i]]
    indices = input_batch.logits_indices
    output = torch.full(
        (input_batch.num_reqs, speculator.total_candidates + 1), -1, dtype=torch.int32, device=hidden_states.device
    )
    num_sampled = torch.zeros(input_batch.num_reqs, dtype=torch.int32, device=hidden_states.device)
    num_rejected = torch.zeros_like(num_sampled)
    if not active_rows:
        empty = SamplerOutput(
            sampled_token_ids=output,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        return empty, num_sampled, num_rejected
    if accept_all:
        # All Transformer rows ran, but only one LM-head row per live request
        # is required. The other policies share the same Target forward.
        rows = torch.tensor([int(boundaries[i + 1]) - 1 for i in active_rows], dtype=torch.long, device=indices.device)
        logits = runner.model.compute_logits(hidden_states[indices[rows]])
    else:
        logits = runner.model.compute_logits(hidden_states[indices])
    target_seconds = 0.0
    events = getattr(runner, "_sparse_target_events", None)
    if config.enable_metrics and events is not None:
        events[1].synchronize()
        target_seconds = events[0].elapsed_time(events[1]) / 1000
    for logit_row, request in enumerate(active_rows):
        begin, end = int(boundaries[request]), int(boundaries[request + 1])
        proposed = end - begin - 1
        # Input [anchor, x0, ...] produces logits predicting [x0, ..., bonus].
        # Never compare a candidate with the logits of its own input row.
        candidates = input_batch.input_ids[indices[begin + 1 : end]].long()
        request_logits = logits[logit_row : logit_row + 1] if accept_all else logits[begin:end]
        start_time = perf_counter()
        try:
            decision = verifier.verify(request_logits, candidates)
        except ValueError as error:
            raise ValueError(
                f"DFlash Target verification req={input_batch.req_ids[request]} "
                f"method={config.final_verification.method} proposed={proposed} "
                f"logits_shape={tuple(request_logits.shape)}: {error}"
            ) from error
        positions = torch.arange(proposed, device=output.device)
        output[request, :proposed] = torch.where(positions < decision.accepted_length, candidates, -1).int()
        output[request].scatter_(0, decision.accepted_length.reshape(1).long(), decision.next_token.reshape(1).int())
        num_sampled[request] = decision.accepted_length + 1
        num_rejected[request] = proposed - decision.accepted_length
        if config.enable_metrics:
            accepted = int(decision.accepted_length.item())
            rejected_at = int(decision.first_rejection.item())
            if rejected_at >= 0:
                logger.info(
                    "DFlash Target rejection req=%s first_rejection=%s details=%s",
                    input_batch.req_ids[request],
                    rejected_at,
                    rejection_logits(request_logits[rejected_at], int(candidates[rejected_at].item())),
                )
            strict_accepted = None
            if not accept_all:
                strict_mask = request_logits[:-1].argmax(dim=-1) == candidates
                strict_accepted = int(strict_mask.int().cumprod(dim=0).sum().item())
            record = FinalMetrics(
                input_batch.req_ids[request],
                config.final_verification.method,
                proposed,
                accepted,
                int(decision.first_rejection.item()),
                target_seconds / len(active_rows),
                perf_counter() - start_time,
                accepted + 1,
                strict_accepted,
                None if strict_accepted is None else accepted - strict_accepted,
            )
            speculator.sparse_metrics.record_final(record)
            logger.info(
                "DFlash final verification req=%s method=%s proposed=%s accepted=%s rejected_at=%s",
                record.request_id,
                record.method,
                proposed,
                accepted,
                record.first_rejection,
            )
    sampler_output = SamplerOutput(
        sampled_token_ids=output,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )
    return sampler_output, num_sampled, num_rejected
