# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import logging
from time import perf_counter

import torch

from vllm_ascend.worker.v2.spec_decode.acceptance import AcceptancePolicy

logger = logging.getLogger(__name__)


class IntermediatePipeline:
    """Refine primary tokens without modifying committed request state."""

    def __init__(self, backend, config, capacity, eos_token_id=None):
        self.backend = backend
        self.config = config
        self.capacity = capacity
        self.policy = AcceptancePolicy(**config.verification)
        self.eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])

    def _decide(self, logits, drafts, lengths):
        # Keep vocabulary-sized logits packed; only the small acceptance mask
        # is padded. One top-k and one decision D2H per model microbatch.
        if self.policy.method == "all":
            sizes = torch.tensor(lengths, device=logits.device)
            replacement = logits.index_select(0, sizes.cumsum(0) - 1).argmax(-1)
            return torch.stack((sizes - 1, replacement), dim=-1).cpu().tolist()
        packed = torch.tensor(
            lengths + [t for draft in drafts for t in (*draft, 0)],
            dtype=torch.int64,
            pin_memory=logits.device.type != "cpu",
        ).to(logits.device, non_blocking=True)
        sizes, tokens = packed[: len(lengths)], packed[len(lengths) :]
        starts = sizes.cumsum(0) - sizes
        flags = self.policy.accept(logits, tokens)
        steps = torch.arange(max(lengths), device=logits.device)
        rows = (starts[:, None] + steps).clamp(max=logits.shape[0] - 1)
        is_draft = steps < sizes[:, None] - 1
        stop = torch.where(is_draft & flags[rows], max(lengths), steps).amin(dim=1)
        replacement = logits.index_select(0, starts + stop).argmax(-1)
        return torch.stack((stop, replacement), dim=-1).cpu().tolist()

    @torch.inference_mode()
    def refine(self, prefixes, primary_tokens, limits, req_ids=None):
        req_ids = req_ids if req_ids is not None else [str(i) for i in range(len(prefixes))]
        results = []
        # Keep a group resident throughout its rounds. Round-major execution
        # across more requests than cache slots would evict every useful prefix.
        width = getattr(self.backend, "max_num_reqs", len(prefixes)) or 1
        for start in range(0, len(prefixes), width):
            end = start + width
            results.extend(
                self._refine(prefixes[start:end], primary_tokens[start:end], limits[start:end], req_ids[start:end])
            )
        return results

    def _refine(self, prefixes, primary_tokens, limits, req_ids):
        limits = [min(limit, self.capacity) for limit in limits]
        accepted = [[] for _ in prefixes]
        drafts = [tokens[:limit] for tokens, limit in zip(primary_tokens, limits)]
        active = [i for i, limit in enumerate(limits) if limit > 0]
        for round_id in range(self.config.num_rounds):
            if not active:
                break
            contexts = [prefixes[i] + accepted[i] for i in active]
            round_drafts = [drafts[i] for i in active]
            forward_before = getattr(self.backend, "forward_tokens", 0)
            reused_before = getattr(self.backend, "reused_tokens", 0)
            started = perf_counter()
            decisions = []
            for logits, lengths in self.backend.verify_batches(
                contexts, round_drafts, req_ids=[req_ids[i] for i in active]
            ):
                offset = len(decisions)
                decisions.extend(self._decide(logits, round_drafts[offset : offset + len(lengths)], lengths))
            intermediate_ms = (perf_counter() - started) * 1000
            continuing = []
            accepted_by_request = [0] * len(prefixes)
            for i, (length, replacement) in zip(active, decisions):
                additions = drafts[i][:length] + [replacement]
                stopped = False
                for position, token in enumerate(additions[: limits[i] - len(accepted[i])]):
                    accepted[i].append(token)
                    accepted_by_request[i] += int(position < length)
                    if token in self.eos_ids:
                        stopped = True
                        break
                if not stopped and len(accepted[i]) < limits[i]:
                    continuing.append(i)
            active = continuing
            secondary_ms = 0.0
            if round_id + 1 < self.config.num_rounds and active:
                started = perf_counter()
                proposals = self.backend.propose(
                    [prefixes[i] + accepted[i] for i in active], req_ids=[req_ids[i] for i in active]
                )
                secondary_ms = (perf_counter() - started) * 1000
                for i, tokens in zip(active, proposals):
                    drafts[i] = tokens[: limits[i] - len(accepted[i])]
            # Host wall times include metadata/model/required D2H, plus the
            # acceptance policy for verification; no extra NPU synchronization.
            logger.debug(
                "multi_stage_intermediate round=%d proposed=%d accepted=%d accepted_by_request=%s "
                "intermediate_model_ms=%.3f secondary_model_ms=%.3f timing=host_wall_with_acceptance "
                "request_ids=%s forward_tokens=%d kv_reused_tokens=%d",
                round_id,
                sum(map(len, round_drafts)),
                sum(accepted_by_request),
                accepted_by_request,
                intermediate_ms,
                secondary_ms,
                req_ids,
                getattr(self.backend, "forward_tokens", 0) - forward_before,
                getattr(self.backend, "reused_tokens", 0) - reused_before,
            )
        return accepted
