# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import logging
from collections import OrderedDict
from time import perf_counter

import torch

from vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance import AcceptancePolicy

logger = logging.getLogger(__name__)


class IntermediatePipeline:
    """Refine primary tokens without modifying committed request state."""

    def __init__(self, backend, config, capacity, eos_token_id=None):
        self.backend = backend
        self.config = config
        self.capacity = capacity
        self.policy = AcceptancePolicy(**config.verification)
        self.eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
        self._decision_shapes = OrderedDict()

    def _decision_shape(self, logits, lengths):
        # Accepted contexts change every round, but the small query shapes
        # usually repeat. Bound retention for heterogeneous request traffic.
        key = (logits.device, tuple(lengths))
        if key not in self._decision_shapes:
            sizes = torch.tensor(lengths, device=logits.device)
            starts = sizes.cumsum(0) - sizes
            steps = torch.arange(max(lengths), device=logits.device)
            rows = (starts[:, None] + steps).clamp(max=sum(lengths) - 1)
            is_draft = steps < sizes[:, None] - 1
            self._decision_shapes[key] = sizes, starts, steps, rows, is_draft
            if len(self._decision_shapes) > 16:
                self._decision_shapes.popitem(last=False)
        self._decision_shapes.move_to_end(key)
        return self._decision_shapes[key]

    def _decide(self, logits, drafts, lengths, tokens=None):
        # Keep vocabulary-sized logits packed; only the small acceptance mask
        # is padded. One top-k and one decision D2H per model microbatch.
        sizes, starts, steps, rows, is_draft = self._decision_shape(logits, lengths)
        if self.policy.method == "all":
            replacement = logits.index_select(0, starts + sizes - 1).argmax(-1)
            return torch.stack((sizes - 1, replacement), dim=-1).cpu().tolist()
        if tokens is None:
            tokens = torch.tensor(
                [t for draft in drafts for t in (*draft, 0)],
                dtype=torch.int64,
                pin_memory=logits.device.type != "cpu",
            ).to(logits.device, non_blocking=True)
        values, indices = logits.topk(min(self.policy.top_k, logits.shape[-1]), dim=-1)
        # Use the values already returned by topk instead of launching a
        # second gather of the vocabulary logits. Keep topk's tie ordering;
        # argmax is deliberately reserved for the replacement token.
        if indices.shape[-1] == 1:
            flags = (indices[:, 0] == tokens) & torch.isfinite(values[:, 0])
        else:
            flags = ((indices == tokens[:, None]) & torch.isfinite(values)).any(dim=-1)
        stop = torch.where(is_draft & flags[rows], max(lengths), steps).amin(dim=1)
        replacement = logits.index_select(0, starts + stop).argmax(-1)
        return torch.stack((stop, replacement), dim=-1).cpu().tolist()

    @torch.inference_mode()
    def refine(self, prefixes, primary_tokens, limits, req_ids=None):
        req_ids = req_ids if req_ids is not None else [str(i) for i in range(len(prefixes))]
        results = [[] for _ in prefixes]
        # Keep a group resident throughout its rounds. Round-major execution
        # across more requests than cache slots would evict every useful prefix.
        width = getattr(self.backend, "max_num_reqs", len(prefixes)) or 1
        resident = getattr(getattr(self.backend, "cache", None), "slots", {})
        order = sorted(range(len(prefixes)), key=lambda i: req_ids[i] not in resident)
        for start in range(0, len(prefixes), width):
            group = order[start : start + width]
            refined = self._refine(
                [prefixes[i] for i in group],
                [primary_tokens[i] for i in group],
                [limits[i] for i in group],
                [req_ids[i] for i in group],
            )
            for i, tokens in zip(group, refined):
                results[i] = tokens
        return results

    def _refine(self, prefixes, primary_tokens, limits, req_ids):
        limits = [min(limit, self.capacity) for limit in limits]
        accepted = [[] for _ in prefixes]
        drafts = [tokens[:limit] for tokens, limit in zip(primary_tokens, limits)]
        active = [i for i, limit in enumerate(limits) if limit > 0]
        contexts_by_request = list(prefixes)
        trace = logger.isEnabledFor(logging.DEBUG)
        for round_id in range(self.config.num_rounds):
            if not active:
                break
            contexts = [contexts_by_request[i] for i in active]
            round_drafts = [drafts[i] for i in active]
            if trace:
                forward_before = getattr(self.backend, "forward_tokens", 0)
                executed_before = getattr(self.backend, "executed_tokens", 0)
                reused_before = getattr(self.backend, "reused_tokens", 0)
                started = perf_counter()
            decisions = []
            decision_tensors = []
            decision_runner = getattr(self.backend, "decision_runner", None)
            for logits, lengths in self.backend.verify_batches(
                contexts,
                round_drafts,
                req_ids=[req_ids[i] for i in active],
                retain_hidden=round_id + 1 < self.config.num_rounds,
                **({"decision": decision_runner} if decision_runner is not None else {}),
            ):
                if decision_runner is not None:
                    # Keep graph output alive until all microbatches finish;
                    # perform one compact D2H copy instead of synchronizing
                    # once per microbatch.
                    decision_tensors.append(logits.clone() if decision_runner.graph_enabled else logits)
                    continue
                offset = len(decisions)
                decisions.extend(
                    self._decide(
                        logits,
                        round_drafts[offset : offset + len(lengths)],
                        lengths,
                        getattr(self.backend, "verification_tokens", None),
                    )
                )
            if decision_tensors:
                decision_output = (
                    decision_tensors[0] if len(decision_tensors) == 1 else torch.cat(decision_tensors, dim=0)
                )
                decisions.extend(decision_output.cpu().tolist())
            intermediate_ms = (perf_counter() - started) * 1000 if trace else 0.0
            continuing = []
            accepted_by_request = [0] * len(prefixes)
            for i, (length, replacement) in zip(active, decisions):
                additions = drafts[i][:length] + [replacement]
                stopped = False
                for position, token in enumerate(additions[: limits[i] - len(accepted[i])]):
                    accepted[i].append(token)
                    if trace:
                        accepted_by_request[i] += int(position < length)
                    if token in self.eos_ids:
                        stopped = True
                        break
                if not stopped and len(accepted[i]) < limits[i]:
                    continuing.append(i)
            active = continuing
            secondary_ms = 0.0
            if round_id + 1 < self.config.num_rounds and active:
                # Share this immutable snapshot with the next verification.
                # Never extend it in place: backends may retain references.
                for i in active:
                    contexts_by_request[i] = prefixes[i] + accepted[i]
                started = perf_counter() if trace else 0.0
                proposals = self.backend.propose(
                    [contexts_by_request[i] for i in active], req_ids=[req_ids[i] for i in active]
                )
                secondary_ms = (perf_counter() - started) * 1000 if trace else 0.0
                for i, tokens in zip(active, proposals):
                    drafts[i] = tokens[: limits[i] - len(accepted[i])]
            # Host wall times include metadata/model/required D2H, plus the
            # acceptance policy for verification; no extra NPU synchronization.
            if trace:
                logger.debug(
                    "multi_stage_intermediate round=%d proposed=%d accepted=%d accepted_by_request=%s "
                    "intermediate_model_ms=%.3f secondary_model_ms=%.3f timing=host_wall_with_acceptance "
                    "request_ids=%s forward_tokens=%d kv_reused_tokens=%d executed_tokens=%d padding_tokens=%d",
                    round_id,
                    sum(map(len, round_drafts)),
                    sum(accepted_by_request),
                    accepted_by_request,
                    intermediate_ms,
                    secondary_ms,
                    req_ids,
                    getattr(self.backend, "forward_tokens", 0) - forward_before,
                    getattr(self.backend, "reused_tokens", 0) - reused_before,
                    getattr(self.backend, "executed_tokens", 0) - executed_before,
                    max(
                        0,
                        getattr(self.backend, "executed_tokens", 0)
                        - executed_before
                        - (getattr(self.backend, "forward_tokens", 0) - forward_before),
                    ),
                )
        return accepted
