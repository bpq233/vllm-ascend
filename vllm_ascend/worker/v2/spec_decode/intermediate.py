# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import logging

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

    @torch.inference_mode()
    def refine(self, prefixes, primary_tokens, limits):
        limits = [min(limit, self.capacity) for limit in limits]
        accepted = [[] for _ in prefixes]
        drafts = [tokens[:limit] for tokens, limit in zip(primary_tokens, limits)]
        active = [i for i, limit in enumerate(limits) if limit > 0]
        for round_id in range(self.config.num_rounds):
            if not active:
                break
            contexts = [prefixes[i] + accepted[i] for i in active]
            logits = self.backend.verify(contexts, [drafts[i] for i in active])
            # One small batched D2H transfer per verification round.
            results = []
            for i, scores in zip(active, logits):
                tokens = torch.tensor(drafts[i], dtype=torch.long, device=scores.device)
                flags = self.policy.accept(scores[:-1], tokens)
                length = flags.to(torch.int32).cumprod(0).sum()
                replacement = scores.index_select(0, length.reshape(1)).argmax(-1).squeeze(0)
                results.append(torch.stack((length, replacement)))
            decisions = torch.stack(results).cpu().tolist()
            continuing = []
            for i, (length, replacement) in zip(active, decisions):
                additions = drafts[i][:length] + [replacement]
                stopped = False
                for token in additions[: limits[i] - len(accepted[i])]:
                    accepted[i].append(token)
                    if token in self.eos_ids:
                        stopped = True
                        break
                if not stopped and len(accepted[i]) < limits[i]:
                    continuing.append(i)
            logger.debug("Intermediate round %d: requests=%d, continuing=%d", round_id, len(active), len(continuing))
            active = continuing
            if round_id + 1 < self.config.num_rounds and active:
                proposals = self.backend.propose([prefixes[i] + accepted[i] for i in active])
                for i, tokens in zip(active, proposals):
                    drafts[i] = tokens[: limits[i] - len(accepted[i])]
        return accepted
