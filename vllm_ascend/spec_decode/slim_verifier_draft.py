# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import inspect
from functools import wraps

import torch


class DraftLogitsCapture:
    """Private raw-logit capture; never replaces the sampler's logits cache.

    Installed on one speculator instance before graph capture, on the metrics
    TP rank only. Device scatters are recorded by graph capture and replayed
    with live request mappings. The extra row absorbs graph padding (-1).
    Replace these wrappers with an upstream observation callback when offered.
    """

    @staticmethod
    def validate(speculator):
        if speculator is None or not all(
            callable(getattr(speculator, name, None)) for name in ("sample_draft", "compute_draft_logits", "propose")
        ):
            raise NotImplementedError(
                "KL mode requires a neural proposer exposing sample_draft and compute_draft_logits; "
                "token-only proposers have no full draft distribution. Use metrics_mode='acceptance' for them."
            )
        if getattr(speculator, "use_local_argmax_reduction", False):
            raise NotImplementedError("KL mode requires use_local_argmax_reduction=False to capture full draft logits")
        if getattr(getattr(speculator, "speculative_config", None), "use_heterogeneous_vocab", False):
            raise NotImplementedError("KL mode requires Draft and Target to share the same vocabulary/token ID space")

    def __init__(self, speculator, vocab_size):
        self.validate(speculator)
        self.max_num_reqs = speculator.max_num_reqs
        self.num_steps = speculator.num_speculative_steps
        self.vocab_size = vocab_size
        shape = (self.max_num_reqs + 1, self.num_steps)
        self.logits = torch.zeros((*shape, vocab_size), dtype=torch.float32, device=speculator.device)
        self.positions = torch.full(shape, -1, dtype=torch.int64, device=speculator.device)
        self.token_ids = torch.full_like(self.positions, -1)
        self._active = None
        self._head_calls = 0
        self._install(speculator)

    def _request_slots(self, slots):
        # Real rows have unique (request slot, draft column) pairs. Writes to
        # the padding-only sink may collide, but this row is never consumed.
        return torch.where((slots >= 0) & (slots < self.max_num_reqs), slots, self.max_num_reqs).long()

    def _install(self, speculator):
        original_sample = speculator.sample_draft
        original_head = speculator.compute_draft_logits
        original_propose = speculator.propose
        signature = inspect.signature(original_sample)

        @wraps(original_head)
        def head(*args, **kwargs):
            logits = original_head(*args, **kwargs)
            if self._active is not None:
                slots, steps, positions = self._active
                if logits is None or logits.shape != (slots.numel(), self.vocab_size):
                    raise RuntimeError("Draft KL requires complete global-vocabulary logits with Target token IDs")
                self.logits[slots, steps] = logits.float()
                self.positions[slots, steps] = positions.long()
                self._head_calls += 1
            return logits

        @wraps(original_sample)
        def sample(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            positions = arguments.get("sample_src_positions", arguments.get("positions"))
            if positions is None:
                raise NotImplementedError("Cannot locate the draft sampler's prediction positions")
            slots = self._request_slots(arguments["idx_mapping"])
            steps = arguments["draft_step"].long()
            self._active = (slots, steps, positions)
            self._head_calls = 0
            try:
                sampled = original_sample(*args, **kwargs)
                if self._head_calls != 1:
                    raise NotImplementedError("Draft KL requires one full-logit head call per sample_draft call")
                self.token_ids[slots, steps] = sampled.long()
                return sampled
            finally:
                self._active = None

        @wraps(original_propose)
        def propose(*args, **kwargs):
            batch = args[0] if args else kwargs["input_batch"]
            slots = self._request_slots(batch.idx_mapping)
            # Clear only slots being reproposed; other requests may still have
            # pending drafts. This also prevents stale data after slot reuse.
            self.positions[slots] = -1
            self.token_ids[slots] = -1
            return original_propose(*args, **kwargs)

        speculator.compute_draft_logits = head
        speculator.sample_draft = sample
        speculator.propose = propose

    def read(self, state_index, prediction_positions, draft_tokens):
        n = draft_tokens.numel()
        if not 0 <= state_index < self.max_num_reqs or n > self.num_steps:
            raise ValueError("Draft KL request slot or draft length exceeds the capture buffers")
        aligned = self.positions[state_index, :n].eq(prediction_positions)
        aligned &= self.token_ids[state_index, :n].eq(draft_tokens)
        return self.logits[state_index, :n], aligned
