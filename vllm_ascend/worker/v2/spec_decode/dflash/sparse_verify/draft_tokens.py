# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Report actual per-request draft widths to the upstream scheduler."""

import torch
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


class SparseDraftTokensHandler(DraftTokensHandler):
    def set_draft_tokens(self, input_batch, draft_tokens: torch.Tensor) -> None:
        if input_batch.has_structured_output_reqs:
            raise ValueError("dflash_sparse_verify does not support structured outputs")
        if draft_tokens.ndim != 2 or draft_tokens.shape[0] != len(input_batch.req_ids):
            raise ValueError("sparse draft rows must match the current request batch")
        # Native DraftTokensHandler records the tensor width for every request.
        # Sparse proposals have variable lengths with a -1 padded suffix. Report
        # the real prefix width, otherwise those sentinels become Target inputs.
        valid = draft_tokens >= 0
        prefix = valid.int().cumprod(dim=1).bool()
        if bool((valid & ~prefix).any()) or bool((draft_tokens < -1).any()):
            raise ValueError("sparse drafts must be valid token prefixes followed only by -1 padding")
        widths = prefix.sum(dim=1).cpu().tolist()
        for req_id, width in zip(input_batch.req_ids, widths):
            # Reuse the native request lifetime and placeholder transport; only
            # its stored width changes. Actual token IDs stay in RequestState.
            self._draft_tokens_by_req[req_id] = int(width)
