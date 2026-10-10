# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""4B DFlash -> 4B verifier -> Target, with no target-conditioned drafter."""

import torch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

from vllm_ascend.worker.v2.spec_decode.multi_stage.adapter import MultiStageDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.multi_stage.config import IntermediateConfig


class IntermediateOnlySpeculator(BaseSpeculator):
    updates_computed_tokens_cpu = True
    uses_primary_drafter = False
    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config, device):
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.final_capacity = vllm_config.speculative_config.num_speculative_tokens
        self.intermediate_config = IntermediateConfig.from_dict(
            vllm_config.additional_config["multi_stage_speculative"]["intermediate"]
        )
        self.candidates = []
        self.req_ids = []
        self._host_histories = {}

    # Reuse the existing history/commit/output adapter without inheriting
    # DraftModelSpeculator: upstream must not bind a draft to the Target model,
    # register its layers in Target KV, or configure Target auxiliary outputs.
    _read_step = MultiStageDFlashSpeculator._read_step
    _refine_and_publish = MultiStageDFlashSpeculator._refine_and_publish
    initialize_intermediate = MultiStageDFlashSpeculator.initialize_intermediate
    set_draft_tokens = MultiStageDFlashSpeculator.set_draft_tokens
    get_draft_tokens = MultiStageDFlashSpeculator.get_draft_tokens

    def init_cudagraph_manager(self, cudagraph_mode):
        # Independent verifier/drafter managers are initialized by the backend.
        pass

    def capture(self):
        # The backend captures its graphs once while loading the pair.
        pass

    @torch.inference_mode()
    def propose(self, input_batch, *args, **kwargs):
        output = torch.empty((input_batch.num_reqs, self.final_capacity), dtype=torch.int64, device=self.device)
        if kwargs.get("dummy_run", False):
            output.zero_()
            if kwargs.get("is_profile", False):
                self.pipeline.backend.profile()
            return output
        # Only history/progress cross this boundary; there is no Primary model
        # output to copy. The first proposal is produced inside each resident
        # pipeline group, after hydrating the matching 4B context KV.
        no_primary = output[:, :0]
        lengths, prefixes, _ = self._read_step(input_batch, no_primary)
        return self._refine_and_publish(input_batch, lengths, prefixes, None, output)
