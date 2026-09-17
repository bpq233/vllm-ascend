# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from copy import copy

import torch
from vllm.v1.outputs import DraftTokenIds

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.intermediate import IntermediatePipeline
from vllm_ascend.worker.v2.spec_decode.intermediate_backend import IntermediateBackend
from vllm_ascend.worker.v2.spec_decode.multi_stage_config import IntermediateConfig


class MultiStageDFlashSpeculator(AscendDFlashSpeculator):
    """An output adapter: the original DFlash propose runs unchanged."""

    def __init__(self, vllm_config, device):
        options = vllm_config.additional_config["multi_stage_speculative"]
        self.final_capacity = vllm_config.speculative_config.num_speculative_tokens
        primary_config = copy(vllm_config)
        primary_config.speculative_config = copy(vllm_config.speculative_config)
        primary_config.speculative_config.num_speculative_tokens = options.get(
            "primary_num_speculative_tokens", self.final_capacity
        )
        super().__init__(primary_config, device)
        self.intermediate_config = IntermediateConfig.from_dict(options["intermediate"])
        self.candidates = []
        self.req_ids = []

    def initialize_intermediate(self, runner):
        self.req_states = runner.req_states
        backend = IntermediateBackend(runner.vllm_config, self.intermediate_config, self.device)
        backend.load_model()
        self.pipeline = IntermediatePipeline(
            backend, self.intermediate_config, self.final_capacity, runner.model_config.hf_text_config.eos_token_id
        )

    @torch.inference_mode()
    def propose(self, input_batch, *args, **kwargs):
        primary = super().propose(input_batch, *args, **kwargs)
        output = primary.new_zeros((input_batch.num_reqs, self.final_capacity))
        # Profiling/capture must exercise the original drafter without accessing
        # live requests or publishing dummy candidate state.
        if kwargs.get("dummy_run", False):
            if kwargs.get("is_profile", False):
                self.pipeline.backend.profile()
            output[:, : primary.shape[1]] = primary
            return output
        indices = input_batch.idx_mapping
        lengths = self.req_states.total_len.gpu[indices].cpu().tolist()
        max_length = max(lengths)
        histories = self.req_states.all_token_ids.gpu[indices, :max_length].cpu().tolist()
        prefixes = [tokens[:length] for tokens, length in zip(histories, lengths)]
        # postprocess_sampled has already committed the current target result.
        # Partial prefill rows must not enter the intermediate pipeline.
        limits = [
            max(
                0,
                min(
                    self.final_capacity,
                    self.pipeline.backend.max_model_len - length - 1,
                    self.max_model_len - length - 1,
                    int(self.req_states.max_seq_len[index]) - length - 1,
                ),
            )
            if length > int(self.req_states.prefill_len.np[index])
            else 0
            for index, length in zip(input_batch.idx_mapping_np, lengths)
        ]
        self.candidates = self.pipeline.refine(prefixes, primary.cpu().tolist(), limits)
        self.req_ids = list(input_batch.req_ids)
        # Padding is only storage; the handler publishes the actual row lengths.
        padded = [tokens + [0] * (self.final_capacity - len(tokens)) for tokens in self.candidates]
        output.copy_(torch.tensor(padded, device=self.device, dtype=output.dtype))
        return output

    def set_draft_tokens(self, input_batch, draft_tokens):
        # The pipeline already owns CPU token lists for synchronous scheduling.
        pass

    def get_draft_tokens(self):
        return DraftTokenIds(self.req_ids, self.candidates)
