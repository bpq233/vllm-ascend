# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import logging
from copy import copy

import torch
from vllm.v1.outputs import DraftTokenIds

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.intermediate import IntermediatePipeline
from vllm_ascend.worker.v2.spec_decode.intermediate_backend import IntermediateBackend
from vllm_ascend.worker.v2.spec_decode.multi_stage_config import IntermediateConfig

logger = logging.getLogger(__name__)


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
        self._host_histories = {}

    def _read_step(self, input_batch, primary):
        """One bounded D2H for warm requests; read full history only on a miss."""
        histories = getattr(self, "_host_histories", {})
        live = self.req_states.req_id_to_index
        histories = {key: value for key, value in histories.items() if key in live}
        indices = input_batch.idx_mapping
        lengths_gpu = self.req_states.total_len.gpu[indices]
        source = self.req_states.all_token_ids.gpu
        width = min(self.final_capacity + 1, source.shape[1])
        offsets = torch.arange(width, device=primary.device)
        positions = (lengths_gpu[:, None] - width + offsets).clamp(min=0)
        tails = source[indices[:, None], positions.long()]
        packed = torch.cat((lengths_gpu[:, None], primary, tails), dim=1).cpu().tolist()
        lengths, prefixes, drafts = [], [], []
        for req_id, index, row in zip(input_batch.req_ids, input_batch.idx_mapping_np, packed):
            length = int(row[0])
            previous = histories.get(req_id)
            delta = length - len(previous) if previous is not None else width + 1
            if previous is None or delta < 0 or delta > width:
                prefix = source[int(index), :length].cpu().tolist()
            else:
                prefix = previous + (row[-delta:] if delta else [])
            histories[req_id] = prefix
            lengths.append(length)
            prefixes.append(prefix)
            drafts.append(row[1 : 1 + primary.shape[1]])
        self._host_histories = histories
        return lengths, prefixes, drafts

    def initialize_intermediate(self, runner):
        self.req_states = runner.req_states
        backend = IntermediateBackend(runner.vllm_config, self.intermediate_config, self.device)
        backend.load_model()
        self.pipeline = IntermediatePipeline(
            backend, self.intermediate_config, self.final_capacity, runner.model_config.hf_text_config.eos_token_id
        )
        logger.info(
            "multi_stage_init primary_drafter_model=%s intermediate_verifier_model=%s secondary_drafter_model=%s "
            "num_intermediate_rounds=%d intermediate_num_speculative_tokens=%d "
            "intermediate_verification_method=%s final_verification_method=%s final_capacity=%d",
            runner.vllm_config.speculative_config.model,
            self.intermediate_config.verifier_model,
            self.intermediate_config.drafter_model,
            self.intermediate_config.num_rounds,
            self.intermediate_config.num_speculative_tokens,
            self.pipeline.policy.method,
            runner.vllm_config.additional_config["multi_stage_speculative"]["final_verification"].get("method", "topk"),
            self.final_capacity,
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
        lengths, prefixes, primary_tokens = self._read_step(input_batch, primary)
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
        self.pipeline.backend.cache.retain(self.req_states.req_id_to_index)
        self.candidates = self.pipeline.refine(prefixes, primary_tokens, limits, req_ids=list(input_batch.req_ids))
        self.req_ids = list(input_batch.req_ids)
        # Padding is only storage; the handler publishes the actual row lengths.
        padded = [tokens + [0] * (self.final_capacity - len(tokens)) for tokens in self.candidates]
        host_output = torch.tensor(padded, dtype=output.dtype, pin_memory=self.device.type != "cpu")
        output.copy_(host_output, non_blocking=True)
        return output

    def set_draft_tokens(self, input_batch, draft_tokens):
        # The pipeline already owns CPU token lists for synchronous scheduling.
        pass

    def get_draft_tokens(self):
        return DraftTokenIds(self.req_ids, self.candidates)
