# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from time import perf_counter

import torch

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator

from .config import MultiStageConfig


class MultiStageDFlashSpeculator(AscendDFlashSpeculator):
    """Primary DFlash unchanged internally; only its returned drafts expand."""

    def __init__(self, vllm_config, device):
        self.multi_stage_config = MultiStageConfig.from_vllm_config(vllm_config)
        self.multi_stage_config.configure_runtime(vllm_config)
        self.multi_stage_config.validate_runtime(vllm_config)
        self.output_width = vllm_config.speculative_config.num_speculative_tokens
        primary_config = copy.copy(vllm_config)
        primary_config.speculative_config = copy.copy(vllm_config.speculative_config)
        primary_config.speculative_config.num_speculative_tokens = (
            self.multi_stage_config.primary_num_speculative_tokens
        )
        # Preserve the target's layer registry so normal KV grouping sees the
        # primary draft layers. Only the *intermediate* registry is independent.
        super().__init__(primary_config, device)
        self.runtime = None

    def load_model(self, target_model):
        super().load_model(target_model)
        assert self.runtime is not None
        self.runtime.load_backend()

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        measure = (
            self.multi_stage_config.summary_logging or self.multi_stage_config.metrics_enabled
        ) and not dummy_run
        if measure:
            torch.npu.synchronize()
        start = perf_counter()
        primary = super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile,
        )
        if dummy_run:
            return torch.zeros((input_batch.num_reqs, self.output_width), dtype=primary.dtype, device=primary.device)
        assert self.runtime is not None
        if measure:
            torch.npu.synchronize()
            elapsed = (perf_counter() - start) * 1000
            self.runtime.last_primary_ms = elapsed
            self.runtime.metrics.record("primary_drafter", primary.numel(), elapsed)
        return self.runtime.expand(input_batch, primary, num_sampled)
