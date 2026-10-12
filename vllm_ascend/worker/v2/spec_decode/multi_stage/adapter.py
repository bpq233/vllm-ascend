# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import logging
from copy import copy
from time import perf_counter

import torch
from vllm.v1.outputs import DraftTokenIds

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.multi_stage.backend import IntermediateBackend
from vllm_ascend.worker.v2.spec_decode.multi_stage.config import IntermediateConfig, primary_draft_width
from vllm_ascend.worker.v2.spec_decode.multi_stage.pipeline import IntermediatePipeline

logger = logging.getLogger(__name__)


class MultiStageDFlashSpeculator(AscendDFlashSpeculator):
    """An output adapter: the original DFlash propose runs unchanged."""

    # The synchronous propose path publishes committed progress along with its
    # existing D2H. The runner must not enqueue a second progress-only transfer.
    updates_computed_tokens_cpu = True
    uses_primary_drafter = True

    def __init__(self, vllm_config, device):
        options = vllm_config.additional_config["multi_stage_speculative"]
        self.final_capacity = vllm_config.speculative_config.num_speculative_tokens
        primary_config = copy(vllm_config)
        primary_config.speculative_config = copy(vllm_config.speculative_config)
        primary_config.speculative_config.num_speculative_tokens = primary_draft_width(options, self.final_capacity)
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
        computed_gpu = self.req_states.num_computed_tokens.gpu[indices]
        source = self.req_states.all_token_ids.gpu
        width = min(self.final_capacity + 1, source.shape[1])
        offsets = getattr(self, "_history_offsets", None)
        if offsets is None or offsets.numel() != width:
            offsets = self._history_offsets = torch.arange(width, device=primary.device)
        positions = (lengths_gpu[:, None] - width + offsets).clamp(min=0)
        tails = source[indices[:, None], positions.long()]
        packed_cpu = torch.cat((lengths_gpu[:, None], computed_gpu[:, None], primary, tails), dim=1).cpu()
        final_verifier = getattr(self, "final_verifier", None)
        if final_verifier is not None:
            final_verifier.flush_trace()
        # Publish only real request rows; unused/recycled slots must retain
        # their add_request state. numpy() views already completed CPU storage.
        self.req_states.num_computed_tokens_cpu.numpy()[input_batch.idx_mapping_np] = packed_cpu.numpy()[:, 1]
        packed = packed_cpu.tolist()
        missing = [
            (req_id, int(index), int(row[0]))
            for req_id, index, row in zip(input_batch.req_ids, input_batch.idx_mapping_np, packed)
            if req_id not in histories or not 0 <= int(row[0]) - len(histories[req_id]) <= width
        ]
        if missing:
            # Cold starts and prefill jumps share one D2H, regardless of batch
            # size. Slices use CPU lengths, so no device scalar is extracted.
            cold = torch.cat([source[index, :length] for _, index, length in missing]).cpu().tolist()
            offset = 0
            for req_id, _, length in missing:
                histories[req_id] = cold[offset : offset + length]
                offset += length
        lengths, prefixes, drafts = [], [], []
        for req_id, index, row in zip(input_batch.req_ids, input_batch.idx_mapping_np, packed):
            length = int(row[0])
            prefix = histories[req_id]
            delta = length - len(prefix)
            # The synchronous pipeline never mutates committed prefixes.
            # Append only the delta, avoiding an O(context) host copy per step.
            if delta:
                prefix.extend(row[-delta:])
            lengths.append(length)
            prefixes.append(prefix)
            drafts.append(row[2 : 2 + primary.shape[1]])
        self._host_histories = histories
        return lengths, prefixes, drafts

    def initialize_intermediate(self, runner):
        self.req_states = runner.req_states
        final_verifier = getattr(runner, "rejection_sampler", None)
        if hasattr(final_verifier, "flush_trace"):
            self.final_verifier = final_verifier
            self.final_verifier.defer_trace = True
        backend = IntermediateBackend(runner.vllm_config, self.intermediate_config, self.device)
        backend.update_stream = runner.update_stream
        backend.load_model()
        self.pipeline = IntermediatePipeline(
            backend, self.intermediate_config, self.final_capacity, runner.model_config.hf_text_config.eos_token_id
        )
        logger.info(
            "multi_stage_init use_primary_drafter=%s primary_drafter_model=%s "
            "intermediate_verifier_model=%s secondary_drafter_model=%s "
            "num_intermediate_rounds=%d max_generated_tokens=%s intermediate_num_speculative_tokens=%d "
            "intermediate_verification_method=%s final_verification_method=%s final_capacity=%d",
            self.uses_primary_drafter,
            runner.vllm_config.speculative_config.model if self.uses_primary_drafter else None,
            self.intermediate_config.verifier_model,
            self.intermediate_config.drafter_model,
            self.intermediate_config.num_rounds,
            self.intermediate_config.max_generated_tokens,
            self.intermediate_config.num_speculative_tokens,
            self.pipeline.policy.method,
            runner.vllm_config.additional_config["multi_stage_speculative"]["final_verification"].get("method", "topk"),
            self.final_capacity,
        )

    @torch.inference_mode()
    def propose(self, input_batch, *args, **kwargs):
        primary = super().propose(input_batch, *args, **kwargs)
        output = primary.new_empty((input_batch.num_reqs, self.final_capacity))
        # Profiling/capture must exercise the original drafter without accessing
        # live requests or publishing dummy candidate state.
        if kwargs.get("dummy_run", False):
            output.zero_()
            if kwargs.get("is_profile", False):
                self.pipeline.backend.profile()
            output[:, : primary.shape[1]] = primary
            return output
        lengths, prefixes, primary_tokens = self._read_step(input_batch, primary)
        return self._refine_and_publish(input_batch, lengths, prefixes, primary_tokens, output)

    def _refine_and_publish(self, input_batch, lengths, prefixes, primary_tokens, output):
        # postprocess_sampled has already committed the current target result.
        # Partial prefill rows must not enter the intermediate pipeline.
        limits = [
            max(
                0,
                min(
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
        # _read_step's blocking D2H has completed all earlier uploads on this
        # stream, including the previous use of this pinned source. Reuse it
        # without adding an event; a different stream/shape gets fresh storage.
        # Padding is only storage; the handler publishes actual row lengths.
        stream = torch.npu.current_stream() if self.device.type == "npu" else None
        host_output = getattr(self, "_host_candidates", None)
        if (
            host_output is None
            or host_output.shape != output.shape
            or host_output.dtype != output.dtype
            or getattr(self, "_host_candidate_stream", None) != stream
        ):
            host_output = self._host_candidates = torch.empty(
                output.shape, dtype=output.dtype, pin_memory=self.device.type != "cpu"
            )
            self._host_candidate_stream = stream
        host_rows = host_output.numpy()
        host_rows.fill(0)
        for row, tokens in enumerate(self.candidates):
            host_rows[row, : len(tokens)] = tokens
        output.copy_(host_output, non_blocking=True)
        refined_at = getattr(self.pipeline, "last_refine_finished_at", None)
        self.proposal_timing = (
            (refined_at, perf_counter(), self.pipeline.last_compute_stream) if refined_at is not None else None
        )
        return output

    def set_draft_tokens(self, input_batch, draft_tokens):
        # The pipeline already owns CPU token lists for synchronous scheduling.
        pass

    def get_draft_tokens(self):
        return DraftTokenIds(self.req_ids, self.candidates)
