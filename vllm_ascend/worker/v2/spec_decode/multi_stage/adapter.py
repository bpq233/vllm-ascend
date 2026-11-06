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
        self._draft_tokens_by_req = {}
        self._deferred_req_ids = None

    def begin_step(self, scheduler_output):
        """Keep pending drafts for live, unscheduled requests like MRV2."""
        self._deferred_req_ids = (
            tuple(scheduler_output.scheduled_spec_decode_tokens)
            if getattr(scheduler_output, "pending_structured_output_tokens", False)
            else None
        )
        context = getattr(self, "proposal_context", None)
        if context is not None:
            context.begin_step(scheduler_output)
        reset_ids = set(getattr(scheduler_output, "finished_req_ids", ()))
        reset_ids.update(request.req_id for request in getattr(scheduler_output, "scheduled_new_reqs", ()))
        self.reset_requests(reset_ids)
        # The previous step's published drafts live in the per-request map.
        # Current-batch views must not be mistaken for a new empty step's output.
        self.req_ids = []
        self.candidates = []

    def discard(self, req_id):
        self.reset_requests((req_id,))

    def reset_requests(self, req_ids):
        """Invalidate histories before a finished request ID can be reused."""
        reset_ids = set(req_ids)
        retained = getattr(self, "_draft_tokens_by_req", {})
        for req_id in reset_ids:
            self._host_histories.pop(req_id, None)
            retained.pop(req_id, None)
            context = getattr(self, "proposal_context", None)
            if context is not None:
                context.discard(req_id)
        if reset_ids:
            current = [
                (req_id, tokens) for req_id, tokens in zip(self.req_ids, self.candidates) if req_id not in reset_ids
            ]
            self.req_ids = [req_id for req_id, _ in current]
            self.candidates = [tokens for _, tokens in current]
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            backend = pipeline.backend
            for req_id in reset_ids:
                slot = backend.cache.slots.get(req_id)
                if slot is not None:
                    backend._context_rows[slot] = None
            backend.cache.retain(set(backend.cache.slots) - reset_ids)

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
        self.proposal_context = getattr(final_verifier, "proposal_context", None)
        if self.proposal_context is not None:
            from vllm_ascend.worker.v2.spec_decode.multi_stage.proposal_pipeline import ProposalPipeline

            self.pipeline = ProposalPipeline(
                backend,
                self.intermediate_config,
                self.final_capacity,
                runner.model_config.hf_text_config.eos_token_id,
                self.proposal_context,
            )
        else:
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
            self.pipeline.policy.method if self.proposal_context is None else "standard_mixture",
            runner.vllm_config.additional_config["multi_stage_speculative"]["final_verification"].get("method", "topk"),
            self.final_capacity,
        )

    @torch.inference_mode()
    def propose(self, input_batch, *args, **kwargs):
        max_candidates = kwargs.get("num_speculative_tokens")
        if max_candidates is not None:
            # The runner passes its Target budget, while Primary owns only a
            # short per-round buffer. Keep those two capacities independent.
            kwargs["num_speculative_tokens"] = min(max_candidates, self.num_speculative_steps)
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
        if getattr(self.pipeline.backend, "decision_runner", None) is not None:
            # Primary owns its output until the next Target cycle; Secondary
            # has separate buffers. Keep these views through the first round's
            # existing decision packet rather than copying Primary to CPU now.
            lengths, prefixes, _ = self._read_step(input_batch, primary[:, :0])
            primary_tokens = list(primary.unbind(0))
        else:
            lengths, prefixes, primary_tokens = self._read_step(input_batch, primary)
        return self._refine_and_publish(
            input_batch, lengths, prefixes, primary_tokens, output, max_candidates=max_candidates
        )

    def _refine_and_publish(self, input_batch, lengths, prefixes, primary_tokens, output, max_candidates=None):
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
        if max_candidates is not None:
            limits = [min(limit, max_candidates) for limit in limits]
        self.pipeline.backend.cache.retain(self.req_states.req_id_to_index)
        publish_device = getattr(self, "proposal_context", None) is not None or (
            getattr(self.pipeline.backend, "decision_runner", None) is not None
            and (primary_tokens is None or all(isinstance(tokens, torch.Tensor) for tokens in primary_tokens))
        )
        self.candidates = self.pipeline.refine(
            prefixes,
            primary_tokens,
            limits,
            req_ids=list(input_batch.req_ids),
            **({"output": output} if publish_device else {}),
        )
        self.req_ids = list(input_batch.req_ids)
        if publish_device and self.pipeline.device_output_ready:
            refined_at = getattr(self.pipeline, "last_refine_finished_at", None)
            self.proposal_timing = (
                (refined_at, perf_counter(), self.pipeline.last_compute_stream) if refined_at is not None else None
            )
            return output
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
        # Respect the runner's final truncation, including skip_drafts (width 0),
        # without transferring the same candidate tensor to CPU a second time.
        width = draft_tokens.shape[1]
        if width < self.final_capacity:
            self.candidates = [tokens[:width] for tokens in self.candidates]
        self._remember_drafts()

    def _remember_drafts(self):
        retained = getattr(self, "_draft_tokens_by_req", None)
        if retained is None:
            retained = self._draft_tokens_by_req = {}
        retained.update(zip(self.req_ids, self.candidates))
        return retained

    def get_draft_tokens(self):
        retained = self._remember_drafts()
        deferred = getattr(self, "_deferred_req_ids", None)
        req_ids = list(retained) if deferred is None else [req_id for req_id in deferred if req_id in retained]
        return DraftTokenIds(req_ids, [retained[req_id] for req_id in req_ids])
