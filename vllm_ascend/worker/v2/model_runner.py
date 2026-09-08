# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import time
from contextlib import contextmanager

import numpy as np
import torch
from vllm.compilation import breakable_cudagraph
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import (
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.model_runner import (
    ExecuteModelState,
    GPUModelRunner,
    sort_batch_req_ids,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import (
    MoECommType,
    get_mc2_tokens_capacity,
    override_mrv2_in_profile_run,
    select_moe_comm_method,
    set_mc2_mask,
    set_mc2_tokens_capacity,
)
from vllm_ascend.core.profiling_chunk_predictor import (
    _finish_profiling_chunk_timing,
    _start_profiling_chunk_timing,
)
from vllm_ascend.ops.rotary_embedding import set_cos_and_sin, update_cos_sin
from vllm_ascend.utils import set_potential_max_tokens, vllm_version_is

if not vllm_version_is("0.27.1"):
    from vllm.v1.worker.gpu.model_runner import BatchReqState

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.eplb import AscendEPLBController
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.spec_decode.via_sd import (
    ViaSdExecutionCoordinator,
    ViaSdRoutePlan,
    ViaSdModel,
    ViaSdVerifier,
    build_via_sd_model,
    build_route_plan,
    normalize_draft_tokens,
)
from vllm_ascend.worker.v2.states import AscendRequestState
from vllm_ascend.worker.v2.utils import torch_cuda_wrapper


class NPUModelRunner(GPUModelRunner):
    """Model runner for Ascend NPUs."""

    execute_model_state: ExecuteModelState | None
    via_sd_model: ViaSdModel | None
    via_sd_verifier: ViaSdVerifier | None

    @property
    def pcp_manager_cls(self) -> type[AscendPCPManager]:
        return AscendPCPManager

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Ascend-specific configurations
        self.ascend_config = get_ascend_config()
        # FusedMoE can be constructed by the parent initializer and reads this
        # capacity while setting up MC2 communication.
        set_potential_max_tokens(vllm_config)
        parallel_config = vllm_config.parallel_config
        if parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("Decode Context parallelism is not supported by Ascend NPU model runner v2.")

        with torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        self.use_spec_pp = (
            self.use_pp and self.speculative_config is not None and self.speculative_config.method == "mtp"
        )

        self.use_aclgraph = (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and (
                self.compilation_config.mode == CompilationMode.VLLM_COMPILE
                or breakable_cudagraph.is_breakable_cudagraph_enabled()
            )
            and not self.model_config.enforce_eager
        )
        load_collection_phase = self.ascend_config.eplb_config.load_collection_phase
        self.eplb = AscendEPLBController(
            parallel_config,
            device,
            load_collection_phase=(load_collection_phase if parallel_config.enable_eplb else "all"),
        )

        self.update_stream = None
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.update_stream = torch.npu.Stream()

        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.req_states
        del self.input_buffers
        del self.speculator

        # we define AscendEagleSpeculator in vllm_ascend.worker.v2.spec_decode.eagle.speculator
        # init_speculator will return AscendEagleSpeculator when eagle is used.
        # so here we just call init_speculator to reinitialize speculator.
        self.speculator: AscendEagleSpeculator | None = None
        if self.speculative_config is not None and (not self.use_spec_pp or self.is_last_pp_rank):
            self.speculator = init_speculator(self.vllm_config, self.device)
            # Shared update_stream: main model (ModelAclGraphManager) and draft
            # (Eagle/DFlash/DSpark AclGraphManager) all use this same stream.
            self.speculator.update_stream = self.update_stream

        # AscendRequestState has extra `num_computed_tokens_cpu` attribute.
        # so reinitialize req_states here.
        self.req_states: AscendRequestState = AscendRequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        if self.use_spec_pp:
            from vllm_ascend.patch.worker.patch_v2.patch_spec_pp import (
                install_spec_pp_token_broadcast,
            )

            assert self.pp_handler is not None
            install_spec_pp_token_broadcast(self.pp_handler, self.req_states)
        # AscendInputBuffers has extra `seq_lens_cpu` attribute.
        # so reinitialize input_buffers here.
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )

        # VIA-SD is an opt-in, read-only q' side pass.  The structural q' view
        # is installed in load_model(), before KV-cache planning, so its
        # optional attention pages are accounted for by MRv2.
        self.via_sd_model = None
        self.via_sd_verifier = None
        self.via_sd_last_logits = None
        self._via_sd_disabled = False
        self._via_sd_error_reported = False
        self._via_sd_qprime_validation_count = 0
        self._via_sd_target_validation_count = 0
        self._via_sd_route_plan: ViaSdRoutePlan | None = None
        self._via_sd_execution_result = None
        self._via_sd_pre_route_batch = None
        self._via_sd_pre_route_scheduler_output = None
        self._via_sd_coordinator: ViaSdExecutionCoordinator | None = None
        # ``sample_tokens`` runs after the parent target forward.  Keep the
        # exact draft block scheduled for that forward so q' never validates
        # a stale block produced for the next iteration.
        self._via_sd_pending_target_drafts: dict[object, tuple[int, ...]] = {}

        # we need to copy num_computed_tokens back to cpu to help
        # update actual seq_lens_cpu. gpu attention backend doesn't need these
        # attributes, cause their attention backends doesn't use seq_lens_cpu.
        # and seq_lens_cpu is deprecated in gpu_model_runner_v2.
        self.num_computed_tokens_event = torch.npu.Event()
        self.num_computed_tokens_stream = torch.npu.Stream()
        self.num_computed_tokens_cpu = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        # NOTE: In GPUModelRunner, decode_query_len is initialized in load_model(),
        # +1 is hardcoded here but not in vllm.
        self.decode_query_len = self.num_speculative_steps + 1
        # Set _mc2_tokens_capacity and _reserved_mc2_mask for MoE communication optimization.
        # TODO: remove set_cos_and_sin (together with update_cos_sin) when mla can properly handle cos/sin internally
        set_cos_and_sin(vllm_config, self.max_num_reqs, self.decode_query_len, self.dtype, self.device)
        set_mc2_tokens_capacity(vllm_config, self.max_num_reqs, self.decode_query_len)
        set_mc2_mask(vllm_config, self.device)
        set_potential_max_tokens(vllm_config)

    def _via_sd_config(self):
        return getattr(self.ascend_config, "via_sd_config", None)

    def _via_sd_enabled(self) -> bool:
        config = self._via_sd_config()
        return bool(
            config is not None
            and getattr(config, "enabled", False)
            and getattr(config, "mode", "observe") != "disabled"
        )

    def _via_sd_mode(self) -> str:
        config = self._via_sd_config()
        if config is None or not getattr(config, "enabled", False):
            return "disabled"
        mode = getattr(config, "mode", "observe")
        return mode if mode in {"disabled", "observe", "hierarchical"} else "disabled"

    def _via_sd_is_hierarchical(self) -> bool:
        return self._via_sd_mode() == "hierarchical" and not self._via_sd_disabled

    def _via_sd_timing_enabled(self) -> bool:
        config = self._via_sd_config()
        return bool(
            config is not None
            and getattr(config, "enabled", False)
            and getattr(config, "log_enabled", True)
            and getattr(config, "log_validation_timing", True)
        )

    def _via_sd_logging_enabled(self) -> bool:
        config = self._via_sd_config()
        return bool(
            config is not None
            and getattr(config, "enabled", False)
            and getattr(config, "log_enabled", True)
        )

    @staticmethod
    def _via_sd_real_drafts(scheduled_drafts) -> dict[object, tuple[int, ...]]:
        """Drop scheduler padding and retain only real draft-token prefixes."""

        return {
            request_id: valid_tokens
            for request_id, tokens in scheduled_drafts.items()
            if (valid_tokens := normalize_draft_tokens(tokens))
        }

    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        """Load target q, then construct q' without loading another checkpoint."""

        super().load_model(load_dummy_weights=load_dummy_weights, *args, **kwargs)
        if not self._via_sd_enabled() or self._via_sd_disabled:
            return
        if self.speculator is None:
            logger.warning_once("VIA-SD q' is enabled but no draft speculator is configured; disabling q'.")
            self._via_sd_disabled = True
            return
        try:
            if self.parallel_config.pipeline_parallel_size != 1:
                raise NotImplementedError("VIA-SD q' currently requires pipeline_parallel_size=1")
            if self.lora_config is not None:
                raise NotImplementedError("VIA-SD q' currently does not support LoRA")
            if getattr(self.vllm_config, "kv_transfer_config", None) is not None:
                raise NotImplementedError("VIA-SD q' currently does not support KV transfer")
            config = self._via_sd_config()
            assert config is not None
            self.via_sd_model = build_via_sd_model(
                self.model,
                self.vllm_config,
                config.layer_ids,
                config.layer_fraction,
            )
            if self._via_sd_logging_enabled():
                logger.info(
                    "VIA-SD q' enabled on MRv2: retained %d/%d target layers (%s), "
                    "kv_cache=%s, validation_timing=%s, mode=%s, "
                    "accept_ratio=%.4f, escalate_ratio=%.4f",
                    len(self.via_sd_model.layer_ids),
                    len(self.model.model.layers),
                    ",".join(str(index) for index in self.via_sd_model.layer_ids),
                    config.kv_cache_enabled,
                    config.log_validation_timing,
                    self._via_sd_mode(),
                    config.accept_ratio,
                    config.escalate_ratio,
                )
        except Exception as exc:
            self._disable_via_sd(str(exc))

    def get_kv_cache_spec(self):
        specs = super().get_kv_cache_spec()
        config = self._via_sd_config()
        if (
            self.via_sd_model is not None
            and config is not None
            and not config.kv_cache_enabled
        ):
            qprime_names = set(self.via_sd_model.attention_layer_names)
            specs = {name: spec for name, spec in specs.items() if name not in qprime_names}
        return specs

    def sample_tokens(self, grammar_output):
        if self._via_sd_mode() == 'hierarchical':
            runtime = getattr(self, '_via_sd_runtime', None)
            return runtime.sample(grammar_output) if runtime is not None else None
        # The parent clears execute_model_state before returning. Keep this
        # batch object so q' can use the target batch ordering after sampling.
        input_batch = self.execute_model_state.input_batch if self.execute_model_state is not None else None
        pending_drafts = self._via_sd_pending_target_drafts
        output = super().sample_tokens(grammar_output)
        # Observe-mode verification runs after parent sampling and never routes.
        if (
            self._via_sd_mode() == "observe"
            and output is not None
            and input_batch is not None
            and pending_drafts
        ):
            try:
                self._run_via_sd_validation(input_batch, pending_drafts)
            finally:
                self._via_sd_pending_target_drafts = {}
        elif pending_drafts:
            # Do not carry a target block into a later execute if sampling was
            # skipped by the parent runner.
            self._via_sd_pending_target_drafts = {}

        self._via_sd_pre_route_batch = None
        self._via_sd_pre_route_scheduler_output = None

        if self.use_spec_pp and self.is_last_pp_rank:
            assert self.pp_handler is not None
            self.pp_handler.broadcast_draft_tokens()
        return output

    def _run_via_sd_validation(
        self,
        input_batch,
        scheduled_drafts: dict[object, tuple[int, ...]] | None = None,
    ) -> ViaSdRoutePlan | None:
        if self._via_sd_disabled or self.via_sd_verifier is None or not self.is_last_pp_rank:
            return None
        try:
            return self._run_via_sd_validation_once(input_batch, scheduled_drafts)
        except Exception as exc:
            config = self._via_sd_config()
            if config is not None and not config.fail_open:
                raise
            self._disable_via_sd(str(exc))
            return None

    def _run_via_sd_validation_once(
        self,
        input_batch,
        scheduled_drafts: dict[object, tuple[int, ...]] | None = None,
    ) -> ViaSdRoutePlan | None:
        assert self.via_sd_verifier is not None
        scheduled_drafts = self._via_sd_real_drafts(
            self._via_sd_pending_target_drafts
            if scheduled_drafts is None
            else scheduled_drafts
        )
        num_reqs = int(input_batch.num_reqs)
        if num_reqs <= 0 or self.num_speculative_steps <= 0 or not scheduled_drafts:
            return None

        batch_request_ids = list(input_batch.req_ids[:num_reqs])
        batch_rows = {
            request_id: row for row, request_id in enumerate(batch_request_ids)
        }
        selected = [
            (row, request_id, scheduled_drafts[request_id])
            for request_id in batch_request_ids
            if request_id in scheduled_drafts
            for row in [batch_rows[request_id]]
        ]
        if not selected:
            return None
        selected_rows = [item[0] for item in selected]
        all_request_indices = [
            int(index)
            for index in input_batch.idx_mapping[:num_reqs]
            .detach()
            .cpu()
            .tolist()
        ]
        idx_mapping = input_batch.idx_mapping[selected_rows].to(dtype=torch.long)
        request_indices = [int(index) for index in idx_mapping.detach().cpu().tolist()]
        request_ids = [item[1] for item in selected]
        table_indices = selected_rows

        # ``num_computed_tokens_np`` is captured while preparing the target
        # batch and therefore names the committed prefix *before* rejection
        # sampling mutates request state.  Falling back to total_len keeps the
        # helper usable with lightweight test batches.
        batch_lengths = getattr(input_batch, "num_computed_tokens_np", None)
        if batch_lengths is None:
            lengths = self.req_states.total_len.gpu[idx_mapping]
            lengths_cpu = [int(length) for length in lengths.detach().cpu().tolist()]
        elif isinstance(batch_lengths, torch.Tensor):
            lengths_cpu = [
                int(length)
                for length in batch_lengths[selected_rows].detach().cpu().tolist()
            ]
        else:
            lengths_cpu = [int(batch_lengths[row]) for row in selected_rows]
        prefixes: list[list[int]] = []
        for state_index, length in zip(request_indices, lengths_cpu):
            if length <= 0:
                prefixes.append([])
                continue
            tokens = self.req_states.all_token_ids.gpu[state_index, :length]
            prefixes.append([int(token) for token in tokens.detach().cpu().tolist()])

        # q' needs one committed token to align the first predicted row.  A
        # target draft on an empty prefix is not a valid validation request.
        valid_rows = [row for row, prefix in enumerate(prefixes) if prefix]
        if not valid_rows:
            return None
        request_ids = [request_ids[row] for row in valid_rows]
        request_indices = [request_indices[row] for row in valid_rows]
        table_indices = [table_indices[row] for row in valid_rows]
        prefixes = [prefixes[row] for row in valid_rows]
        selected = [selected[row] for row in valid_rows]
        draft_steps = max(len(item[2]) for item in selected)
        draft_tokens = torch.full(
            (len(selected), draft_steps),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        valid_lengths: list[int] = []
        for row, (_, _, tokens) in enumerate(selected):
            valid_lengths.append(len(tokens))
            draft_tokens[row, : len(tokens)] = torch.as_tensor(
                tokens, dtype=torch.long, device=self.device
            )

        # MRv2 has already gathered the current batch block tables for target
        # attention.  q' uses the same page IDs, while its layer names select
        # separate physical pages when caching is enabled.
        block_tables = getattr(self.block_tables, "input_block_tables", None)
        backend = self.via_sd_verifier.backend
        setter = getattr(backend, "set_request_block_tables", None)
        if setter is not None:
            # ``block_tables`` rows are gathered in batch order, whereas the
            # allocator's ``num_blocks`` uses request-state indices.
            # Keep the complete batch mapping because ``table_indices`` still
            # refer to the original input-batch rows after filtering requests.
            setter(block_tables, request_indices=all_request_indices)

        validation_start = None
        if self._via_sd_timing_enabled():
            # NPU execution is asynchronous. Synchronizing at both boundaries
            # makes this a real per-validation latency rather than enqueue time.
            torch.npu.synchronize()
            validation_start = time.perf_counter()
        self.via_sd_last_logits = self.via_sd_verifier.verify(
            draft_tokens,
            request_ids=request_ids,
            request_indices=request_indices,
            prefix_token_ids=prefixes,
            valid_lengths=valid_lengths,
            table_indices=table_indices,
        )
        self._via_sd_qprime_validation_count += 1
        stats = self.via_sd_verifier.last_stats
        elapsed_ms = None
        if validation_start is not None:
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - validation_start) * 1000.0
        if self._via_sd_logging_enabled():
            logger.info(
                "[VIA-SD] q' validation #%d: elapsed_ms=%s, cache=%s, "
                "requests=%d, scored_requests=%d, draft_tokens=%d, "
                "cache_hits=%d, cache_misses=%d, cached_prefix_tokens=%d, "
                "recomputed_prefix_tokens=%d, model_input_tokens=%d, layers=%d, "
                "request_ids=%s, draft_token_ids=%s, draft_positions=%s, logits_shape=%s",
                self._via_sd_qprime_validation_count,
                "disabled" if elapsed_ms is None else f"{elapsed_ms:.3f}",
                "on" if stats.cache_enabled else "off",
                stats.request_count,
                stats.scored_request_count,
                stats.draft_tokens,
                stats.cache_hits,
                stats.cache_misses,
                stats.cached_prefix_tokens,
                stats.recomputed_prefix_tokens,
                stats.model_input_tokens,
                len(self.via_sd_model.layer_ids) if self.via_sd_model is not None else 0,
                stats.request_ids,
                stats.draft_token_ids,
                stats.positions,
                tuple(self.via_sd_last_logits.shape),
            )
        config = self._via_sd_config()
        if config is None or self._via_sd_mode() == 'observe':
            return None
        # Keep the original InputBatch row in the plan.  The q' verifier may
        # compact requests with ragged prefixes, but target fallback must be
        # scattered back through this stable row mapping.
        plan = self.via_sd_verifier.build_route_plan(
            self.via_sd_last_logits,
            draft_tokens,
            request_ids=request_ids,
            valid_lengths=valid_lengths,
            batch_rows=table_indices,
        )
        self._via_sd_route_plan = plan
        return plan

    def get_via_sd_route_plan(self) -> ViaSdRoutePlan | None:
        return self._via_sd_route_plan

    def get_via_sd_target_fallback_rows(self) -> tuple[int, ...]:
        plan = self._via_sd_route_plan
        return () if plan is None else plan.fallback_rows

    def execute_via_sd_hierarchical(
        self,
        request_ids,
        prefix_token_ids,
        draft_token_ids,
        qprime_logits,
        **kwargs,
    ):
        """Execute the device-independent route coordinator for a batch.

        Production integrations can pass target catch-up/verification callbacks
        through ``kwargs``.  Keeping this public runner seam small also gives
        tests and profiling tools a way to exercise mixed-row routing without
        constructing a full scheduler output.
        """

        config = self._via_sd_config()
        accept = 0.7 if config is None else getattr(config, "accept_ratio", 0.7)
        escalate = 0.5 if config is None else getattr(config, "escalate_ratio", 0.5)
        mode = self._via_sd_mode()
        if self._via_sd_coordinator is None or (
            self._via_sd_coordinator.accept_ratio != float(accept)
            or self._via_sd_coordinator.escalate_ratio != float(escalate)
            or self._via_sd_coordinator.mode != mode
        ):
            self._via_sd_coordinator = ViaSdExecutionCoordinator(
                accept,
                escalate,
                mode=mode,
            )
        result = self._via_sd_coordinator.run(
            request_ids,
            prefix_token_ids,
            draft_token_ids,
            qprime_logits,
            **kwargs,
        )
        self._via_sd_execution_result = result
        self._via_sd_route_plan = result.plan
        return result

    def _disable_via_sd(self, reason: str) -> None:
        """Disable only q'; target sampling and rejection remain untouched."""

        if self._via_sd_disabled:
            return
        self._via_sd_disabled = True
        self._via_sd_pending_target_drafts = {}
        self._via_sd_route_plan = None
        self._via_sd_execution_result = None
        self._via_sd_pre_route_batch = None
        self._via_sd_pre_route_scheduler_output = None
        if self._via_sd_coordinator is not None:
            self._via_sd_coordinator.clear()
        if self.via_sd_verifier is not None:
            self.via_sd_verifier.clear()
        if self.via_sd_model is not None:
            self.via_sd_model.unregister_attention_layers(self.vllm_config)
        self.via_sd_verifier = None
        self.via_sd_model = None
        if not self._via_sd_error_reported:
            logger.warning(
                "VIA-SD q' side pass disabled; target verification continues unchanged: %s",
                reason,
            )
            self._via_sd_error_reported = True

    def get_via_sd_last_logits(self):
        """Return the latest ``[batch, draft_steps, vocab]`` q' logits tensor."""

        return self.via_sd_last_logits

    def _discard_via_sd_requests(self, scheduler_output: SchedulerOutput) -> None:
        request_ids = set(getattr(scheduler_output, "finished_req_ids", None) or ())
        request_ids.update(getattr(scheduler_output, "preempted_req_ids", None) or ())
        if request_ids and self.via_sd_verifier is not None:
            self.via_sd_verifier.discard(tuple(request_ids))
        if request_ids and self._via_sd_coordinator is not None:
            self._via_sd_coordinator.discard(tuple(request_ids))
        if request_ids:
            self._via_sd_route_plan = None
            self._via_sd_execution_result = None

    def shutdown(self) -> None:
        self._via_sd_runtime = None
        # Remove q' registrations before the parent clears target model state
        # and the shared static forward context.
        if self.via_sd_verifier is not None:
            self.via_sd_verifier.clear()
        if self.via_sd_model is not None:
            self.via_sd_model.unregister_attention_layers(self.vllm_config)
        self.via_sd_verifier = None
        self.via_sd_model = None
        self.via_sd_last_logits = None
        self._via_sd_pending_target_drafts = {}
        self._via_sd_route_plan = None
        self._via_sd_execution_result = None
        self._via_sd_pre_route_batch = None
        self._via_sd_pre_route_scheduler_output = None
        if self._via_sd_coordinator is not None:
            self._via_sd_coordinator.clear()
        super().shutdown()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        with graph_manager_wrapper(self):
            super().initialize_kv_cache(kv_cache_config)
            if self.pcp_manager is not None:
                assert isinstance(self.pcp_manager, AscendPCPManager)
                self.pcp_manager.vllm_config = self.vllm_config
                self.model_state.pcp_manager = self.pcp_manager
        if self.via_sd_model is not None and not self._via_sd_disabled:
            try:
                config = self._via_sd_config()
                assert config is not None
                self.via_sd_verifier = ViaSdVerifier(
                    self,
                    self.via_sd_model,
                    config,
                )
            except Exception as exc:
                self._disable_via_sd(str(exc))
        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
    ):
        if self._via_sd_mode() == 'hierarchical' and not dummy_run and not is_profile:
            self._cpp_execution_time_ms = None
            self._via_sd_route_plan = None
            self._via_sd_execution_result = None
            from vllm_ascend.worker.v2.spec_decode.via_sd.runtime import ViaSdRuntime

            if getattr(self, '_via_sd_runtime', None) is None:
                self._via_sd_runtime = ViaSdRuntime(self)
            return self._via_sd_runtime.execute(scheduler_output)
        self._cpp_execution_time_ms = None
        if not dummy_run:
            self._discard_via_sd_requests(scheduler_output)

        scheduled_drafts = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
        real_scheduled_drafts = self._via_sd_real_drafts(scheduled_drafts)
        target_request_ids = tuple(scheduled_drafts.keys())

        # A route plan belongs to exactly one prepared batch.  Clear any
        # previous plan before lifecycle/update calls can reorder request rows.
        self._via_sd_route_plan = None
        self._via_sd_execution_result = None
        self._via_sd_pre_route_batch = None
        self._via_sd_pre_route_scheduler_output = None

        target_draft_tokens = sum(
            len(tokens)
            for tokens in scheduled_drafts.values()
        )
        if not dummy_run and not is_profile:
            self._via_sd_pending_target_drafts = real_scheduled_drafts
        else:
            self._via_sd_pending_target_drafts = {}
        profiling_config = self.ascend_config.scheduler_config.profiling_chunk_config
        execution_start_time = _start_profiling_chunk_timing(
            profiling_config,
            scheduler_output,
        )
        # This measures parent preparation and execution, not pure forward.
        # Observe-mode q' runs later in sample_tokens, outside this interval.
        target_validation_start = None
        if (
            self._via_sd_timing_enabled()
            and not dummy_run
            and not is_profile
            and target_draft_tokens > 0
        ):
            torch.npu.synchronize()
            target_validation_start = time.perf_counter()

        if vllm_version_is("0.27.1"):
            output = super().execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
            )
        else:
            output = super().execute_model(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
                context_len=context_len,
            )

        elapsed_ms = None
        if target_validation_start is not None:
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - target_validation_start) * 1000.0
        if target_draft_tokens > 0 and not dummy_run and not is_profile:
            self._via_sd_target_validation_count += 1
            if self._via_sd_logging_enabled():
                logger.info(
                    "[VIA-SD] target execute #%d: elapsed_ms=%s, "
                    "requests=%d, draft_tokens=%d, request_ids=%s, "
                    "scope=parent_execute_model",
                    self._via_sd_target_validation_count,
                    "disabled" if elapsed_ms is None else f"{elapsed_ms:.3f}",
                    len(target_request_ids),
                    target_draft_tokens,
                    target_request_ids,
                )

        self._cpp_execution_time_ms = _finish_profiling_chunk_timing(
            profiling_config,
            execution_start_time,
        )
        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Override GPUModelRunner.profile_run for Ascend NPUs.
        When running moe models, we need an extra dummy run with mc2_tokens_capacity tokens to reserve
        necessary HCCL buffer for the MC2 operator before standard `profile_run`. Additionally, we set
        override_mrv2_in_profile_run to True to force moe load to be balanced when executing `profile_run`
        """
        mc2_tokens_capacity = get_mc2_tokens_capacity()
        with override_mrv2_in_profile_run(True):
            if (
                mc2_tokens_capacity is not None
                and self.max_num_tokens > mc2_tokens_capacity
                and select_moe_comm_method(mc2_tokens_capacity, self.vllm_config)
                in {MoECommType.MC2, MoECommType.FUSED_MC2}
            ):
                self._dummy_run(mc2_tokens_capacity, skip_attn=True, skip_eplb=True, is_profile=True)
            super().profile_run()

    if vllm_version_is("0.27.1"):

        def prepare_inputs(
            self,
            scheduler_output: SchedulerOutput,
            batch_desc: BatchExecutionDescriptor,
        ) -> AscendInputBatch:
            """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
            npu attention backends need seq_lens_cpu to work.
            so we need to prepare seq_lens_cpu here.
            """
            num_tokens = scheduler_output.total_num_scheduled_tokens
            num_tokens_after_padding = batch_desc.num_tokens
            assert num_tokens > 0
            num_tokens_per_req = scheduler_output.num_scheduled_tokens
            num_reqs = len(num_tokens_per_req)

            req_ids = sort_batch_req_ids(num_tokens_per_req, self.decode_query_len)

            self._update_seq_lens_cpu(scheduler_output, req_ids)

            numtoks_iter = map(num_tokens_per_req.get, req_ids)
            num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)
            num_valid_tokens = num_scheduled_tokens
            if scheduler_output.scheduled_spec_decode_tokens:
                num_valid_tokens = np.array(
                    [
                        num_tokens - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                        for num_tokens, i in zip(num_scheduled_tokens, req_ids)
                    ],
                    dtype=np.int32,
                )
            attn_state = build_attn_state(
                self.vllm_config,
                self.input_buffers.seq_lens_np,
                num_reqs,
                num_scheduled_tokens,
                num_valid_tokens,
            )
            idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
            idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
            idx_mapping_cpu = torch.from_numpy(idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_cpu, device=self.device)

            # Get the number of draft tokens for each request.
            draft_tokens = scheduler_output.scheduled_spec_decode_tokens
            num_draft_tokens_per_req = None
            if not draft_tokens:
                # No draft token scheduled (common case).
                total_num_draft_tokens = 0
                total_num_logits = num_reqs
                cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
                cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
                expanded_idx_mapping = idx_mapping
                expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
            else:
                num_draft_tokens_per_req = np.fromiter(
                    (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                    dtype=np.int32,
                    count=num_reqs,
                )
                num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
                total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
                total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
                num_logits = num_draft_tokens_per_req + num_bonus_tokens
                cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
                cu_num_logits_np[0] = 0
                np.cumsum(num_logits, out=cu_num_logits_np[1:])
                cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

                max_expand_len = self.decode_query_len
                expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                    idx_mapping, total_num_logits, cu_num_logits, max_expand_len
                )

            # Get query_start_loc.
            # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
            # See _pad_query_start_loc_for_fia.
            num_reqs_padded = batch_desc.num_reqs or num_reqs
            query_start_loc_np = np.empty(self.max_num_reqs + 2, dtype=np.int32)
            query_start_loc_np[0] = 0
            np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
            # Pad for full CUDA graph mode.
            # Some attention backends like FA3 require query_start_loc to be non-decreasing.
            query_start_loc_np[num_reqs + 1 :] = num_tokens

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                # This is only required for vllm-ascend.
                query_start_loc_np, num_reqs_padded = self._pad_query_start_loc_for_fia(
                    num_tokens_after_padding,
                    num_reqs_padded,
                    num_reqs,
                    query_start_loc_np,
                    batch_desc.cg_mode,
                    batch_desc.num_reqs,
                )

            async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)

            query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
            query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]
            prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
            num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[idx_mapping_np]
            is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np
            batch_has_prefill = bool(np.any(is_prefilling_np))
            self.eplb.set_batch_phase(batch_has_prefill)

            # Get prefill tokens if any.
            if batch_has_prefill:
                prepare_prefill_inputs(
                    self.input_buffers.input_ids,
                    self.req_states.next_prefill_tokens,
                    idx_mapping,
                    query_start_loc,
                    self.req_states.all_token_ids.gpu,
                    self.req_states.prefill_len.gpu,
                    self.req_states.num_computed_tokens.gpu,
                )

            # Prepare positions and seq_lens.
            prepare_pos_seq_lens(
                idx_mapping,
                query_start_loc,
                self.req_states.num_computed_tokens.gpu,
                self.input_buffers.positions,
                self.input_buffers.seq_lens,
            )
            seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

            # Pad for full CUDA graph mode.
            self.input_buffers.seq_lens_np[num_reqs_padded:] = 0

            # Some input token ids are directly read from the last sampled tokens
            # and draft tokens. Also, get the logits indices to sample tokens from.
            logits_indices = combine_sampled_and_draft_tokens(
                self.input_buffers.input_ids,
                idx_mapping,
                self.req_states.last_sampled_tokens,
                query_start_loc,
                seq_lens,
                self.req_states.prefill_len.gpu,
                self.req_states.draft_tokens,
                cu_num_logits,
                total_num_logits,
                self.model_state.num_new_sampled_tokens_per_step,
            )

            # CPU upper bound on seq_lens (num_computed_tokens + num_scheduled_tokens).
            # Added by vLLM PR #40654 to avoid GPU->CPU sync for seq_lens.
            seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
            np.add(
                self.req_states.num_computed_tokens_np[idx_mapping_np],
                num_scheduled_tokens,
                out=seq_lens_cpu_upper_bound_np[:num_reqs],
            )
            seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)
            num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]

            max_seq_len_np = None
            if self.use_pp:
                # max_seq_len is only consumed by the PP `compute_need_sampled_mask`
                max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

            prompt_lens = None
            if self.model_config.rswa_window is not None:
                # prompt_lens is only used in R-SWA case.
                prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

            input_batch = AscendInputBatch(
                req_ids=req_ids,
                num_reqs=num_reqs,
                num_reqs_after_padding=num_reqs_padded,
                idx_mapping=idx_mapping,
                idx_mapping_np=idx_mapping_np,
                expanded_idx_mapping=expanded_idx_mapping,
                expanded_local_pos=expanded_local_pos,
                num_scheduled_tokens=num_scheduled_tokens,
                num_tokens=num_tokens,
                num_tokens_after_padding=num_tokens_after_padding,
                num_draft_tokens=total_num_draft_tokens,
                num_draft_tokens_per_req=num_draft_tokens_per_req,
                query_start_loc=query_start_loc,
                query_start_loc_np=query_start_loc_np,
                seq_lens=seq_lens,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                dcp_local_seq_lens=None,  # TODO(Ronald1995): support cp.
                is_prefilling_np=is_prefilling_np,
                num_computed_tokens_np=num_computed_tokens_np,
                prefill_len_np=prefill_len_np,
                num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
                max_seq_len_np=max_seq_len_np,
                input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
                positions=self.input_buffers.positions[:num_tokens_after_padding],
                is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
                logits_indices=logits_indices,
                cu_num_logits=cu_num_logits,
                cu_num_logits_np=cu_num_logits_np,
                has_structured_output_reqs=scheduler_output.has_structured_output_requests,
                # TODO: only populated for R-SWA (not supported yet).
                prompt_lens=prompt_lens,
                # extra attributes for ascend npus.
                seq_lens_np=self.input_buffers.seq_lens_np,
                attn_state=attn_state,
            )

            input_batch = vllm_model_runner.pcp.maybe_partition_pcp_batch(self.pcp_manager, input_batch)

            # For mla/sfa, update cos/sin. Here is for execute_model.
            update_cos_sin(input_batch.positions)

            return input_batch

    else:

        def prepare_inputs(  # type: ignore[misc]
            self,
            scheduler_output: SchedulerOutput,
            batch_req_state: BatchReqState,
            batch_desc: BatchExecutionDescriptor,
        ) -> AscendInputBatch:
            """Override GPUModelRunner.prepare_inputs for Ascend NPUs.
            npu attention backends need seq_lens_cpu to work.
            so we need to prepare seq_lens_cpu here.
            """
            num_tokens = scheduler_output.total_num_scheduled_tokens
            num_tokens_after_padding = batch_desc.num_tokens
            assert num_tokens > 0
            num_tokens_per_req = scheduler_output.num_scheduled_tokens
            num_reqs = len(num_tokens_per_req)

            req_ids = sort_batch_req_ids(
                num_tokens_per_req,
                scheduler_output.scheduled_spec_decode_tokens,
                self.decode_query_len,
            )

            self._update_seq_lens_cpu(scheduler_output, req_ids)

            numtoks_iter = map(num_tokens_per_req.get, req_ids)
            num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)
            num_valid_tokens = num_scheduled_tokens
            if scheduler_output.scheduled_spec_decode_tokens:
                num_valid_tokens = np.array(
                    [
                        num_tokens - len(scheduler_output.scheduled_spec_decode_tokens.get(i, []))
                        for num_tokens, i in zip(num_scheduled_tokens, req_ids)
                    ],
                    dtype=np.int32,
                )
            attn_state = build_attn_state(
                self.vllm_config,
                self.input_buffers.seq_lens_np,
                num_reqs,
                num_scheduled_tokens,
                num_valid_tokens,
            )
            idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
            idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
            idx_mapping_cpu = torch.from_numpy(idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_cpu, device=self.device)

            # Get the number of draft tokens for each request.
            draft_tokens = scheduler_output.scheduled_spec_decode_tokens
            num_draft_tokens_per_req = None
            if not draft_tokens:
                # No draft token scheduled (common case).
                total_num_draft_tokens = 0
                total_num_logits = num_reqs
                cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
                cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
                expanded_idx_mapping = idx_mapping
                expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
            else:
                num_draft_tokens_per_req = np.fromiter(
                    (len(draft_tokens.get(req_id, ())) for req_id in req_ids),
                    dtype=np.int32,
                    count=num_reqs,
                )
                if self._via_sd_enabled():
                    real_drafts = {}

                    for batch_row, req_id in enumerate(req_ids):
                        num_drafts = int(
                            num_draft_tokens_per_req[batch_row]
                        )

                        if num_drafts <= 0:
                            continue

                        state_index = int(
                            idx_mapping_np[batch_row]
                        )

                        tokens = (
                            self.req_states.draft_tokens[
                                state_index, :num_drafts
                            ]
                            .detach()
                            .cpu()
                            .tolist()
                        )

                        real_drafts[req_id] = tuple(
                            int(token) for token in tokens
                        )

                    self._via_sd_pending_target_drafts = real_drafts

                    # logger.warning(
                    #     "[VIA DEBUG] real worker drafts1=%s",
                    #     real_drafts,
                    # )
                # ===== VIA-SD end =====
                num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
                total_num_draft_tokens = int(num_draft_tokens_per_req.sum())
                total_num_logits = num_reqs * num_bonus_tokens + total_num_draft_tokens
                num_logits = num_draft_tokens_per_req + num_bonus_tokens
                cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
                cu_num_logits_np[0] = 0
                np.cumsum(num_logits, out=cu_num_logits_np[1:])
                cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

                max_expand_len = self.decode_query_len
                expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                    idx_mapping, total_num_logits, cu_num_logits, max_expand_len
                )

            # Get query_start_loc.
            # NOTE: For FULL mode we change +1 to +2 to reserve extra space for padding.
            # See _pad_query_start_loc_for_fia.
            num_reqs_padded = batch_desc.num_reqs or num_reqs
            query_start_loc_np = np.empty(self.max_num_reqs + 2, dtype=np.int32)
            query_start_loc_np[0] = 0
            np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
            # Pad for full CUDA graph mode.
            # Some attention backends like FA3 require query_start_loc to be non-decreasing.
            query_start_loc_np[num_reqs + 1 :] = num_tokens

            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                # This is only required for vllm-ascend.
                query_start_loc_np, num_reqs_padded = self._pad_query_start_loc_for_fia(
                    num_tokens_after_padding,
                    num_reqs_padded,
                    num_reqs,
                    query_start_loc_np,
                    batch_desc.cg_mode,
                    batch_desc.num_reqs,
                )

            async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)

            query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
            query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]
            prefill_len_np = self.req_states.prefill_len.np[idx_mapping_np]
            num_computed_prefill_tokens_np = self.req_states.num_computed_prefill_tokens[idx_mapping_np]
            is_prefilling_np = num_computed_prefill_tokens_np < prefill_len_np
            batch_has_prefill = bool(np.any(is_prefilling_np))
            self.eplb.set_batch_phase(batch_has_prefill)

            # Get prefill tokens if any.
            if batch_has_prefill:
                prepare_prefill_inputs(
                    self.input_buffers.input_ids,
                    self.req_states.next_prefill_tokens,
                    idx_mapping,
                    query_start_loc,
                    self.req_states.all_token_ids.gpu,
                    self.req_states.prefill_len.gpu,
                    self.req_states.num_computed_tokens.gpu,
                )

            # Prepare positions and seq_lens.
            prepare_pos_seq_lens(
                idx_mapping,
                query_start_loc,
                self.req_states.num_computed_tokens.gpu,
                self.input_buffers.positions,
                self.input_buffers.seq_lens,
            )
            seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]

            # Pad for full CUDA graph mode.
            self.input_buffers.seq_lens_np[num_reqs_padded:] = 0

            # Some input token ids are directly read from the last sampled tokens
            # and draft tokens. Also, get the logits indices to sample tokens from.
            logits_indices = combine_sampled_and_draft_tokens(
                self.input_buffers.input_ids,
                idx_mapping,
                self.req_states.last_sampled_tokens,
                query_start_loc,
                seq_lens,
                self.req_states.prefill_len.gpu,
                self.req_states.draft_tokens,
                cu_num_logits,
                total_num_logits,
                self.model_state.num_new_sampled_tokens_per_step,
            )

            # CPU upper bound on seq_lens (num_computed_tokens + num_scheduled_tokens).
            # Added by vLLM PR #40654 to avoid GPU->CPU sync for seq_lens.
            seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
            np.add(
                self.req_states.num_computed_tokens_np[idx_mapping_np],
                num_scheduled_tokens,
                out=seq_lens_cpu_upper_bound_np[:num_reqs],
            )
            seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)
            num_computed_tokens_np = self.req_states.num_computed_tokens_np[idx_mapping_np]

            max_seq_len_np = None
            if self.use_pp:
                # max_seq_len is only consumed by the PP `compute_need_sampled_mask`
                max_seq_len_np = self.req_states.max_seq_len[idx_mapping_np]

            prompt_lens = None
            if self.model_config.rswa_window is not None:
                # prompt_lens is only used in R-SWA case.
                prompt_lens = self.req_states.prompt_len.gpu[idx_mapping]

            input_batch = AscendInputBatch(
                req_ids=req_ids,
                num_reqs=num_reqs,
                num_reqs_after_padding=num_reqs_padded,
                idx_mapping=idx_mapping,
                idx_mapping_np=idx_mapping_np,
                expanded_idx_mapping=expanded_idx_mapping,
                expanded_local_pos=expanded_local_pos,
                num_scheduled_tokens=num_scheduled_tokens,
                num_tokens=num_tokens,
                num_tokens_after_padding=num_tokens_after_padding,
                num_draft_tokens=total_num_draft_tokens,
                num_draft_tokens_per_req=num_draft_tokens_per_req,
                query_start_loc=query_start_loc,
                query_start_loc_np=query_start_loc_np,
                seq_lens=seq_lens,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                dcp_local_seq_lens=None,  # TODO(Ronald1995): support cp.
                is_prefilling_np=is_prefilling_np,
                has_prefill=batch_has_prefill,
                num_computed_tokens_np=num_computed_tokens_np,
                prefill_len_np=prefill_len_np,
                num_computed_prefill_tokens_np=num_computed_prefill_tokens_np,
                max_seq_len_np=max_seq_len_np,
                input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
                positions=self.input_buffers.positions[:num_tokens_after_padding],
                is_padding=self.input_buffers.is_padding[:num_tokens_after_padding],
                logits_indices=logits_indices,
                cu_num_logits=cu_num_logits,
                cu_num_logits_np=cu_num_logits_np,
                has_structured_output_reqs=scheduler_output.has_structured_output_requests,
                # TODO: only populated for R-SWA (not supported yet).
                prompt_lens=prompt_lens,
                # extra attributes for ascend npus.
                seq_lens_np=self.input_buffers.seq_lens_np,
                attn_state=attn_state,
            )

            input_batch = vllm_model_runner.pcp.maybe_partition_pcp_batch(self.pcp_manager, input_batch)

            # For mla/sfa, update cos/sin. Here is for execute_model.
            update_cos_sin(input_batch.positions)

            return input_batch

    def postprocess_sampled(
        self,
        idx_mapping,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc=None,
    ):
        """Override GPUModelRunner.postprocess_sampled for Ascend NPUs.
        npu attention backends need seq_lens_cpu to work.
        so we need to copy num_computed_tokens back to cpu here.
        """
        super().postprocess_sampled(
            idx_mapping,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
        )

        # Skip D2H copy without MTP: num_computed_tokens_cpu is synced
        # from num_computed_tokens_np in _update_seq_lens_cpu instead.
        if self.speculator is not None:
            self._copy_num_computed_tokens_to_cpu()

    def _copy_num_computed_tokens_to_cpu(self):
        # npu attention backend still need to use seq_lens_cpu,
        # we need to copy num_computed_tokens back to cpu.
        default_stream = torch.cuda.current_stream()
        assert self.num_computed_tokens_stream is not None
        assert self.num_computed_tokens_cpu is not None
        with torch.npu.stream(self.num_computed_tokens_stream):
            self.num_computed_tokens_stream.wait_stream(default_stream)
            self.num_computed_tokens_cpu.copy_(
                self.req_states.num_computed_tokens.gpu,
                non_blocking=True,
            )
            self.num_computed_tokens_event.record()

    def _update_seq_lens_cpu(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: list[str],
    ):
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        # MTP needs D2H copy to get reverted num_computed_tokens after rejection.
        # Without MTP, num_computed_tokens_np is already correct from update_requests.
        if self.speculator is not None:
            self.num_computed_tokens_event.synchronize()
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.num_computed_tokens_cpu[req_index]
        else:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                req_index = self.req_states.req_id_to_index[req_id]
                self.req_states.num_computed_tokens_cpu[req_index] = self.req_states.num_computed_tokens_np[req_index]

        # update seq_lens_cpu
        for i, req_id in enumerate(req_ids):  # type: ignore
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens = self.req_states.num_computed_tokens_cpu[req_index]
            self.input_buffers.seq_lens_cpu[i] = num_computed_tokens + num_scheduled_tokens[req_id]

    def _pad_query_start_loc_for_fia(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_reqs: int,
        query_start_loc_np: np.ndarray,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        batch_desc_num_reqs: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        # TODO: need refactor later, related to vllm PR #34043 this pr delete func
        # relax_for_mixed_batch_cudagraphs, num_reqs no longer equals the actual number of requests.
        if (
            cudagraph_runtime_mode == CUDAGraphMode.FULL
            and self.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        ):
            num_reqs_padded = num_reqs
        else:
            num_reqs_padded = batch_desc_num_reqs if batch_desc_num_reqs is not None else num_reqs

        if num_tokens_padded == num_reqs_padded * self.decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc_np[num_reqs]
            query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
                np.arange(1, num_reqs_padded + 1 - num_reqs) * self.decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            query_start_loc_np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1

        return query_start_loc_np, num_reqs_padded


@contextmanager
def graph_manager_wrapper(model_runner):
    """Context manager to override graph manager."""
    original_graph_manager = vllm_model_runner.ModelCudaGraphManager

    if vllm_version_is("0.27.1"):

        def factory(
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            lora_capture_cases: list[int] | None = None,
        ):
            return ModelAclGraphManager(
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                model_runner,
                lora_capture_cases=lora_capture_cases,
            )

    else:

        def factory(  # type: ignore[misc]
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            lora_capture_cases: list[int] | None = None,
            varlen_decode: bool = False,
        ):
            return ModelAclGraphManager(
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                model_runner,
                lora_capture_cases=lora_capture_cases,
                varlen_decode=varlen_decode,  # type: ignore[call-arg]
            )

    try:
        vllm_model_runner.ModelCudaGraphManager = factory
        yield
    finally:
        vllm_model_runner.ModelCudaGraphManager = original_graph_manager
