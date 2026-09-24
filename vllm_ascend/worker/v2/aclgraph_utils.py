# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/aclgraph_utils.py
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
from collections.abc import Callable
from contextlib import contextmanager
from inspect import signature
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.spec_decode import MAX_DECODE_QUERY_LEN, uses_long_speculative_queries
from vllm_ascend.compilation.acl_graph import set_graph_params, update_full_graph_params
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.utils import communicator_switch

MAX_LONG_VERIFICATION_CAPTURE_TOKENS = 128


def collect_sorted_captured_token_sizes(capture_descs: dict) -> list[int]:
    """Collect the actual per-graph token counts that will be captured.

    With speculative decoding under FULL_DECODE_ONLY, each raw
    ``cudagraph_capture_size`` is rounded up to a multiple of
    ``decode_query_len`` (see ``CudaGraphManager._init_candidates``), so the
    real graph sizes differ from ``compilation_config.cudagraph_capture_sizes``.
    The attention backend keys its per-size graph params (events/handles/...)
    by these rounded token counts, so they must be derived from the actual
    capture descriptors, not the raw config sizes.
    """
    return sorted({desc.num_tokens for descs in capture_descs.values() for desc in descs})


def target_graph_mode(vllm_config, cudagraph_mode):
    return cudagraph_mode


def long_verification_capture_shapes(vllm_config):
    """Return TP-aligned FULL graph shapes for long Target verification.

    Long verification is cached-prefill rather than ordinary decode, so the
    regular ``cudagraph_capture_sizes`` do not cover it when the query is
    wider than ``MAX_DECODE_QUERY_LEN``. Keep this list sparse: one graph per
    resident request count is enough because runtime queries are padded to a
    compatible token bucket.
    """
    compilation = vllm_config.compilation_config
    if (
        getattr(vllm_config.model_config, "enforce_eager", False)
        or compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
        or not uses_long_speculative_queries(vllm_config)
    ):
        return []

    query_width = vllm_config.speculative_config.num_speculative_tokens + 1
    if query_width <= MAX_DECODE_QUERY_LEN:
        return []

    capture_sizes = compilation.cudagraph_capture_sizes or ()
    capture_limit = compilation.max_cudagraph_capture_size
    if not capture_limit:
        capture_limit = max(capture_sizes, default=0)
    if not capture_limit:
        return []
    scheduler_limit = vllm_config.scheduler_config.max_num_batched_tokens
    # Long verification uses only the target's existing capture budget. Do not
    # synthesize a larger graph for a ragged batch: unsupported long batches
    # must follow the normal eager fallback path.
    capture_limit = min(capture_limit, scheduler_limit, MAX_LONG_VERIFICATION_CAPTURE_TOKENS)
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    scheduler_limit = scheduler_limit // tp_size * tp_size
    capture_limit = min(capture_limit // tp_size * tp_size, scheduler_limit)
    max_tokens = capture_limit
    if max_tokens <= MAX_DECODE_QUERY_LEN:
        return []

    shapes = []
    for num_reqs in range(1, vllm_config.scheduler_config.max_num_seqs + 1):
        requested_tokens = (num_reqs * query_width + tp_size - 1) // tp_size * tp_size
        if requested_tokens > max_tokens:
            break
        shapes.append((num_reqs, requested_tokens))
    return shapes


def select_long_verification_graph(candidates, num_reqs, num_tokens, num_active_loras):
    for desc in candidates:
        if (
            desc.num_tokens >= num_tokens
            and desc.num_reqs == num_reqs
            and desc.num_active_loras == num_active_loras
        ):
            return desc
    return None


def eager_execution_descriptor(desc, num_reqs, num_tokens, uniform_token_count, max_query_len):
    """Clone a dispatch descriptor while disabling graph execution."""
    parameters = signature(BatchExecutionDescriptor).parameters
    values = {name: getattr(desc, name) for name in parameters if hasattr(desc, name)}
    values.update(
        cg_mode=CUDAGraphMode.NONE,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        uniform_token_count=uniform_token_count,
        max_query_len=max_query_len,
    )
    return BatchExecutionDescriptor(**values)


def _get_graph_update_backend(
    attn_groups: list[list[AttentionGroup]],
) -> type[AttentionBackend]:
    for groups in attn_groups:
        for group in groups:
            backend = group.backend
            if backend.get_impl_cls() is not None:
                return backend
    raise RuntimeError("No executable attention backend is available for full-graph parameter updates.")


class ModelAclGraphManager(ModelCudaGraphManager):
    """ACL Model Cuda Graph Manager for Ascend NPUs."""

    # FULL FIA graphs refresh lengths and page tables through task updates.
    # DFlash retains its own independent full decode manager.

    if vllm_version_is("0.27.1"):

        def __init__(
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            model_runner: Any,
            lora_capture_cases: list[int] | None = None,
        ):
            super().__init__(
                vllm_config,
                device,
                target_graph_mode(vllm_config, cudagraph_mode),
                decode_query_len,
                lora_capture_cases=lora_capture_cases,
            )
            self.model_runner = model_runner
            self.update_stream = self.model_runner.update_stream
            self._dispatch_parameters = signature(super().dispatch).parameters
            self.long_verification_graphs = self._add_long_verification_graphs(vllm_config)
            self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
            if super().needs_capture():
                set_graph_params(self.capture_sizes)

    else:

        def __init__(  # type: ignore[misc]
            self,
            vllm_config: VllmConfig,
            device: torch.device,
            cudagraph_mode: CUDAGraphMode,
            decode_query_len: int,
            model_runner: Any,
            lora_capture_cases: list[int] | None = None,
            varlen_decode: bool = False,
        ):
            super().__init__(
                vllm_config,
                device,
                target_graph_mode(vllm_config, cudagraph_mode),
                decode_query_len,
                lora_capture_cases=lora_capture_cases,
                varlen_decode=varlen_decode,
            )
            self.breakable_cg_runner: BreakableACLGraphWrapper | None = None
            self.model_runner = model_runner
            self.update_stream = self.model_runner.update_stream
            self._dispatch_parameters = signature(super().dispatch).parameters
            self.long_verification_graphs = self._add_long_verification_graphs(vllm_config)
            self.capture_sizes = collect_sorted_captured_token_sizes(self._capture_descs)
            if super().needs_capture():
                set_graph_params(self.capture_sizes)

    def _add_long_verification_graphs(self, vllm_config):
        shapes = long_verification_capture_shapes(vllm_config)
        descriptor_parameters = signature(BatchExecutionDescriptor).parameters
        descs = []
        for num_reqs, num_tokens in shapes:
            values = {
                "cg_mode": CUDAGraphMode.FULL,
                "num_tokens": num_tokens,
                "num_reqs": num_reqs,
                "uniform_token_count": None,
                "max_query_len": vllm_config.speculative_config.num_speculative_tokens + 1,
            }
            descs.append(
                BatchExecutionDescriptor(
                    **{key: value for key, value in values.items() if key in descriptor_parameters}
                )
            )
        if descs:
            full_descs = self._capture_descs.setdefault(CUDAGraphMode.FULL, [])
            full_descs.extend(descs)
            full_descs.sort(key=lambda desc: desc.num_tokens, reverse=True)
        return tuple(reversed(descs))

    def dispatch(
        self,
        num_reqs,
        num_tokens,
        uniform_token_count,
        num_active_loras,
        max_query_len=None,
        num_ubatches=1,
    ):
        dispatch_kwargs = {}
        if "max_query_len" in self._dispatch_parameters:
            dispatch_kwargs["max_query_len"] = max_query_len
        if "num_ubatches" in self._dispatch_parameters:
            dispatch_kwargs["num_ubatches"] = num_ubatches
        desc = super().dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
            **dispatch_kwargs,
        )
        long_verification = (
            getattr(self, "long_verification_active", False)
            and max_query_len is not None
            and max_query_len > MAX_DECODE_QUERY_LEN
            and bool(self.long_verification_graphs)
        )
        if not long_verification:
            return desc

        effective_loras = self._resolve_effective_loras(num_active_loras)
        graph_desc = select_long_verification_graph(
            self.long_verification_graphs,
            num_reqs,
            num_tokens,
            effective_loras,
        )
        if graph_desc is None:
            logger.warning_once(
                "Long Target verification shape is outside captured ACL graph buckets; "
                "falling back to eager: num_reqs=%s num_tokens=%s max_query_len=%s captured_shapes=%s",
                num_reqs,
                num_tokens,
                max_query_len,
                [(getattr(item, "num_reqs", None), item.num_tokens) for item in self.long_verification_graphs],
            )
            if desc.cg_mode == CUDAGraphMode.NONE:
                return desc
            return eager_execution_descriptor(desc, num_reqs, num_tokens, uniform_token_count, max_query_len)
        return graph_desc

    def init_breakable_cg_runner(self, model: nn.Module) -> None:
        if self.breakable_cg_runner is None:
            self.breakable_cg_runner = BreakableACLGraphWrapper(model, self.vllm_config)

    def run_fullgraph(self, desc: BatchExecutionDescriptor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Override run_fullgraph to update full graph params in run_fullgraph."""
        num_tokens = desc.num_tokens
        logger.info_once("run_fullgraph with num_tokens=%s", num_tokens)
        assert self.update_stream is not None
        current_stream = torch.npu.current_stream()
        self.update_stream.wait_stream(current_stream)
        logger.debug("Target FULL graph %d: replay begin", num_tokens)
        ret = super().run_fullgraph(desc)
        logger.debug("Target FULL graph %d: replay complete; updating attention parameters", num_tokens)

        # refer to vllm.v1.worker.gpu.dp_utils.sync_cudagraph_and_dp_padding to
        # calculate num_tokens_across_dp.
        num_tokens_across_dp = torch.full([self.model_runner.dp_size], num_tokens)
        # sfa_v1.py:AscendSFABackend.get_impl_cls reaches
        # sfa_cp.py:resolve_sfa_impl, whose SFA CP selector reads the current
        # ModelConfig. Publish the target config because set_forward_context()
        # does not update it.
        # TODO: Remove this explicit current-config scope once ACL graph replay
        # passes VllmConfig directly through the graph-update interfaces.
        with (
            set_current_vllm_config(self.vllm_config),
            set_forward_context(
                self.model_runner.model_state.attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=desc.cg_mode,
                num_tokens_across_dp=num_tokens_across_dp,
                batch_descriptor=None,  # Full graph model don't need batch_descriptor
                slot_mapping=None,
            ),
        ):
            forward_context = get_forward_context()
            attn_backend = _get_graph_update_backend(self.model_runner.attn_groups)
            update_full_graph_params(
                # FIXME(Ronald1995): support hybrid attn backend
                attn_backend,
                self.update_stream,
                forward_context,
                num_tokens,
                self.vllm_config,
                self.model_runner.speculative_config,
            )
        logger.debug("Target FULL graph %d: attention parameter update enqueued", num_tokens)
        return ret

    def capture(
        self,
        model: nn.Module,
        model_state: ModelState,
        input_buffers: InputBuffers,
        intermediate_tensors: IntermediateTensors | None,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        has_lora: bool = False,
        use_aux_hidden_state_outputs: bool = False,
        lora_capture_hook: Callable[[int, int, int], None] | None = None,
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        """Capture CUDA graphs for model forward pass."""
        model = ModelWithContext(model)
        with communicator_switch():
            result = super().capture(
                model,
                model_state,
                input_buffers,
                intermediate_tensors,
                block_tables,
                attn_groups,
                kv_cache_config,
                has_lora=has_lora,
                use_aux_hidden_state_outputs=use_aux_hidden_state_outputs,
                lora_capture_hook=lora_capture_hook,
                progress_bar_desc=progress_bar_desc,
            )
        if getattr(self, "_max_full_descs_to_capture", None) is None:
            missing = [desc for desc in self.long_verification_graphs if desc not in self.graphs]
            if missing:
                raise RuntimeError(
                    "Long Target verification graph capture is incomplete: "
                    f"missing_descriptors={missing}. Reduce max_num_seqs or speculative width, "
                    "or increase max_num_batched_tokens/max_cudagraph_capture_size."
                )
        return result


class ModelWithContext(nn.Module):
    """Define a wrapper model to inject forward context.
    so we can inherit vllm's CudaGraphManager._capture_full_graph.
    """

    def __init__(self, original_model, is_draft_model=False, is_draft_model_prefill=False):
        super().__init__()
        self.original_model = original_model
        self.is_draft_model = is_draft_model
        self.is_draft_model_prefill = is_draft_model_prefill

    def forward(self, *args, **kwargs):
        forward_context = get_forward_context()
        # In warmup phase, capturing=False by default.
        # when capturing, we need to set capturing=True in forward context.
        _EXTRA_CTX.capturing = (
            torch.npu.is_current_stream_capturing()
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        )
        if self.is_draft_model:
            _EXTRA_CTX.is_draft_model = True
        if self.is_draft_model_prefill:
            _EXTRA_CTX.is_draft_model_prefill = True

        return self.original_model(*args, **kwargs)

    def get_original_model(self):
        return self.original_model

    def compute_logits(self, hidden_states: torch.Tensor):
        # draft model has `compute_logits`, which is not in ModelWithContext
        return self.original_model.compute_logits(hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor):
        return self.original_model.compute_draft_logits(hidden_states)

    def markov_embed(self, token_ids: torch.Tensor):
        return self.original_model.markov_embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor):
        return self.original_model.markov_bias(markov_embed)

    def map_draft_to_target(self, draft_ids: torch.Tensor):
        return self.original_model.map_draft_to_target(draft_ids)


@contextmanager
def model_capture_wrapper(speculator, is_draft_model_prefill):
    """Context manager to override speculator's model for speculator capturing."""
    try:
        speculator.model = ModelWithContext(speculator.model, True, is_draft_model_prefill)
        yield
    finally:
        speculator.model = speculator.model.get_original_model()
