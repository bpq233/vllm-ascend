# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""A Qwen decoder execution view with shared weights and separate attention.

The runner must register this view *before* KV-cache profiling, and supply its
attention names in the forward context. The cloned attention modules receive
their own scheduler-allocated caches; no target cache is swapped during a
forward or graph replay. Token positions, slot mappings, and causal attention
metadata remain the responsibility of the runner, just as for the target.
"""

import copy
import logging
from collections.abc import Iterable, MutableMapping
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _module_view(module: nn.Module) -> nn.Module:
    """Copy module bookkeeping, retaining every parameter and tensor storage."""
    view = copy.copy(module)
    view._modules = module._modules.copy()
    view._parameters = module._parameters.copy()
    view._buffers = module._buffers.copy()
    return view


def _clone_decoder_attention(
    layer: nn.Module,
    static_forward_context: MutableMapping[str, Any],
    prefix: str,
) -> tuple[nn.Module, str, nn.Module]:
    """Clone only the mutable attention/cache owners, without copying weights."""
    try:
        original_attention = layer.self_attn.attn
        original_name = original_attention.layer_name
        original_impl = original_attention.impl
    except AttributeError as exc:
        raise ValueError("Sparse DFlash requires a Qwen-style decoder attention layer") from exc
    if getattr(original_attention, "kv_sharing_target_layer_name", None) is not None:
        raise ValueError("Sparse DFlash does not support target layers sharing KV caches")
    name = f"{prefix}.{original_name}"
    if name in static_forward_context:
        raise ValueError(f"Sparse attention name already registered: {name}")

    view = _module_view(layer)
    view.self_attn = _module_view(layer.self_attn)
    attention = _module_view(original_attention)
    attention.layer_name = name
    attention.impl = copy.copy(original_impl)
    # Ascend attention memoizes cache views in its implementation. Merely
    # replacing Attention.kv_cache would keep reading the target's cache.
    for attr in ("key_cache", "value_cache"):
        if hasattr(attention.impl, attr):
            setattr(attention.impl, attr, None)
    if hasattr(attention.impl, "_layer_name"):
        attention.impl._layer_name = name
    if hasattr(attention.impl, "kv_sharing_target_layer_name"):
        attention.impl.kv_sharing_target_layer_name = None
    if hasattr(attention, "kv_sharing_target_layer_name"):
        attention.kv_sharing_target_layer_name = None
    # Upstream binds the real cache after memory profiling. Never retain even
    # an empty-container alias to the target's cache list.
    if hasattr(original_attention, "kv_cache"):
        original_cache = original_attention.kv_cache
        if isinstance(original_cache, list):
            attention.kv_cache = [torch.empty(0) for _ in original_cache]
        elif isinstance(original_cache, tuple):
            attention.kv_cache = tuple(torch.empty(0) for _ in original_cache)
        else:
            attention.kv_cache = torch.empty(0)
    view.self_attn.attn = attention
    # Keep the original transformer index in the registry name. Renumbering
    # selected layers would break model/back-end layer-index conventions.
    static_forward_context[name] = attention
    return view, name, attention


class SparseTargetVerifier(nn.Module):
    """Execute selected target layers using independent attention/cache owners.

    ``aux_layer_ids`` use vLLM's capture boundaries: zero means the embedding,
    and ``i + 1`` means the residual stream after transformer layer ``i``.
    DFlash's ``target_layer_ids`` are converted to these ids by upstream's
    ``get_eagle3_aux_layers_from_config``. Use the *resolved target* list here,
    rather than interpreting checkpoint ids a second time.

    The returned feature list follows ``aux_layer_ids`` exactly. A forward on
    ``[anchor, *candidates]`` produces candidate-prediction logits in rows
    ``[:-1]`` and the bonus prediction in row ``[-1]``. This alignment differs
    from DFlash's parallel mask predictions at their own token positions.
    """

    def __init__(
        self,
        target_model: nn.Module,
        selected_layers: Iterable[int],
        aux_layer_ids: Iterable[int],
        static_forward_context: MutableMapping[str, Any],
        *,
        prefix: str = "dflash_sparse_verify",
    ) -> None:
        super().__init__()
        decoder = target_model.model
        total_layers = len(decoder.layers)
        selected = tuple(selected_layers)
        aux_ids = tuple(aux_layer_ids)
        if (
            not selected
            or any(type(i) is not int or not 0 <= i < total_layers for i in selected)
            or tuple(sorted(set(selected))) != selected
        ):
            raise ValueError("selected_layers must contain unique, increasing valid transformer indices")
        if any(type(i) is not int or not 0 <= i <= total_layers for i in aux_ids):
            raise ValueError("aux_layer_ids must be valid vLLM capture-boundary indices")
        if len(set(aux_ids)) != len(aux_ids):
            raise ValueError("aux_layer_ids must not contain duplicates")
        if any(i > 0 and i - 1 not in selected for i in aux_ids):
            raise ValueError("selected_layers must include every requested DFlash feature layer")
        if getattr(decoder, "start_layer", 0) != 0 or getattr(decoder, "end_layer", total_layers) != total_layers:
            raise ValueError("Sparse DFlash requires all target decoder layers on this pipeline rank")
        if getattr(getattr(decoder, "config", None), "is_causal", True) is not True:
            raise ValueError("Sparse DFlash requires a causal target decoder")
        self.selected_layers = selected
        self.aux_layer_ids = aux_ids
        self.embed_tokens = decoder.embed_tokens
        self.norm = decoder.norm
        self.lm_head = target_model.lm_head
        self.logits_processor = target_model.logits_processor
        self.layers = nn.ModuleDict()
        self.attention_layers: dict[str, nn.Module] = {}
        registered: list[str] = []
        try:
            for i in selected:
                layer, name, attention = _clone_decoder_attention(decoder.layers[i], static_forward_context, prefix)
                self.layers[str(i)] = layer
                self.attention_layers[name] = attention
                registered.append(name)
        except Exception:
            # A failed initialization must not leave half a sparse model in
            # the target registry used for memory profiling.
            for name in registered:
                del static_forward_context[name]
            raise
        logger.info("DFlash sparse verifier selected target transformer layers (0-based): %s", selected)
        logger.info("DFlash sparse verifier resolved auxiliary capture boundaries: %s", aux_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if intermediate_tensors is not None:
            raise ValueError("Sparse DFlash does not support pipeline intermediate tensors")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            hidden_states = self.embed_input_ids(input_ids)
        else:
            hidden_states = inputs_embeds
        if positions.ndim != 1 or positions.shape[0] != hidden_states.shape[0]:
            raise ValueError("Sparse Qwen verifier requires one position per input token")
        residual = None
        features: dict[int, torch.Tensor] = {}
        if 0 in self.aux_layer_ids:
            features[0] = hidden_states
        for layer_index in self.selected_layers:
            hidden_states, residual = self.layers[str(layer_index)](positions, hidden_states, residual)
            boundary = layer_index + 1
            if boundary in self.aux_layer_ids:
                # Qwen's fused residual stream is split across these tensors.
                # This is exactly EagleModelMixin's capture-after convention.
                features[boundary] = hidden_states if residual is None else hidden_states + residual
        if residual is None:
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, [features[i] for i in self.aux_layer_ids]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            raise RuntimeError("Sparse verifier requires target logits on this rank")
        return logits

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states, features = self(input_ids, positions)
        if logits_indices is not None:
            hidden_states = hidden_states[logits_indices]
        return self.compute_logits(hidden_states), features
