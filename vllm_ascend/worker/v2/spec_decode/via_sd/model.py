"""A parameter-sharing routed view of a dense target decoder."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn
from vllm.config import set_current_vllm_config

from .adapters import DecoderLayerAdapter, find_adapter, validate_layer_forward


def resolve_layer_ids(
    total_layers: int,
    configured_layer_ids: Sequence[int] | None,
    fraction: float = 0.4,
) -> tuple[int, ...]:
    """Resolve an explicit layer list or an evenly spaced fraction.

    The default for Qwen3-8B (36 layers) is 14 layers, which is the nearest
    integer to 40%.  Explicit IDs are kept intact so layer-search experiments
    can be reproduced exactly.
    """

    if total_layers <= 0:
        raise ValueError(f"target model has no decoder layers: {total_layers}")
    if configured_layer_ids:
        ids = tuple(int(index) for index in configured_layer_ids)
        if ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
            raise ValueError("via_sd_config.layer_ids must be sorted and unique")
        if ids[0] < 0 or ids[-1] >= total_layers:
            raise ValueError(
                f"via_sd_config.layer_ids={list(ids)} is outside target layer range "
                f"[0, {total_layers})"
            )
        return ids

    if not 0 < fraction <= 1:
        raise ValueError(f"via_sd_config.layer_fraction must be in (0, 1], got {fraction}")
    count = max(1, min(total_layers, math.floor(total_layers * fraction + 0.5)))
    if count == 1:
        return (total_layers // 2,)
    # Include both ends and spread the retained layers over the full target.
    ids = {round(index * (total_layers - 1) / (count - 1)) for index in range(count)}
    # ``round`` can collide for very small models.  Fill any missing slots in
    # order so the requested count remains deterministic.
    for index in range(total_layers):
        if len(ids) >= count:
            break
        ids.add(index)
    return tuple(sorted(ids))


def _share_parameters(destination: nn.Module, source: nn.Module) -> None:
    """Alias destination parameters to source parameters without loading weights."""

    source_parameters = dict(source.named_parameters())
    destination_parameters = list(destination.named_parameters())
    missing = [name for name, _ in destination_parameters if name not in source_parameters]
    if missing:
        raise ValueError(f"target layer is missing parameters required by q': {missing[:4]}")
    for name, old_parameter in destination_parameters:
        source_parameter = source_parameters[name]
        if old_parameter.shape != source_parameter.shape:
            raise ValueError(
                f"parameter shape mismatch for {name}: "
                f"q'={tuple(old_parameter.shape)}, target={tuple(source_parameter.shape)}"
            )
        owner_name, _, leaf_name = name.rpartition(".")
        owner = destination.get_submodule(owner_name) if owner_name else destination
        owner._parameters[leaf_name] = source_parameter


def _attention_name(layer: nn.Module) -> str:
    attention = getattr(getattr(layer, "self_attn", None), "attn", None)
    name = getattr(attention, "layer_name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("VIA-SD requires a self_attn.attn layer with a registered layer_name")
    return name


class ViaSdModel(nn.Module):
    """Sparse decoder view sharing target weights and output head.

    Only the small decoder-layer objects and their runtime buffers are new.
    Every trainable parameter is replaced with the corresponding target
    parameter object, so this class never loads a second model checkpoint.
    """

    def __init__(
        self,
        target: nn.Module,
        vllm_config: Any,
        layer_ids: Sequence[int] | None = None,
        layer_fraction: float = 0.4,
    ) -> None:
        super().__init__()
        target_model = getattr(target, "model", None)
        target_layers = getattr(target_model, "layers", None)
        if target_model is None or target_layers is None:
            raise TypeError("VIA-SD target must expose model.layers")
        if getattr(vllm_config, "quant_config", None) is not None:
            raise NotImplementedError("VIA-SD q' currently requires an unquantized target")

        total_layers = len(target_layers)
        self.total_layers = total_layers
        self.aux_hidden_state_layers = tuple(getattr(target_model, 'aux_hidden_state_layers', ()))
        if any(index < 0 or index > total_layers for index in self.aux_hidden_state_layers):
            raise ValueError('Draft auxiliary feature boundary is outside target depth')
        self.last_aux_hidden_states: list[torch.Tensor] = []
        self.capture_draft_features = False
        self.layer_ids = resolve_layer_ids(total_layers, layer_ids, layer_fraction)
        adapter = find_adapter(target_layers[self.layer_ids[0]])
        if adapter is None:
            layer_type = type(target_layers[self.layer_ids[0]]).__name__
            raise NotImplementedError(f"VIA-SD has no adapter for {layer_type}")
        self.adapter: DecoderLayerAdapter = adapter
        for index in self.layer_ids:
            if not adapter.matches(target_layers[index]):
                raise TypeError(
                    "VIA-SD layer selection contains incompatible decoder layer types: "
                    f"layer {index} is {type(target_layers[index]).__name__}, "
                    f"expected {adapter.layer_type.__name__}"
                )
            validate_layer_forward(adapter, target_layers[index])

        # These modules are shared references.  They are intentionally not a
        # reference to the complete target model, which keeps q'.state_dict()
        # from registering all target layers a second time.
        self.embed_tokens = target_model.embed_tokens
        self.norm = target_model.norm
        self.logits_processor = target.logits_processor
        self.lm_head = target.lm_head
        self.layers = nn.ModuleList()
        self.source_attention_layer_names: dict[str, str] = {}
        self.attention_layer_names: tuple[str, ...]

        target_dtype = next(target.parameters()).dtype
        target_device = next(target.parameters()).device
        try:
            with set_current_vllm_config(vllm_config), torch.device(target_device):
                for index in self.layer_ids:
                    layer = adapter.build(target, vllm_config, f"via_sd.layers.{index}")
                    layer = layer.to(dtype=target_dtype)
                    _share_parameters(layer, target_layers[index])
                    self.layers.append(layer)
                    qprime_name = _attention_name(layer)
                    target_name = _attention_name(target_layers[index])
                    self.source_attention_layer_names[qprime_name] = target_name
        except Exception:
            self.unregister_attention_layers(vllm_config)
            raise

        self.attention_layer_names = tuple(self.source_attention_layer_names)
        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual: torch.Tensor | None = None
        self.last_aux_hidden_states = []
        requested = set(self.aux_hidden_state_layers) if self.capture_draft_features else set()
        retained = dict(zip(self.layer_ids, self.layers))
        # Boundary k is the residual stream after k original layers. Skipped
        # layers are identity operations, including at auxiliary draft taps.
        for boundary in range(self.total_layers + 1):
            if boundary in requested:
                value = hidden_states if residual is None else hidden_states + residual
                self.last_aux_hidden_states.append(value.clone())
            layer = retained.get(boundary)
            if layer is not None:
                hidden_states, residual = self.adapter.forward(
                    layer, positions, hidden_states, residual
                )
        normalized = self.norm(hidden_states, residual)
        if isinstance(normalized, tuple):
            hidden_states = normalized[0]
        else:
            hidden_states = normalized
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            raise RuntimeError("VIA-SD target output head returned no logits")
        return logits

    def unregister_attention_layers(self, vllm_config: Any) -> None:
        context = vllm_config.compilation_config.static_forward_context
        for name in getattr(self, "attention_layer_names", ()):
            context.pop(name, None)
        # During construction an exception can happen before the tuple is set.
        for name in getattr(self, "source_attention_layer_names", {}):
            context.pop(name, None)


def build_via_sd_model(
    target: nn.Module,
    vllm_config: Any,
    layer_ids: Sequence[int] | None,
    layer_fraction: float = 0.4,
) -> ViaSdModel:
    """Factory kept separate so future model adapters can be injected."""

    return ViaSdModel(target, vllm_config, layer_ids, layer_fraction)


__all__ = ["ViaSdModel", "build_via_sd_model", "resolve_layer_ids"]
