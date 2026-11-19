# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections.abc import Callable
from copy import copy

import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig

PrefixKVProvider = Callable[[int], tuple[torch.Tensor, torch.Tensor]]


class ReplayAttention(nn.Module):
    """One unpadded query, optionally attending to a read-only Target prefix.

    This leaf deliberately has no access to an Attention instance, its backend,
    static forward-context registration, or its paged KV. A caller can supply
    gathered prefix K/V; all query K/V are private. All other operations
    (including Q/K normalization and RoPE) remain the target's operations.
    """

    def __init__(self, attention: nn.Module):
        super().__init__()
        self.num_heads = attention.num_heads
        self.num_kv_heads = attention.num_kv_heads
        self.head_dim = attention.head_dim
        self.scale = attention.scaling
        self.prefix_kv: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        length = query.shape[0]
        q = query.reshape(length, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        k = key.reshape(length, self.num_kv_heads, self.head_dim)
        v = value.reshape(length, self.num_kv_heads, self.head_dim)
        prefix_length = 0
        if self.prefix_kv is not None:
            prefix_k, prefix_v = self.prefix_kv
            if (
                prefix_k.ndim != 3
                or prefix_k.shape != prefix_v.shape
                or prefix_k.shape[1:] != (self.num_kv_heads, self.head_dim)
                or prefix_k.dtype != k.dtype
                or prefix_v.dtype != v.dtype
                or prefix_k.device != k.device
                or prefix_v.device != v.device
            ):
                raise ValueError("Prefix K/V must match the query's local KV heads, dtype and device")
            prefix_length = prefix_k.shape[0]
            k = torch.cat((prefix_k, k), dim=0)
            v = torch.cat((prefix_v, v), dim=0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        # Explicit GQA expansion also works on torch_npu versions whose SDPA
        # does not expose enable_gqa. There are no random/dropout operations.
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        # SDPA's non-square is_causal mask is upper-left aligned. The query
        # starts AFTER the prefix, so it needs an explicitly offset mask.
        mask = None
        if prefix_length:
            rows = torch.arange(length, device=q.device)[:, None] + prefix_length
            columns = torch.arange(prefix_length + length, device=q.device)[None, :]
            mask = columns <= rows
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=mask is None, scale=self.scale
        )
        return output.squeeze(0).transpose(0, 1).reshape(length, -1)


def _module_view(module: nn.Module) -> nn.Module:
    """Copy only Python containers; every parameter/buffer remains shared.

    Bound CustomOp dispatchers must be rebound to the view, otherwise __call__
    may still dispatch into the original instance and its original children.
    """
    view = copy(module)
    for name, attr in vars(module).items():
        if name in ("forward", "_forward_method") and callable(attr) and getattr(attr, "__self__", None) is not module:
            raise NotImplementedError("Slim Verifier cannot safely copy closure-wrapped forwards (e.g. CPU offload)")
        if isinstance(attr, (dict, set)):
            # Includes module/parameter maps and hook registries. Registering an
            # instrumentation hook on a view must not register it on the target.
            setattr(view, name, attr.copy())
        elif getattr(attr, "__self__", None) is module and hasattr(attr, "__func__"):
            setattr(view, name, attr.__func__.__get__(view, type(view)))
    return view


class SublayerDecoder(nn.Module):
    """Qwen fused-residual contract with independently bypassed residual branches.

    Skipping attention also skips its input norm; skipping MLP also skips its
    post-attention norm. The logical state remains hidden + residual, and each
    unexecuted branch contributes exactly zero without clearing the residual.
    """

    def __init__(self, layer, skip_attention: bool, skip_mlp: bool):
        super().__init__()
        self.self_attn = layer.self_attn
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.skip_attention, self.skip_mlp = skip_attention, skip_mlp

    def forward(self, positions, hidden_states, residual):
        if not self.skip_attention:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        if not self.skip_mlp:
            if residual is None:
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
            else:
                hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SlimVerifier:
    """Shared-weight Qwen2/Qwen3 dense decoder with an isolated shadow path.

    A separate module *view* is needed only for the selected decoder and
    self-attention containers so their Attention leaf never writes Target KV. This
    does not copy weights, initialize modules, or modify the target layer list
    or target forward. Final norm and embedding are the original instances.
    """

    def __init__(self, target: nn.Module, config: SlimVerifierConfig):
        self.target = target
        self.config = config
        backbone = target.model
        model_type = backbone.config.model_type
        if model_type not in ("qwen2", "qwen3"):
            raise NotImplementedError("Slim Verifier currently supports dense text Qwen2/Qwen3 targets")
        num_layers = len(backbone.layers)
        if backbone.start_layer != 0 or backbone.end_layer != num_layers:
            raise NotImplementedError("Slim Verifier requires all target layers on each TP rank (PP=1)")
        if (
            getattr(backbone.config, "sliding_window", None)
            or getattr(backbone.config, "dual_chunk_attention_config", None)
            or not getattr(backbone.config, "is_causal", True)
        ):
            raise NotImplementedError("Slim Verifier requires standard full causal attention")
        if any(i < 0 or i >= num_layers for i in config.keep_layers):
            raise ValueError("keep_layers is outside the loaded target's actual layer range")
        self.embedding = backbone.embed_tokens
        self.norm = backbone.norm
        self.layers = []
        for index in config.keep_layers:
            original = backbone.layers[index]
            if type(original).__name__ not in ("Qwen2DecoderLayer", "Qwen3DecoderLayer"):
                raise NotImplementedError(f"Unsupported decoder layer: {type(original).__name__}")
            layer = _module_view(original)
            attention = _module_view(original.self_attn)
            attention.attn = ReplayAttention(original.self_attn)
            layer.self_attn = attention
            if index in config.skip_attention_layers or index in config.skip_mlp_layers:
                layer = SublayerDecoder(layer, index in config.skip_attention_layers, index in config.skip_mlp_layers)
            self.layers.append(layer)

    @torch.inference_mode()
    def hidden_states(
        self,
        input_ids: torch.Tensor,
        *,
        position_offset: int = 0,
        prefix_kv_provider: PrefixKVProvider | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError("Slim Verifier expects one nonempty, unpadded token sequence")
        if position_offset < 0 or (position_offset and prefix_kv_provider is None):
            raise ValueError("A nonnegative position offset and its prefix K/V are required")
        positions = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.int64) + position_offset
        hidden = self.embedding(input_ids)
        residual = None
        for original_index, layer in zip(self.config.keep_layers, self.layers):
            attention = layer.self_attn.attn
            try:
                if prefix_kv_provider is not None and original_index not in self.config.skip_attention_layers:
                    attention.prefix_kv = prefix_kv_provider(original_index)
                    if any(kv.shape[0] != position_offset for kv in attention.prefix_kv):
                        raise ValueError("Prefix K/V length must equal the query's absolute position offset")
                hidden, residual = layer(positions=positions, hidden_states=hidden, residual=residual)
            finally:
                # No persistent Slim cache; release each layer's gathered prefix
                # immediately, including on a failed forward.
                attention.prefix_kv = None
        return self.norm(hidden) if residual is None else self.norm(hidden, residual)[0]

    @torch.inference_mode()
    def logits(
        self,
        input_ids: torch.Tensor,
        prediction_positions: torch.Tensor,
        *,
        position_offset: int = 0,
        prefix_kv_provider: PrefixKVProvider | None = None,
    ) -> torch.Tensor | None:
        # prediction_positions index input_ids locally, independent of the
        # absolute RoPE offset. Only these rows go through the shared LM head.
        hidden = self.hidden_states(input_ids, position_offset=position_offset, prefix_kv_provider=prefix_kv_provider)
        return self.target.compute_logits(hidden[prediction_positions])
