# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from copy import copy

import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig


class ReplayAttention(nn.Module):
    """One unpadded request, with private ephemeral K/V from the full history.

    This leaf deliberately has no access to an Attention instance, its backend,
    static forward-context registration, or its paged KV. All other operations
    (including Q/K normalization and RoPE) remain the target's operations.
    """

    def __init__(self, attention: nn.Module):
        super().__init__()
        self.num_heads = attention.num_heads
        self.num_kv_heads = attention.num_kv_heads
        self.head_dim = attention.head_dim
        self.scale = attention.scaling

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        length = query.shape[0]
        q = query.reshape(length, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        k = key.reshape(length, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        v = value.reshape(length, self.num_kv_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        # Explicit GQA expansion also works on torch_npu versions whose SDPA
        # does not expose enable_gqa. There are no random/dropout operations.
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=self.scale)
        return output.squeeze(0).transpose(0, 1).reshape(length, -1)


def _module_view(module: nn.Module) -> nn.Module:
    """Copy only Python containers; every parameter/buffer remains shared.

    Bound CustomOp dispatchers must be rebound to the view, otherwise __call__
    may still dispatch into the original instance and its original children.
    """
    view = copy(module)
    for name, attr in vars(module).items():
        if isinstance(attr, (dict, set)):
            # Includes module/parameter maps and hook registries. Registering an
            # instrumentation hook on a view must not register it on the target.
            setattr(view, name, attr.copy())
        elif getattr(attr, "__self__", None) is module and hasattr(attr, "__func__"):
            setattr(view, name, attr.__func__.__get__(view, type(view)))
    return view


class SlimVerifier:
    """Shared-weight Qwen2/Qwen3 dense decoder with an isolated replay path.

    A separate module *view* is needed only for the selected decoder and
    self-attention containers so their Attention leaf can be cacheless. This
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
            self.layers.append(layer)

    @torch.inference_mode()
    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError("Replay expects one nonempty, unpadded complete token history")
        positions = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.int64)
        hidden = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions=positions, hidden_states=hidden, residual=residual)
        return self.norm(hidden) if residual is None else self.norm(hidden, residual)[0]

    @torch.inference_mode()
    def logits(self, input_ids: torch.Tensor, prediction_positions: torch.Tensor) -> torch.Tensor | None:
        # Only candidate-prediction rows go through the shared global LM head.
        hidden = self.hidden_states(input_ids)
        return self.target.compute_logits(hidden[prediction_positions])
