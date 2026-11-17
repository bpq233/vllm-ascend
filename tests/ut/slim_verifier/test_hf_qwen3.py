# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Physical layer deletion in the official HF Qwen3 implementation.

Uses a 36-layer small-dimension RANDOM model, not pretrained Qwen3-8B. The
adapter shares its weights and exposes vLLM's packed-QKV/fused-residual
interface. This validates architecture/skip semantics, not NPU kernels or
the quality of a pruned pretrained checkpoint. No checkpoint is downloaded.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.spec_decode.slim_verifier import ReplayAttention, SlimVerifier
from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3ForCausalLM"):
    pytest.skip("Requires transformers with Qwen3 support", allow_module_level=True)


class PackedQKV(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.q, self.k, self.v = attn.q_proj, attn.k_proj, attn.v_proj

    def forward(self, hidden):
        return torch.cat((self.q(hidden), self.k(hidden), self.v(hidden)), -1), None


class ResidualNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, hidden, residual=None):
        if residual is None:
            return self.norm(hidden)
        residual = hidden + residual
        return self.norm(residual), residual


class Rotary(nn.Module):
    def __init__(self, rotary, head_dim):
        super().__init__()
        self.rotary, self.head_dim = rotary, head_dim

    def forward(self, positions, q, k):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        cos, sin = self.rotary(q, positions.unsqueeze(0))
        length = positions.numel()
        q = q.reshape(1, length, -1, self.head_dim).transpose(1, 2)
        k = k.reshape(1, length, -1, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q.transpose(1, 2).reshape(length, -1), k.transpose(1, 2).reshape(length, -1)


class AttentionAdapter(nn.Module):
    def __init__(self, original, rotary, config):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.q_size, self.kv_size = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
        self.scaling = original.scaling
        self.qkv_proj = PackedQKV(original)
        self.o_proj = original.o_proj
        self.q_norm, self.k_norm = original.q_norm, original.k_norm
        self.rotary_emb = Rotary(rotary, self.head_dim)
        self.attn = ReplayAttention(self)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), -1)
        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(-1, self.q_size)
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(-1, self.kv_size)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v))


class Qwen3DecoderLayer(nn.Module):
    """Adapt HF weights to the vLLM decoder calling convention."""

    def __init__(self, original, rotary, config):
        super().__init__()
        self.input_layernorm = ResidualNorm(original.input_layernorm)
        self.post_attention_layernorm = ResidualNorm(original.post_attention_layernorm)
        self.self_attn = AttentionAdapter(original.self_attn, rotary, config)
        self.mlp = original.mlp

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_all_36_single_layer_deletions_against_hf(dtype, record_property):
    torch.manual_seed(20261116)
    config = transformers.Qwen3Config(
        num_hidden_layers=36,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
        max_position_embeddings=128,
        use_cache=False,
        sliding_window=None,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    hf = transformers.Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    original_layers = hf.model.layers
    target = SimpleNamespace(
        config=config,
        model=SimpleNamespace(
            config=config,
            start_layer=0,
            end_layer=36,
            embed_tokens=hf.model.embed_tokens,
            norm=ResidualNorm(hf.model.norm),
            layers=nn.ModuleList(Qwen3DecoderLayer(layer, hf.model.rotary_emb, config) for layer in original_layers),
        ),
        compute_logits=hf.lm_head,
    )
    ids = torch.tensor([1, 4, 2, 7, 9, 3, 11, 12, 4, 8, 15, 20, 2, 6, 3, 5, 9])
    largest_difference = 0.0
    for skipped in (None, *range(36)):
        kept = tuple(i for i in range(36) if i != skipped)
        # Actually remove modules from the official implementation. No adapter
        # or SlimVerifier code participates in computing the reference logits.
        try:
            hf.model.layers = nn.ModuleList(original_layers[i] for i in kept)
            expected = hf(ids.unsqueeze(0), use_cache=False).logits[0]
        finally:
            hf.model.layers = original_layers
        actual = SlimVerifier(target, SlimVerifierConfig(kept)).logits(ids, torch.arange(ids.numel()))
        largest_difference = max(largest_difference, (actual.float() - expected.float()).abs().max().item())
        rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (0.05, 0.03)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=f"skipped={skipped}, dtype={dtype}")
    record_property("transformers_version", transformers.__version__)
    record_property("cases", 37)
    record_property("largest_logit_abs_difference", largest_difference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_target_prefix_kv_against_hf_cached_layer_deletion(dtype):
    torch.manual_seed(20261117)
    config = transformers.Qwen3Config(
        num_hidden_layers=36,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
        max_position_embeddings=128,
        sliding_window=None,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    hf = transformers.Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    original_layers = hf.model.layers
    target = SimpleNamespace(
        config=config,
        model=SimpleNamespace(
            config=config,
            start_layer=0,
            end_layer=36,
            embed_tokens=hf.model.embed_tokens,
            norm=ResidualNorm(hf.model.norm),
            layers=nn.ModuleList(Qwen3DecoderLayer(layer, hf.model.rotary_emb, config) for layer in original_layers),
        ),
        compute_logits=hf.lm_head,
    )
    prefix = torch.tensor([[1, 4, 2, 7, 9, 3, 11]])
    query = torch.tensor([12, 4, 8, 15])
    # The official FULL model creates the history. A fresh copy goes into each
    # reduced HF forward, exactly the initialization we want to reproduce.
    full_cache = hf(prefix, use_cache=True).past_key_values
    saved = [tuple(t.clone() for t in full_cache[i]) for i in range(36)]

    def prefix_kv(index):
        return tuple(t[0].transpose(0, 1) for t in full_cache[index])

    for kept in (tuple(range(36)), tuple(i for i in range(36) if i != 18), (0, 35), (1, 18, 35)):
        try:
            hf.model.layers = nn.ModuleList(original_layers[i] for i in kept)
            expected = hf(query.unsqueeze(0), past_key_values=deepcopy(full_cache), use_cache=True).logits[0]
        finally:
            hf.model.layers = original_layers
        actual = SlimVerifier(target, SlimVerifierConfig(kept)).logits(
            query, torch.arange(query.numel()), position_offset=prefix.shape[1], prefix_kv_provider=prefix_kv
        )
        rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (0.05, 0.03)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=f"kept={kept}, dtype={dtype}")
    for index in range(36):
        for actual, expected in zip(full_cache[index], saved[index]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
