# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch


def _linear(module, hidden):
    output = module(hidden)
    return output[0] if isinstance(output, tuple) else output


@torch.inference_mode()
def reference_hidden_states(target, input_ids, config, *, position_offset=0, prefix_kv_provider=None):
    """Independent explicit residual + FP32 grouped attention Qwen reference.

    Uses original projection/norm/RoPE/MLP modules, without copied decoders,
    fused residual additions, SDPA, or Target's paged attention backend.
    """
    length = input_ids.numel()
    positions = torch.arange(length, device=input_ids.device) + position_offset
    hidden = target.model.embed_tokens(input_ids)
    for index in config.keep_layers:
        layer = target.model.layers[index]
        if index not in config.skip_attention_layers:
            attn = layer.self_attn
            qkv = _linear(attn.qkv_proj, layer.input_layernorm(hidden))
            q, k, v = qkv.split((attn.q_size, attn.kv_size, attn.kv_size), dim=-1)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q.reshape(length, attn.num_heads, attn.head_dim)).reshape(length, attn.q_size)
                k = attn.k_norm(k.reshape(length, attn.num_kv_heads, attn.head_dim)).reshape(length, attn.kv_size)
            q, k = attn.rotary_emb(positions, q, k)
            k = k.reshape(length, attn.num_kv_heads, attn.head_dim)
            v = v.reshape(length, attn.num_kv_heads, attn.head_dim)
            if prefix_kv_provider is not None:
                past_k, past_v = prefix_kv_provider(index)
                k, v = torch.cat((past_k, k)), torch.cat((past_v, v))
            q = q.float().reshape(length, attn.num_kv_heads, attn.num_heads // attn.num_kv_heads, attn.head_dim)
            q = q.permute(1, 2, 0, 3)
            keys = k.float().permute(1, 2, 0).unsqueeze(1)
            values = v.float().permute(1, 0, 2).unsqueeze(1)
            future = torch.arange(k.shape[0], device=k.device)[None, :] > positions[:, None]
            scores = q @ keys * attn.scaling
            output = scores.masked_fill(future, -torch.inf).softmax(-1) @ values
            output = output.permute(2, 0, 1, 3).reshape(length, attn.q_size).to(v.dtype)
            hidden = hidden + _linear(attn.o_proj, output)
        if index not in config.skip_mlp_layers:
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return target.model.norm(hidden)


def compare_tensors(actual, expected, rtol, atol):
    actual, expected = actual.float(), expected.float()
    if not bool((torch.isfinite(actual).all() & torch.isfinite(expected).all()).item()):
        return {"finite": False, "within_tolerance": False, "mean_abs_diff": None, "max_abs_diff": None}
    delta = (actual - expected).abs()
    return {
        "finite": True,
        "within_tolerance": bool((delta <= atol + rtol * expected.abs()).all().item()),
        "mean_abs_diff": delta.mean().item(),
        "max_abs_diff": delta.max().item(),
    }
