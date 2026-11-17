# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from test_slim_verifier import Target, make_batch, make_hook

from vllm_ascend.spec_decode.slim_verifier import ReplayAttention, SlimVerifier
from vllm_ascend.spec_decode.slim_verifier_cache import TargetPrefixKV, gather_prefix_kv
from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig


@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("prefix_length", [0, 1, 5, 6])
def test_paged_prefix_layout_and_partial_block(head_first, packed, prefix_length):
    # Nontrivial physical order and an unused negative tail in the block table.
    key = torch.arange(5 * 3 * 2 * 4).reshape(5, 3, 2, 4).float()
    value = key + 1000
    table = torch.tensor([4, 1, -1], dtype=torch.int32)
    expected_k = torch.cat((key[4], key[1]))[:prefix_length]
    expected_v = torch.cat((value[4], value[1]))[:prefix_length]
    if head_first:
        key, value = key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()
    cache = torch.stack((key, value)) if packed else (key, value)
    before = key.clone(), value.clone()
    k, v = gather_prefix_kv(
        cache, table, prefix_length, block_size=3, num_kv_heads=2, head_dim=4, head_first=head_first
    )
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(v, expected_v)
    k.zero_()
    v.zero_()
    torch.testing.assert_close(key, before[0])
    torch.testing.assert_close(value, before[1])


def test_packed_nz_cache_rejected():
    with pytest.raises(NotImplementedError, match="NZ or packed"):
        gather_prefix_kv(
            (torch.zeros(2, 1, 1, 3, 16),) * 2,
            torch.tensor([0]),
            1,
            block_size=3,
            num_kv_heads=1,
            head_dim=16,
            head_first=False,
        )


@pytest.mark.parametrize("prefix_length", [1, 5])
def test_offset_causal_mask_against_independent_attention(prefix_length):
    torch.manual_seed(900)
    length, heads, kv_heads, dim = 4, 4, 2, 8
    q = torch.randn(length, heads * dim)
    k, v = (torch.randn(prefix_length + length, kv_heads * dim) for _ in range(2))
    attention = ReplayAttention(
        SimpleNamespace(num_heads=heads, num_kv_heads=kv_heads, head_dim=dim, scaling=dim**-0.5)
    )
    attention.prefix_kv = tuple(t[:prefix_length].reshape(prefix_length, kv_heads, dim) for t in (k, v))
    actual = attention(q, k[prefix_length:], v[prefix_length:])
    # Each query explicitly sees exactly prefix_length + i + 1 keys. Group
    # heads without SDPA, repeat_interleave, or a rectangular masking helper.
    expected = []
    for i in range(length):
        end = prefix_length + i + 1
        qi = q[i].reshape(kv_heads, heads // kv_heads, dim)
        ki = k[:end].reshape(end, kv_heads, dim).permute(1, 2, 0)
        vi = v[:end].reshape(end, kv_heads, dim).transpose(0, 1)
        expected.append(((qi @ ki * dim**-0.5).softmax(-1) @ vi).reshape(-1))
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-6)
    # Alter every future key/value; the first query must remain unchanged.
    changed_k, changed_v = k[prefix_length:].clone(), v[prefix_length:].clone()
    changed_k[1:], changed_v[1:] = 10000, -10000
    torch.testing.assert_close(attention(q, changed_k, changed_v)[0], actual[0])


@torch.inference_mode()
def bind_target_cache(target, runner, histories, prefix_lengths, head_first):
    """Capture real full-model K/V; pack them into unrelated physical blocks."""
    all_kv = []
    for ids in histories:
        captured = {}
        handles = []
        for index, layer in enumerate(target.model.layers):

            def capture(module, args, i=index, saved=captured):
                saved[i] = tuple(t.reshape(-1, 1, 4).clone() for t in args[1:])

            handles.append(layer.self_attn.attn.register_forward_pre_hook(capture))
        try:
            target(ids)
        finally:
            for handle in handles:
                handle.remove()
        all_kv.append(captured)
    # Intentionally reorder both groups and layers; compressed Slim indices or
    # idx_mapping state slots would read the wrong cache.
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer.3", "layer.1"]),
            SimpleNamespace(layer_names=["layer.2", "layer.0"]),
        ]
    )
    tables = [torch.tensor([[4, 0, 3], [2, 5, 1]], dtype=torch.int32) for _ in range(2)]
    runner.block_tables = SimpleNamespace(input_block_tables=tables, kernel_block_sizes=[2, 3])
    for index, layer in enumerate(target.model.layers):
        group = 0 if index in (1, 3) else 1
        block_size = runner.block_tables.kernel_block_sizes[group]
        key, value = (torch.full((6, block_size, 1, 4), float("nan")) for _ in range(2))
        for req, ids in enumerate(histories):
            for pos in range(len(ids)):
                block = tables[group][req, pos // block_size]
                for cache, saved in zip((key, value), all_kv[req][index]):
                    # Current-query and stale tail entries are deliberately bad:
                    # reading them instead of computing Slim K/V must fail.
                    cache[block, pos % block_size] = saved[pos] if pos < prefix_lengths[req] else float("nan")
        attention = layer.self_attn.attn
        attention.layer_name = f"layer.{index}"
        attention.impl = SimpleNamespace(use_bnsd_kv_cache=head_first)
        attention.kv_cache = tuple(t.transpose(1, 2).contiguous() for t in (key, value)) if head_first else (key, value)
    return all_kv


@pytest.mark.parametrize("head_first", [False, True])
def test_target_prefix_hook_matches_full_forward_and_preserves_target(head_first):
    torch.manual_seed(17)
    target = Target().eval()
    hook = make_hook(target, kv_source="target")
    runner, hidden, batch, histories = make_batch(target)
    bind_target_cache(target, runner, histories, [2, 4], head_first)
    before = [tuple(t.clone() for t in layer.self_attn.attn.kv_cache) for layer in target.model.layers]
    # New mode must not even access the token-history store.
    del runner.req_states
    observed = []
    original_logits = hook.verifier.logits

    def capture(ids, positions, **kwargs):
        observed.append((ids.clone(), positions.clone(), kwargs["position_offset"]))
        return original_logits(ids, positions, **kwargs)

    hook.verifier.logits = capture
    hook.evaluate(runner, hidden, batch)
    assert observed[0][0].tolist() == [3, 1, 2]
    assert observed[0][1].tolist() == [0, 1]
    assert observed[0][2] == 2
    assert observed[1][0].tolist() == [0, 3]
    assert observed[1][1].tolist() == [0]
    assert observed[1][2] == 4
    assert hook.last_results[1].prediction_positions.tolist() == [4]
    stats = hook.metrics.snapshot()
    assert stats["draft_tokens"] == 3
    assert stats["prediction_agreement_rate"] == 1
    assert stats["target_slim_logits_max_abs_diff"] < 1e-5
    for layer, saved in zip(target.model.layers, before):
        for actual, expected in zip(layer.self_attn.attn.kv_cache, saved):
            torch.testing.assert_close(actual, expected, equal_nan=True)
    assert all(layer.self_attn.attn.prefix_kv is None for layer in hook.verifier.layers)


def test_reduced_model_reads_original_layers_and_restarts_from_target_prefix():
    torch.manual_seed(17)
    target = Target().eval()
    runner, _, _, histories = make_batch(target)
    captured = bind_target_cache(target, runner, histories, [2, 4], False)
    slim = SlimVerifier(target, SlimVerifierConfig((1, 3)))
    calls = []
    provider = TargetPrefixKV(runner, target, 0, 2)

    def read_prefix(index):
        calls.append(index)
        actual = provider(index)
        for kv, reference in zip(actual, captured[0][index]):
            torch.testing.assert_close(kv, reference[:2])
        return actual

    query = histories[0][2:]
    actual = slim.logits(query, torch.arange(3), position_offset=2, prefix_kv_provider=read_prefix)
    assert calls == [1, 3]
    assert torch.isfinite(actual).all()
    replay = slim.logits(histories[0], torch.arange(2, 5))
    assert not torch.allclose(actual, replay, atol=1e-4, rtol=1e-4)
    again = slim.logits(query, torch.arange(3), position_offset=2, prefix_kv_provider=provider)
    torch.testing.assert_close(actual, again)


def test_prefix_released_on_error_and_offset_requires_cache():
    target = Target().eval()
    slim = SlimVerifier(target, SlimVerifierConfig((1,)))
    with pytest.raises(ValueError, match="position offset"):
        slim.hidden_states(torch.tensor([1]), position_offset=3)
    with pytest.raises(ValueError, match="length"):
        slim.hidden_states(
            torch.tensor([1]), position_offset=3, prefix_kv_provider=lambda i: (torch.zeros(2, 1, 4),) * 2
        )
    assert slim.layers[0].self_attn.attn.prefix_kv is None


def test_target_kv_config_default_and_replay_option():
    raw = {"enabled": True, "skip_layers": [1]}
    assert SlimVerifierConfig.from_additional_config({"slim_verifier_eval": raw}, 4).kv_source == "target"
    raw["kv_source"] = "replay"
    assert SlimVerifierConfig.from_additional_config({"slim_verifier_eval": raw}, 4).kv_source == "replay"
    raw["kv_source"] = "live"
    with pytest.raises(ValueError, match="kv_source"):
        SlimVerifierConfig.from_additional_config({"slim_verifier_eval": raw}, 4)


def test_closure_forward_cannot_redirect_shadow_into_target():
    target = Target().eval()
    original = target.model.layers[0].forward
    target.model.layers[0].forward = lambda *args, **kwargs: original(*args, **kwargs)
    with pytest.raises(NotImplementedError, match="closure"):
        SlimVerifier(target, SlimVerifierConfig((0,)))
