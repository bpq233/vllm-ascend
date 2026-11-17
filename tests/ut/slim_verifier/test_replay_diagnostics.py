# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.spec_decode.slim_verifier import ReplayAttention
from vllm_ascend.spec_decode.slim_verifier_metrics import SlimVerifierMetrics


def test_logit_differences_weight_tokens_and_reset_window_maximum():
    metrics = SlimVerifierMetrics(log_interval=10)
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    slim = torch.tensor([[1.0, -4.0], [-1.0, 2.0]])
    metrics.record(torch.tensor([0, 1]), target, slim)
    first = metrics.snapshot()
    assert first["target_slim_logits_mean_abs_diff"] == 2.0
    assert first["target_slim_logits_max_abs_diff"] == 4.0
    assert first["target_first_token_acceptance_rate"] == 1.0
    assert first["slim_first_token_acceptance_rate"] == 1.0
    metrics.flush()

    # A shorter block weights the mean by tokens, not by blocks. Its maximum
    # belongs only to the new window; cumulative retains the previous maximum.
    metrics.record(torch.tensor([1]), target[:1], target[:1] + 1)
    window, total = metrics.snapshot(False), metrics.snapshot()
    assert window["target_slim_logits_mean_abs_diff"] == 1.0
    assert window["target_slim_logits_max_abs_diff"] == 1.0
    assert total["target_slim_logits_mean_abs_diff"] == pytest.approx(5 / 3)
    assert total["target_slim_logits_max_abs_diff"] == 4.0
    assert total["target_first_token_acceptance_rate"] == 0.5
    assert total["slim_first_token_acceptance_rate"] == 0.5


def test_first_token_uses_selected_strategy_and_ignores_later_matches():
    metrics = SlimVerifierMetrics(acceptance_strategy="probability_ratio", probability_ratio_threshold=0.5)
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    slim = torch.tensor([[0.0, -0.1], [0.0, 1.0]])
    metrics.record(torch.ones(2, dtype=torch.long), target, slim)
    stats = metrics.snapshot()
    assert stats["slim_token_match_rate"] == 0.5
    assert stats["slim_first_token_acceptance_rate"] == 1.0

    slim[0, 1] = -10
    metrics.record(torch.ones(2, dtype=torch.long), target, slim)
    stats = metrics.snapshot()
    assert stats["slim_token_acceptance_rate"] == 0.75
    assert stats["slim_first_token_acceptance_rate"] == 0.5
    assert stats["slim_avg_accepted_length"] == 1.0


def test_equal_masked_logits_have_zero_difference():
    metrics = SlimVerifierMetrics()
    logits = torch.tensor([[0.0, -torch.inf]])
    metrics.record(torch.tensor([0]), logits, logits)
    assert metrics.snapshot()["target_slim_logits_max_abs_diff"] == 0
    assert metrics.snapshot()["target_slim_logits_mean_abs_diff"] == 0


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_qwen3_8b_gqa_shapes_match_fp32_reference_and_do_not_see_future(tp_size):
    # Qwen3-8B dimensions, partitioned as ordinary TP. CPU BF16 verifies the
    # layout and causal math; actual NPU SDPA still needs a hardware control.
    num_heads, num_kv_heads, head_dim, length = 32 // tp_size, 8 // tp_size, 128, 17
    attention = ReplayAttention(
        SimpleNamespace(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, scaling=head_dim**-0.5)
    )
    generator = torch.Generator().manual_seed(2026)
    q = torch.randn(length, num_heads * head_dim, generator=generator).bfloat16()
    k = torch.randn(length, num_kv_heads * head_dim, generator=generator).bfloat16()
    v = torch.randn(length, num_kv_heads * head_dim, generator=generator).bfloat16()
    # Group query heads explicitly rather than mirroring the implementation's
    # repeated K/V expansion, and calculate the reference attention in FP32.
    q_ref = q.float().reshape(length, num_kv_heads, num_heads // num_kv_heads, head_dim).permute(1, 2, 0, 3)
    k_ref = k.float().reshape(length, num_kv_heads, head_dim).permute(1, 0, 2).unsqueeze(1)
    v_ref = v.float().reshape(length, num_kv_heads, head_dim).permute(1, 0, 2).unsqueeze(1)
    scores = q_ref @ k_ref.transpose(-1, -2) * head_dim**-0.5
    future = torch.ones(length, length, dtype=torch.bool).triu(1)
    expected = (scores.masked_fill(future, -torch.inf).softmax(-1) @ v_ref).permute(2, 0, 1, 3)
    actual = attention(q, k, v)
    torch.testing.assert_close(actual.float(), expected.reshape(length, -1), rtol=1e-2, atol=1e-2)
    # Appending future drafts must not change earlier predictions.
    torch.testing.assert_close(actual[:8], attention(q[:8], k[:8], v[:8]), rtol=1e-2, atol=1e-2)
