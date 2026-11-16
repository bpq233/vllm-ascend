# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import math

import pytest
import torch

from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig
from vllm_ascend.spec_decode.slim_verifier_metrics import SlimVerifierMetrics


def test_probability_ratio_config_and_top1_default():
    assert SlimVerifierConfig((0,)).acceptance_strategy == "top1"
    parsed = SlimVerifierConfig.from_additional_config(
        {
            "slim_verifier_eval": {
                "enabled": True,
                "keep_layers": [0, 2],
                "acceptance_strategy": "probability_ratio",
                "probability_ratio_threshold": 0.8,
            }
        },
        4,
    )
    assert parsed.acceptance_strategy == "probability_ratio"
    assert parsed.probability_ratio_threshold == 0.8


@pytest.mark.parametrize(
    "threshold", [-0.01, 1.01, 10**400, float("nan"), float("inf"), -float("inf"), True, "0.5", None]
)
def test_invalid_ratio_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="probability_ratio_threshold"):
        SlimVerifierConfig((0,), acceptance_strategy="probability_ratio", probability_ratio_threshold=threshold)
    with pytest.raises(ValueError, match="probability_ratio_threshold"):
        SlimVerifierMetrics(acceptance_strategy="probability_ratio", probability_ratio_threshold=threshold)


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError, match="acceptance_strategy"):
        SlimVerifierConfig((0,), acceptance_strategy="ratio_typo")
    with pytest.raises(ValueError, match="acceptance_strategy"):
        SlimVerifierMetrics(acceptance_strategy="ratio_typo")


def test_ratio_matches_softmax_and_strict_boundary():
    logits = torch.tensor([[0.0, -0.5], [0.0, -1.0], [0.0, -1.5], [0.0, 0.0]])
    draft = torch.ones(4, dtype=torch.long)
    metrics = SlimVerifierMetrics(
        acceptance_strategy="probability_ratio",
        probability_ratio_threshold=math.exp(-1),
    )
    result = metrics.record(draft, logits, logits)
    probabilities = logits.softmax(-1)
    expected_ratio = probabilities[:, 1] / probabilities.max(-1).values
    torch.testing.assert_close(result.slim_probability_ratios, expected_ratio)
    assert result.slim_accept_mask.tolist() == [True, False, False, True]
    assert result.slim_top1_match_mask.tolist() == [False, False, False, False]
    assert result.slim_acceptance_strategy == "probability_ratio"
    # Equality at exp(-1) is REJECTED. A tie for max has ratio 1 even
    # if the deterministic argmax selects the other tied vocabulary token.
    assert metrics.snapshot()["slim_avg_accepted_length"] == 1
    assert metrics.snapshot()["slim_token_acceptance_rate"] == 0.5
    assert metrics.snapshot()["slim_token_match_rate"] == 0


def test_ratio_statistics_compare_policy_decisions_with_target_top1():
    slim = torch.tensor([[0.0, -0.1], [0.0, -0.3], [0.0, -0.05], [0.0, -1.0]])
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    draft = torch.ones(4, dtype=torch.long)
    metrics = SlimVerifierMetrics(
        acceptance_strategy="probability_ratio",
        probability_ratio_threshold=math.exp(-0.25),
    )
    result = metrics.record(draft, target, slim)
    assert result.target_accept_mask.tolist() == [True, True, False, False]
    assert result.slim_accept_mask.tolist() == [True, False, True, False]
    stats = metrics.snapshot()
    assert stats["target_avg_accepted_length"] == 2
    assert stats["slim_avg_accepted_length"] == 1
    assert stats["slim_prefix_acceptance_rate"] == 0.25
    assert stats["slim_token_match_rate"] == 0
    assert stats["slim_token_acceptance_rate"] == 0.5
    assert stats["slim_full_block_acceptance_rate"] == 0
    assert stats["prediction_agreement_rate"] == 0.5
    assert stats["decision_agreement_rate"] == 0.5
    assert stats["slim_only_rate"] == stats["target_only_rate"] == 0.25
    assert stats["slim_acceptance_strategy"] == "probability_ratio"
    assert stats["probability_ratio_threshold"] == math.exp(-0.25)


def test_threshold_zero_preserves_small_positive_probabilities_without_underflow():
    logits = torch.tensor([[0.0, -1000.0], [0.0, -torch.inf]])
    metrics = SlimVerifierMetrics(acceptance_strategy="probability_ratio", probability_ratio_threshold=0)
    result = metrics.record(torch.ones(2, dtype=torch.long), logits, logits)
    assert result.slim_probability_ratios.tolist() == [0.0, 0.0]  # FP32 exp underflows in the first row.
    assert result.slim_accept_mask.tolist() == [True, False]


def test_threshold_one_rejects_even_the_most_likely_token():
    logits = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    metrics = SlimVerifierMetrics(acceptance_strategy="probability_ratio", probability_ratio_threshold=1)
    result = metrics.record(torch.ones(2, dtype=torch.long), logits, logits)
    assert result.slim_probability_ratios.tolist() == [1.0, 1.0]
    assert result.slim_accept_mask.tolist() == [False, False]


def test_ratio_is_invariant_to_large_logit_offsets():
    logits = torch.tensor([[1.0, 0.5], [1.0, -0.5]])
    draft = torch.ones(2, dtype=torch.long)
    masks, ratios = [], []
    for offset in (0.0, 1000.0, -1000.0):
        metrics = SlimVerifierMetrics(acceptance_strategy="probability_ratio", probability_ratio_threshold=0.5)
        result = metrics.record(draft, logits + offset, logits + offset)
        masks.append(result.slim_accept_mask)
        ratios.append(result.slim_probability_ratios)
    assert masks[0].tolist() == [True, False]
    for mask, ratio in zip(masks[1:], ratios[1:]):
        torch.testing.assert_close(mask, masks[0])
        torch.testing.assert_close(ratio, ratios[0])


def test_ratio_window_and_cumulative_metrics_are_labeled(caplog):
    metrics = SlimVerifierMetrics(
        log_interval=2,
        acceptance_strategy="probability_ratio",
        probability_ratio_threshold=0.5,
    )
    logits = torch.tensor([[0.0, -0.1]])
    with caplog.at_level("INFO"):
        for _ in range(2):
            metrics.record(torch.tensor([1]), logits, logits)
    assert "Shadow Diagnostic" in caplog.text
    assert "q(x)/q(max) > 0.5" in caplog.text
    assert metrics.window_blocks == 0
    stats = metrics.snapshot()
    assert stats["slim_avg_accepted_length"] == 1
    assert stats["slim_full_block_acceptance_rate"] == 1
    assert stats["slim_token_acceptance_rate"] == 1
    assert stats["slim_token_match_rate"] == 0


def test_default_top1_masks_remain_unchanged():
    logits = torch.tensor([[0.0, -0.1], [0.0, 1.0]])
    metrics = SlimVerifierMetrics()
    result = metrics.record(torch.ones(2, dtype=torch.long), logits, logits)
    assert result.slim_accept_mask.tolist() == [False, True]
    torch.testing.assert_close(result.slim_accept_mask, result.slim_top1_match_mask)
    assert result.slim_probability_ratios is None
    assert metrics.snapshot()["slim_token_acceptance_rate"] == metrics.snapshot()["slim_token_match_rate"]
