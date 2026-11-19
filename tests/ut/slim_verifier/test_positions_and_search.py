# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import math

import pytest
import torch

from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig
from vllm_ascend.spec_decode.slim_verifier_metrics import SlimVerifierMetrics
from vllm_ascend.spec_decode.slim_verifier_search import search_masks


def test_position_metrics_exclude_suffix_after_target_rejection():
    metrics = SlimVerifierMetrics()
    # Draft is always token 0. Target accepts TTFT, Slim accepts TFTF.
    target = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
    slim = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]])
    metrics.record(torch.zeros(4, dtype=torch.long), target, slim)
    result = metrics.snapshot()
    assert result["target_prefix_tokens"] == 2
    assert result["slim_rejection_rate_in_target_prefix"] == 0.5
    assert result["joint_avg_accepted_length"] == 1
    assert result["target_prefix_preservation_rate"] == 0.5
    assert result["target_rejection_boundaries"] == 1
    assert result["slim_acceptance_at_target_rejection_rate"] == 1
    assert result["per_position"][3]["target_prefix_tokens"] == 0
    assert result["per_position"][3]["slim_rejection_rate_in_target_prefix"] is None
    assert result["per_position"][0]["mean_kl_target_to_slim"] == 0
    assert result["mean_kl_target_to_slim"] > 0
    metrics.flush()
    # A short new window must not retain long-block denominators or scores.
    metrics.record(torch.zeros(1, dtype=torch.long), target[:1], target[:1])
    assert len(metrics.snapshot(False)["per_position"]) == 1
    assert metrics.snapshot(False)["mean_kl_target_to_slim"] == 0
    assert metrics.snapshot()["per_position"][0]["tokens"] == 2
    assert metrics.snapshot()["per_position"][3]["tokens"] == 1


def test_position_metrics_no_target_prefix_and_probability_ratio():
    metrics = SlimVerifierMetrics(acceptance_strategy="probability_ratio", probability_ratio_threshold=0.5)
    target, slim = torch.tensor([[0.0, 1.0], [2.0, 0.0]]), torch.tensor([[1.0, 0.9], [1.0, 0.9]])
    metrics.record(torch.tensor([0, 1]), target, slim)
    stats = metrics.snapshot()
    assert stats["target_prefix_tokens"] == 0
    assert stats["slim_rejection_rate_in_target_prefix"] is None
    assert stats["target_prefix_preservation_rate"] is None
    assert stats["slim_acceptance_at_target_rejection_rate"] == 1
    assert stats["per_position"][1]["slim_acceptance_rate"] == 1
    metrics.record(torch.tensor([0, 1, 1]), target[[0, 0, 0]], slim[[0, 0, 0]])
    assert len(metrics.snapshot()["per_position"]) == 3


def test_kl_is_invariant_to_logit_shift_and_common_negative_infinity():
    metrics = SlimVerifierMetrics()
    logits = torch.tensor([[2.0, 0.0, -torch.inf]])
    metrics.record(torch.tensor([0]), logits, logits + 10)
    assert metrics.snapshot()["mean_kl_target_to_slim"] == pytest.approx(0, abs=1e-6)
    assert metrics.snapshot()["target_slim_logits_mean_abs_diff"] > 0


def test_search_rechecks_joint_masks_instead_of_combining_single_ranking():
    called = []

    def cost(actions):
        called.append(actions)
        ids = tuple(i for _, i in actions)
        if ids == (1, 2):
            return 100.0
        if ids == (1, 3):
            return 0.1
        return float(sum(ids))

    result = search_masks(6, 2, cost)
    assert result["completed"]
    assert result["best_actions"] == [("layer", 1), ("layer", 3)]
    assert len(called) == len(set(called))
    assert all(i not in (0, 5) for actions in called for _, i in actions)


def test_search_budget_does_not_claim_requested_mask_was_found():
    result = search_masks(6, 3, lambda actions: len(actions), max_evaluations=1)
    assert not result["completed"]
    assert result["budget_exhausted"]
    assert result["achieved_skip_count"] == 0
    with pytest.raises(ValueError, match="search space"):
        search_masks(2, 1, lambda actions: 0)
    with pytest.raises(ValueError, match="Nonfinite"):
        search_masks(4, 1, lambda actions: math.nan)


def test_sublayer_search_can_select_two_branches_at_the_same_layer():
    result = search_masks(3, 2, lambda actions: len(actions), scope="sublayer")
    assert result["best_actions"] == [("attention", 1), ("mlp", 1)]


def test_sublayer_config_defaults_to_all_whole_layers():
    raw = {"enabled": True, "skip_attention_layers": [2, 0], "skip_mlp_layers": [1]}
    config = SlimVerifierConfig.from_additional_config({"slim_verifier_eval": raw}, 4)
    assert config.keep_layers == (0, 1, 2, 3)
    assert config.skip_attention_layers == (0, 2)
    assert config.skip_mlp_layers == (1,)


@pytest.mark.parametrize(
    "raw",
    [
        {"skip_attention_layers": [True]},
        {"skip_mlp_layers": [4]},
        {"skip_attention_layers": [1, 1]},
        {"skip_mlp_layers": "1"},
        {"skip_layers": [1], "skip_mlp_layers": [1]},
    ],
)
def test_invalid_sublayer_configuration(raw):
    with pytest.raises(ValueError):
        SlimVerifierConfig.from_additional_config({"slim_verifier_eval": {"enabled": True, **raw}}, 4)
