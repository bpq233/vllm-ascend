# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU tests for past-only mixture feedback and vocabulary-free host updates."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage/proposal_feedback.py"
SPEC = importlib.util.spec_from_file_location("_test_proposal_feedback", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def policy(**config):
    return MODULE.ProposalFeedback({"mode": "adaptive", "min_observations": 2, **config})


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_fixed_weight_never_changes_from_target_feedback(weight):
    feedback = policy(mode="fixed", mixture_lambda=weight)
    for _ in range(4):
        feedback.update("state", [1.0] + [0.0] * 10, target_verified=True)
    assert feedback.select("state") == weight


def test_cold_bucket_default_and_ema_update_after_target_only():
    feedback = policy(ema_decay=0.5)
    winner = [0.0] * 10 + [1.0]
    assert feedback.select("a") == 0.5
    with pytest.raises(ValueError, match="completed Target"):
        feedback.update("a", winner, target_verified=False)
    assert feedback.version == 0 and not feedback.buckets
    feedback.update("a", winner, target_verified=True)
    assert feedback.select("a") == 0.5
    feedback.update("a", winner, target_verified=True)
    assert feedback.select("a") == 1.0
    assert feedback.select("b") == 0.5
    feedback.update("a", [1.0] + [0.0] * 10, target_verified=True)
    assert feedback.buckets["a"].overlaps[0] == feedback.buckets["a"].overlaps[-1] == 0.5
    feedback.update("a", [1.0] + [0.0] * 10, target_verified=True)
    assert feedback.select("a") == 0.0


def test_snapshot_holds_inflight_choices_across_later_target_completions():
    feedback = policy(min_observations=1)
    feedback.update("a", [0.0] * 10 + [1.0], target_verified=True)
    snapshot = feedback.snapshot()
    assert feedback.select("a", snapshot) == 1.0
    for _ in range(20):
        feedback.update("a", [1.0] + [0.0] * 10, target_verified=True)
    assert feedback.select("a") == 0.0
    assert feedback.select("a", snapshot) == 1.0
    assert snapshot.version == 1
    with pytest.raises(TypeError):
        snapshot.buckets["b"] = MODULE.BucketStats(1, (1.0,))


def test_disabled_feedback_remains_at_default_with_no_state_growth():
    feedback = policy(feedback_enabled=False)
    for _ in range(20):
        feedback.update("a", [1.0] + [0.0] * 10, target_verified=True)
    assert feedback.select("a") == 0.5 and feedback.version == 0 and not feedback.buckets


def test_equal_overlap_ties_prefer_default_then_candidate_order():
    feedback = policy(min_observations=1)
    feedback.update("a", [0.3] * 11, target_verified=True)
    assert feedback.select("a") == 0.5
    feedback = policy(min_observations=1, mixture_lambda=0.5, lambda_candidates=[0.25, 0.75])
    feedback.update("a", [0.8, 0.8], target_verified=True)
    assert feedback.select("a") == 0.25


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "oracle"},
        {"mixture_lambda": math.nan},
        {"mixture_lambda": True},
        {"mixture_lambda": -0.1},
        {"lambda_candidates": []},
        {"lambda_candidates": [0.0, 0.0]},
        {"lambda_candidates": [0.0, 1.1]},
        {"min_observations": 0},
        {"min_observations": True},
        {"ema_decay": 1.0},
        {"feedback_enabled": "true"},
        {"debug": 1},
    ],
)
def test_invalid_policy_config_fails(config):
    with pytest.raises(ValueError):
        policy(**config)


def test_updates_accept_scalar_packets_only_and_validate_overlap_range():
    feedback = policy()
    with pytest.raises(TypeError, match="scalar reduction"):
        feedback.update("a", torch.ones(11), target_verified=True)
    for values in ([0.1], [math.nan] * 11, [-0.1] * 11, [1.1] * 11):
        with pytest.raises(ValueError):
            feedback.update("a", values, target_verified=True)
    assert not feedback.buckets


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_greedy_device_features_equal_true_delta_distribution_features(dtype):
    intermediate = torch.tensor([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]], dtype=dtype)
    intermediate = intermediate.float()
    intermediate /= intermediate.sum(-1, keepdim=True)
    tokens = torch.tensor([2, 0])
    delta = torch.nn.functional.one_hot(tokens, num_classes=3).to(dtype)
    actual = MODULE.device_features(intermediate, tokens)
    expected = MODULE.device_features(intermediate, delta)
    assert actual.dtype == torch.float32 and actual.shape == (2, 5)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 1], torch.zeros(2))
    torch.testing.assert_close(actual[:, 4], torch.ones(2))


def test_feature_buckets_separate_source_round_position_and_uncertainty():
    state = MODULE.make_state([0.2, 0.0, 0.1, 0.8, 1.0], 0, 1, "primary")
    assert state != MODULE.make_state([0.9, 0.0, 0.1, 0.8, 1.0], 0, 1, "primary")
    assert state != MODULE.make_state([0.2, 0.0, 0.1, 0.8, 1.0], 1, 1, "primary")
    assert state != MODULE.make_state([0.2, 0.0, 0.1, 0.8, 1.0], 0, 4, "primary")
    assert state != MODULE.make_state([0.2, 0.0, 0.1, 0.8, 1.0], 0, 1, "secondary")
    assert MODULE.make_state([0.2, 0.0, 0.1, 0.8, 1.0], 100, 1000, "primary")[-2:] == (4, 7)
    with pytest.raises(TypeError, match="small feature packet"):
        MODULE.make_state(torch.zeros(5), 0, 0, "primary")
    with pytest.raises(ValueError):
        MODULE.make_state([math.nan] * 5, 0, 0, "primary")


@pytest.mark.parametrize("delta", [True, False])
def test_overlap_reduction_matches_full_probability_reference_without_mutating_inputs(delta):
    target = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.1, 0.7]])
    intermediate = torch.tensor([[0.1, 0.6, 0.3], [0.7, 0.2, 0.1]])
    draft = torch.tensor([2, 0]) if delta else torch.tensor([[0.7, 0.1, 0.2], [0.4, 0.3, 0.3]])
    full_draft = torch.nn.functional.one_hot(draft, num_classes=3).float() if delta else draft
    candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
    copies = [tensor.clone() for tensor in (target, intermediate, draft)]
    actual = MODULE.overlap_scores(target, intermediate, draft, candidates)
    expected_m = 1 - (target - intermediate).abs().sum(-1) * 0.5
    expected_d = 1 - (target - full_draft).abs().sum(-1) * 0.5
    expected = [expected_m, expected_d]
    for weight in candidates:
        expected.append(1 - (target - ((1 - weight) * intermediate + weight * full_draft)).abs().sum(-1) * 0.5)
    torch.testing.assert_close(actual, torch.stack(expected, -1))
    for original, copied in zip((target, intermediate, draft), copies):
        torch.testing.assert_close(original, copied)


def test_single_token_vocabulary_has_finite_features_and_full_overlap():
    one = torch.ones(3, 1)
    features = MODULE.device_features(one, torch.zeros(3, dtype=torch.long))
    assert torch.isfinite(features).all()
    torch.testing.assert_close(MODULE.overlap_scores(one, one, one), torch.ones(3, 13))


def test_oracle_exists_only_as_post_hoc_analysis_and_cannot_be_online_policy():
    assert MODULE.offline_oracle_lambda([0.1, 0.9, 0.8], [0.0, 0.5, 1.0]) == 0.5
    with pytest.raises(ValueError, match="offline only"):
        policy(mode="oracle")
    with pytest.raises(TypeError):
        MODULE.offline_oracle_lambda(torch.zeros(11))


def test_debug_records_all_fields_without_implicit_tensor_copy(caplog):
    quiet = policy(debug=False)
    quiet.emit_record({"request_id": "r"})
    assert not caplog.records
    feedback = policy(debug=True)
    with caplog.at_level("INFO", logger=MODULE.logger.name):
        feedback.emit_record({"request_id": "r", "lambda": 0.25, "target_accept": True})
    record = json.loads(caplog.records[-1].getMessage().split("proposal_sampling ", 1)[1])
    assert set(MODULE.LOG_FIELDS).issubset(record)
    assert record["request_id"] == "r" and record["lambda"] == 0.25 and record["target_accept"]
    assert record["target_verify_time_ms"] is None
    with pytest.raises(TypeError):
        feedback.emit_record({"pM_draft_token": torch.tensor(0.5)})
