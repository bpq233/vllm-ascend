# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Validate opt-in exact proposal configuration without loading NPU modules."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
from test_intermediate import settings, vllm_config

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage"


@pytest.fixture
def configs(monkeypatch):
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    proposal = load("vllm_ascend.worker.v2.spec_decode.multi_stage.proposal_config", "proposal_config.py")
    load("vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance", "acceptance.py")
    multi = load("_proposal_multi_config_contract", "config.py")
    return proposal, multi


def enabled_options(**proposal):
    options = settings(num_speculative_tokens=4, num_rounds=3)
    options.pop("final_verification")
    options["primary_num_speculative_tokens"] = 4
    options["proposal_sampling"] = {"enabled": True, **proposal}
    return options


def test_default_stays_disabled_and_has_no_mutable_candidate_alias(configs):
    proposal, _ = configs
    first, second = proposal.ProposalSamplingConfig.from_dict(), proposal.ProposalSamplingConfig.from_dict()
    assert not first.enabled
    assert first.mode == "fixed" and first.mixture_lambda == 0.5
    assert first.lambda_candidates == [index / 10 for index in range(11)]
    assert first.lambda_candidates is not second.lambda_candidates
    assert not proposal.proposal_sampling_enabled({})
    assert not proposal.proposal_sampling_enabled({"proposal_sampling": None})


@pytest.mark.parametrize("weight", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("primary", [False, True])
def test_fixed_weight_and_capacity_are_idempotent(configs, weight, primary):
    _, multi = configs
    cfg = vllm_config()
    options = enabled_options(mixture_lambda=weight)
    options["use_primary_drafter"] = primary
    if not primary:
        options.pop("primary_num_speculative_tokens")
    cfg.additional_config = {"multi_stage_speculative": options}
    for _ in range(2):
        multi.prepare_multi_stage_config(cfg)
        multi.validate_multi_stage(cfg, options)
        assert cfg.speculative_config.num_speculative_tokens == 12
        assert options["final_verification"] == {"method": "standard"}


@pytest.mark.parametrize("primary", [False, True])
def test_fifteen_token_three_round_capacity_excludes_intermediate_bonus(configs, primary):
    _, multi = configs
    cfg = vllm_config()
    options = enabled_options()
    options["use_primary_drafter"] = primary
    options["intermediate"]["num_speculative_tokens"] = 15
    if primary:
        options["primary_num_speculative_tokens"] = 15
    else:
        options.pop("primary_num_speculative_tokens")
    cfg.additional_config = {"multi_stage_speculative": options}
    for _ in range(2):
        multi.prepare_multi_stage_config(cfg)
        multi.validate_multi_stage(cfg, options)
        assert cfg.speculative_config.num_speculative_tokens == 45


@pytest.mark.parametrize("rounds,cap,expected", [(3, 100, 45), (10, 100, 100), (3, 7, 7), (3, 16, 16)])
def test_max_generated_tokens_is_a_hard_candidate_cap(configs, rounds, cap, expected):
    _, multi = configs
    cfg, options = vllm_config(), enabled_options()
    options["primary_num_speculative_tokens"] = 15
    options["intermediate"].update(num_rounds=rounds, num_speculative_tokens=15, max_generated_tokens=cap)
    cfg.additional_config = {"multi_stage_speculative": options}
    for _ in range(2):
        multi.prepare_multi_stage_config(cfg)
        multi.validate_multi_stage(cfg, options)
        assert cfg.speculative_config.num_speculative_tokens == expected


@pytest.mark.parametrize("cap", [0, -1, True, 2.5, "100"])
def test_invalid_candidate_caps_are_rejected(configs, cap):
    proposal, _ = configs
    options = enabled_options()
    options["intermediate"]["max_generated_tokens"] = cap
    with pytest.raises(ValueError, match="positive integer candidate cap"):
        proposal.validate_proposal_sampling(vllm_config(), options)


@pytest.mark.parametrize("policy", [None, {"method": "standard"}])
def test_explicit_standard_policy_or_omission_is_supported(configs, policy):
    _, multi = configs
    options = enabled_options()
    if policy is not None:
        options["intermediate"]["verification"] = policy.copy()
        options["final_verification"] = policy.copy()
    multi.validate_multi_stage(vllm_config(), options)
    assert options["final_verification"] == {"method": "standard"}


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_adaptive_defaults_do_not_require_training_or_oracle(configs, mode):
    proposal, multi = configs
    options = enabled_options(mode=mode)
    config = proposal.validate_proposal_sampling(vllm_config(), options)
    assert config.mode == mode
    assert config.feedback_enabled
    multi.validate_multi_stage(vllm_config(), options)


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", 1),
        ("enabled", "true"),
        ("feedback_enabled", None),
        ("debug", 0),
        ("mixture_lambda", True),
        ("mixture_lambda", -0.1),
        ("mixture_lambda", 1.1),
        ("mixture_lambda", math.nan),
        ("mixture_lambda", math.inf),
        ("ema_decay", 1),
        ("ema_decay", -1),
        ("min_observations", 0),
        ("min_observations", True),
        ("min_observations", 2.0),
        ("lambda_candidates", []),
        ("lambda_candidates", (0.0, 1.0)),
        ("lambda_candidates", [0.1, 0.1]),
        ("lambda_candidates", [True]),
        ("lambda_candidates", [math.nan]),
        ("lambda_candidates", [1.01]),
        ("mode", "oracle"),
        ("mode", "unknown"),
    ],
)
def test_invalid_sampling_options_fail_clearly(configs, field, value):
    proposal, _ = configs
    with pytest.raises(ValueError, match="proposal_sampling"):
        proposal.ProposalSamplingConfig.from_dict({field: value})


@pytest.mark.parametrize("values", [False, [], "enabled"])
def test_sampling_options_must_be_a_mapping(configs, values):
    proposal, _ = configs
    with pytest.raises(ValueError, match="mapping"):
        proposal.ProposalSamplingConfig.from_dict(values)


def test_unknown_option_and_adaptive_without_feedback_are_rejected(configs):
    proposal, _ = configs
    with pytest.raises(ValueError, match="Unknown proposal_sampling"):
        proposal.ProposalSamplingConfig.from_dict({"oracle": True})
    with pytest.raises(ValueError, match="requires feedback_enabled"):
        proposal.ProposalSamplingConfig.from_dict({"enabled": True, "mode": "adaptive", "feedback_enabled": False})


@pytest.mark.parametrize("location", ["final_verification", "intermediate"])
def test_approximate_policy_cannot_be_mixed_with_exact_proposal_mode(configs, location):
    proposal, _ = configs
    options = enabled_options()
    if location == "intermediate":
        options[location]["verification"] = {"method": "topk", "top_k": 1}
    else:
        options[location] = {"method": "prob_ratio"}
    with pytest.raises(ValueError, match="verification"):
        proposal.validate_proposal_sampling(vllm_config(), options)


@pytest.mark.parametrize("kind", ["absent", "zero_rounds", "threshold", "probabilistic", "synthetic", "trace"])
def test_unsupported_probability_and_stopping_contracts_are_rejected(configs, kind):
    proposal, _ = configs
    cfg, options = vllm_config(), enabled_options()
    if kind == "absent":
        options.pop("intermediate")
    elif kind == "zero_rounds":
        options["intermediate"]["num_rounds"] = 0
    elif kind == "threshold":
        options["intermediate"]["max_generated_tokens"] = 0
    elif kind == "probabilistic":
        cfg.speculative_config.draft_sample_method = "probabilistic"
    elif kind == "synthetic":
        cfg.speculative_config.rejection_sample_method = "synthetic"
    else:
        cfg.model_config.enable_trace_replay = True
    with pytest.raises(ValueError):
        proposal.validate_proposal_sampling(cfg, options)


def test_disabled_mode_preserves_legacy_approximate_configuration(configs):
    _, multi = configs
    cfg = vllm_config()
    options = settings(num_rounds=2)
    options["primary_num_speculative_tokens"] = 2
    options["proposal_sampling"] = {"enabled": False}
    multi.validate_multi_stage(cfg, options)
    assert options["final_verification"] == {"method": "all"}
    assert cfg.speculative_config.num_speculative_tokens == 8


@pytest.mark.parametrize(
    "intermediate", ["invalid", [1], {"num_rounds": "3"}, {"num_rounds": True}, {"num_rounds": -1}]
)
def test_invalid_intermediate_shape_or_rounds_fail_before_probability_setup(configs, intermediate):
    proposal, _ = configs
    options = enabled_options()
    options["intermediate"] = intermediate
    with pytest.raises(ValueError, match="active intermediate"):
        proposal.validate_proposal_sampling(vllm_config(), options)
