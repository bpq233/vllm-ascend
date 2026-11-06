# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Lightweight opt-in configuration; safe during platform initialization."""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProposalSamplingConfig:
    enabled: bool = False
    mode: str = "fixed"
    mixture_lambda: float = 0.5
    lambda_candidates: list[float] = field(default_factory=lambda: [index / 10 for index in range(11)])
    feedback_enabled: bool = True
    debug: bool = False
    ema_decay: float = 0.9
    min_observations: int = 8

    @classmethod
    def from_dict(cls, values=None):
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError("proposal_sampling must be a mapping.")
        if set(values) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown proposal_sampling option.")
        result = cls(**values)
        for name in ("enabled", "feedback_enabled", "debug"):
            if type(getattr(result, name)) is not bool:
                raise ValueError(f"proposal_sampling.{name} must be boolean.")
        if result.mode not in ("fixed", "adaptive"):
            raise ValueError("proposal_sampling.mode must be fixed or adaptive; oracle is offline-only.")
        for name in ("mixture_lambda", "ema_decay"):
            value = getattr(result, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"proposal_sampling.{name} must be finite and in [0, 1].")
        if result.ema_decay == 1:
            raise ValueError("proposal_sampling.ema_decay must be less than 1.")
        if type(result.min_observations) is not int or result.min_observations < 1:
            raise ValueError("proposal_sampling.min_observations must be a positive integer.")
        candidates = result.lambda_candidates
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in candidates)
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("proposal_sampling.lambda_candidates must be distinct finite values in [0, 1].")
        if result.enabled and result.mode == "adaptive" and not result.feedback_enabled:
            raise ValueError("Adaptive proposal sampling requires feedback_enabled=true.")
        return result


def proposal_sampling_enabled(options):
    return ProposalSamplingConfig.from_dict(options.get("proposal_sampling")).enabled


def validate_proposal_sampling(vllm_config, options):
    config = ProposalSamplingConfig.from_dict(options.get("proposal_sampling"))
    if not config.enabled:
        return config
    intermediate = options.get("intermediate", {})
    rounds = intermediate.get("num_rounds", 3) if isinstance(intermediate, dict) else None
    if not intermediate or type(rounds) is not int or rounds <= 0:
        raise ValueError("Proposal sampling requires an active intermediate pipeline.")
    cap = intermediate.get("max_generated_tokens")
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ValueError("Proposal sampling max_generated_tokens must be a positive integer candidate cap.")
    for name, policy in (
        ("intermediate", intermediate.get("verification")),
        ("final", options.get("final_verification")),
    ):
        if policy is not None and policy != {"method": "standard"}:
            raise ValueError(f"Proposal sampling requires {name} verification={{'method': 'standard'}} or omission.")
    spec = vllm_config.speculative_config
    if spec is None or getattr(spec, "draft_sample_method", "greedy") != "greedy":
        raise ValueError("Proposal sampling currently requires greedy DFlash, whose true qD is a delta distribution.")
    if getattr(spec, "rejection_sample_method", "standard") != "standard":
        raise ValueError("Proposal sampling requires rejection_sample_method='standard'.")
    if getattr(vllm_config.model_config, "enable_trace_replay", False):
        raise ValueError("Proposal sampling cannot force trace-replay tokens while claiming the Target sampling law.")
    options["final_verification"] = {"method": "standard"}
    return config
