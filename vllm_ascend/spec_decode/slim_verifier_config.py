# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import math
from collections.abc import Mapping
from dataclasses import dataclass


def validate_acceptance_strategy(strategy: str, threshold: float) -> None:
    if strategy not in ("top1", "probability_ratio"):
        raise ValueError("acceptance_strategy must be 'top1' or 'probability_ratio'")
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1 or not math.isfinite(threshold):
        raise ValueError("probability_ratio_threshold must be a finite number in [0, 1]")


@dataclass(frozen=True)
class SlimVerifierConfig:
    """Opt-in, diagnostic-only layer selection. Indices refer to target layers."""

    keep_layers: tuple[int, ...]
    log_interval: int = 100
    top_k: int = 5
    profile_latency: bool = False
    acceptance_strategy: str = "top1"
    probability_ratio_threshold: float = 0.5
    kv_source: str = "target"
    skip_attention_layers: tuple[int, ...] = ()
    skip_mlp_layers: tuple[int, ...] = ()

    def __post_init__(self):
        if any(type(i) is not int or i < 0 for i in self.keep_layers) or len(set(self.keep_layers)) != len(
            self.keep_layers
        ):
            raise ValueError("keep_layers must contain unique nonnegative integer indices")
        object.__setattr__(self, "keep_layers", tuple(sorted(self.keep_layers)))
        for field in ("skip_attention_layers", "skip_mlp_layers"):
            indices = getattr(self, field)
            if any(type(i) is not int or i not in self.keep_layers for i in indices) or len(set(indices)) != len(
                indices
            ):
                raise ValueError(f"{field} must contain unique indices from keep_layers")
            object.__setattr__(self, field, tuple(sorted(indices)))
        if type(self.log_interval) is not int or self.log_interval <= 0:
            raise ValueError("log_interval must be a positive integer")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if type(self.profile_latency) is not bool:
            raise ValueError("profile_latency must be a boolean")
        validate_acceptance_strategy(self.acceptance_strategy, self.probability_ratio_threshold)
        if self.kv_source not in ("target", "replay"):
            raise ValueError("kv_source must be 'target' or 'replay'")

    @classmethod
    def from_additional_config(cls, additional_config: Mapping | None, num_layers: int):
        raw = (additional_config or {}).get("slim_verifier_eval", {})
        if not isinstance(raw, Mapping):
            raise ValueError("slim_verifier_eval must be an object")
        enabled = raw.get("enabled", False)
        if type(enabled) is not bool:
            raise ValueError("slim_verifier_eval.enabled must be a boolean")
        if not enabled:
            return None
        unknown = raw.keys() - {
            "enabled",
            "keep_layers",
            "skip_layers",
            "log_interval",
            "top_k",
            "profile_latency",
            "acceptance_strategy",
            "probability_ratio_threshold",
            "kv_source",
            "skip_attention_layers",
            "skip_mlp_layers",
        }
        if unknown:
            raise ValueError(f"Unknown slim_verifier_eval options: {sorted(unknown)}")
        if "keep_layers" in raw and "skip_layers" in raw:
            raise ValueError("Specify exactly one of keep_layers and skip_layers")
        if not any(k in raw for k in ("keep_layers", "skip_layers", "skip_attention_layers", "skip_mlp_layers")):
            raise ValueError("Specify a whole-layer or sublayer selection")
        if num_layers <= 0:
            raise ValueError("Target must contain decoder layers")
        field = "keep_layers" if "keep_layers" in raw else "skip_layers"
        layers = raw.get(field, [])
        if not isinstance(layers, (list, tuple)) or any(type(i) is not int for i in layers):
            raise ValueError(f"{field} must be a list of integer layer indices")
        if len(set(layers)) != len(layers) or any(i < 0 or i >= num_layers for i in layers):
            raise ValueError(f"{field} must contain unique indices in [0, {num_layers})")
        keep = sorted(layers) if field == "keep_layers" else [i for i in range(num_layers) if i not in layers]
        for sublayer in ("skip_attention_layers", "skip_mlp_layers"):
            if not isinstance(raw.get(sublayer, []), (list, tuple)):
                raise ValueError(f"{sublayer} must be a list of integer layer indices")
        for key, default in (("log_interval", 100), ("top_k", 5)):
            if type(raw.get(key, default)) is not int or raw.get(key, default) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(raw.get("profile_latency", False)) is not bool:
            raise ValueError("profile_latency must be a boolean")
        return cls(
            keep_layers=tuple(keep),
            log_interval=raw.get("log_interval", 100),
            top_k=raw.get("top_k", 5),
            profile_latency=raw.get("profile_latency", False),
            acceptance_strategy=raw.get("acceptance_strategy", "top1"),
            probability_ratio_threshold=raw.get("probability_ratio_threshold", 0.5),
            kv_source=raw.get("kv_source", "target"),
            skip_attention_layers=tuple(raw.get("skip_attention_layers", [])),
            skip_mlp_layers=tuple(raw.get("skip_mlp_layers", [])),
        )


def slim_verifier_enabled(additional_config: Mapping | None) -> bool:
    raw = (additional_config or {}).get("slim_verifier_eval", {})
    if not isinstance(raw, Mapping) or type(raw.get("enabled", False)) is not bool:
        raise ValueError("slim_verifier_eval must be an object with a boolean enabled field")
    return raw.get("enabled", False)
