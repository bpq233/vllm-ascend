# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SlimVerifierConfig:
    """Opt-in, diagnostic-only layer selection. Indices refer to target layers."""

    keep_layers: tuple[int, ...]
    log_interval: int = 100
    top_k: int = 5
    profile_latency: bool = False

    def __post_init__(self):
        if any(type(i) is not int or i < 0 for i in self.keep_layers) or len(set(self.keep_layers)) != len(
            self.keep_layers
        ):
            raise ValueError("keep_layers must contain unique nonnegative integer indices")
        object.__setattr__(self, "keep_layers", tuple(sorted(self.keep_layers)))
        if type(self.log_interval) is not int or self.log_interval <= 0:
            raise ValueError("log_interval must be a positive integer")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if type(self.profile_latency) is not bool:
            raise ValueError("profile_latency must be a boolean")

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
        unknown = raw.keys() - {"enabled", "keep_layers", "skip_layers", "log_interval", "top_k", "profile_latency"}
        if unknown:
            raise ValueError(f"Unknown slim_verifier_eval options: {sorted(unknown)}")
        if ("keep_layers" in raw) == ("skip_layers" in raw):
            raise ValueError("Specify exactly one of keep_layers and skip_layers")
        if num_layers <= 0:
            raise ValueError("Target must contain decoder layers")
        field = "keep_layers" if "keep_layers" in raw else "skip_layers"
        layers = raw[field]
        if not isinstance(layers, (list, tuple)) or any(type(i) is not int for i in layers):
            raise ValueError(f"{field} must be a list of integer layer indices")
        if len(set(layers)) != len(layers) or any(i < 0 or i >= num_layers for i in layers):
            raise ValueError(f"{field} must contain unique indices in [0, {num_layers})")
        keep = sorted(layers) if field == "keep_layers" else [i for i in range(num_layers) if i not in layers]
        for key, default in (("log_interval", 100), ("top_k", 5)):
            if type(raw.get(key, default)) is not int or raw.get(key, default) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(raw.get("profile_latency", False)) is not bool:
            raise ValueError("profile_latency must be a boolean")
        return cls(tuple(keep), raw.get("log_interval", 100), raw.get("top_k", 5), raw.get("profile_latency", False))


def slim_verifier_enabled(additional_config: Mapping | None) -> bool:
    raw = (additional_config or {}).get("slim_verifier_eval", {})
    if not isinstance(raw, Mapping) or type(raw.get("enabled", False)) is not bool:
        raise ValueError("slim_verifier_eval must be an object with a boolean enabled field")
    return raw.get("enabled", False)
