# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Validated configuration for ``additional_config.dflash_sparse_verify``."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from math import isfinite
from typing import Any


def _integer(value: Any, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")


def _number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _check_keys(values: Mapping[str, Any], cls: type, path: str) -> None:
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {path} options: {sorted(unknown, key=str)}")


@dataclass(frozen=True)
class FinalVerificationConfig:
    method: str = "strict"
    topk: int = 2
    target_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.method not in ("strict", "topk", "confidence", "accept_all"):
            raise ValueError("final_verification.method must be strict, topk, confidence or accept_all")
        _integer(self.topk, "final_verification.topk")
        _number(self.target_threshold, "final_verification.target_threshold")
        if self.method == "topk" and self.topk < 1:
            raise ValueError("final_verification.topk must be positive")
        if self.method == "confidence" and not 0 < self.target_threshold <= 1:
            raise ValueError("final_verification.target_threshold must be in (0, 1]")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None = None) -> "FinalVerificationConfig":
        if values is None:
            return cls()
        values = _mapping(values, "final_verification")
        _check_keys(values, cls, "final_verification")
        return cls(**values)


@dataclass(frozen=True)
class SparseVerifyConfig:
    enabled: bool = False
    mode: str = "fixed"
    num_intermediate_layers: int = 12
    num_rounds: int = 4
    threshold_high: float = 0.9
    threshold_low: float = 0.5
    feature_source: str = "sparse"
    enable_metrics: bool = True
    final_verification: FinalVerificationConfig = field(default_factory=FinalVerificationConfig)

    def __post_init__(self) -> None:
        for name in ("enabled", "enable_metrics"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"dflash_sparse_verify.{name} must be a boolean")
        for name in ("num_intermediate_layers", "num_rounds"):
            value = getattr(self, name)
            _integer(value, name)
            if value < 1:
                raise ValueError(f"dflash_sparse_verify.{name} must be positive")
        if self.mode not in ("fixed", "confidence"):
            raise ValueError("dflash_sparse_verify.mode must be fixed or confidence")
        if self.feature_source not in ("sparse", "target"):
            raise ValueError("dflash_sparse_verify.feature_source must be sparse or target")
        _number(self.threshold_high, "threshold_high")
        _number(self.threshold_low, "threshold_low")
        if self.mode == "confidence" and not 0 <= self.threshold_low < self.threshold_high <= 1:
            raise ValueError("confidence mode requires 0 <= threshold_low < threshold_high <= 1")
        if not isinstance(self.final_verification, FinalVerificationConfig):
            raise TypeError("final_verification must be a FinalVerificationConfig")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None = None) -> "SparseVerifyConfig":
        if values is None:
            return cls()
        values = dict(_mapping(values, "dflash_sparse_verify"))
        _check_keys(values, cls, "dflash_sparse_verify")
        if "final_verification" in values:
            values["final_verification"] = FinalVerificationConfig.from_dict(values["final_verification"])
        return cls(**values)

    @classmethod
    def from_additional_config(cls, additional_config: Mapping[str, Any] | None) -> "SparseVerifyConfig":
        if additional_config is None:
            return cls()
        additional_config = _mapping(additional_config, "additional_config")
        return cls.from_dict(additional_config.get("dflash_sparse_verify"))

    def total_token_budget(self, draft_block_size: int) -> int:
        """Scheduler capacity; this does not change the checkpoint's block size."""
        _integer(draft_block_size, "draft_block_size")
        if draft_block_size < 1:
            raise ValueError("draft_block_size must be positive")
        return draft_block_size * self.num_rounds

    def validate_temperature(self, temperature: float) -> None:
        _number(temperature, "temperature")
        if self.enabled and temperature != 0:
            raise ValueError("dflash_sparse_verify supports temperature=0 only")
