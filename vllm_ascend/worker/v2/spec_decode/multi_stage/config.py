# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in configuration, kept outside upstream SpeculativeConfig."""

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class VerificationConfig:
    method: str = "topk"
    top_k: int = 5

    def __post_init__(self):
        if self.method not in ("topk", "all"):
            raise ValueError("verification.method must be 'topk' or 'all'")
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("verification.top_k must be a positive integer")


@dataclass(frozen=True)
class MultiStageConfig:
    enabled: bool = False
    intermediate_model: str = ""
    secondary_model: str = ""
    primary_num_speculative_tokens: int = 8
    secondary_num_speculative_tokens: int = 4
    num_intermediate_rounds: int = 3
    intermediate_revision: str | None = None
    secondary_revision: str | None = None
    # Explicit private KV budget, deducted before target memory profiling.
    kv_cache_memory_bytes: int = 1 << 30
    intermediate_verification: VerificationConfig = VerificationConfig()
    final_verification: VerificationConfig = VerificationConfig()
    debug_logging: bool = False
    metrics_enabled: bool = False

    def __post_init__(self):
        for name in ("enabled", "debug_logging", "metrics_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"multi_stage_spec_config.{name} must be a boolean")
        for name in (
            "primary_num_speculative_tokens",
            "secondary_num_speculative_tokens",
            "num_intermediate_rounds",
            "kv_cache_memory_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"multi_stage_spec_config.{name} must be a positive integer")
        if self.enabled and (not self.intermediate_model or not self.secondary_model):
            raise ValueError("Enabled multi-stage decoding requires intermediate_model and secondary_model")
        for name in ("intermediate_model", "secondary_model"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"multi_stage_spec_config.{name} must be a model path or identifier")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "MultiStageConfig":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("additional_config.multi_stage_spec_config must be an object")
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown multi-stage options: {sorted(unknown)}")
        values = dict(raw)
        for name in ("intermediate_verification", "final_verification"):
            if name in values:
                if not isinstance(values[name], dict):
                    raise ValueError(f"{name} must be an object")
                values[name] = VerificationConfig(**values[name])
        return cls(**values)

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "MultiStageConfig":
        return cls.from_dict((vllm_config.additional_config or {}).get("multi_stage_spec_config"))

    def validate_runtime(self, vllm_config: Any) -> None:
        if not self.enabled:
            return
        if not getattr(vllm_config, "use_v2_model_runner", True):
            raise ValueError("Multi-stage decoding requires VLLM_USE_V2_MODEL_RUNNER=1")
        if getattr(vllm_config.model_config, "is_moe", False):
            raise ValueError("Multi-stage decoding currently supports dense models only")
        spec = vllm_config.speculative_config
        if spec is None or not spec.use_dflash():
            raise ValueError("Multi-stage decoding requires speculative_config.method='dflash'")
        if spec.num_speculative_tokens < self.primary_num_speculative_tokens:
            raise ValueError("speculative_config.num_speculative_tokens must cover the primary draft width")
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError("Multi-stage decoding currently requires async_scheduling=False")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("Multi-stage decoding currently requires enforce_eager=True")
        parallel = vllm_config.parallel_config
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        ):
            if getattr(parallel, name, 1) != 1:
                raise ValueError(f"Multi-stage decoding currently requires {name}=1")
        for name in ("lora_config", "kv_transfer_config", "ec_transfer_config"):
            if getattr(vllm_config, name, None) is not None:
                raise ValueError(f"Multi-stage decoding does not yet support {name}")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Multi-stage DFlash currently requires enable_prefix_caching=False")
        if getattr(vllm_config.model_config, "enable_trace_replay", False):
            raise ValueError("Multi-stage decoding does not support trace replay")
        if getattr(vllm_config.model_config, "return_sampling_mask", False):
            raise ValueError("Multi-stage decoding does not support return_sampling_mask")
        if getattr(spec, "enable_adaptive_verification", False):
            raise ValueError("Multi-stage decoding does not support adaptive verification")


def make_policy(config: VerificationConfig):
    # Lazy import keeps config validation usable without a device runtime.
    from .acceptance import AcceptAllPolicy, TopKPolicy

    return TopKPolicy(config.top_k) if config.method == "topk" else AcceptAllPolicy()
