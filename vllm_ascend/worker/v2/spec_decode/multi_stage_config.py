# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from dataclasses import dataclass, field


@dataclass
class IntermediateConfig:
    verifier_model: str
    drafter_model: str
    num_rounds: int = 3
    num_speculative_tokens: int = 4
    max_num_seqs: int = 4
    max_model_len: int | None = None
    verification: dict = field(default_factory=lambda: {"method": "topk", "top_k": 5})

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        verifier = values.pop("verifier")
        drafter = values.pop("drafter")
        result = cls(verifier_model=verifier["model"], drafter_model=drafter["model"], **values)
        if not result.verifier_model or not result.drafter_model:
            raise ValueError("Both intermediate model paths are required.")
        for name in ("num_rounds", "num_speculative_tokens", "max_num_seqs", "max_model_len"):
            value = getattr(result, name)
            minimum = 0 if name == "num_rounds" else 1
            if value is None and name == "max_model_len":
                continue
            if type(value) is not int or value < minimum:
                raise ValueError(f"intermediate.{name} must be an integer >= {minimum}.")
        return result


def validate_multi_stage(vllm_config, config):
    """Validate before loading weights; the upstream width is the final budget."""
    if not config:
        return
    if set(config) - {"primary_num_speculative_tokens", "intermediate", "final_verification"}:
        raise ValueError("Unknown multi_stage_speculative option.")
    from vllm_ascend.worker.v2.spec_decode.acceptance import AcceptancePolicy

    if "final_verification" in config:
        AcceptancePolicy(**config["final_verification"])
    spec = vllm_config.speculative_config
    if not vllm_config.use_v2_model_runner or spec is None:
        raise ValueError("multi_stage_speculative requires MRV2 and speculative decoding.")
    if "intermediate" not in config:
        if "primary_num_speculative_tokens" in config:
            raise ValueError("primary_num_speculative_tokens requires intermediate.")
        return
    intermediate = IntermediateConfig.from_dict(config["intermediate"])
    AcceptancePolicy(**intermediate.verification)
    width = config.get("primary_num_speculative_tokens", spec.num_speculative_tokens)
    if type(width) is not int or not 0 < width <= spec.num_speculative_tokens:
        raise ValueError("primary_num_speculative_tokens must be within the final speculative budget.")
    if intermediate.num_rounds == 0:
        if width != spec.num_speculative_tokens:
            raise ValueError("With zero intermediate rounds, primary and final widths must match.")
        return
    if not spec.use_dflash():
        raise ValueError("The intermediate pipeline currently requires a DFlash primary drafter.")
    if "final_verification" not in config:
        raise ValueError("Intermediate candidates require explicit final_verification: topk or all.")
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    ):
        raise ValueError("Intermediate decoding currently supports TP only (PP/DP/CP must be 1).")
    if vllm_config.scheduler_config.async_scheduling:
        raise ValueError("Variable-length intermediate candidates require --no-async-scheduling.")
    if vllm_config.lora_config or vllm_config.model_config.is_multimodal_model:
        raise ValueError("Intermediate decoding currently supports text-only models without LoRA.")
    if getattr(spec, "enable_adaptive_verification", False):
        raise ValueError("Intermediate decoding cannot reuse primary adaptive-verification confidences.")
