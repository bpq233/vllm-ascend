# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from dataclasses import dataclass, field

MAX_DFLASH_DRAFT_TOKENS = 15


def intermediate_capture_sizes(max_tokens, max_reqs, draft_width, requested=None):
    """Sparse gears: four resident models share the device's stream budget."""
    if requested is not None:
        if any(size > max_tokens for size in requested):
            raise ValueError("intermediate.cudagraph_capture_sizes must fit the intermediate token buffer.")
        sizes = set(requested)
    else:
        sizes = {1}
        size = 16
        while size < max_tokens:
            sizes.add(size)
            size *= 4
    # Cover the largest secondary batch even if the verifier gears are sparse.
    # Smaller DFlash batches pad to a captured request count.
    sizes.update((max_reqs * (draft_width + 1), max_tokens))
    return sorted(sizes)


def primary_draft_width(options, final_capacity):
    # A long final budget accumulates multiple short drafts. Do not silently
    # turn that budget into an unsupported single DFlash query.
    default = 4 if final_capacity > MAX_DFLASH_DRAFT_TOKENS else final_capacity
    return options.get("primary_num_speculative_tokens", default)


def configure_long_target_graphs(vllm_config):
    """Enable attention splitting before compilation and graph sizing.

    The target manager uses PIECEWISE for variable cached-prefill queries.
    Keep FULL decode enabled at the config level so the independent DFlash
    manager still captures its short, uniform queries and gets an update stream.
    """
    from vllm.config import CompilationMode
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.attention.spec_decode import uses_long_speculative_queries

    compilation = vllm_config.compilation_config
    if (
        not uses_long_speculative_queries(vllm_config)
        or vllm_config.model_config.enforce_eager
        or compilation.cudagraph_mode == CUDAGraphMode.NONE
    ):
        return
    if compilation.cudagraph_mode.has_full_cudagraphs():
        compilation.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    compilation.mode = CompilationMode.VLLM_COMPILE
    # FULL configurations may already carry an empty split list. Upstream
    # otherwise silently resolves FULL_AND_PIECEWISE back to FULL, leaving no
    # compiled pieces to capture. Preserve custom splits and add the dynamic
    # attention boundary; None lets upstream install its complete default set.
    if not compilation.splitting_ops:
        compilation.splitting_ops = None
    elif "vllm::unified_attention_with_output" not in compilation.splitting_ops:
        compilation.splitting_ops = [*compilation.splitting_ops, "vllm::unified_attention_with_output"]


@dataclass
class IntermediateConfig:
    verifier_model: str
    drafter_model: str
    num_rounds: int = 3
    num_speculative_tokens: int = 4
    max_num_seqs: int = 4
    max_model_len: int | None = None
    cudagraph_capture_sizes: list[int] | None = None
    verification: dict = field(default_factory=lambda: {"method": "topk", "top_k": 5})

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        verifier = values.pop("verifier")
        drafter = values.pop("drafter")
        result = cls(verifier_model=verifier["model"], drafter_model=drafter["model"], **values)
        if not result.verifier_model or not result.drafter_model:
            raise ValueError("Both intermediate model paths are required.")
        sizes = result.cudagraph_capture_sizes
        if sizes is not None and (
            not isinstance(sizes, list) or not sizes or any(type(size) is not int or size <= 0 for size in sizes)
        ):
            raise ValueError("intermediate.cudagraph_capture_sizes must be a nonempty list of positive integers.")
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
    from vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance import AcceptancePolicy

    if "final_verification" in config:
        AcceptancePolicy(**config["final_verification"])
    spec = vllm_config.speculative_config
    if not vllm_config.use_v2_model_runner or spec is None:
        raise ValueError("multi_stage_speculative requires MRV2 and speculative decoding.")
    # DFlash query kernels retain their original short-query contract. Only
    # the target's extended candidates may use cached-prefill verification.
    long_candidates = spec.num_speculative_tokens > MAX_DFLASH_DRAFT_TOKENS
    if long_candidates and ("intermediate" not in config or config["intermediate"].get("num_rounds", 3) == 0):
        raise ValueError("Long candidates require an active intermediate DFlash pipeline.")
    if "intermediate" not in config:
        if "primary_num_speculative_tokens" in config:
            raise ValueError("primary_num_speculative_tokens requires intermediate.")
        return
    intermediate = IntermediateConfig.from_dict(config["intermediate"])
    AcceptancePolicy(**intermediate.verification)
    width = primary_draft_width(config, spec.num_speculative_tokens)
    if type(width) is not int or not 0 < width <= spec.num_speculative_tokens:
        raise ValueError("primary_num_speculative_tokens must be within the final speculative budget.")
    if long_candidates:
        if width > MAX_DFLASH_DRAFT_TOKENS or intermediate.num_speculative_tokens > MAX_DFLASH_DRAFT_TOKENS:
            raise ValueError(
                f"DFlash per-round widths must be <= 15: primary_num_speculative_tokens={width}, "
                f"intermediate.num_speculative_tokens={intermediate.num_speculative_tokens}; "
                f"final speculative_config.num_speculative_tokens={spec.num_speculative_tokens} is allowed. "
                "Set both draft widths to 4 and increase intermediate.num_rounds to accumulate long candidates."
            )
        if getattr(vllm_config.model_config, "use_mla", False) or getattr(vllm_config.model_config, "is_hybrid", False):
            raise ValueError("Long candidate verification currently requires a full-attention target.")
        if vllm_config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
            raise ValueError("Long candidate verification requires unquantized target KV cache.")
        if vllm_config.scheduler_config.max_num_batched_tokens < spec.num_speculative_tokens + 1:
            raise ValueError("max_num_batched_tokens must fit the final candidate budget plus its context token.")
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
