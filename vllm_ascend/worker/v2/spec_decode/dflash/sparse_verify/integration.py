# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Early scheduler sizing and request validation for the opt-in experiment."""

from copy import copy

from .config import SparseVerifyConfig


def configure_sparse_verification(vllm_config) -> None:
    config = SparseVerifyConfig.from_additional_config(vllm_config.additional_config)
    if not config.enabled:
        return
    spec = vllm_config.speculative_config
    if spec is None or spec.method != "dflash":
        raise ValueError("dflash_sparse_verify requires speculative_config.method=dflash")
    architectures = spec.draft_model_config.architectures or []
    if architectures != ["DFlashDraftModel"] and architectures != ["Qwen3DFlashDraftModel"]:
        # The actual registered upstream class is DFlashDraftModel. In particular,
        # DFlash2/DSpark have different anchors and must never enter this path.
        if "DFlashDraftModel" not in architectures:
            raise ValueError("dflash_sparse_verify requires an original DFlashDraftModel checkpoint")
    if vllm_config.model_config.hf_text_config.model_type != "qwen3":
        raise ValueError("dflash_sparse_verify currently supports dense Qwen3 targets only")
    if not getattr(vllm_config, "use_v2_model_runner", True):
        raise ValueError("dflash_sparse_verify requires the V2 model runner")
    if getattr(spec, "rejection_sample_method", "standard") != "standard":
        raise ValueError("dflash_sparse_verify requires standard rejection sampling (no synthetic/block/fly)")
    if getattr(vllm_config, "watermark_config", None) is not None:
        raise ValueError("dflash_sparse_verify does not support watermarking")
    if getattr(spec, "draft_sample_method", "greedy") != "greedy":
        raise ValueError("dflash_sparse_verify requires draft_sample_method=greedy")
    additional = vllm_config.additional_config or {}
    if additional.get("enable_kvpp", False) or additional.get("sparse_kv_offload_config", {}).get("enabled", False):
        raise ValueError("dflash_sparse_verify does not support KVPP or sparse KV offloading")
    if (vllm_config.additional_config or {}).get("finegrained_tp_config", {}).get("lmhead_tensor_parallel_size", 0):
        raise ValueError("dflash_sparse_verify does not support finegrained LM-head tensor parallelism")
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
        raise ValueError("dflash_sparse_verify requires PP=DP=PCP=DCP=1 (tensor parallelism is supported)")
    if vllm_config.kv_transfer_config is not None or vllm_config.lora_config is not None:
        raise ValueError("dflash_sparse_verify does not support KV transfer or LoRA")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("dflash_sparse_verify requires enable_prefix_caching=False for its private sparse KV history")
    if vllm_config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
        raise ValueError("dflash_sparse_verify requires unquantized KV caches")
    if spec.enable_adaptive_verification or spec.num_speculative_tokens_per_batch_size:
        raise ValueError("dflash_sparse_verify cannot be combined with upstream adaptive verification or dynamic K")
    hf_config = spec.draft_model_config.hf_config
    # Save the checkpoint block width on the serializable HF config. Platform
    # validation is called in the driver and workers; expansion is idempotent.
    block_tokens = getattr(hf_config, "_ascend_sparse_block_tokens", spec.num_speculative_tokens)
    budget = config.total_token_budget(block_tokens)
    if block_tokens + 1 > 16:
        raise ValueError("the original DFlash block must contain at most 15 candidates; use multiple rounds")
    if budget + 1 > vllm_config.scheduler_config.max_num_batched_tokens:
        raise ValueError("max_num_batched_tokens must accommodate num_rounds*K+1 in one logical Target verification")
    if budget + 1 >= vllm_config.model_config.max_model_len:
        raise ValueError("max_model_len must exceed num_rounds*K+1")
    hf_config._ascend_sparse_block_tokens = block_tokens
    spec.num_speculative_tokens = budget
    if hasattr(spec, "speculative_token_tree"):
        # Parallel DFlash has no autoregressive tree. Keep the scheduler's
        # capacity separate from the copied drafter's fixed checkpoint width.
        spec.speculative_token_tree = None


def draft_block_config(vllm_config):
    result = copy(vllm_config)
    result.speculative_config = copy(vllm_config.speculative_config)
    result.speculative_config.num_speculative_tokens = (
        result.speculative_config.draft_model_config.hf_config._ascend_sparse_block_tokens
    )
    return result


def validate_sparse_sampling(config: SparseVerifyConfig, params) -> None:
    config.validate_temperature(params.temperature)
    if getattr(params, "min_tokens", 0):
        raise ValueError("dflash_sparse_verify does not yet support min_tokens EOS masking")
    if params.logprobs is not None or params.prompt_logprobs is not None:
        raise ValueError("dflash_sparse_verify does not yet support logprobs")
    if (
        params.presence_penalty != 0
        or params.frequency_penalty != 0
        or params.repetition_penalty != 1
        or getattr(params, "logit_bias", None)
        or getattr(params, "allowed_token_ids", None)
        or getattr(params, "bad_words", None)
        or getattr(params, "structured_outputs", None)
    ):
        raise ValueError(
            "dflash_sparse_verify currently requires unmodified greedy logits (no penalties or constraints)"
        )
