# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

MAX_DECODE_QUERY_LEN = 16


def uses_long_speculative_queries(vllm_config) -> bool:
    """Only the multi-stage target may exceed the short decode kernel budget."""
    spec = vllm_config.speculative_config
    options = (getattr(vllm_config, "additional_config", None) or {}).get("multi_stage_speculative", {})
    return bool(
        getattr(vllm_config, "use_v2_model_runner", False)
        and spec is not None
        and spec.num_speculative_tokens + 1 > MAX_DECODE_QUERY_LEN
        and "intermediate" in options
        and options["intermediate"].get("num_rounds", 3) > 0
    )


def speculative_decode_threshold(vllm_config) -> int:
    spec = vllm_config.speculative_config
    width = 1 + spec.num_speculative_tokens if spec else 1
    if uses_long_speculative_queries(vllm_config):
        # This is the classification boundary, NOT the candidate/storage width.
        # Longer queries are executed as causal cached prefill by MRV2.
        return min(width, MAX_DECODE_QUERY_LEN)
    return width
