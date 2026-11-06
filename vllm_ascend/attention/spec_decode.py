# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import numpy as np

MAX_DECODE_QUERY_LEN = 16


def uses_long_speculative_queries(vllm_config) -> bool:
    """Only an active multi-stage target may exceed FIA's decode limit."""
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
    return min(width, MAX_DECODE_QUERY_LEN) if uses_long_speculative_queries(vllm_config) else width


def prepare_verification_batch_req_state(batch_state):
    """Keep short FIA queries first, before inputs and graph dispatch.

    Upstream sorts verification requests ahead of decodes. A long verification
    must instead follow every short query for the attention backend's split.
    Reorder all per-request arrays together, including new upstream fields.
    """
    long_queries = batch_state.num_scheduled_tokens > MAX_DECODE_QUERY_LEN
    if not np.any(long_queries):
        return batch_state
    order = np.argsort(long_queries, kind="stable")
    if np.array_equal(order, np.arange(len(order))):
        return batch_state
    changes = {"req_ids": [batch_state.req_ids[index] for index in order]}
    for name in batch_state._fields:
        values = getattr(batch_state, name)
        if isinstance(values, np.ndarray) and values.ndim > 0 and values.shape[0] == len(order):
            changes[name] = values[order]
    return batch_state._replace(**changes)
