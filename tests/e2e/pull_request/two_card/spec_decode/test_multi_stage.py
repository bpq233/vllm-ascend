# SPDX-License-Identifier: Apache-2.0
"""Two-rank tensor-parallel smoke test for four-model decoding."""

import os
from unittest.mock import patch

from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_four_model_tensor_parallel_generation(multi_stage_models):
    target, primary, intermediate, secondary = multi_stage_models
    prompts = ["The capital of France is", "Complete the sequence: 1, 2, 3,"]
    params = SamplingParams(temperature=0.0, max_tokens=16)
    common = dict(
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        max_model_len=512,
        max_num_seqs=2,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
    )

    with VllmRunner(target, **common) as baseline:
        expected = [result.outputs[0].token_ids for result in baseline.model.generate(prompts, params)]

    with VllmRunner(
        target,
        **common,
        speculative_config={
            "method": "dflash",
            "model": primary,
            "num_speculative_tokens": 6,
            "draft_tensor_parallel_size": 2,
        },
        additional_config={
            "multi_stage_spec_config": {
                "enabled": True,
                "intermediate_model": intermediate,
                "secondary_model": secondary,
                "primary_num_speculative_tokens": 2,
                "secondary_num_speculative_tokens": 2,
                "num_intermediate_rounds": 3,
                "kv_cache_memory_bytes": 268435456,
                "intermediate_verification": {"method": "topk", "top_k": 1},
                "final_verification": {"method": "topk", "top_k": 1},
            }
        },
    ) as runner:
        outputs = runner.model.generate(prompts, params)

    assert [result.outputs[0].token_ids for result in outputs] == expected
