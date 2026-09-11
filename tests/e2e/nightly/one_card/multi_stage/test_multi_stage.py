# SPDX-License-Identifier: Apache-2.0
"""Real four-model NPU smoke tests and a top-k=1 greedy reference comparison."""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


@pytest.mark.parametrize("rounds", [1, 3])
@pytest.mark.parametrize("method,top_k", [("topk", 1), ("topk", 5), ("all", 5)])
@pytest.mark.parametrize("batch_size", [1, 4])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_four_model_generation(multi_stage_models, rounds, method, top_k, batch_size):
    target, primary, intermediate, secondary = multi_stage_models
    prompts = [
        "The capital of France is",
        "Explain briefly why leaves are green:",
        "Complete the sequence: 1, 2, 3,",
        "A short story about a lighthouse:\n",
    ][:batch_size]
    params = SamplingParams(temperature=0.0, max_tokens=16)
    common = dict(
        max_model_len=512, max_num_seqs=4, enforce_eager=True, async_scheduling=False, enable_prefix_caching=False
    )
    expected = None
    if method == "topk" and top_k == 1:
        with VllmRunner(target, **common) as baseline:
            expected = [r.outputs[0].token_ids for r in baseline.model.generate(prompts, params)]
    with VllmRunner(
        target,
        **common,
        speculative_config={"method": "dflash", "model": primary, "num_speculative_tokens": 2 + (rounds - 1) * 2},
        additional_config={
            "multi_stage_spec_config": {
                "enabled": True,
                "intermediate_model": intermediate,
                "secondary_model": secondary,
                "primary_num_speculative_tokens": 2,
                "secondary_num_speculative_tokens": 2,
                "num_intermediate_rounds": rounds,
                "intermediate_verification": {"method": method, "top_k": top_k},
                "debug_logging": True,
                "metrics_enabled": True,
            }
        },
    ) as runner:
        outputs = runner.model.generate(prompts, params)
        actual = [r.outputs[0].token_ids for r in outputs]
        assert len(actual) == batch_size
        assert all(0 < len(tokens) <= 16 for tokens in actual)
        assert all(r.finished for r in outputs)
        if expected is not None:
            assert actual == expected
        # Run a second batch to exercise request cleanup and KV block reuse.
        again = runner.model.generate(prompts[::-1], params)
        assert [r.outputs[0].token_ids for r in again] == actual[::-1]
