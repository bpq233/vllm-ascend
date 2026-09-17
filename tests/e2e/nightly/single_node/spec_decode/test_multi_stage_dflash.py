# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import DFLASH


@pytest.mark.parametrize("method", ["topk", "all"])
def test_multi_stage_dflash(method, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    models = DFLASH["dflash"]
    prompts = ["The capital of France is", "List three prime numbers:"]
    params = [SamplingParams(temperature=0, max_tokens=n, ignore_eos=True) for n in (17, 31)]
    common = dict(
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
    )
    reference = None
    if method == "topk":
        with VllmRunner(models["main"], **common) as runner:
            reference = [out.outputs[0].token_ids for out in runner.model.generate(prompts, params)]
    options = {
        "primary_num_speculative_tokens": 2,
        "intermediate": {
            "verifier": {"model": models["main"]},
            "drafter": {"model": models["spec"]},
            "num_rounds": 2,
            "num_speculative_tokens": 2,
            "max_num_seqs": 2,
            "verification": {"method": "topk", "top_k": 1},
        },
        "final_verification": {"method": method, "top_k": 1},
    }
    with VllmRunner(
        models["main"],
        **common,
        speculative_config={"method": "dflash", "model": models["spec"], "num_speculative_tokens": 6},
        additional_config={"multi_stage_speculative": options},
    ) as runner:
        tokens = [out.outputs[0].token_ids for out in runner.model.generate(prompts, params)]
        assert [len(row) for row in tokens] == [17, 31]
        if reference is not None:
            assert tokens == reference
        # Recycle request slots and scratch KV with a different prefix.
        output = runner.model.generate(
            ["One plus one equals"], SamplingParams(temperature=0, max_tokens=9, ignore_eos=True)
        )
        assert len(output[0].outputs[0].token_ids) == 9
