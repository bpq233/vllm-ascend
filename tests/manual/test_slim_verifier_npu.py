# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real NPU regression: eager/graph, TP=1/2, all layers/reduced layers.

Run on an installed Ascend environment with the Qwen3-0.6B checkpoint:
pytest tests/manual/test_slim_verifier_npu.py -v -s
"""

from typing import Any

import pytest
import torch

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "Continue the sequence: 1, 2, 3, 4, 5,",
    "Write a short story about a small robot exploring a garden.",
    "Repeat the following words several times: red blue green red blue green.",
]


class SlimVerifierTestWorker:
    model_runner: Any

    def check_slim_disabled(self):
        assert not hasattr(self.model_runner, "slim_verifier_eval")
        return True

    def install_slim_probe(self, check_full_logits):
        shadow = self.model_runner.slim_verifier_eval
        self.slim_calls = []
        for index, layer in zip(shadow.config.keep_layers, shadow.verifier.layers):
            layer.register_forward_pre_hook(lambda module, args, i=index: self.slim_calls.append(i))
        assert shadow.verifier.embedding is self.model_runner.model.model.embed_tokens
        assert shadow.verifier.norm is self.model_runner.model.model.norm
        if shadow.metrics is not None and check_full_logits:
            original_record = shadow.metrics.record

            def check_and_record(draft, target, slim, slim_ms=None):
                # Separate attention kernels and prefill/decode shapes can differ
                # numerically in BF16. Compare actual global-vocabulary logits.
                torch.testing.assert_close(slim, target, rtol=3e-2, atol=1e-1)
                return original_record(draft, target, slim, slim_ms)

            shadow.metrics.record = check_and_record

    def read_slim_probe(self):
        shadow = self.model_runner.slim_verifier_eval
        shadow.flush()
        return {
            "layers_executed": sorted(set(self.slim_calls)),
            "keep_layers": list(shadow.config.keep_layers),
            "metrics": shadow.metrics.snapshot() if shadow.metrics is not None else None,
        }


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graph"])
@pytest.mark.parametrize("acceptance_strategy", ["top1", "probability_ratio"])
@wait_until_npu_memory_free()
def test_slim_verifier_npu(tp_size, enforce_eager, acceptance_strategy, monkeypatch):
    if not hasattr(torch, "npu") or torch.npu.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} actual Ascend NPUs")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    settings = dict(
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=3,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.65,
        seed=123,
        enforce_eager=enforce_eager,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        # The same small checkpoint makes this test self-contained. The hook
        # accepts any proposer; no production code inspects the draft method.
        speculative_config={"method": "draft_model", "model": MODEL, "num_speculative_tokens": 3},
        worker_extension_cls=f"{__name__}.SlimVerifierTestWorker",
    )
    baseline = None
    for mode in ("disabled", "full", "reduced"):
        config = {
            "enabled": mode != "disabled",
            "log_interval": 1,
            "profile_latency": True,
            "acceptance_strategy": acceptance_strategy,
            "probability_ratio_threshold": 0.5,
        }
        config.update({"skip_layers": []} if mode == "full" else {"keep_layers": [0]})
        with VllmRunner(MODEL, additional_config={"slim_verifier_eval": config}, **settings) as runner:
            if mode != "disabled":
                runner.model.collective_rpc("install_slim_probe", args=(mode == "full",))
            else:
                assert all(runner.model.collective_rpc("check_slim_disabled"))
            # Includes unequal prompt lengths, chunked prefill, and a second
            # generation call to exercise prefix hits and request-slot reuse.
            outputs = runner.generate_greedy(PROMPTS, max_tokens=48)
            second = runner.generate_greedy(PROMPTS[:1], max_tokens=16)
            if mode == "disabled":
                baseline = (outputs, second)
            else:
                assert (outputs, second) == baseline
                probes = runner.model.collective_rpc("read_slim_probe")
                for probe in probes:
                    assert probe["layers_executed"] == probe["keep_layers"]
                    if probe["metrics"] is not None:
                        assert probe["metrics"]["blocks"] > 0
                        assert probe["metrics"]["draft_tokens"] > 0
                        assert probe["metrics"]["slim_acceptance_strategy"] == acceptance_strategy
                        assert probe["metrics"]["slim_replay_ms_per_block"] > 0
