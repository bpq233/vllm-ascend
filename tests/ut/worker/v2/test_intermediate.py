# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode"

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("vllm_ascend.worker.v2.spec_decode.acceptance", "acceptance.py")
    config = load("_intermediate_test_config", "multi_stage_config.py")
    pipeline = load("_intermediate_test_pipeline", "intermediate.py")
    return config, pipeline.IntermediatePipeline


def scores(predictions):
    logits = torch.full((len(predictions), 32), -10.0)
    logits[torch.arange(len(predictions)), predictions] = 10.0
    return logits


class Backend:
    def __init__(self, verification, proposals=()):
        self.verification = iter(verification)
        self.proposals = iter(proposals)
        self.calls = []

    def verify(self, contexts, drafts):
        self.calls.append(("verify", contexts, drafts))
        return [scores(row) for row in next(self.verification)]

    def propose(self, contexts):
        self.calls.append(("propose", contexts))
        return next(self.proposals)


def test_three_rounds_preserve_primary_and_use_only_accepted_context(modules):
    config, Pipeline = modules
    backend = Backend(
        [[[1, 7, 8]], [[3, 4, 5]], [[9, 10]]],
        proposals=[[[3, 4]], [[9]]],
    )
    options = config.IntermediateConfig("verifier", "drafter", verification={"method": "topk", "top_k": 1})
    primary, prefixes = [[1, 2]], [[20]]
    result = Pipeline(backend, options, capacity=8).refine(prefixes, primary, [8])
    assert result == [[1, 7, 3, 4, 5, 9, 10]]
    assert primary == [[1, 2]]
    assert prefixes == [[20]]
    assert backend.calls == [
        ("verify", [[20]], [[1, 2]]),
        ("propose", [[20, 1, 7]]),
        ("verify", [[20, 1, 7]], [[3, 4]]),
        ("propose", [[20, 1, 7, 3, 4, 5]]),
        ("verify", [[20, 1, 7, 3, 4, 5]], [[9]]),
    ]


def test_heterogeneous_eos_limit_and_inactive_requests(modules):
    config, Pipeline = modules
    backend = Backend([[[1, 2, 3], [4, 5, 6], [7]], [[8, 9]]], proposals=[[[8]]])
    options = config.IntermediateConfig("v", "d", verification={"method": "all"})
    result = Pipeline(backend, options, capacity=5, eos_token_id=[2]).refine(
        [[20], [21], [22], [23]], [[1, 2], [4, 5], [], [9]], [5, 2, 3, 0]
    )
    assert result == [[1, 2], [4, 5], [7, 8, 9], []]
    assert backend.calls == [
        ("verify", [[20], [21], [22]], [[1, 2], [4, 5], []]),
        ("propose", [[22, 7]]),
        ("verify", [[22, 7]], [[8]]),
    ]


def test_empty_active_batch_does_not_run_backend(modules):
    config, Pipeline = modules
    backend = Backend([])
    assert Pipeline(backend, config.IntermediateConfig("v", "d"), 4).refine([[1]], [[2]], [0]) == [[]]
    assert backend.calls == []


def vllm_config():
    return NS(
        use_v2_model_runner=True,
        speculative_config=NS(num_speculative_tokens=8, use_dflash=lambda: True),
        parallel_config=NS(),
        scheduler_config=NS(async_scheduling=False),
        lora_config=None,
        model_config=NS(is_multimodal_model=False),
    )


def settings(**intermediate):
    return {
        "intermediate": {"verifier": {"model": "v"}, "drafter": {"model": "d"}, **intermediate},
        "final_verification": {"method": "all"},
    }


def test_config_disabled_and_zero_rounds(modules):
    config, _ = modules
    config.validate_multi_stage(None, None)
    config.validate_multi_stage(None, {})
    cfg = vllm_config()
    cfg.speculative_config.use_dflash = lambda: False
    config.validate_multi_stage(cfg, settings(num_rounds=0))
    invalid = settings(num_rounds=0)
    invalid["primary_num_speculative_tokens"] = 4
    with pytest.raises(ValueError, match="zero intermediate rounds"):
        config.validate_multi_stage(cfg, invalid)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_rounds", -1),
        ("num_rounds", True),
        ("num_speculative_tokens", 0),
        ("max_num_seqs", 0),
        ("max_model_len", 0),
    ],
)
def test_invalid_intermediate_numbers(modules, field, value):
    config, _ = modules
    with pytest.raises(ValueError, match=field):
        config.IntermediateConfig.from_dict(settings(**{field: value})["intermediate"])


@pytest.mark.parametrize(
    "invalid",
    [
        {"unexpected": True},
        {"primary_num_speculative_tokens": 4},
        {**settings(), "primary_num_speculative_tokens": 9},
        {"intermediate": settings()["intermediate"]},
    ],
)
def test_invalid_multi_stage_options(modules, invalid):
    config, _ = modules
    with pytest.raises(ValueError):
        config.validate_multi_stage(vllm_config(), invalid)


@pytest.mark.parametrize("attribute,value", [("use_v2_model_runner", False), ("lora_config", object())])
def test_unsupported_runner_modes(modules, attribute, value):
    config, _ = modules
    cfg = vllm_config()
    setattr(cfg, attribute, value)
    with pytest.raises(ValueError):
        config.validate_multi_stage(cfg, settings())
