# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise the independent DFlash chain without a Target-conditioned model."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_intermediate import Backend, vllm_config
from test_intermediate import modules as modules
from test_intermediate_backend import backend as backend
from test_intermediate_backend import load_class


def options(**intermediate):
    return {
        "intermediate": {"verifier": {"model": "4B"}, "drafter": {"model": "DFlash-4B"}, **intermediate},
        "final_verification": {"method": "topk", "top_k": 1},
    }


@pytest.mark.parametrize("rounds,threshold,expected", [(1, None, 3), (3, None, 9), (3, 0, 3), (3, 3, 6), (3, 4, 7)])
def test_capacity_uses_intermediate_width_and_never_adds_a_primary(modules, rounds, threshold, expected):
    config, _ = modules
    cfg = vllm_config()
    cfg.additional_config = {
        "multi_stage_speculative": options(num_rounds=rounds, num_speculative_tokens=2, max_generated_tokens=threshold)
    }
    values = cfg.additional_config["multi_stage_speculative"]
    config.prepare_multi_stage_config(cfg)
    config.validate_multi_stage(cfg, values)
    assert cfg.speculative_config.num_speculative_tokens == expected
    assert "primary_num_speculative_tokens" not in values
    config.prepare_multi_stage_config(cfg)
    config.validate_multi_stage(cfg, values)
    assert cfg.speculative_config.num_speculative_tokens == expected


@pytest.mark.parametrize("invalid", [None, 0, 1, "false"])
def test_primary_switch_requires_a_boolean(modules, invalid):
    config, _ = modules
    values = {**options(), "use_primary_drafter": invalid}
    with pytest.raises(ValueError, match="boolean"):
        config.validate_multi_stage(vllm_config(), values)


def test_primary_width_is_rejected_without_a_primary_model(modules):
    config, _ = modules
    with pytest.raises(ValueError, match="requires use_primary_drafter"):
        config.validate_multi_stage(vllm_config(), {**options(), "primary_num_speculative_tokens": 2})


@pytest.fixture
def standalone(modules):
    config, Pipeline = modules

    class ForbiddenPrimary:
        def __init__(self, *args):
            pytest.fail("The Target-conditioned drafter must not be constructed")

        def propose(self, *args, **kwargs):
            pytest.fail("The Target-conditioned drafter must not run")

    legacy = load_class(
        "adapter.py",
        "MultiStageDFlashSpeculator",
        dict(
            torch=torch,
            AscendDFlashSpeculator=ForbiddenPrimary,
            DraftTokenIds=lambda ids, tokens: (ids, tokens),
            logger=Mock(),
        ),
    )
    base = type("BaseSpeculator", (), {})
    cls = load_class(
        "intermediate_speculator.py",
        "IntermediateOnlySpeculator",
        dict(
            torch=torch,
            BaseSpeculator=base,
            MultiStageDFlashSpeculator=legacy,
            IntermediateConfig=config.IntermediateConfig,
        ),
    )
    cfg = NS(
        model_config=NS(dtype=torch.float32, max_model_len=16),
        speculative_config=NS(num_speculative_tokens=6),
        additional_config={
            "multi_stage_speculative": options(
                num_rounds=2, num_speculative_tokens=2, verification={"method": "topk", "top_k": 1}
            )
        },
    )
    obj = cls(cfg, torch.device("cpu"))
    assert isinstance(obj, base) and not isinstance(obj, ForbiddenPrimary)
    assert not hasattr(obj, "model") and not hasattr(obj, "query_cudagraph_manager")
    return obj, config, Pipeline


@pytest.mark.parametrize("method", ["topk", "all", "prob_ratio"])
def test_first_draft_is_from_4b_then_verify_extend_and_publish(standalone, backend, method):
    obj, config, Pipeline = standalone
    back, _, _ = backend
    acceptance = sys.modules["vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance"]
    obj.intermediate_config.verification = {"method": method, "top_k": 1}
    back.model.compute_logits = lambda h: torch.nn.functional.one_hot(h[:, 0].long() + 1, 32).float() * 20
    back.decision_runner = acceptance.IntermediateDecisionRunner(
        back.model.compute_logits, acceptance.AcceptancePolicy(method, 1), 3
    )
    back.drafter.propose.side_effect = [torch.tensor([[4, 5], [14, 15]]), torch.tensor([[7, 8], [17, 18]])]
    obj.pipeline = Pipeline(back, obj.intermediate_config, obj.final_capacity)
    history = torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0], [11, 12, 13, 0, 0, 0, 0, 0]])
    committed = history.clone()
    obj.req_states = NS(
        total_len=NS(gpu=torch.tensor([3, 3])),
        num_computed_tokens=NS(gpu=torch.tensor([2, 2])),
        num_computed_tokens_cpu=torch.zeros(2, dtype=torch.int32),
        all_token_ids=NS(gpu=history),
        max_seq_len=np.array([16, 16]),
        prefill_len=NS(np=np.array([2, 2])),
        req_id_to_index={"a": 0, "b": 1},
    )
    batch = NS(num_reqs=2, idx_mapping=torch.tensor([0, 1]), idx_mapping_np=np.array([0, 1]), req_ids=["a", "b"])
    target_hidden = torch.full((2, 8), float("nan"))  # Must never condition the 4B drafter.
    result = obj.propose(batch, None, None, target_hidden, [target_hidden])
    assert result.tolist() == [[4, 5, 6, 7, 0, 0], [14, 15, 16, 17, 0, 0]]
    assert obj.get_draft_tokens() == (["a", "b"], [[4, 5, 6, 7], [14, 15, 16, 17]])
    assert back.drafter.propose.call_count == 2
    assert torch.equal(history, committed)
    assert back.cache.tokens[back.cache.slots["a"]] == [1, 2, 3, 4, 5, 6, 7]
    assert obj.req_states.num_computed_tokens_cpu.tolist() == [2, 2]
    # Target rejects the candidate tail and commits its own replacement. The
    # next independent proposal must hydrate that new prefix, not the old draft.
    history[:, 3] = torch.tensor([9, 19])
    obj.req_states.total_len.gpu.fill_(4)
    obj.req_states.num_computed_tokens.gpu.fill_(3)
    back.drafter.propose.side_effect = [torch.tensor([[10, 11], [20, 21]])]
    changed_history = history.clone()
    assert obj.propose(batch).tolist() == [[10, 11, 12, 0, 0, 0], [20, 21, 22, 0, 0, 0]]
    assert back.cache.tokens[back.cache.slots["a"]] == [1, 2, 3, 9, 10, 11]
    assert torch.equal(history, changed_history)


def test_initial_proposal_runs_inside_each_resident_group(modules):
    config, Pipeline = modules
    back = Backend([[[2, 3]], [[4, 5]], [[12, 13]], [[14, 15]]], proposals=[[[2]], [[4]], [[12]], [[14]]])
    back.max_num_reqs = 1
    pipe = Pipeline(back, config.IntermediateConfig("4B", "DFlash-4B", num_rounds=2), 6)
    assert pipe.refine([[1], [11]], None, [6, 6], ["a", "b"]) == [[2, 3, 4, 5], [12, 13, 14, 15]]
    assert [label for label, *_ in back.calls] == ["propose", "verify", "propose", "verify"] * 2
    assert back.calls[0][1] == [[1]] and back.calls[4][1] == [[11]]
    assert back.retain_hidden == [True] * 4  # The next cycle's initial drafter consumes these rows.


def test_dummy_does_not_read_history_or_publish_and_profiles_only_pair(standalone):
    obj, _, _ = standalone
    obj.pipeline = NS(backend=NS(profile=Mock()))
    obj._read_step = Mock(side_effect=AssertionError("dummy history read"))
    obj.candidates, obj.req_ids = [[99]], ["live"]
    result = obj.propose(NS(num_reqs=2), dummy_run=True, is_profile=True)
    assert result.tolist() == [[0] * 6, [0] * 6]
    assert obj.candidates == [[99]] and obj.req_ids == ["live"]
    obj.pipeline.backend.profile.assert_called_once_with()
    obj.init_cudagraph_manager("full")
    obj.capture()


@pytest.mark.parametrize(
    "primary,rounds,expected",
    [(None, 2, "independent"), (False, 2, "independent"), (True, 2, "legacy"), (False, 0, "ordinary")],
)
def test_factory_selects_only_the_required_model_path(monkeypatch, primary, rounds, expected):
    root = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode"
    tree = ast.parse((root / "__init__.py").read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(torch=NS(device=object), VllmConfig=object)
    exec(compile(tree, "spec_decode_factory", "exec"), ns)
    choices = {}
    for path, name, label in (
        ("multi_stage.intermediate_speculator", "IntermediateOnlySpeculator", "independent"),
        ("multi_stage.adapter", "MultiStageDFlashSpeculator", "legacy"),
        ("dflash.speculator", "AscendDFlashSpeculator", "ordinary"),
    ):
        choices[label] = Mock(return_value=label)
        monkeypatch.setitem(sys.modules, "vllm_ascend.worker.v2.spec_decode." + path, NS(**{name: choices[label]}))
    values = options(num_rounds=rounds)
    if primary is not None:
        values["use_primary_drafter"] = primary
    cfg = NS(
        speculative_config=NS(use_dspark=lambda: False, use_dflash=lambda: True),
        additional_config={"multi_stage_speculative": values},
    )
    assert ns["init_speculator"](cfg, "cpu") == expected
    assert {key: value.call_count for key, value in choices.items()} == {key: int(key == expected) for key in choices}
