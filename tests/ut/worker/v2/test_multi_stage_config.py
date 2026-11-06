# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Capacity must reach upstream defaults before graph sizes become fixed."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from test_intermediate import modules as modules
from test_intermediate import settings, vllm_config


@pytest.fixture
def platform_defaults(modules, monkeypatch):
    config, _ = modules
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.v2.spec_decode.multi_stage.config", config)
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/platform.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        n
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "apply_config_platform_defaults"
    )
    method.decorator_list = []

    def reduced_cap(cfg):
        # Platform-specific graph defaults must see the expanded width.
        assert cfg.speculative_config.num_speculative_tokens == 40
        return None

    namespace = {"VllmConfig": object, "_get_reduced_cg_cap": reduced_cap}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["apply_config_platform_defaults"]


@pytest.mark.parametrize("explicit", [None, "max", "sizes"])
def test_capacity_precedes_graph_defaults_and_preserves_user_gears(platform_defaults, modules, explicit):
    config, _ = modules
    cfg = vllm_config()
    cfg.speculative_config.num_speculative_tokens = 4
    cfg.scheduler_config.max_num_seqs = 1
    cfg.additional_config = {"multi_stage_speculative": settings(num_rounds=8)}
    cfg.compilation_config = NS(
        max_cudagraph_capture_size=128 if explicit == "max" else None,
        cudagraph_capture_sizes=[8, 64] if explicit == "sizes" else None,
    )
    platform_defaults(None, cfg)
    assert cfg.speculative_config.num_speculative_tokens == 40
    assert cfg.compilation_config.max_cudagraph_capture_size == (128 if explicit == "max" else None)
    assert cfg.compilation_config.cudagraph_capture_sizes == ([8, 64] if explicit == "sizes" else None)
    # Parent serialization and worker initialization see the resolved width.
    assert cfg.additional_config["multi_stage_speculative"]["primary_num_speculative_tokens"] == 4
    config.validate_multi_stage(cfg, cfg.additional_config["multi_stage_speculative"])
    platform_defaults(None, cfg)
    assert cfg.speculative_config.num_speculative_tokens == 40


@pytest.mark.parametrize("options", [{}, {"final_verification": {"method": "all"}}, settings(num_rounds=0)])
def test_inactive_pipeline_leaves_upstream_capacity_alone(modules, options):
    config, _ = modules
    cfg = vllm_config()
    cfg.additional_config = {"multi_stage_speculative": options}
    config.prepare_multi_stage_config(cfg)
    assert cfg.speculative_config.num_speculative_tokens == 8
    assert "primary_num_speculative_tokens" not in options


@pytest.mark.parametrize("mode", ["block", "entropy", "dynamic", "watermark", "nonstandard"])
def test_conflicting_verification_modes_fail_before_initializing_models(modules, mode):
    config, _ = modules
    cfg = vllm_config()
    if mode in ("block", "entropy"):
        cfg.additional_config["rejection_sampler_config"] = {f"enable_{mode}_verify": True}
    elif mode == "dynamic":
        cfg.additional_config["dynamic_spec_config"] = {"method": "dynamic"}
    elif mode == "watermark":
        cfg.watermark_config = object()
    else:
        cfg.speculative_config.rejection_sample_method = "synthetic"

    with pytest.raises(ValueError, match="cannot|does not support"):
        config.validate_multi_stage(cfg, settings(num_rounds=2))


@pytest.mark.parametrize("architecture", ["DFlash2DraftModel", "LiLiCorrDraftModel"])
def test_incompatible_dflash_architectures_are_rejected(modules, architecture):
    config, _ = modules
    cfg = vllm_config()
    cfg.speculative_config.draft_model_config = NS(architectures=[architecture])

    with pytest.raises(ValueError, match="original DFlash"):
        config.validate_multi_stage(cfg, settings(num_rounds=2))


@pytest.mark.parametrize("chain,emit_bonus", [("ordinary", True), ("three", False), ("four", True)])
def test_runner_disables_target_bonus_only_for_three_model_chain(modules, monkeypatch, chain, emit_bonus):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "NPUModelRunner"
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name == "load_model"
    )
    # Exercise the integration after upstream/NZ weight loading has completed.
    start = next(
        index
        for index, node in enumerate(method.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "options" for target in node.targets)
    )
    method.body = method.body[start:]
    base = type("Base", (), {"load_model": lambda *args, **kwargs: None})
    cls = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    namespace = {"Base": base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    sampler = Mock()
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.v2.spec_decode.multi_stage.final_verification",
        NS(FinalVerificationSampler=sampler),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.utils.mem_utils", NS(DeviceMemoryProfiler=lambda: nullcontext(NS(consumed_memory=0)))
    )
    runner = namespace["Runner"]()
    options = {"final_verification": {"method": "all"}}
    runner.ascend_config = NS(multi_stage_speculative=options)
    runner.is_last_pp_rank = True
    active = chain != "ordinary"
    runner.speculator = NS(initialize_intermediate=Mock(), uses_primary_drafter=chain == "four") if active else NS()
    runner.sampler, runner.speculative_config, runner.device = object(), object(), "cpu"
    runner.model_memory_usage = 0
    runner.load_model()
    assert sampler.call_args.kwargs["emit_bonus_token"] == emit_bonus
    if active:
        runner.speculator.initialize_intermediate.assert_called_once_with(runner)
