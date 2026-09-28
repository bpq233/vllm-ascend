# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for verifier graph policy and independent drafter configuration."""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2"


class Mode(Enum):
    NONE = 0
    FULL = 1
    FULL_DECODE_ONLY = 2
    FULL_AND_PIECEWISE = 3
    PIECEWISE = 4


@pytest.fixture
def policy():
    path = ROOT / "aclgraph_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("verification_graph_mode", "target_graph_mode")
    ]
    routing = ast.parse((ROOT.parents[1] / "attention/spec_decode.py").read_text(encoding="utf-8"))
    namespace = dict(CUDAGraphMode=Mode)
    exec(compile(routing, "spec_decode.py", "exec"), namespace)
    exec(compile(tree, str(path), "exec"), namespace)
    return NS(**namespace)


@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize("rounds", [0, 5, None])
@pytest.mark.parametrize("eager", [False, True])
def test_target_variable_verification_uses_mixed_full_without_changing_draft_mode(policy, mode, rounds, eager):
    options = {"intermediate": {"num_rounds": rounds}} if rounds is not None else {}
    cfg = NS(
        use_v2_model_runner=True,
        speculative_config=NS(num_speculative_tokens=55),
        additional_config={"multi_stage_speculative": options},
        model_config=NS(enforce_eager=eager),
        compilation_config=NS(cudagraph_mode=mode),
    )
    expected = Mode.FULL if mode == Mode.FULL_DECODE_ONLY and rounds == 5 and not eager else mode
    assert policy.target_graph_mode(cfg, mode) == expected
    # Primary DFlash initializes its own manager from this same configuration.
    assert cfg.compilation_config.cudagraph_mode == mode
    cfg.use_v2_model_runner = False
    assert policy.target_graph_mode(cfg, mode) == mode
    cfg.use_v2_model_runner = True
    cfg.speculative_config.num_speculative_tokens = 4
    assert policy.target_graph_mode(cfg, mode) == mode


@pytest.mark.parametrize("mode", list(Mode))
def test_intermediate_capture_uses_verifier_policy_without_changing_secondary(policy, mode):
    path = ROOT / "spec_decode/multi_stage/backend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IntermediateBackend")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_capture_graphs")
    method.decorator_list = []

    class StopAfterManagerInit(Exception):
        pass

    manager_factory = Mock(side_effect=StopAfterManagerInit)
    namespace = dict(
        ModelAclGraphManager=manager_factory,
        verification_graph_mode=policy.verification_graph_mode,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    config = NS(compilation_config=NS(cudagraph_mode=mode))
    backend = NS(vllm_config=config, device="npu:0", decode_query_len=5, drafter=NS(update_stream=object()))
    with pytest.raises(StopAfterManagerInit):
        namespace["_capture_graphs"](backend)
    expected = Mode.FULL if mode == Mode.FULL_DECODE_ONLY else mode
    assert manager_factory.call_args.args == (config, "npu:0", expected, 5, backend)
    # Secondary DFlash retains the user's mode, independently of the verifier.
    assert config.compilation_config.cudagraph_mode == mode
