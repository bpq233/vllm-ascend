# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU lifecycle checks for independently captured intermediate graphs."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


@pytest.fixture
def graph_helpers():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage/backend.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
        and n.name
        in (
            "IntermediateGraphState",
            "init_secondary_graphs",
            "capture_secondary_graphs",
            "_validate_full_graph_capture",
        )
    ]
    params = NS(_graph_params=object(), _draft_graph_params=object(), _draft_graph_prefill_params=object())
    stream = Mock(return_value=object())
    namespace = dict(
        contextmanager=contextmanager,
        acl_graph=params,
        torch=NS(npu=NS(Stream=stream)),
        CUDAGraphMode=NS(
            NONE="none", FULL="full", FULL_AND_PIECEWISE="full_and_piecewise", FULL_DECODE_ONLY="full_decode"
        ),
    )
    exec(compile(tree, str(path), "exec"), namespace)
    return NS(**namespace)


def test_primary_handles_preserved_and_secondary_handles_reused(graph_helpers):
    h = graph_helpers
    primary = vars(h.acl_graph).copy()
    state = h.IntermediateGraphState()
    secondary = {name: object() for name in primary}
    with state.context():
        assert all(value is None for value in vars(h.acl_graph).values())
        for name, value in secondary.items():
            setattr(h.acl_graph, name, value)
        with state.context():
            assert vars(h.acl_graph) == secondary
        assert vars(h.acl_graph) == secondary
    assert vars(h.acl_graph) == primary
    with state.context():
        assert vars(h.acl_graph) == secondary
    assert vars(h.acl_graph) == primary


def test_capture_failure_restores_main_graphs(graph_helpers):
    h = graph_helpers
    primary = vars(h.acl_graph).copy()
    state = h.IntermediateGraphState()
    with pytest.raises(RuntimeError, match="capture failed"), state.context():
        h.acl_graph._draft_graph_params = object()
        raise RuntimeError("capture failed")
    assert vars(h.acl_graph) == primary
    assert state._depth == 0


def test_secondary_stream_precedes_manager_and_capture(graph_helpers):
    h = graph_helpers
    drafter = NS(query_cudagraph_manager=NS(needs_capture=lambda: True), capture=Mock())

    def init(mode):
        assert mode == "full_decode"
        assert drafter.update_stream is h.torch.npu.Stream.return_value

    drafter.init_cudagraph_manager = Mock(side_effect=init)
    h.init_secondary_graphs(drafter, "full_and_piecewise", "npu:0")
    h.torch.npu.Stream.assert_called_once_with(device="npu:0")
    drafter.capture.assert_not_called()
    h.capture_secondary_graphs(drafter)
    drafter.capture.assert_called_once_with()


def test_explicit_eager_never_creates_stream_or_captures(graph_helpers):
    h = graph_helpers
    drafter = NS(
        query_cudagraph_manager=NS(needs_capture=lambda: False),
        init_cudagraph_manager=Mock(),
        capture=Mock(),
    )
    h.init_secondary_graphs(drafter, "none", "npu:0")
    h.capture_secondary_graphs(drafter)
    drafter.init_cudagraph_manager.assert_called_once_with("none")
    h.torch.npu.Stream.assert_not_called()
    assert drafter.update_stream is None
    drafter.capture.assert_not_called()


def test_requested_graphs_do_not_silently_fall_back(graph_helpers):
    h = graph_helpers
    drafter = NS(query_cudagraph_manager=NS(needs_capture=lambda: False), init_cudagraph_manager=Mock())
    with pytest.raises(ValueError, match="full graph attention support"):
        h.init_secondary_graphs(drafter, "full", "npu:0")


def test_full_graph_capture_validation_requires_every_planned_size(graph_helpers):
    validate = graph_helpers._validate_full_graph_capture
    validate("full", [1, 16, 64], [1, 16, 64])
    with pytest.raises(RuntimeError, match=r"missing_sizes=\[64\]"):
        validate("full", [1, 16, 64], [1, 16])
    validate("full_and_piecewise", [1, 16, 64], [1])


@pytest.mark.parametrize("draft", [False, True])
def test_full_graph_dispatch_rejects_piecewise_and_eager_after_capture(draft):
    root = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2"
    path = root / ("spec_decode/dflash/aclgraph.py" if draft else "aclgraph_utils.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    name = "DFlashAclGraphManager" if draft else "ModelAclGraphManager"
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "dispatch"]
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]

    class Base:
        def dispatch(self, desc):
            return desc

    ns = dict(Base=Base, CUDAGraphMode=NS(FULL="full"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "dispatch", "exec"), ns)
    manager = ns[name]()
    manager._graphs_captured = True
    manager.require_full_graph = True
    manager.vllm_config = NS(
        compilation_config=NS(cudagraph_mode="full"),
        model_config=NS(enforce_eager=False),
        additional_config={"multi_stage_speculative": {"intermediate": {"verifier": "v"}}},
    )
    full = NS(cg_mode="full", num_tokens=55)
    assert manager.dispatch(full) is full
    for mode in (None, "piecewise"):
        with pytest.raises(RuntimeError, match="refusing eager/PIECEWISE"):
            manager.dispatch(NS(cg_mode=mode, num_tokens=55))
    # Initialization warmups and ordinary non-multi-stage paths keep their API.
    manager._graphs_captured = False
    assert manager.dispatch(NS(cg_mode=None, num_tokens=55)).cg_mode is None
    manager._graphs_captured = True
    manager.require_full_graph = False
    manager.vllm_config.additional_config = {}
    assert manager.dispatch(NS(cg_mode=None, num_tokens=55)).cg_mode is None
