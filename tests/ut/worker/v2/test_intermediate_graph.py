# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU lifecycle checks for independently captured intermediate graphs."""

import ast
from contextlib import contextmanager
from enum import Enum
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
            "intermediate_graph_mode",
        )
    ]
    params = NS(_graph_params=object(), _draft_graph_params=object(), _draft_graph_prefill_params=object())
    stream = Mock(return_value=object())
    class GraphMode(Enum):
        NONE = "none"
        FULL = "full"
        FULL_AND_PIECEWISE = "full_and_piecewise"
        FULL_DECODE_ONLY = "full_decode_only"

        def decode_mode(self):
            return GraphMode.FULL if self in (GraphMode.FULL, GraphMode.FULL_DECODE_ONLY) else GraphMode.NONE

    namespace = dict(
        contextmanager=contextmanager,
        acl_graph=params,
        torch=NS(npu=NS(Stream=stream)),
        CUDAGraphMode=GraphMode,
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
        assert mode is h.CUDAGraphMode.FULL_AND_PIECEWISE
        assert drafter.update_stream is h.torch.npu.Stream.return_value

    drafter.init_cudagraph_manager = Mock(side_effect=init)
    h.init_secondary_graphs(drafter, h.CUDAGraphMode.FULL_AND_PIECEWISE, "npu:0")
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
    h.init_secondary_graphs(drafter, h.CUDAGraphMode.NONE, "npu:0")
    h.capture_secondary_graphs(drafter)
    drafter.init_cudagraph_manager.assert_called_once_with(h.CUDAGraphMode.NONE)
    h.torch.npu.Stream.assert_not_called()
    assert drafter.update_stream is None
    drafter.capture.assert_not_called()


def test_secondary_preserves_full_decode_only_mode(graph_helpers):
    h = graph_helpers
    drafter = NS(query_cudagraph_manager=NS(needs_capture=lambda: False), init_cudagraph_manager=Mock())
    h.init_secondary_graphs(drafter, h.CUDAGraphMode.FULL_DECODE_ONLY, "npu:0")
    drafter.init_cudagraph_manager.assert_called_once_with(h.CUDAGraphMode.FULL_DECODE_ONLY)
    assert drafter.update_stream is h.torch.npu.Stream.return_value


def test_full_graph_capture_validation_requires_every_planned_size(graph_helpers):
    validate = graph_helpers._validate_full_graph_capture
    validate(graph_helpers.CUDAGraphMode.FULL, [1, 16, 64], [1, 16, 64])
    with pytest.raises(RuntimeError, match=r"missing_sizes=\[64\]"):
        validate(graph_helpers.CUDAGraphMode.FULL_DECODE_ONLY, [1, 16, 64], [1, 16])
    validate(graph_helpers.CUDAGraphMode.FULL_AND_PIECEWISE, [1, 16, 64], [1])


def test_full_decode_only_uses_full_graph_for_intermediate_verifier(graph_helpers):
    assert (
        graph_helpers.intermediate_graph_mode(graph_helpers.CUDAGraphMode.FULL_DECODE_ONLY)
        is graph_helpers.CUDAGraphMode.FULL
    )
    assert (
        graph_helpers.intermediate_graph_mode(graph_helpers.CUDAGraphMode.FULL_AND_PIECEWISE)
        is graph_helpers.CUDAGraphMode.FULL_AND_PIECEWISE
    )
