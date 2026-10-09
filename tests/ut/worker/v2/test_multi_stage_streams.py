# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for stream ownership and progress-copy dependencies."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from test_intermediate_graph import graph_helpers as graph_helpers

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2"


@pytest.fixture
def progress_runner():
    path = SOURCE / "model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    tree.body = [
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_init_num_computed_tokens_copy"
    ]
    torch = NS(npu=NS(Stream=Mock(), Event=Mock()), empty=Mock(), int32=object())
    namespace = {"torch": torch}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["_init_num_computed_tokens_copy"], torch


@pytest.mark.parametrize("speculator", [None, NS(updates_computed_tokens_cpu=True)])
def test_no_progress_copy_resources_when_cpu_progress_is_already_available(progress_runner, speculator):
    initialize, torch = progress_runner
    runner = NS(speculator=speculator, max_num_reqs=8)
    initialize(runner)
    assert runner.num_computed_tokens_stream is None
    assert runner.num_computed_tokens_event is None
    assert runner.num_computed_tokens_cpu is None
    torch.npu.Stream.assert_not_called()
    torch.npu.Event.assert_not_called()
    torch.empty.assert_not_called()


@pytest.mark.parametrize("speculator", [NS(), NS(updates_computed_tokens_cpu=False)])
def test_ordinary_speculation_keeps_progress_copy_resources(progress_runner, speculator):
    initialize, torch = progress_runner
    runner = NS(speculator=speculator, max_num_reqs=8)
    initialize(runner)
    assert runner.num_computed_tokens_stream is torch.npu.Stream.return_value
    assert runner.num_computed_tokens_event is torch.npu.Event.return_value
    assert runner.num_computed_tokens_cpu is torch.empty.return_value
    torch.npu.Stream.assert_called_once_with()
    torch.npu.Event.assert_called_once_with()
    torch.empty.assert_called_once_with(8, dtype=torch.int32, device="cpu", pin_memory=True)


def test_secondary_uses_parent_update_stream_before_initializing_graph_manager(graph_helpers):
    h = graph_helpers
    shared = object()
    drafter = NS()

    def initialize(mode):
        assert mode is h.CUDAGraphMode.FULL
        assert drafter.update_stream is shared

    drafter.init_cudagraph_manager = Mock(side_effect=initialize)
    h.init_secondary_graphs(drafter, h.CUDAGraphMode.FULL, "npu:0", shared)
    drafter.init_cudagraph_manager.assert_called_once_with(h.CUDAGraphMode.FULL)
    h.torch.npu.Stream.assert_not_called()


def test_eager_secondary_does_not_attach_parent_stream(graph_helpers):
    h = graph_helpers
    drafter = NS(init_cudagraph_manager=Mock())
    h.init_secondary_graphs(drafter, h.CUDAGraphMode.NONE, "npu:0", object())
    assert drafter.update_stream is None
    h.torch.npu.Stream.assert_not_called()


def test_adapter_supplies_stream_before_backend_load():
    path = SOURCE / "spec_decode/multi_stage/adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    tree.body = [
        node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name == "initialize_intermediate"
    ]
    shared = object()
    backend = NS()

    def load():
        assert backend.update_stream is shared

    backend.load_model = Mock(side_effect=load)
    namespace = dict(
        IntermediateBackend=Mock(return_value=backend),
        IntermediatePipeline=Mock(return_value=NS(policy=NS(method="all"))),
        logger=Mock(),
    )
    exec(compile(tree, str(path), "exec"), namespace)
    adapter = NS(
        intermediate_config=NS(
            verifier_model="verifier",
            drafter_model="secondary",
            num_rounds=3,
            max_generated_tokens=8,
            num_speculative_tokens=2,
        ),
        device="npu:0",
        final_capacity=12,
    )
    runner = NS(
        req_states=object(),
        update_stream=shared,
        model_config=NS(hf_text_config=NS(eos_token_id=1)),
        vllm_config=NS(
            speculative_config=NS(model="primary"),
            additional_config={"multi_stage_speculative": {"final_verification": {"method": "all"}}},
        ),
    )
    namespace["initialize_intermediate"](adapter, runner)
    backend.load_model.assert_called_once_with()
    assert adapter.req_states is runner.req_states
