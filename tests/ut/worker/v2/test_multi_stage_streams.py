# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for stream ownership and progress-copy dependencies."""

import ast
from contextlib import nullcontext
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


@pytest.mark.parametrize("use_primary", [False, True])
def test_adapter_supplies_stream_before_backend_load(use_primary):
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
        uses_primary_drafter=use_primary,
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
        rejection_sampler=NS(flush_trace=Mock(), defer_trace=False),
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
    assert adapter.final_verifier is runner.rejection_sampler
    assert adapter.final_verifier.defer_trace
    logged = namespace["logger"].info.call_args.args
    assert logged[1:3] == (use_primary, "primary" if use_primary else None)


@pytest.mark.parametrize("trace", [False, True])
def test_target_replay_keeps_external_event_updater_separate_without_host_sync(trace):
    path = SOURCE / "aclgraph_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelAclGraphManager")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run_fullgraph"]
    events = []
    output = object()

    class Parent:
        def run_fullgraph(self, desc):
            events.append("replay")
            return output

    compute = NS(synchronize=Mock(side_effect=AssertionError("Host synchronization")))
    update = NS(wait_stream=Mock(side_effect=lambda stream: events.append("wait")))
    updater = Mock(side_effect=lambda *args: events.append("update"))
    logger = Mock()
    logger.isEnabledFor.return_value = trace
    timing = Mock(side_effect=[1.0, 1.002, 1.005])
    namespace = dict(
        ModelCudaGraphManager=Parent,
        torch=NS(npu=NS(current_stream=lambda: compute), full=Mock()),
        logger=logger,
        DEBUG=10,
        perf_counter=timing,
        set_current_vllm_config=lambda *a: nullcontext(),
        set_forward_context=lambda *a, **kw: nullcontext(),
        get_forward_context=lambda: object(),
        _get_graph_update_backend=lambda groups: "backend",
        update_full_graph_params=updater,
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    manager = namespace["ModelAclGraphManager"]()
    manager.update_stream = update
    manager.vllm_config = object()
    manager.model_runner = NS(dp_size=1, model_state=NS(attn_metadata={}), attn_groups=[], speculative_config=None)
    desc = NS(num_tokens=17, cg_mode="full")
    assert manager.run_fullgraph(desc) is output
    assert events == ["wait", "replay", "update"]
    update.wait_stream.assert_called_once_with(compute)
    assert updater.call_args.args[1] is update and update is not compute
    compute.synchronize.assert_not_called()
    assert timing.call_count == (3 if trace else 0)
    assert logger.debug.call_count == int(trace)
