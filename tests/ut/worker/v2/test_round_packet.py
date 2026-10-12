# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Verify candidate publication against CPU control, including ragged rounds."""

import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from test_intermediate import modules as modules
from test_intermediate_backend import backend as backend


@pytest.fixture
def kernels():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage/round_packet.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    kernel_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pack_round_kernel")
    constexpr_args = {arg.arg for arg in kernel_node.args.args if arg.annotation is not None}
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None

    class Address:
        def __init__(self, data, offset=0):
            self.data, self.offset = data.reshape(-1), offset

        def __add__(self, offset):
            return Address(self.data, self.offset + offset)

    class Language:
        row = 0
        int1 = np.bool_
        arange = staticmethod(np.arange)
        static_range = staticmethod(range)
        minimum = staticmethod(np.minimum)
        where = staticmethod(np.where)
        min = staticmethod(np.min)

        def program_id(self, axis):
            return self.row

        @staticmethod
        def full(shape, value, dtype):
            return np.full(shape, value, dtype=dtype)

        @staticmethod
        def load(address, mask=True, other=0):
            offsets, mask = np.broadcast_arrays(address.offset, mask)
            values = np.full(offsets.shape, other, dtype=address.data.dtype)
            values[mask] = address.data[offsets[mask]]
            return values

        @staticmethod
        def store(address, values, mask=True):
            offsets, values, mask = np.broadcast_arrays(address.offset, values, mask)
            address.data[offsets[mask]] = values[mask]

    language = Language()
    namespace = dict(torch=torch, tl=language)
    exec(compile(tree, str(source), "exec"), namespace)
    return NS(
        pack=namespace["pack_round"],
        kernel=namespace["_pack_round_kernel"],
        namespace=namespace,
        language=language,
        Address=Address,
        constexpr_args=constexpr_args,
    )


@pytest.mark.parametrize("eos", [[], [0], [5, 7]])
@pytest.mark.parametrize("room", [0, 1, 3, 8])
@pytest.mark.parametrize("initialize", [False, True])
def test_actual_kernel_body_matches_reference_with_masks_strides_and_prior_rounds(kernels, eos, room, initialize):
    k = kernels
    decisions = torch.tensor([[2, 99, 7, 99], [0, 99, 0, 99], [1, 99, 9, 99]])[:, ::2]
    drafts = [
        torch.tensor([4, 99, 5, 99])[:, None].reshape(-1)[::2],
        torch.empty(0, dtype=torch.int64),
        torch.tensor([8]),
    ]
    expected = torch.zeros(4, 9, dtype=torch.int64)
    expected[2, :2] = torch.tensor([11, 12])
    counts = torch.tensor([0, 0, 2, 0], dtype=torch.int32)
    limits = torch.tensor([room, 9, room + 2, 9], dtype=torch.int32)
    if initialize:
        expected.zero_()
        counts.fill_(-777)  # First-round kernel must not read uninitialized counts.
    actual = expected.clone()
    actual_counts = counts.clone()
    row_ids = [2, 0, 3]
    packet = k.pack(decisions, drafts, expected, counts, limits, row_ids, eos, initialize=initialize)
    actual_packet = torch.empty_like(packet)
    addr = lambda tensor: k.Address(tensor.numpy())
    # Simulate the kernel body itself, not a second copy of its math. Address
    # masks check empty/strided draft loads and candidate write boundaries.
    # decisions uses row/column strides, so expose its full underlying storage.
    storage = torch.as_strided(decisions, (3, 4), (4, 1))
    pointers = [
        addr(torch.as_strided(t, (max(1, len(t) * t.stride(0)),), (1,))) if len(t) else addr(t) for t in drafts
    ] + [addr(actual_packet)]
    for row in range(3):
        k.language.row = row
        k.kernel(
            addr(storage),
            *pointers,
            addr(actual_packet),
            addr(actual),
            addr(actual_counts),
            *[int(limits[index]) for index in row_ids],
            0,
            *row_ids,
            0,
            BASE_ROW=0,
            NUM_ROWS=3,
            DECISION_STRIDE=decisions.stride(0),
            DECISION_COL_STRIDE=decisions.stride(1),
            CAPACITY=9,
            WIDTHS=(2, 0, 1),
            STRIDES=tuple(t.stride(0) for t in drafts),
            OFFSETS=(6, 8, 8),
            EOS_IDS=tuple(eos),
            INITIALIZE=initialize,
            BLOCK=4,
        )
    assert torch.equal(actual_packet, packet)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_counts, counts)


@pytest.mark.parametrize("num_reqs", [1, 3, 4, 5, 9])
@pytest.mark.parametrize("initialize", [False, True])
def test_launch_arguments_work_with_legacy_dtype_binder_and_chunked_batches(kernels, num_reqs, initialize):
    k = kernels
    calls = []

    class DeviceTensor:
        device = NS(type="npu")

        def __init__(self, data):
            self.data, self.dtype, self.shape = data, data.dtype, data.shape

    class LegacyLauncher:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                bound = inspect.signature(k.kernel).bind(*args, **kwargs)
                for name, argument in bound.arguments.items():
                    if name not in k.constexpr_args and not isinstance(argument, (int, float, bool)):
                        # Reproduce the user's Triton mangle_type contract.
                        # A tuple at this point raises the reported error.
                        assert argument.dtype == torch.int64 or argument.dtype == torch.int32
                calls.append(grid[0])
                converted = [
                    k.Address((arg.data if isinstance(arg, DeviceTensor) else arg).numpy())
                    if isinstance(arg, (DeviceTensor, torch.Tensor))
                    else arg
                    for arg in args
                ]
                for row in range(grid[0]):
                    k.language.row = row
                    k.kernel(*converted, **kwargs)

            return launch

    decisions = torch.tensor([[i % 3, 70 + i] for i in range(num_reqs)])
    drafts = [torch.arange(10 + i * 3, 10 + i * 3 + i % 3) for i in range(num_reqs)]
    row_ids = list(reversed(range(num_reqs)))
    limits = [2 + i % 3 for i in range(num_reqs)]
    expected = torch.zeros(num_reqs, 8, dtype=torch.int64)
    counts = torch.full((num_reqs,), -777 if initialize else 1, dtype=torch.int32)
    if not initialize:
        expected[:, 0] = 99
    actual, actual_counts = expected.clone(), counts.clone()
    packet = k.pack(decisions, drafts, expected, counts, limits, row_ids, [14, 72], initialize=initialize)
    k.namespace["_pack_round_kernel"] = LegacyLauncher()
    k.namespace["triton"] = NS(next_power_of_2=lambda value: 1 << (value - 1).bit_length())
    k.namespace["torch"] = NS(
        int64=torch.int64,
        empty=lambda shape, dtype, device: torch.empty(shape, dtype=dtype),
    )
    result = k.pack(
        decisions, drafts, DeviceTensor(actual), actual_counts, limits, row_ids, [14, 72], initialize=initialize
    )
    assert calls == [min(4, num_reqs - start) for start in range(0, num_reqs, 4)]
    assert torch.equal(result, packet)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_counts, counts)


@pytest.mark.parametrize("method", ["topk", "all", "prob_ratio"])
@pytest.mark.parametrize("eos,limits,threshold", [(None, [6, 6], None), ([5, 14], [6, 6], None), (None, [1, 4], 0)])
def test_device_publication_matches_pipeline_control(backend, modules, method, eos, limits, threshold):
    obj, _, _ = backend
    config, Pipeline = modules
    acceptance = sys.modules["vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance"]
    obj.model.compute_logits = lambda h: torch.nn.functional.one_hot(h[:, 0].long() + 1, 32).float() * 20
    obj.decision_runner = acceptance.IntermediateDecisionRunner(
        obj.model.compute_logits, acceptance.AcceptancePolicy(method, 1), 3
    )
    obj.drafter.propose.side_effect = [torch.tensor([[4, 5], [14, 15]]), torch.tensor([[7, 8], [17, 18]])]
    pipe = Pipeline(
        obj,
        config.IntermediateConfig("v", "d", num_rounds=2, max_generated_tokens=threshold),
        6,
        eos_token_id=eos,
    )
    output = torch.full((2, 6), -99, dtype=torch.int64)
    result = pipe.refine([[1, 2, 3], [11, 12, 13]], None, limits, ["a", "b"], device_output=output)
    assert output.tolist() == [row + [0] * (6 - len(row)) for row in result]


@pytest.mark.parametrize("group_width", [1, 2])
def test_device_publication_restores_reordered_resident_groups(backend, modules, group_width):
    obj, _, _ = backend
    config, Pipeline = modules
    acceptance = sys.modules["vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance"]
    obj.model.compute_logits = lambda h: torch.nn.functional.one_hot(h[:, 0].long() + 1, 32).float() * 20
    obj.decision_runner = acceptance.IntermediateDecisionRunner(
        obj.model.compute_logits, acceptance.AcceptancePolicy("all", 1), 3
    )
    # Put b first in the resident ordering, regardless of input order.
    slots, _ = obj.cache.plan(["b"], [[11, 12]], [1])
    obj.cache.commit(slots, [[11, 12]])
    obj.max_num_reqs = group_width
    proposals = (
        [torch.tensor([[14, 15]]), torch.tensor([[17, 18]]), torch.tensor([[4, 5]]), torch.tensor([[7, 8]])]
        if group_width == 1
        else [torch.tensor([[14, 15], [4, 5]]), torch.tensor([[17, 18], [7, 8]])]
    )
    obj.drafter.propose.side_effect = proposals
    pipe = Pipeline(obj, config.IntermediateConfig("v", "d", num_rounds=2), 6)
    output = torch.full((2, 6), -99, dtype=torch.int64)
    result = pipe.refine([[1, 2, 3], [11, 12, 13]], None, [6, 6], ["a", "b"], device_output=output)
    assert result == [[4, 5, 6, 7, 8, 9], [14, 15, 16, 17, 18, 19]]
    assert output.tolist() == [row + [0] * (6 - len(row)) for row in result]
