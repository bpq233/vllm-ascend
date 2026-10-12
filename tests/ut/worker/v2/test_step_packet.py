# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Check step-packet values and storage reuse without importing NPU libraries."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def pack_step():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage/step_packet.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pack_step"]
    namespace = {"torch": torch}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["pack_step"]


def make_state(lengths=(2, 5, 1)):
    return NS(
        total_len=NS(gpu=torch.tensor(lengths, dtype=torch.int32)),
        num_computed_tokens=NS(gpu=torch.tensor(lengths, dtype=torch.int32) - 1),
        all_token_ids=NS(gpu=torch.arange(24, dtype=torch.int32).reshape(3, 8)),
    )


def make_batch(indices):
    return NS(num_reqs=len(indices), idx_mapping=torch.tensor(indices, dtype=torch.int32))


def test_reordered_short_histories_and_primary(pack_step):
    primary = torch.tensor([[91, 92], [93, 94], [95, 96]])
    actual = pack_step(make_state(), make_batch([2, 0, 1]), primary, 4)
    assert actual.dtype == torch.int64
    assert actual.tolist() == [
        [1, 0, 91, 92, 16, 16, 16, 16],
        [2, 1, 93, 94, 0, 0, 0, 1],
        [5, 4, 95, 96, 9, 10, 11, 12],
    ]


@pytest.mark.parametrize("width", [0, 1, 4, 8])
def test_without_primary_and_zero_history(pack_step, width):
    actual = pack_step(make_state((0, 5, 1)), make_batch([0, 1]), torch.empty((2, 0), dtype=torch.int64), width)
    assert actual.shape == (2, 2 + width)
    assert actual[0].tolist() == [0, -1] + [0] * width
    assert actual[1].tolist() == [5, 4] + [8 + max(5 - width + i, 0) for i in range(width)]


def test_strided_inputs(pack_step):
    state = make_state()
    state.total_len.gpu = torch.tensor([2, 99, 5, 99, 1, 99], dtype=torch.int32)[::2]
    state.num_computed_tokens.gpu = torch.tensor([1, 99, 4, 99, 0, 99], dtype=torch.int32)[::2]
    state.all_token_ids.gpu = torch.arange(48, dtype=torch.int32).reshape(3, 16)[:, ::2]
    batch = NS(num_reqs=2, idx_mapping=torch.tensor([1, 99, 0, 99], dtype=torch.int32)[::2])
    primary = torch.tensor([[91, 99, 92, 99], [93, 99, 94, 99]])[:, ::2]
    actual = pack_step(state, batch, primary, 3)
    assert actual.tolist() == [[5, 4, 91, 92, 20, 22, 24], [2, 1, 93, 94, 0, 0, 2]]


def test_matching_output_is_reused_and_fully_overwritten(pack_step):
    state, batch = make_state(), make_batch([2, 0])
    primary = torch.empty((2, 0), dtype=torch.int64)
    out = torch.full((2, 6), -500, dtype=torch.int64)
    assert pack_step(state, batch, primary, 4, out=out) is out
    state.total_len.gpu[0] = 4
    batch.idx_mapping[:] = torch.tensor([0, 2], dtype=torch.int32)
    assert pack_step(state, batch, primary, 4, out=out) is out
    assert out.tolist() == [[4, 1, 0, 1, 2, 3], [1, 0, 16, 16, 16, 16]]


@pytest.mark.parametrize(
    "out",
    [
        torch.empty((2, 5), dtype=torch.int64),
        torch.empty((2, 6), dtype=torch.int32),
        torch.empty((6, 2), dtype=torch.int64).T,
    ],
)
def test_incompatible_output_is_replaced(pack_step, out):
    actual = pack_step(make_state(), make_batch([0, 1]), torch.empty((2, 0), dtype=torch.int64), 4, out=out)
    assert actual is not out
    assert actual.shape == (2, 6)
    assert actual.dtype == torch.int64
    assert actual.is_contiguous()


def test_empty_request_batch(pack_step):
    actual = pack_step(make_state(), make_batch([]), torch.empty((0, 2), dtype=torch.int64), 4)
    assert actual.shape == (0, 8)


@pytest.mark.parametrize("width", [-1, 9])
def test_invalid_history_width(pack_step, width):
    with pytest.raises(ValueError, match="width"):
        pack_step(make_state(), make_batch([0]), torch.empty((1, 0), dtype=torch.int64), width)


def test_primary_batch_mismatch(pack_step):
    with pytest.raises(ValueError, match="Primary"):
        pack_step(make_state(), make_batch([0, 1]), torch.empty((1, 0), dtype=torch.int64), 4)
