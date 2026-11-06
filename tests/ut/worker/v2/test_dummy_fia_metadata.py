# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Keep upstream ragged dummy request ownership through Ascend adaptation."""

import ast
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("padding", [0, 5])
@pytest.mark.parametrize("lengths", [[4080, 14], [1], [4, 4, 5]])
def test_dummy_kv_lengths_follow_upstream_query_lengths(lengths, padding):
    @dataclass
    class InputBatch:
        num_scheduled_tokens: np.ndarray
        positions: torch.Tensor

        @classmethod
        def make_dummy(cls, num_reqs, num_tokens, input_buffers, **kwargs):
            assert num_reqs == len(lengths)
            assert num_tokens == sum(lengths)
            return cls(np.array(lengths, dtype=np.int32), input_buffers.positions[:num_tokens])

    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/input_batch.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendInputBatch"]
    namespace = dict(
        dataclass=dataclass,
        fields=fields,
        np=np,
        torch=torch,
        InputBatch=InputBatch,
        AscendInputBuffers=object,
        AscendAttentionState=type("AscendAttentionState", (), {"DecodeOnly": "decode"}),
        update_cos_sin=Mock(),
    )
    exec(compile(tree, str(path), "exec"), namespace)
    num_tokens = sum(lengths)
    buffers = NS(
        seq_lens_np=np.full(len(lengths) + 2, -1, dtype=np.int32),
        dummy_num_tokens=num_tokens if padding else None,
        profile_cpp=False,
        input_ids=torch.full((num_tokens + padding,), -1, dtype=torch.int32),
        positions=torch.full((num_tokens + padding,), -1, dtype=torch.int64),
        is_padding=torch.zeros(num_tokens + padding, dtype=torch.bool),
    )
    batch = namespace["AscendInputBatch"].make_dummy(len(lengths), num_tokens + padding, buffers)
    np.testing.assert_array_equal(batch.seq_lens_np, lengths)
    np.testing.assert_array_equal(buffers.seq_lens_np[len(lengths) :], 0)
    namespace["update_cos_sin"].assert_called_once_with(batch.positions)
    if padding:
        assert batch.positions.shape == (num_tokens + padding,)
        assert batch.positions[num_tokens:].tolist() == [0] * padding
        assert buffers.input_ids[num_tokens:].tolist() == [0] * padding
        assert buffers.is_padding[num_tokens:].all()
