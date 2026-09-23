# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contract tests for upstream dummy batch adaptation."""

import ast
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("is_release", [True, False])
@pytest.mark.parametrize("lengths", [[4080, 14], [1], [4, 4, 5]])
def test_dummy_kv_lengths_follow_upstream_query_lengths(is_release, lengths):
    @dataclass
    class InputBatch:
        num_scheduled_tokens: np.ndarray
        positions: torch.Tensor

        @classmethod
        def make_dummy(cls, num_reqs, num_tokens, input_buffers, **kwargs):
            assert num_reqs == len(lengths)
            assert num_tokens == sum(lengths)
            return cls(np.array(lengths, dtype=np.int32), torch.zeros(num_tokens, dtype=torch.long))

    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/input_batch.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendInputBatch"]
    namespace = dict(
        dataclass=dataclass,
        fields=fields,
        np=np,
        InputBatch=InputBatch,
        AscendInputBuffers=object,
        AscendAttentionState=type("AscendAttentionState", (), {"DecodeOnly": "decode"}),
        vllm_version_is=lambda version: is_release,
        update_cos_sin=Mock(),
    )
    exec(compile(tree, str(path), "exec"), namespace)
    buffers = NS(seq_lens_np=np.full(len(lengths) + 2, -1, dtype=np.int32))
    batch = namespace["AscendInputBatch"].make_dummy(len(lengths), sum(lengths), buffers)
    np.testing.assert_array_equal(batch.seq_lens_np, lengths)
    np.testing.assert_array_equal(buffers.seq_lens_np[len(lengths):], 0)
    namespace["update_cos_sin"].assert_called_once_with(batch.positions)
