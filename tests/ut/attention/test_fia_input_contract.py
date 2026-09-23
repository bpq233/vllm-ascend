# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU regressions for FIA shape validation, without importing NPU libraries."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def attention():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/attention_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_fit_fia_query_to_output", "_normalize_fia_query_metadata"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionBackendImpl":
            forward = next(
                method for method in node.body if isinstance(method, ast.FunctionDef) and method.name == "forward"
            )
            body.append(forward)
    namespace = dict(torch=torch, _EXTRA_CTX=NS(capturing=False), AttentionLayer=object, AscendMetadata=object)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return NS(**{name: namespace[name] for name in names}, forward=namespace["forward"])


@pytest.mark.parametrize("boundaries", [[4094], [4080, 4094], [2, 4]])
def test_missing_request_boundary_is_not_invented(attention, boundaries):
    original = boundaries.copy()
    with pytest.raises(RuntimeError, match="Cannot infer request ownership"):
        attention._normalize_fia_query_metadata(1, boundaries, [4094] * len(boundaries), None)
    assert boundaries == original


@pytest.mark.parametrize("query_tokens,boundaries", [(1, [1, 4094]), (3, [1, 3, 4094])])
def test_exact_padding_boundary_keeps_request_ownership(attention, query_tokens, boundaries):
    blocks = torch.arange(len(boundaries) * 4).reshape(len(boundaries), 4)
    lengths = [10, 20, 30][:len(boundaries)]
    q, kv, table = attention._normalize_fia_query_metadata(query_tokens, boundaries, lengths, blocks)
    assert q == boundaries[:-1]
    assert kv == lengths[:-1]
    torch.testing.assert_close(table, blocks[:-1])
    assert table.data_ptr() == blocks.data_ptr()


def test_matching_inputs_are_not_copied(attention):
    q = torch.empty(5, 2, 4)
    output = torch.empty_like(q)
    fitted, count = attention._fit_fia_query_to_output(q, output)
    assert fitted is q
    assert count == 5


@pytest.mark.parametrize("query_tokens,output_tokens", [(4094, 1), (1, 1)])
def test_original_shapes_are_reported_before_cache_write(attention, query_tokens, output_tokens):
    impl = NS(_use_layer_aware_fia_graph_replay=False, key_cache=None, reshape_and_cache=Mock(), forward_impl=Mock())
    layer = NS(layer_name="target.layers.0.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    metadata = NS(actual_seq_lengths_q=[4080, 4094], num_actual_tokens=4094, attn_state="prefill")
    q = torch.empty(query_tokens, 32, 128, device="meta")
    key = torch.empty(4094, 8, 128, device="meta")
    output = torch.empty(output_tokens, 32, 128, device="meta")
    with pytest.raises(RuntimeError, match="FIA input contract mismatch") as exc:
        attention.forward(impl, layer, q, key, key, (object(), object()), metadata, output)
    message = str(exc.value)
    assert f"query_shape=({query_tokens}, 32, 128)" in message
    assert f"output_shape=({output_tokens}, 32, 128)" in message
    assert "actual_seq_lengths_q=[4080, 4094]" in message
    assert "layer=target.layers.0.attn" in message
    impl.reshape_and_cache.assert_not_called()
    impl.forward_impl.assert_not_called()


def test_empty_boundaries_are_not_treated_as_one_request(attention):
    with pytest.raises(RuntimeError, match="boundaries are missing"):
        attention._normalize_fia_query_metadata(1, [], [], None)


def test_real_requests_are_not_mistaken_for_padding(attention):
    q = torch.empty(1, 2, 4)
    metadata = NS(actual_seq_lengths_q=[1, 4094], num_actual_tokens=4094, attn_state="prefill")
    with pytest.raises(RuntimeError, match="num_actual_tokens=4094"):
        attention._fit_fia_query_to_output(q, torch.empty_like(q), attn_metadata=metadata)
