# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU regressions for FIA shape validation, without importing NPU libraries."""

import ast
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def attention():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/attention_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_fit_fia_query_to_output", "_normalize_fia_query_metadata"}
    methods = {
        "forward",
        "forward_impl",
        "reshape_and_cache",
        "_get_fia_params",
        "forward_fused_infer_attention",
        "_get_kv_cache_view",
    }
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionBackendImpl":
            body.extend(
                method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in methods
            )
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {"FIAParamProvider", "FIAV2ParamProvider"}:
            body.append(node)
    namespace = dict(
        dataclass=dataclass,
        Any=Any,
        attention_transfer_window=nullcontext,
        envs_vllm=NS(VLLM_BATCH_INVARIANT=False),
        get_current_hardware_profile=lambda: NS(supports=lambda capability: False),
        HardwareCapability=NS(CHUNKED_PREFILL_PHASE_SPLIT="split"),
        torch=torch,
        _EXTRA_CTX=NS(capturing=False),
        AttentionLayer=object,
        AscendMetadata=object,
        AttentionType=NS(ENCODER_DECODER="cross"),
        AscendAttentionState=NS(
            PrefillNoCache="prefill", PrefillCacheHit="cached", DecodeOnly="decode", ChunkedPrefill="chunked"
        ),
        DeviceOperator=NS(reshape_and_cache=Mock(), npu_fused_infer_attention_score=Mock()),
        notify_kv_cache_written=Mock(),
        record_attention_compute_start=Mock(),
        using_paged_attention=lambda *args: False,
        is_950=lambda: False,
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return NS(
        **{name: namespace[name] for name in names | methods | {"FIAParamProvider", "FIAV2ParamProvider"}},
        namespace=namespace,
    )


@pytest.mark.parametrize("boundaries", [[4094], [4080, 4094], [2, 4]])
def test_missing_request_boundary_is_not_invented(attention, boundaries):
    original = boundaries.copy()
    with pytest.raises(RuntimeError, match="Cannot infer request ownership"):
        attention._normalize_fia_query_metadata(1, boundaries, [4094] * len(boundaries), None)
    assert boundaries == original


@pytest.mark.parametrize("query_tokens,boundaries", [(1, [1, 4094]), (3, [1, 3, 4094])])
def test_exact_padding_boundary_keeps_request_ownership(attention, query_tokens, boundaries):
    blocks = torch.arange(len(boundaries) * 4).reshape(len(boundaries), 4)
    lengths = [10, 20, 30][: len(boundaries)]
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


@pytest.mark.parametrize("boundaries", [[43], [19, 43]])
@pytest.mark.parametrize("state", ["prefill", "cached", "chunked", "decode"])
def test_padded_forward_keeps_real_tokens_and_output_buffer(attention, boundaries, state):
    """Reproduce the 43-token batch padded to 48 with no dummy request."""
    q = torch.randn(48, 2, 4)
    key = torch.randn(48, 1, 4)
    value = torch.randn_like(key)
    output = torch.full_like(q, -123)
    cache = (torch.zeros(1, 128, 1, 4), torch.zeros(1, 128, 1, 4))
    lengths = [b - a for a, b in zip([0] + boundaries, boundaries)]
    metadata = NS(
        actual_seq_lengths_q=boundaries.copy(),
        num_actual_tokens=43,
        attn_state=state,
        seq_lens=torch.tensor(lengths),
        seq_lens_list=lengths,
        block_tables=torch.zeros(len(boundaries), 1, dtype=torch.int32),
        slot_mapping=torch.arange(43),
        model_runner_type="generate",
        causal=True,
        uses_pcp_sharded_layout=False,
        attn_mask=None,
    )
    impl = NS(
        _use_layer_aware_fia_graph_replay=False,
        key_cache=None,
        value_cache=None,
        use_bnsd_kv_cache=False,
        kv_sharing_target_layer_name=None,
        attn_type="decoder",
        sliding_window=None,
        sinks=None,
        num_heads=2,
        num_kv_heads=1,
        head_size=4,
        scale=0.5,
        vllm_config=None,
    )
    for name in (
        "forward_impl",
        "reshape_and_cache",
        "_get_fia_params",
        "forward_fused_infer_attention",
        "_get_kv_cache_view",
    ):
        setattr(impl, name, getattr(attention, name).__get__(impl))
    ops = attention.namespace["DeviceOperator"]

    def fia(**kwargs):
        assert kwargs["query"].shape == (43, 2, 4)
        assert kwargs["actual_seq_lengths"] == boundaries
        if state == "prefill":
            torch.testing.assert_close(kwargs["key"], key[:43])
            torch.testing.assert_close(kwargs["value"], value[:43])
        else:
            assert kwargs["key"].data_ptr() == cache[0].data_ptr()
            assert kwargs["actual_seq_lengths_kv"] == lengths
        return kwargs["query"] + 1, None

    ops.npu_fused_infer_attention_score.side_effect = fia
    layer = NS(layer_name="model.layers.0.self_attn.attn", _k_scale_float=1.0, _v_scale_float=1.0)
    result = attention.forward(impl, layer, q, key, value, cache, metadata, output)
    assert result is output
    torch.testing.assert_close(output[:43], q[:43] + 1)
    assert torch.all(output[43:] == -123)
    assert metadata.actual_seq_lengths_q == boundaries
    written = ops.reshape_and_cache.call_args.kwargs
    torch.testing.assert_close(written["key"], key[:43])
    torch.testing.assert_close(written["value"], value[:43])
    assert written["slot_mapping"].numel() == 43
    ops.npu_fused_infer_attention_score.assert_called_once()


@pytest.mark.parametrize("capturing", [False, True])
def test_explicit_dummy_request_retains_graph_shape(attention, capturing):
    attention.namespace["_EXTRA_CTX"].capturing = capturing
    q = torch.empty(48, 2, 4)
    metadata = NS(actual_seq_lengths_q=[43, 48], num_actual_tokens=43)
    fitted, count = attention._fit_fia_query_to_output(q, torch.empty_like(q), attn_metadata=metadata)
    assert fitted is q
    assert count == 48


@pytest.mark.parametrize("actual_tokens", [0, 42, 44, 49])
def test_padding_requires_agreement_with_real_token_count(attention, actual_tokens):
    q = torch.empty(48, 2, 4)
    metadata = NS(actual_seq_lengths_q=[43], num_actual_tokens=actual_tokens)
    with pytest.raises(RuntimeError, match="FIA input contract mismatch"):
        attention._fit_fia_query_to_output(q, torch.empty_like(q), attn_metadata=metadata)


def test_padding_does_not_hide_undersized_output(attention):
    q = torch.empty(48, 2, 4)
    metadata = NS(actual_seq_lengths_q=[43], num_actual_tokens=43)
    with pytest.raises(RuntimeError, match="FIA input contract mismatch"):
        attention._fit_fia_query_to_output(q, torch.empty(1, 2, 4), attn_metadata=metadata)


@pytest.mark.parametrize("provider_name", ["FIAParamProvider", "FIAV2ParamProvider"])
@pytest.mark.parametrize("non_pa", [False, True])
def test_graph_provider_trims_padding_to_captured_query(attention, provider_name, non_pa):
    blocks = torch.arange(8).reshape(2, 4)
    metadata = NS(actual_seq_lengths_q=[1, 4094], seq_lens_list=[32, 4094], block_tables=blocks)
    provider_class = getattr(attention, provider_name)
    kwargs = {"sliding_window": None} if provider_name == "FIAParamProvider" else {}
    provider = provider_class("layer", query_tokens=1, is_non_pa=non_pa, **kwargs)
    params = provider.resolve({"layer": metadata})
    query_key = "actual_seq_lengths" if provider_name == "FIAParamProvider" else "actual_seq_qlen"
    kv_key = "actual_seq_lengths_kv" if provider_name == "FIAParamProvider" else "actual_seq_kvlen"
    assert params[query_key] == [1]
    assert params[kv_key] == [32]
    if non_pa:
        assert params["block_table"] is None
    else:
        torch.testing.assert_close(params["block_table"], blocks[:1])
    assert metadata.actual_seq_lengths_q == [1, 4094]


@pytest.mark.parametrize("provider_name", ["FIAParamProvider", "FIAV2ParamProvider"])
def test_graph_provider_rejects_missing_captured_boundary(attention, provider_name):
    metadata = NS(actual_seq_lengths_q=[4, 8], seq_lens_list=[32, 36], block_tables=torch.zeros(2, 1))
    provider_class = getattr(attention, provider_name)
    kwargs = {"sliding_window": None} if provider_name == "FIAParamProvider" else {}
    provider = provider_class("layer", query_tokens=3, **kwargs)
    with pytest.raises(RuntimeError, match="Cannot infer request ownership"):
        provider.resolve({"layer": metadata})
