# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import DFLASH
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState


def test_packed_metadata_views_and_block_table_out_on_npu():
    """Check dtype reinterpretation and out= on the actual NPU backend."""
    host = torch.empty(36, dtype=torch.uint8, pin_memory=True)
    host[:24].view(torch.int64).copy_(torch.tensor([2, 0, 1]))
    host[24:].view(torch.int32).copy_(torch.tensor([2147483647, 16777217, 3], dtype=torch.int32))
    packed = host.to("npu", non_blocking=True)
    indices, ids = packed[:24].view(torch.int64), packed[24:].view(torch.int32)
    assert indices.untyped_storage().data_ptr() == ids.untyped_storage().data_ptr()
    source = torch.arange(12, dtype=torch.int32, device="npu").reshape(3, 4)
    output = torch.empty_like(source)
    pointer = output.data_ptr()
    torch.index_select(source, 0, indices, out=output)
    assert output.data_ptr() == pointer
    assert output.cpu().tolist() == [[8, 9, 10, 11], [0, 1, 2, 3], [4, 5, 6, 7]]
    assert ids.cpu().tolist() == [2147483647, 16777217, 3]


def _target_graph_probe(worker, install=False):
    """Run inside each worker, including subprocess/TP executors."""
    from vllm.compilation.counter import compilation_counter
    from vllm.config.compilation import CUDAGraphMode

    manager = worker.model_runner.cudagraph_manager
    backend = worker.model_runner.speculator.pipeline.backend
    if install:
        worker._long_target_full_calls = 0
        original = manager.run_fullgraph

        def record_replay(*args, **kwargs):
            worker._long_target_full_calls += 1
            return original(*args, **kwargs)

        manager.run_fullgraph = record_replay
        for label, draft_manager in (
            ("primary", worker.model_runner.speculator.query_cudagraph_manager),
            ("secondary", backend.drafter.query_cudagraph_manager),
        ):
            setattr(worker, f"_{label}_graph_calls", 0)
            original_draft = draft_manager.run_fullgraph

            def record_draft(*args, _label=label, _original=original_draft, **kwargs):
                attr = f"_{_label}_graph_calls"
                setattr(worker, attr, getattr(worker, attr) + 1)
                return _original(*args, **kwargs)

            draft_manager.run_fullgraph = record_draft
    return {
        "piecewise_sizes": [desc.num_tokens for desc in manager._capture_descs.get(CUDAGraphMode.PIECEWISE, [])],
        "full_sizes": [desc.num_tokens for desc in manager._capture_descs.get(CUDAGraphMode.FULL, [])],
        "captures": compilation_counter.num_cudagraph_captured,
        "calls": worker._long_target_full_calls,
        "forward_tokens": backend.forward_tokens,
        "reused_tokens": backend.reused_tokens,
        "reused_hidden_tokens": backend.reused_hidden_tokens,
        "intermediate_calls": backend.graph_replays,
        "intermediate_sizes": [
            desc.num_tokens for desc in backend.cudagraph_manager._capture_descs.get(CUDAGraphMode.FULL, [])
        ],
        "secondary_graphs": len(backend.drafter.query_cudagraph_manager.graphs),
        "primary_calls": worker._primary_graph_calls,
        "secondary_calls": worker._secondary_graph_calls,
        "non_full_capture_count": sum(
            len(descs)
            for graph_manager in (
                manager,
                backend.cudagraph_manager,
                worker.model_runner.speculator.query_cudagraph_manager,
                backend.drafter.query_cudagraph_manager,
            )
            for mode, descs in graph_manager._capture_descs.items()
            if mode != CUDAGraphMode.FULL
        ),
    }


def test_long_cached_prefill_attention_matches_causal_reference():
    """Exercise actual FIA with mixed 1/16/17/33 queries and paged prefix KV."""
    query_lengths, seq_lengths = [1, 16, 17, 33], [22, 44, 61, 90]
    generator = torch.Generator().manual_seed(19)
    query = torch.randn(sum(query_lengths), 4, 128, generator=generator, dtype=torch.float16)
    key = torch.randn(5, 128, 2, 128, generator=generator, dtype=torch.float16)
    value = torch.randn(5, 128, 2, 128, generator=generator, dtype=torch.float16)
    blocks = [4, 2, 1, 3]
    expected, offset = [], 0
    for block, qlen, slen in zip(blocks, query_lengths, seq_lengths):
        q = query[offset : offset + qlen].float().transpose(0, 1)
        k = key[block, :slen].float().repeat_interleave(2, dim=1).transpose(0, 1)
        v = value[block, :slen].float().repeat_interleave(2, dim=1).transpose(0, 1)
        scores = q @ k.transpose(-1, -2) / (128**0.5)
        causal_mask = torch.arange(slen)[None, :] > (slen - qlen + torch.arange(qlen))[:, None]
        expected.append((scores.masked_fill(causal_mask, -torch.inf).softmax(-1) @ v).transpose(0, 1))
        offset += qlen
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 4, 2, 128
    impl.scale, impl.sinks, impl.sliding_window = 128**-0.5, None, None
    impl.key_cache, impl.value_cache = key.npu(), value.npu()
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.ChunkedPrefill,
        block_tables=torch.tensor(blocks, dtype=torch.int32, device="npu")[:, None],
        actual_seq_lengths_q=torch.tensor(query_lengths).cumsum(0).tolist(),
        seq_lens_list=seq_lengths,
        attn_mask=torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device="npu"), diagonal=1),
        num_decodes=2,
        num_prefills=2,
        num_decode_tokens=17,
        causal=True,
    )
    query_npu = query.npu()
    output = torch.empty_like(query_npu)
    impl.forward_fused_infer_attention(query_npu, None, None, metadata, output)
    torch.testing.assert_close(output.cpu().float(), torch.cat(expected), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("method", ["topk", "all"])
@pytest.mark.parametrize("long_candidates,graph", [(False, False), (False, True), (True, False), (True, True)])
def test_multi_stage_dflash(method, long_candidates, graph, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    models = DFLASH["dflash"]
    prompts = ["The capital of France is", "List three prime numbers:"]
    params = [SamplingParams(temperature=0, max_tokens=n, ignore_eos=True) for n in (37, 61)]
    common = dict(
        max_model_len=256,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
    )
    reference = None
    if method == "topk":
        with VllmRunner(models["main"], **common) as runner:
            reference = [out.outputs[0].token_ids for out in runner.model.generate(prompts, params)]
    options = {
        "primary_num_speculative_tokens": 4 if long_candidates else 2,
        "intermediate": {
            "verifier": {"model": models["main"]},
            "drafter": {"model": models["spec"]},
            "num_rounds": 4 if long_candidates else 2,
            "num_speculative_tokens": 4 if long_candidates else 2,
            "max_num_seqs": 2,
            # Ensure >15 candidates regardless of the draft model's accuracy.
            "verification": {"method": "all" if long_candidates else "topk", "top_k": 1},
        },
        "final_verification": {"method": method, "top_k": 1},
    }
    with VllmRunner(
        models["main"],
        **{**common, "enforce_eager": not graph},
        compilation_config={"cudagraph_mode": "FULL" if graph else "NONE"},
        speculative_config={
            "method": "dflash",
            "model": models["spec"],
            "num_speculative_tokens": 20 if long_candidates else 6,
        },
        additional_config={"multi_stage_speculative": options},
    ) as runner:
        if graph:
            before = runner.model.llm_engine.collective_rpc(_target_graph_probe, kwargs={"install": True})
            assert all(
                not row["piecewise_sizes"]
                and row["full_sizes"]
                and row["captures"] > 0
                and row["intermediate_sizes"]
                and row["secondary_graphs"] > 0
                and row["non_full_capture_count"] == 0
                for row in before
            )
        tokens = [out.outputs[0].token_ids for out in runner.model.generate(prompts, params)]
        assert [len(row) for row in tokens] == [37, 61]
        if reference is not None:
            assert tokens == reference
        # Recycle request slots and scratch KV with a different prefix.
        output = runner.model.generate(
            ["One plus one equals"], SamplingParams(temperature=0, max_tokens=9, ignore_eos=True)
        )
        assert len(output[0].outputs[0].token_ids) == 9
        if graph:
            after = runner.model.llm_engine.collective_rpc(_target_graph_probe)
            assert all(row["calls"] > 0 for row in after)
            assert all(row["primary_calls"] > 0 and row["secondary_calls"] > 0 for row in after)
            assert all(end["intermediate_calls"] > begin["intermediate_calls"] for begin, end in zip(before, after))
            assert all(end["reused_tokens"] > begin["reused_tokens"] for begin, end in zip(before, after))
            assert all(end["reused_hidden_tokens"] > begin["reused_hidden_tokens"] for begin, end in zip(before, after))
            # Changed request shapes and slot reuse replay warmed graphs.
            assert [row["captures"] for row in before] == [row["captures"] for row in after]
