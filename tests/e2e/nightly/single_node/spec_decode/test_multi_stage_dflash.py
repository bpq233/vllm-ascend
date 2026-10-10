# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import DFLASH
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState
from vllm_ascend.worker.v2 import model_runner as runner_module
from vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance import AcceptancePolicy
from vllm_ascend.worker.v2.spec_decode.multi_stage.final_verification import FinalVerificationSampler


def _candidate_probe(worker, install=False):
    runner = worker.model_runner
    pipeline = runner.speculator.pipeline
    if install:
        worker._largest_candidate = 0
        worker._context_kv_tokens = 0
        worker._secondary_precomputed_calls = 0
        worker._initial_verifier_tokens = pipeline.backend.forward_tokens
        refine = pipeline.refine
        hydrate = pipeline.backend.drafter.model.precompute_and_store_context_kv
        propose = pipeline.backend.drafter.propose_precomputed

        def record_hydration(states, *args, **kwargs):
            worker._context_kv_tokens += states.shape[0]
            return hydrate(states, *args, **kwargs)

        def record_proposal(*args, **kwargs):
            worker._secondary_precomputed_calls += 1
            return propose(*args, **kwargs)

        pipeline.backend.drafter.model.precompute_and_store_context_kv = record_hydration
        pipeline.backend.drafter.propose_precomputed = record_proposal

        def record(*args, **kwargs):
            candidates = refine(*args, **kwargs)
            worker._largest_candidate = max(worker._largest_candidate, max(map(len, candidates), default=0))
            return candidates

        pipeline.refine = record
    return {
        "capacity": runner.speculator.final_capacity,
        "largest_candidate": worker._largest_candidate,
        "unused_progress_stream": runner.num_computed_tokens_stream is None,
        "shared_update_stream": pipeline.backend.drafter.update_stream is runner.update_stream,
        "context_kv_tokens": worker._context_kv_tokens,
        "verifier_tokens": pipeline.backend.forward_tokens - worker._initial_verifier_tokens,
        "secondary_precomputed_calls": worker._secondary_precomputed_calls,
    }


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


def test_deferred_final_trace_uses_existing_stream_completion_on_npu(monkeypatch):
    """Verify pinned async statistics before/after the adapter's D2H boundary."""
    verifier = FinalVerificationSampler.__new__(FinalVerificationSampler)
    verifier.policy = AcceptancePolicy("all")
    verifier.num_speculative_steps = 2
    verifier._verification_steps = torch.arange(3, device="npu")
    verifier.trace_forward = verifier.defer_trace = True
    verifier.verification_path = "decode"
    verifier._pending_traces = []
    verifier.forward_events = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
    verifier.forward_events[0].record()
    logits = torch.zeros(3, 4, device="npu")
    targets = torch.tensor([1, 2, 3], device="npu")
    verifier.forward_events[1].record()
    verifier.sampler = SimpleNamespace(sample=lambda **kwargs: (targets, logits))
    messages = []
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.multi_stage.final_verification.logger.debug",
        lambda *args: messages.append(args),
    )
    mapping = torch.tensor([0], device="npu")
    _, sampled, counts = verifier._verify(
        logits,
        None,
        torch.tensor([0, 1, 2], device="npu"),
        torch.arange(3, device="npu"),
        torch.tensor([0, 3], dtype=torch.int32, device="npu"),
        mapping,
        mapping.cpu().numpy(),
        torch.zeros(3, dtype=torch.int32, device="npu"),
        torch.arange(3, device="npu"),
    )
    assert not messages
    # This queued D2H is the same ordering boundary used by _read_step. No
    # stream/event synchronize is needed specifically for tracing.
    assert sampled.cpu().tolist() == [[1, 2, 3]]
    verifier.flush_trace()
    assert len(messages) == 1 and messages[0][1:4] == ([2], [2], "decode")
    assert counts.cpu().tolist() == [3] and not verifier._pending_traces


def test_target_metadata_cache_on_npu(monkeypatch):
    """Check immutable indices and stable graph query storage on real hardware."""
    runner = runner_module.NPUModelRunner.__new__(runner_module.NPUModelRunner)
    runner.speculator = SimpleNamespace(updates_computed_tokens_cpu=True)
    runner.device = torch.device("npu:0")
    upload = runner_module.async_copy_to_gpu
    calls = []

    def record(*args, **kwargs):
        calls.append(args[0].copy())
        return upload(*args, **kwargs)

    monkeypatch.setattr(runner_module, "async_copy_to_gpu", record)
    inputs = [np.array(values, dtype=np.int32) for values in ([1, 0], [0, 2, 5], [0, 2, 5, 8])]
    query = torch.empty(4, dtype=torch.int32, device=runner.device)
    pointer = query.data_ptr()

    def stage():
        return [
            runner._copy_spec_metadata(name, values, out=query if index == 2 else None)
            for index, (name, values) in enumerate(zip(("idx_mapping", "cu_num_logits", "query_start_loc"), inputs))
        ]

    first, second = stage(), stage()
    assert len(calls) == 3
    assert all(a is b for a, b in zip(first, second))
    assert [tensor.cpu().tolist() for tensor in second] == [values.tolist() for values in inputs]
    inputs[0][:] = [0, 1]
    inputs[1][:] = [0, 1, 5]
    inputs[2][:] = [0, 1, 5, 8]
    changed = stage()
    assert len(calls) == 6 and query.data_ptr() == pointer
    assert first[0].cpu().tolist() == [1, 0]  # Old read-only entries cannot change.
    assert [tensor.cpu().tolist() for tensor in changed] == [values.tolist() for values in inputs]
    runner._invalidate_spec_metadata()
    stage()
    assert len(calls) == 9 and query.data_ptr() == pointer


def _target_graph_probe(worker, install=False):
    """Run inside each worker, including subprocess/TP executors."""
    from vllm.compilation.counter import compilation_counter
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    manager = worker.model_runner.cudagraph_manager
    backend = worker.model_runner.speculator.pipeline.backend
    if install:
        worker._long_target_full_calls = 0
        worker._long_target_cached_full_calls = 0
        original = manager.run_fullgraph

        def record_replay(*args, **kwargs):
            worker._long_target_full_calls += 1
            metadata = next(iter(worker.model_runner.model_state.attn_metadata.values()))
            if metadata.attn_state == AscendAttentionState.ChunkedPrefill and metadata.max_query_len > 16:
                worker._long_target_cached_full_calls += 1
            return original(*args, **kwargs)

        manager.run_fullgraph = record_replay
        worker._verification_dispatches = {"target": {}, "intermediate": {}}
        worker._long_target_dispatches = {"full": 0, "miss": 0}
        for label, verifier_manager in (("target", manager), ("intermediate", backend.cudagraph_manager)):
            original_dispatch = verifier_manager.dispatch

            def record_dispatch(
                num_reqs,
                num_tokens,
                uniform_token_count,
                num_active_loras,
                *args,
                _label=label,
                _original=original_dispatch,
                **kwargs,
            ):
                desc = _original(num_reqs, num_tokens, uniform_token_count, num_active_loras, *args, **kwargs)
                counts = worker._verification_dispatches[_label]
                counts[desc.cg_mode.name] = counts.get(desc.cg_mode.name, 0) + 1
                max_query_len = kwargs.get("max_query_len", args[0] if args else None)
                # Older dispatch signatures expose only uniform length. A
                # larger average also proves at least one query exceeds 16.
                long_query = (max_query_len or uniform_token_count or 0) > 16 or num_tokens > 16 * num_reqs
                if _label == "target" and long_query:
                    key = "full" if desc.cg_mode == CUDAGraphMode.FULL else "miss"
                    worker._long_target_dispatches[key] += 1
                return desc

            verifier_manager.dispatch = record_dispatch
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
        "cached_long_calls": worker._long_target_cached_full_calls,
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
        "verification_dispatches": {label: counts.copy() for label, counts in worker._verification_dispatches.items()},
        "long_target_dispatches": worker._long_target_dispatches.copy(),
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


@pytest.mark.parametrize("method", ["topk", "all", "prob_ratio"])
@pytest.mark.parametrize(
    "long_candidates,graph_mode",
    [(False, None), (False, "FULL"), (True, None), (True, "FULL"), (True, "FULL_DECODE_ONLY")],
)
def test_multi_stage_dflash(method, long_candidates, graph_mode, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    models = DFLASH["dflash"]
    prompts = ["The capital of France is", "List three prime numbers:", "Count from one to ten:", "A triangle has"]
    params = [SamplingParams(temperature=0, max_tokens=n, ignore_eos=True) for n in (37, 61, 19, 45)]
    common = dict(
        max_model_len=256,
        max_num_seqs=4,
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
            "num_rounds": 5 if long_candidates else 2,
            "max_generated_tokens": 18 if long_candidates else None,
            "num_speculative_tokens": 4 if long_candidates else 2,
            "max_num_seqs": 2,
            # Exercise resident slots beyond the two-row compute block table.
            "cache_max_num_seqs": 4,
            # Ensure >15 candidates regardless of the draft model's accuracy.
            "verification": {"method": "all" if long_candidates else "topk", "top_k": 1},
        },
        "final_verification": {"method": method, "threshold": 0.5}
        if method == "prob_ratio"
        else {"method": method, "top_k": 1},
    }
    with VllmRunner(
        models["main"],
        **{**common, "enforce_eager": graph_mode is None},
        compilation_config={
            "cudagraph_mode": graph_mode or "NONE",
            "cudagraph_capture_sizes": [1, 5, 10, 16, 21, 32, 64, 128, 256],
        },
        speculative_config={
            "method": "dflash",
            "model": models["spec"],
            # This is no longer a target cap: long candidates cross 18 after
            # four rounds and preserve all 20 tokens. Storage is derived as 23.
            "num_speculative_tokens": 4 if long_candidates else 2,
        },
        additional_config={"multi_stage_speculative": options},
    ) as runner:
        initial = runner.model.llm_engine.collective_rpc(_candidate_probe, kwargs={"install": True})
        assert all(row["capacity"] == (23 if long_candidates else 6) for row in initial)
        assert all(row["unused_progress_stream"] and row["shared_update_stream"] for row in initial)
        if graph_mode:
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
        candidates = runner.model.llm_engine.collective_rpc(_candidate_probe)
        # Each new verifier row hydrates context KV once. Secondary proposals
        # must not repeat the projection/cache update, including in eager mode.
        assert all(row["secondary_precomputed_calls"] > 0 for row in candidates)
        assert all(row["context_kv_tokens"] == row["verifier_tokens"] > 0 for row in candidates)
        if long_candidates:
            assert all(row["largest_candidate"] == 20 for row in candidates)
        assert [len(row) for row in tokens] == [p.max_tokens for p in params]
        if reference is not None:
            assert tokens == reference
        # Recycle request slots and scratch KV with a different prefix.
        output = runner.model.generate(
            ["One plus one equals"], SamplingParams(temperature=0, max_tokens=9, ignore_eos=True)
        )
        assert len(output[0].outputs[0].token_ids) == 9
        if graph_mode:
            after = runner.model.llm_engine.collective_rpc(_target_graph_probe)
            assert all(row["calls"] > 0 for row in after)
            assert all(row["primary_calls"] > 0 and row["secondary_calls"] > 0 for row in after)
            assert all(end["intermediate_calls"] > begin["intermediate_calls"] for begin, end in zip(before, after))
            assert all(end["reused_tokens"] > begin["reused_tokens"] for begin, end in zip(before, after))
            assert all(end["reused_hidden_tokens"] > begin["reused_hidden_tokens"] for begin, end in zip(before, after))
            for row in after:
                for label in ("target", "intermediate"):
                    dispatches = row["verification_dispatches"][label]
                    assert dispatches.get("FULL", 0) > 0, (label, dispatches)
                    assert set(dispatches) == {"FULL"}, (label, dispatches)
                if long_candidates:
                    assert row["cached_long_calls"] > 0
                    assert row["long_target_dispatches"]["full"] > 0
                    assert row["long_target_dispatches"]["miss"] == 0
            # Changed request shapes and slot reuse replay warmed graphs.
            assert [row["captures"] for row in before] == [row["captures"] for row in after]
