# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Sparse graph prefix warming batches requests without changing causal KV."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import test_intermediate_backend as backend_tests
import torch


@pytest.fixture
def backend():
    return backend_tests.backend.__wrapped__()


def test_warmup_batches_short_prefixes_and_preserves_causal_outputs(backend):
    obj, metadata, _ = backend
    obj.cudagraph_manager = NS(
        capture_sizes=[8, 128],
        cudagraph_mode=obj.CUDAGraphMode.FULL,
        dispatch=lambda n, total, *a, **kw: NS(cg_mode=obj.CUDAGraphMode.FULL, num_tokens=8),
        run_fullgraph=lambda desc: obj.model(
            input_ids=obj.input_buffers.input_ids[: desc.num_tokens],
            positions=obj.input_buffers.positions[: desc.num_tokens],
        ),
    )
    values = torch.zeros(20)

    def causal_forward(input_ids, positions):
        meta = metadata.call_args.kwargs
        actual = meta["num_actual_tokens"]
        values[meta["slot_mappings"][0, :actual]] = input_ids[:actual].float()
        output = torch.zeros(len(input_ids), 1)
        boundaries = meta["query_start_loc_cpu"].tolist()
        for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            for i in range(start, end):
                prefix = torch.arange(int(positions[i]) + 1)
                physical = meta["block_tables"][0][row, prefix // 4] * 4 + prefix % 4
                output[i, 0] = values[physical].sum()
        return output

    obj.model.side_effect = causal_forward
    prefixes = [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]]
    drafts = [[6], [16]]
    result = list(obj.verify(prefixes, drafts, req_ids=["a", "b"]))
    assert [row.tolist() for row in result] == [[[15], [21]], [[65], [81]]]
    assert obj.model.call_count == 2  # One shared warmup + verification, previously three.
    assert obj.forward_tokens == 12
    assert obj.executed_tokens == 16  # Previously three eight-token graph replays.
    warm_meta = metadata.call_args_list[0].kwargs
    assert warm_meta["num_reqs"] == 2
    assert warm_meta["query_start_loc_cpu"].tolist() == [0, 4, 8]
    assert warm_meta["seq_lens_np"].tolist() == [4, 4]  # Predictor stays in verification.
    assert obj.cudagraph_manager.capture_sizes == [8, 128]
    assert [obj.cache.tokens[obj.cache.slots[r]] for r in ["a", "b"]] == [p + d for p, d in zip(prefixes, drafts)]


@pytest.mark.parametrize("cached_kind", ["cold", "partial", "divergent"])
def test_warmup_packs_chunks_and_reuses_only_matching_prefixes(backend, cached_kind):
    obj, _, _ = backend
    obj.cache = type(obj.cache)(3)
    obj.max_num_reqs = 2
    obj.max_num_tokens = 64
    obj.cudagraph_manager = NS(capture_sizes=[8, 512])
    ids = ["a", "b", "c"]
    sequences = [list(range(17)), list(range(20, 29)), list(range(40, 47))]
    required = [len(s) - 2 for s in sequences]
    for req_id, sequence in zip(ids, sequences):
        if cached_kind == "cold":
            continue
        cached = sequence[:4] if cached_kind == "partial" else sequence[:2] + [999] * 4
        slots, _ = obj.cache.plan([req_id], [cached], [len(cached) - 1])
        obj.cache.commit(slots, [cached])
    slots, _ = obj.cache.plan(["outside"], [[999]], [0])
    obj.cache.commit(slots, [[999]])
    initial = [obj.cache.query_start(r, s, p) for r, s, p in zip(ids, sequences, required)]
    seen = {r: [] for r in ids}
    calls = []

    def forward(rows, req_ids, starts, *, retain_hidden):
        assert retain_hidden is False
        assert all(r in obj.cache.slots for r in ids)
        slots, computed = obj.cache.plan(req_ids, rows, starts)
        counts = [len(s) - c for s, c in zip(rows, computed)]
        assert 0 < len(rows) <= obj.max_num_reqs
        assert all(c > 0 for c in counts)
        assert sum(counts) <= 8
        for r, s, c in zip(req_ids, rows, computed):
            seen[r].extend(s[c:])
        calls.append((req_ids, counts))
        obj.cache.commit(slots, rows)

    obj._forward = forward
    obj._warm_prefixes(sequences, ids, required)
    assert any(len(rows) > 1 for rows, _ in calls)
    assert set(obj.cache.slots) == set(ids)
    assert seen == {r: s[c:p] for r, s, c, p in zip(ids, sequences, initial, required)}
    assert [obj.cache.tokens[obj.cache.slots[r]] for r in ids] == [s[:p] for s, p in zip(sequences, required)]


def test_warmup_failure_keeps_completed_chunks_and_retries_uncommitted_rows(backend):
    obj, _, _ = backend
    obj.max_num_tokens = 64
    obj.cudagraph_manager = NS(capture_sizes=[8, 512])
    sequences = [list(range(17)), list(range(30, 47))]
    ids = ["a", "b"]
    required = [15, 15]
    calls = []

    def forward(rows, req_ids, starts, *, retain_hidden):
        slots, computed = obj.cache.plan(req_ids, rows, starts)
        calls.append([(r, c, len(s)) for r, c, s in zip(req_ids, computed, rows)])
        if len(calls) == 2:
            raise RuntimeError("context KV store failed")
        obj.cache.commit(slots, rows)

    obj._forward = forward
    with pytest.raises(RuntimeError, match="context KV store failed"):
        obj._warm_prefixes(sequences, ids, required)
    assert [len(obj.cache.tokens[obj.cache.slots[r]]) for r in ids] == [4, 4]
    obj._warm_prefixes(sequences, ids, required)
    assert calls[2] == calls[1]
    assert [obj.cache.tokens[obj.cache.slots[r]] for r in ids] == [s[:15] for s in sequences]


@pytest.mark.parametrize(
    ("sizes", "required"),
    [([], [4, 4]), ([8, 96], [4, 4]), ([128], [4, 4]), ([8, 128], [0, 0])],
)
def test_warmup_preserves_sparse_graph_trigger(backend, sizes, required):
    obj, _, _ = backend
    obj.cudagraph_manager = NS(capture_sizes=sizes)
    obj._forward = Mock()
    obj._warm_prefixes([[1] * 6, [2] * 6], ["a", "b"], required)
    obj._forward.assert_not_called()
    assert not obj.cache.slots
