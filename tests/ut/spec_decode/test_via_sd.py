from types import SimpleNamespace

import torch

from vllm_ascend.worker.v2.spec_decode.via_sd.kv_cache import ViaSdKVCacheManager
from vllm_ascend.worker.v2.spec_decode.via_sd.model import resolve_layer_ids
from vllm_ascend.worker.v2.spec_decode.via_sd.verifier import ViaSdVerifier


def test_resolve_layer_ids_uses_evenly_spaced_fraction_and_explicit_override():
    assert resolve_layer_ids(36, [], 0.4) == (
        0,
        3,
        5,
        8,
        11,
        13,
        16,
        19,
        22,
        24,
        27,
        30,
        32,
        35,
    )
    assert resolve_layer_ids(8, [1, 4], 0.4) == (1, 4)


def test_via_sd_cache_manager_checks_request_and_page_identity():
    manager = ViaSdKVCacheManager(enabled=True)
    manager.commit("r0", 2, [1, 2, 3], block_signature=((7, 8),))
    assert manager.reusable_prefix("r0", 2, [1, 2, 4], 2, ((7,),)) == 2
    assert manager.reusable_prefix("r0", 3, [1, 2, 3], 3, ((7, 8),)) == 0
    assert manager.reusable_prefix("r0", 2, [1, 2, 3], 3, ((9, 8),)) == 0
    manager.discard(["r0"])
    assert manager.reusable_prefix("r0", 2, [1, 2, 3], 3, ((7, 8),)) == 0


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def block_signature(self, table_index, end):
        num_pages = (end + 3) // 4
        return ((table_index, *range(num_pages)),)

    def forward(self, token_ids, start, table_index, return_logits=True):
        self.calls.append((list(token_ids), start, table_index, return_logits))
        if not return_logits:
            return None
        rows = torch.arange(start, start + len(token_ids), dtype=torch.float32)
        return rows[:, None].expand(-1, 5)


class _FakeBatchBackend:
    def __init__(self):
        self.batch_calls = []

    def block_signature(self, table_index, end):
        num_pages = (end + 3) // 4
        return ((table_index, *range(num_pages)),)

    def forward_batch(self, token_ids, starts, table_indices):
        self.batch_calls.append(
            ([list(tokens) for tokens in token_ids], list(starts), list(table_indices))
        )
        return [
            torch.arange(start, start + len(tokens), dtype=torch.float32)[:, None].expand(-1, 5)
            for tokens, start in zip(token_ids, starts)
        ]


def test_via_sd_verifier_returns_only_aligned_per_position_logits():
    backend = _FakeBackend()
    runner = SimpleNamespace(max_num_tokens=16, vocab_size=5, device=torch.device("cpu"))
    config = SimpleNamespace(kv_cache_enabled=True)
    verifier = ViaSdVerifier(runner, object(), config, backend=backend)

    draft = torch.tensor([[4, 5, 6]], dtype=torch.int64)
    logits = verifier.verify(
        draft,
        request_ids=["r0"],
        request_indices=[0],
        prefix_token_ids=[[10, 11, 12]],
        valid_lengths=[3],
        table_indices=[0],
    )

    assert logits.shape == (1, 3, 5)
    # input = prefix + draft[:-1], so rows 2, 3, 4 predict draft[0:3].
    assert torch.equal(logits[0, :, 0], torch.tensor([2.0, 3.0, 4.0]))
    assert all(len(call) == 4 for call in backend.calls)


def test_via_sd_verifier_packs_ragged_batch_into_one_backend_call():
    backend = _FakeBatchBackend()
    runner = SimpleNamespace(max_num_tokens=16, vocab_size=5, device=torch.device("cpu"))
    verifier = ViaSdVerifier(
        runner,
        object(),
        SimpleNamespace(kv_cache_enabled=True),
        backend=backend,
    )

    logits = verifier.verify(
        torch.tensor([[4, 5, 6], [7, 8, -1]], dtype=torch.int64),
        request_ids=["r0", "r1"],
        request_indices=[0, 1],
        prefix_token_ids=[[10, 11, 12], [20, 21]],
        valid_lengths=[3, 2],
        table_indices=[0, 1],
    )

    assert len(backend.batch_calls) == 1
    assert backend.batch_calls[0] == (
        [[10, 11, 12, 4, 5], [20, 21, 7]],
        [0, 0],
        [0, 1],
    )
    assert logits.shape == (2, 3, 5)
    assert torch.equal(logits[0, :, 0], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(logits[1, :2, 0], torch.tensor([1.0, 2.0]))
    assert torch.isneginf(logits[1, 2]).all()


def test_via_sd_cache_reuses_warm_prefix_and_matches_cache_off_logits():
    runner = SimpleNamespace(max_num_tokens=16, vocab_size=5, device=torch.device("cpu"))
    cache_on_backend = _FakeBackend()
    cache_off_backend = _FakeBackend()
    cache_on = ViaSdVerifier(
        runner,
        object(),
        SimpleNamespace(kv_cache_enabled=True),
        backend=cache_on_backend,
    )
    cache_off = ViaSdVerifier(
        runner,
        object(),
        SimpleNamespace(kv_cache_enabled=False),
        backend=cache_off_backend,
    )

    initial_draft = torch.tensor([[4, 5, 6]], dtype=torch.int64)
    for verifier in (cache_on, cache_off):
        verifier.verify(
            initial_draft,
            request_ids=["r0"],
            request_indices=[0],
            prefix_token_ids=[[10, 11, 12]],
            table_indices=[0],
        )

    # A live request can be absent from one scheduler batch. Its q' prefix
    # must survive until the scheduler explicitly finishes or preempts it.
    cache_on.verify(
        torch.tensor([[1, 2, 3]], dtype=torch.int64),
        request_ids=["r1"],
        request_indices=[1],
        prefix_token_ids=[[20, 21]],
        table_indices=[1],
    )

    cache_on_backend.calls.clear()
    cache_off_backend.calls.clear()
    next_draft = torch.tensor([[7, 8, 9]], dtype=torch.int64)
    common_args = {
        "request_ids": ["r0"],
        "request_indices": [0],
        "prefix_token_ids": [[10, 11, 12, 4, 99]],
        "table_indices": [0],
    }
    cache_on_logits = cache_on.verify(next_draft, **common_args)
    cache_off_logits = cache_off.verify(next_draft, **common_args)

    torch.testing.assert_close(cache_on_logits, cache_off_logits)
    assert cache_on_backend.calls == [([99, 7, 8], 4, 0, True)]
    assert cache_off_backend.calls == [
        ([10, 11, 12, 4, 99, 7, 8], 0, 0, True)
    ]
    assert cache_on.last_stats.cache_hits == 1
    assert cache_on.last_stats.cached_prefix_tokens == 4
    assert cache_on.last_stats.model_input_tokens == 3
    assert cache_off.last_stats.recomputed_prefix_tokens == 4
    assert cache_off.last_stats.model_input_tokens == 7


def test_via_sd_verifier_ignores_negative_draft_padding():
    backend = _FakeBackend()
    runner = SimpleNamespace(max_num_tokens=16, vocab_size=5, device=torch.device("cpu"))
    verifier = ViaSdVerifier(
        runner,
        object(),
        SimpleNamespace(kv_cache_enabled=True),
        backend=backend,
    )

    logits = verifier.verify(
        torch.tensor([[4, -1, -1]], dtype=torch.int64),
        request_ids=["r0"],
        request_indices=[0],
        prefix_token_ids=[[10, 11]],
        valid_lengths=[3],
        table_indices=[0],
    )

    assert verifier.last_stats.draft_tokens == 1
    assert verifier.last_stats.positions == ((2,),)
    assert all(-1 not in call[0] for call in backend.calls)
    assert torch.isfinite(logits[0, 0]).all()
    assert torch.isneginf(logits[0, 1:]).all()
