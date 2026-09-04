from types import SimpleNamespace

import torch

from vllm_ascend.worker.v2.spec_decode.via_sd.kv_cache import ViaSdKVCacheManager
from vllm_ascend.worker.v2.spec_decode.via_sd.model import resolve_layer_ids
from vllm_ascend.worker.v2.spec_decode.via_sd.verifier import ViaSdVerifier


def test_resolve_layer_ids_uses_nearest_fraction_and_explicit_override():
    assert len(resolve_layer_ids(36, [], 0.4)) == 14
    assert resolve_layer_ids(8, [1, 4], 0.4) == (1, 4)


def test_via_sd_cache_manager_checks_request_and_page_identity():
    manager = ViaSdKVCacheManager(enabled=True)
    manager.commit("r0", 2, [1, 2, 3], block_signature=((7, 8),))
    assert manager.reusable_prefix("r0", 2, [1, 2, 4], 2, ((7, 8),)) == 2
    assert manager.reusable_prefix("r0", 3, [1, 2, 3], 3, ((7, 8),)) == 0
    assert manager.reusable_prefix("r0", 2, [1, 2, 3], 3, ((9, 8),)) == 0


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def block_signature(self, table_index, end):
        return ((table_index, end),)

    def forward(self, token_ids, start, table_index, return_logits=True):
        self.calls.append((list(token_ids), start, table_index, return_logits))
        if not return_logits:
            return None
        rows = torch.arange(start, start + len(token_ids), dtype=torch.float32)
        return rows[:, None].expand(-1, 5)


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
