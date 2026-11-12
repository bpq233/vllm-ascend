# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""CPU reference checks for the shared-weight sparse decoder execution view.

These tests use real causal attention, RoPE and a token-addressed KV cache.
They establish host-side semantics, not CANN kernel or ACL graph correctness.
Run without the repository's NPU conftest using ``--confcutdir`` on this folder.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

MODEL_PATH = Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify/model.py"
SPEC = importlib.util.spec_from_file_location("sparse_verifier_model_test", MODEL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
SparseTargetVerifier = MODEL.SparseTargetVerifier

HIDDEN_SIZE = 8
VOCAB_SIZE = 19
CACHE_SIZE = 64


class ToyRMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(HIDDEN_SIZE).abs() + 0.5)

    def forward(self, x, residual=None):
        stream = x if residual is None else x + residual
        normalized = stream * (stream.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * self.weight
        return normalized if residual is None else (normalized, stream)


class ToyAttention(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.layer_name = f"model.layers.{index}.self_attn.attn"
        self.kv_sharing_target_layer_name = None
        self.qkv = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 3, bias=False)
        self.kv_cache = torch.zeros(2, CACHE_SIZE, HIDDEN_SIZE)
        self.impl = SimpleNamespace(key_cache=None, value_cache=None, _layer_name=self.layer_name)

    @staticmethod
    def rope(x, positions):
        half = x.shape[-1] // 2
        angle = positions[:, None].float() / torch.arange(1, half + 1).float()[None, :]
        a, b = x[:, :half], x[:, half:]
        return torch.cat((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1)

    def forward(self, x, positions):
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k = self.rope(q, positions), self.rope(k, positions)
        if self.impl.key_cache is None:
            self.impl.key_cache, self.impl.value_cache = self.kv_cache[0], self.kv_cache[1]
        self.impl.key_cache[positions] = k
        self.impl.value_cache[positions] = v
        rows = []
        for query, position in zip(q, positions.tolist()):
            keys = self.impl.key_cache[: position + 1]
            values = self.impl.value_cache[: position + 1]
            scores = query @ keys.T / HIDDEN_SIZE**0.5
            rows.append(scores.softmax(-1) @ values)
        return torch.stack(rows)


class ToySelfAttention(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.attn = ToyAttention(index)
        self.output = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, positions, hidden_states):
        return self.output(self.attn(hidden_states, positions))


class ToyDecoderLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.self_attn = ToySelfAttention(index)
        self.input_layernorm = ToyRMSNorm()
        self.post_attention_layernorm = ToyRMSNorm()
        self.mlp = nn.Sequential(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU())

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class ToyTarget(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        torch.manual_seed(141)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.model.layers = nn.ModuleList(ToyDecoderLayer(i) for i in range(num_layers))
        self.model.norm = ToyRMSNorm()
        self.model.start_layer, self.model.end_layer = 0, num_layers
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    @staticmethod
    def logits_processor(lm_head, hidden_states):
        return lm_head(hidden_states)

    def reference(self, tokens, positions, selected=None, aux=(1, 3, 4)):
        selected = range(len(self.model.layers)) if selected is None else selected
        hidden_states = self.model.embed_tokens(tokens)
        residual = None
        features = {0: hidden_states}
        for i in selected:
            hidden_states, residual = self.model.layers[i](positions, hidden_states, residual)
            features[i + 1] = hidden_states + residual
        hidden_states, _ = self.model.norm(hidden_states, residual)
        return self.lm_head(hidden_states), [features[i] for i in aux]


def make_sparse(target=None, selected=(0, 2, 3), aux=(1, 3, 4)):
    target = ToyTarget() if target is None else target
    registry = {layer.self_attn.attn.layer_name: layer.self_attn.attn for layer in target.model.layers}
    sparse = SparseTargetVerifier(target, selected, aux, registry)
    for attention in sparse.attention_layers.values():
        attention.kv_cache = torch.zeros(2, CACHE_SIZE, HIDDEN_SIZE)
    return target, sparse, registry


@torch.inference_mode()
def test_prefix_features_equal_same_layers_of_full_target():
    target, prefix, _ = make_sparse(selected=(0, 1, 2), aux=(1, 2, 3))
    tokens, positions = torch.tensor([3, 8, 1, 9, 11]), torch.arange(5)
    actual_logits, prefix_features = prefix.forward_logits(tokens, positions)
    expected_logits, _ = target.reference(tokens, positions, selected=(0, 1, 2), aux=(1, 2, 3))
    _, target_features = target.reference(tokens, positions, aux=(1, 2, 3))
    torch.testing.assert_close(actual_logits, expected_logits)
    for actual, expected in zip(prefix_features, target_features):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@torch.inference_mode()
def test_all_layers_equal_full_target_logits_and_hidden_states():
    target, sparse, _ = make_sparse(selected=(0, 1, 2, 3), aux=(0, 1, 3, 4))
    tokens, positions = torch.tensor([3, 8, 1, 9, 11]), torch.arange(5)
    actual_logits, actual_features = sparse.forward_logits(tokens, positions)
    expected_logits, expected_features = target.reference(tokens, positions, aux=(0, 1, 3, 4))
    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-6, rtol=1e-6)
    for actual, expected in zip(actual_features, expected_features):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@torch.inference_mode()
def test_sparse_layers_equal_independent_reference_and_keep_original_indices():
    target, sparse, registry = make_sparse()
    tokens, positions = torch.tensor([5, 10, 4]), torch.arange(3)
    actual, features = sparse.forward_logits(tokens, positions)
    expected, reference_features = target.reference(tokens, positions, selected=(0, 2, 3))
    torch.testing.assert_close(actual, expected)
    assert tuple(sparse.layers) == ("0", "2", "3")
    assert set(sparse.attention_layers) == {f"dflash_sparse_verify.model.layers.{i}.self_attn.attn" for i in (0, 2, 3)}
    assert len(registry) == 7
    assert all(x.shape == (3, HIDDEN_SIZE) for x in features)
    for actual_feature, reference in zip(features, reference_features):
        torch.testing.assert_close(actual_feature, reference)


@torch.inference_mode()
def test_incremental_kv_matches_full_context_and_rollback_discards_suffix():
    _, sparse_incremental, _ = make_sparse()
    _, sparse_full, _ = make_sparse()
    prefix = torch.tensor([2, 7, 1])
    sparse_incremental.forward_logits(prefix, torch.arange(3))
    # A rejected speculative suffix remains physically present. Overwrite it
    # at the same positions; causal lengths must prevent its later use.
    sparse_incremental.forward_logits(torch.tensor([13, 14, 15]), torch.arange(3, 6))
    replacement = torch.tensor([5, 3])
    incremental, incremental_features = sparse_incremental.forward_logits(replacement, torch.arange(3, 5))
    full, full_features = sparse_full.forward_logits(torch.cat((prefix, replacement)), torch.arange(5))
    torch.testing.assert_close(incremental, full[-2:])
    for actual, expected in zip(incremental_features, full_features):
        torch.testing.assert_close(actual, expected[-2:])
    later, _ = sparse_incremental.forward_logits(torch.tensor([8]), torch.tensor([5]))
    _, sparse_fresh, _ = make_sparse()
    full_later, _ = sparse_fresh.forward_logits(torch.tensor([2, 7, 1, 5, 3, 8]), torch.arange(6))
    torch.testing.assert_close(later, full_later[-1:])


@torch.inference_mode()
def test_future_tokens_cannot_change_previous_logits_and_shift_is_explicit():
    _, first, _ = make_sparse()
    _, second, _ = make_sparse()
    left, _ = first.forward_logits(torch.tensor([2, 7, 1, 5]), torch.arange(4))
    right, _ = second.forward_logits(torch.tensor([2, 7, 1, 12]), torch.arange(4))
    torch.testing.assert_close(left[:3], right[:3])
    # The anchor's output predicts candidate 0. Candidate 0's own hidden
    # state is used to predict candidate 1, never itself.
    _, incremental, _ = make_sparse()
    prefix_logits, _ = incremental.forward_logits(torch.tensor([2, 7, 1]), torch.arange(3))
    torch.testing.assert_close(left[2], prefix_logits[-1])


@torch.inference_mode()
def test_aux_order_and_last_boundary_are_preserved():
    target, sparse, _ = make_sparse(aux=(4, 0, 3, 1))
    tokens, positions = torch.tensor([2, 7]), torch.arange(2)
    _, actual = sparse.forward_logits(tokens, positions)
    _, expected = target.reference(tokens, positions, selected=(0, 2, 3), aux=(4, 0, 3, 1))
    assert len(actual) == 4
    for actual_feature, expected_feature in zip(actual, expected):
        torch.testing.assert_close(actual_feature, expected_feature)


def test_weight_aliases_and_cache_owners_are_independent():
    target, sparse, _ = make_sparse()
    target_parameter_ids = {id(p) for p in target.parameters()}
    assert all(id(p) in target_parameter_ids for p in sparse.parameters())
    assert sparse.embed_tokens is target.model.embed_tokens
    assert sparse.lm_head is target.lm_head
    assert sparse.norm is target.model.norm
    for i in sparse.selected_layers:
        original = target.model.layers[i]
        view = sparse.layers[str(i)]
        assert view.mlp is original.mlp
        assert view.self_attn.output is original.self_attn.output
        assert view.self_attn.attn is not original.self_attn.attn
        assert view.self_attn.attn.impl is not original.self_attn.attn.impl
        assert view.self_attn.attn.kv_cache.data_ptr() != original.self_attn.attn.kv_cache.data_ptr()
        assert original.self_attn.attn.impl._layer_name == original.self_attn.attn.layer_name


@torch.inference_mode()
def test_populated_target_cache_is_untouched_by_sparse_forward():
    target = ToyTarget()
    target.reference(torch.tensor([2, 4, 7]), torch.arange(3))
    original_caches = [layer.self_attn.attn.kv_cache.clone() for layer in target.model.layers]
    _, sparse, _ = make_sparse(target=target)
    assert all(attention.impl.key_cache is None for attention in sparse.attention_layers.values())
    sparse.forward_logits(torch.tensor([8, 10, 11]), torch.arange(3))
    for layer, expected in zip(target.model.layers, original_caches):
        torch.testing.assert_close(layer.self_attn.attn.kv_cache, expected)


@pytest.mark.parametrize(
    "selected,aux",
    [((0, 0), (1,)), ((1, 0), (1,)), ((0, 4), (1,)), ((False, 2), (1,)), ((0, 2), (4,)), ((0,), (-1,)), ((0,), (1, 1))],
)
def test_invalid_layer_views_fail_before_registration(selected, aux):
    registry = {}
    with pytest.raises(ValueError):
        SparseTargetVerifier(ToyTarget(), selected, aux, registry)
    assert registry == {}


def test_failed_registration_removes_partial_sparse_entries():
    target = ToyTarget()
    existing_name = "dflash_sparse_verify.model.layers.2.self_attn.attn"
    existing = object()
    registry = {existing_name: existing}
    with pytest.raises(ValueError, match="already registered"):
        SparseTargetVerifier(target, (0, 2), (1, 3), registry)
    assert registry == {existing_name: existing}


@torch.inference_mode()
def test_bonus_only_logits_and_embedding_inputs():
    target, sparse, _ = make_sparse()
    tokens, positions = torch.tensor([2, 3, 5]), torch.arange(3)
    logits, _ = sparse.forward_logits(tokens, positions, logits_indices=torch.tensor([2]))
    hidden_states, _ = sparse(None, positions, inputs_embeds=target.model.embed_tokens(tokens))
    torch.testing.assert_close(logits, sparse.compute_logits(hidden_states[-1:]))
    with pytest.raises(ValueError, match="one position"):
        sparse(tokens, torch.arange(4))
    with pytest.raises(ValueError, match="pipeline"):
        sparse(tokens, positions, intermediate_tensors={})
