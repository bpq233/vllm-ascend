# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from test_hf_qwen3 import Qwen3DecoderLayer, ResidualNorm, transformers
from torch import nn

from vllm_ascend.spec_decode.slim_verifier import SlimVerifier
from vllm_ascend.spec_decode.slim_verifier_calibration import _run_block, audit_target, prepare_blocks
from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig


def models(dtype=torch.float32):
    torch.manual_seed(20261117)
    config = transformers.Qwen3Config(
        num_hidden_layers=4,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        max_position_embeddings=128,
        sliding_window=None,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    hf = transformers.Qwen3ForCausalLM(config).to(dtype=dtype).eval()
    target = SimpleNamespace(
        config=config,
        compute_logits=hf.lm_head,
        model=SimpleNamespace(
            config=config,
            start_layer=0,
            end_layer=4,
            embed_tokens=hf.model.embed_tokens,
            norm=ResidualNorm(hf.model.norm),
            layers=nn.ModuleList(Qwen3DecoderLayer(layer, hf.model.rotary_emb, config) for layer in hf.model.layers),
        ),
    )
    return hf, target


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "attention,mlp", [((0,), ()), ((), (0,)), ((3,), ()), ((), (3,)), ((0, 2), (1, 3)), ((0, 1, 2, 3), (0, 1, 2, 3))]
)
@torch.inference_mode()
def test_sublayer_bypass_matches_independent_hf_residual_branches(dtype, attention, mlp):
    hf, target = models(dtype)
    prefix, query = torch.tensor([[1, 4, 2, 7]]), torch.tensor([9, 3, 11, 5])
    cache = hf(prefix, use_cache=True).past_key_values
    handles = []
    for index in attention:
        handles.append(
            hf.model.layers[index].self_attn.register_forward_hook(
                lambda module, args, output: (torch.zeros_like(output[0]), output[1])
            )
        )
    for index in mlp:
        handles.append(
            hf.model.layers[index].mlp.register_forward_hook(lambda module, args, output: torch.zeros_like(output))
        )
    try:
        expected = hf(query[None, :], past_key_values=deepcopy(cache), use_cache=True).logits[0]
    finally:
        for handle in handles:
            handle.remove()
    config = SlimVerifierConfig(tuple(range(4)), skip_attention_layers=attention, skip_mlp_layers=mlp)
    requested = []

    def read(index):
        requested.append(index)
        return tuple(t[0].transpose(0, 1) for t in cache[index])

    actual = SlimVerifier(target, config).logits(query, torch.arange(4), position_offset=4, prefix_kv_provider=read)
    torch.testing.assert_close(
        actual, expected, rtol=1e-5 if dtype == torch.float32 else 0.05, atol=1e-6 if dtype == torch.float32 else 0.03
    )
    assert requested == [i for i in range(4) if i not in attention]


def test_bypassing_both_branches_equals_whole_layer_deletion():
    _, target = models()
    ids = torch.tensor([1, 3, 2, 4])
    a = SlimVerifier(target, SlimVerifierConfig((0, 1, 2, 3), skip_attention_layers=(0, 2), skip_mlp_layers=(0, 2)))
    b = SlimVerifier(target, SlimVerifierConfig((1, 3)))
    torch.testing.assert_close(a.hidden_states(ids), b.hidden_states(ids))


def test_target_kv_audit_search_validation_and_refresh_are_separate():
    _, target = models()
    report = audit_target(
        target,
        [[1, 4, 2, 7, 9, 3, 11], [7, 6, 3, 2, 4, 9, 8, 1]],
        draft_length=3,
        mask={"skip_layers": [1]},
        search_skip_count=2,
        beam_width=2,
        max_evaluations=12,
        validation_token_batches=[[2, 9, 8, 3, 6, 4]],
        compare_kv_refresh=True,
        rtol=1e-4,
        atol=1e-5,
    )
    assert report["kv_source"] == "target"
    assert report["reference_mismatch_count"] == 0
    assert len(report["single_ablation_effects_least_kl_first"]) == 12
    assert report["full_layer_control"]["metrics"]["mean_kl_target_to_slim"] < 1e-6
    best = report["selected_masks"]["search_best"]
    assert best["candidate_additional_config"]["slim_verifier_eval"]["keep_layers"] == [0, 3]
    assert best["held_out_evaluated"]
    assert best["numerical_reference_checks_passed"]
    assert best["refresh_comparison_dataset"] == "validation"
    assert len(best["validation"]["metrics"]["per_position"]) == 3
    assert best["target_kv_refreshed_each_position"]["kv_mode"] == "target_per_token"
    json.dumps(report, allow_nan=False)


def test_replay_control_and_budget_exhaustion_do_not_export_partial_mask():
    _, target = models()
    report = audit_target(
        target,
        [[1, 4, 2, 7, 9]],
        draft_length=2,
        kv_source="replay",
        ablations=False,
        search_skip_count=2,
        max_evaluations=1,
    )
    assert not report["search"]["completed"]
    assert report["selected_masks"] == {}
    assert report["full_layer_control"]["metrics"]["mean_kl_target_to_slim"] < 1e-6


def test_kv_refresh_preserves_first_position_but_changes_later_conditioning():
    _, target = models()
    context = lambda n: nullcontext()
    block = prepare_blocks(target, [[1, 4, 2, 7, 9, 3, 11]], context, 3)[0]
    verifier = SlimVerifier(target, SlimVerifierConfig((0, 2, 3)))
    normal, _ = _run_block(target, verifier, block, context, "target", False, True)
    refreshed, _ = _run_block(target, verifier, block, context, "target_per_token", False, True)
    torch.testing.assert_close(normal[0], refreshed[0], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(normal[1:], refreshed[1:], rtol=1e-5, atol=1e-6)


def test_nonprimary_rank_uses_broadcast_search_scores_and_same_collectives():
    _, target = models()
    messages, primary_calls, follower_calls = [], [], []
    original = target.compute_logits

    def head(hidden):
        primary_calls.append(hidden.shape)
        return original(hidden)

    def publish(value):
        messages.append(deepcopy(value))
        return value

    target.compute_logits = head
    options = dict(ablations=False, search_skip_count=2, max_evaluations=8, draft_length=2)
    audit_target(target, [[1, 3, 4, 5, 7]], sync=publish, **options)

    def nonprimary_head(hidden):
        follower_calls.append(hidden.shape)
        original(hidden)
        return None

    target.compute_logits = nonprimary_head
    result = audit_target(target, [[1, 3, 4, 5, 7]], primary=False, sync=lambda value: messages.pop(0), **options)
    assert result is None and not messages
    assert primary_calls == follower_calls


def test_audit_rejects_local_vocab_shard():
    _, target = models()
    original = target.compute_logits
    target.compute_logits = lambda hidden: original(hidden)[:, :16]
    with pytest.raises(RuntimeError, match="global-vocabulary"):
        audit_target(target, [[1, 3, 4, 5, 7]], ablations=False)


def test_failed_sublayer_reference_blocks_search(monkeypatch):
    import vllm_ascend.spec_decode.slim_verifier_calibration as calibration

    _, target = models()
    reference = calibration.reference_hidden_states

    def bad_reduced_reference(model, ids, config, **kwargs):
        hidden = reference(model, ids, config, **kwargs)
        return hidden * 10 if config.skip_attention_layers else hidden

    monkeypatch.setattr(calibration, "reference_hidden_states", bad_reduced_reference)
    with pytest.raises(RuntimeError, match="Candidate failed"):
        audit_target(target, [[1, 3, 4, 5, 7]], search_scope="attention", search_skip_count=1)


def test_failed_refresh_reference_is_not_exported_as_a_valid_candidate(monkeypatch):
    import vllm_ascend.spec_decode.slim_verifier_calibration as calibration

    _, target = models()
    reference = calibration.reference_hidden_states

    def bad_single_token_reference(model, ids, config, **kwargs):
        hidden = reference(model, ids, config, **kwargs)
        return hidden * 10 if config.skip_mlp_layers and ids.numel() == 1 else hidden

    monkeypatch.setattr(calibration, "reference_hidden_states", bad_single_token_reference)
    report = audit_target(
        target,
        [[1, 3, 4, 5, 7]],
        ablations=False,
        draft_length=2,
        mask={"skip_mlp_layers": [1]},
        compare_kv_refresh=True,
    )
    selected = report["selected_masks"]["configured"]
    assert selected["calibration"]["reference_checks_passed"]
    assert not selected["target_kv_refreshed_each_position"]["reference_checks_passed"]
    assert selected["candidate_additional_config"] is None
    assert not selected["numerical_reference_checks_passed"]
    assert report["reference_mismatch_count"] == 1
