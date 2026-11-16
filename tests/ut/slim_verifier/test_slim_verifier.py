# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU tests: pytest --confcutdir=tests/ut/slim_verifier tests/ut/slim_verifier -q"""

import ast
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vllm_ascend.spec_decode.slim_verifier import SlimVerifier, _module_view
from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig, slim_verifier_enabled
from vllm_ascend.spec_decode.slim_verifier_eval import SlimVerifierEval
from vllm_ascend.spec_decode.slim_verifier_metrics import SlimVerifierMetrics


class RMSNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size))

    def forward(self, x, residual=None):
        total = x if residual is None else x + residual
        normed = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6) * self.weight
        return normed if residual is None else (normed, total)


class ReferenceAttention(nn.Module):
    """Independent dense matmul reference, with a sentinel for target KV writes."""

    def __init__(self):
        super().__init__()
        self.register_buffer("kv_sentinel", torch.tensor(0))

    def forward(self, q, k, v):
        self.kv_sentinel += 1
        length = q.shape[0]
        q = q.reshape(length, 2, 4).transpose(0, 1)
        k = k.reshape(length, 1, 4).transpose(0, 1).expand(2, -1, -1)
        v = v.reshape(length, 1, 4).transpose(0, 1).expand(2, -1, -1)
        scores = q @ k.transpose(-1, -2) * 0.5
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        return (scores.masked_fill(mask, -torch.inf).softmax(-1) @ v).transpose(0, 1).reshape(length, 8)


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim, self.scaling = 2, 1, 4, 0.5
        self.qkv_proj = nn.Linear(8, 16, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.q_norm, self.k_norm = RMSNorm(4), RMSNorm(4)
        self.attn = ReferenceAttention()

    def forward(self, positions, hidden_states):
        q, k, v = self.qkv_proj(hidden_states).split((8, 4, 4), -1)
        q, k = self.q_norm(q.reshape(-1, 2, 4)).reshape(-1, 8), self.k_norm(k.reshape(-1, 1, 4)).reshape(-1, 4)

        # A position-dependent rotation tests absolute-position preservation.
        def rotate(x):
            x = x.reshape(x.shape[0], -1, 4)
            angle = positions.float()[:, None, None] / 10
            half = torch.cat((-x[..., 2:], x[..., :2]), -1)
            return (x * angle.cos() + half * angle.sin()).reshape(x.shape[0], -1)

        return self.o_proj(self.attn(rotate(q), rotate(k), v))


class Qwen3DecoderLayer(nn.Module):
    """vLLM's fused-residual Qwen decoder contract, using CPU operations."""

    def __init__(self, index, calls):
        super().__init__()
        self.index, self.calls = index, calls
        self.self_attn = SelfAttention()
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(8), RMSNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))

    def forward(self, positions, hidden_states, residual):
        self.calls.append(self.index)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.config = SimpleNamespace(model_type="qwen3", vocab_size=11)
        self.model = nn.Module()
        self.model.config = self.config
        self.model.start_layer, self.model.end_layer = 0, 4
        self.model.embed_tokens = nn.Embedding(11, 8)
        self.model.layers = nn.ModuleList(Qwen3DecoderLayer(i, self.calls) for i in range(4))
        self.model.norm = RMSNorm(8)
        self.lm_head = nn.Linear(8, 11, bias=False)

    def forward(self, ids, layers=None):
        h, r = self.model.embed_tokens(ids), None
        p = torch.arange(ids.shape[0])
        for i in range(4) if layers is None else layers:
            h, r = self.model.layers[i](p, h, r)
        return self.model.norm(h) if r is None else self.model.norm(h, r)[0]

    def compute_logits(self, hidden):
        return self.lm_head(hidden)


@pytest.fixture
def target():
    torch.manual_seed(17)
    return Target().eval()


def test_config_disabled():
    assert SlimVerifierConfig.from_additional_config(None, 0) is None
    assert SlimVerifierConfig.from_additional_config({"slim_verifier_eval": {"enabled": False}}, 0) is None
    assert not slim_verifier_enabled({})


@pytest.mark.parametrize(
    "raw",
    [
        {"keep_layers": [-1]},
        {"keep_layers": [4]},
        {"keep_layers": [1, 1]},
        {"keep_layers": [True]},
        {"keep_layers": "1"},
        {"keep_layers": [0], "skip_layers": []},
        {},
        {"keep_layers": [1], "log_interval": 0},
        {"keep_layers": [1], "top_k": True},
        {"keep_layers": [1], "profile_latency": "true"},
        {"keep_layers": [1], "typo": 3},
    ],
)
def test_config_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        SlimVerifierConfig.from_additional_config({"slim_verifier_eval": {"enabled": True, **raw}}, 4)


def test_selection_sorted_and_skip_complement():
    def parse(raw):
        return SlimVerifierConfig.from_additional_config({"slim_verifier_eval": {"enabled": True, **raw}}, 4)

    assert parse({"keep_layers": [3, 0, 2]}).keep_layers == (0, 2, 3)
    assert parse({"skip_layers": [3, 1]}).keep_layers == (0, 2)
    assert parse({"keep_layers": []}).keep_layers == ()


def test_full_layer_logits_match_and_target_kv_untouched(target):
    ids = torch.tensor([1, 4, 2, 7, 0, 3])
    expected = target.compute_logits(target(ids))
    kv_before = [layer.self_attn.attn.kv_sentinel.clone() for layer in target.model.layers]
    pointers = {name: p.data_ptr() for name, p in target.named_parameters()}
    original_cores = [layer.self_attn.attn for layer in target.model.layers]
    slim = SlimVerifier(target, SlimVerifierConfig((0, 1, 2, 3)))
    actual = slim.logits(ids, torch.arange(len(ids)))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert pointers == {name: p.data_ptr() for name, p in target.named_parameters()}
    assert slim.embedding is target.model.embed_tokens
    assert slim.norm is target.model.norm
    assert slim.target.lm_head is target.lm_head
    for index, view in enumerate(slim.layers):
        original = target.model.layers[index]
        assert view.self_attn.qkv_proj is original.self_attn.qkv_proj
        assert view.mlp is original.mlp
        assert original.self_attn.attn is original_cores[index]
        torch.testing.assert_close(original.self_attn.attn.kv_sentinel, kv_before[index])


@pytest.mark.parametrize("kept", [(0, 3), (1,), ()])
def test_only_selected_layers_execute(target, kept):
    ids = torch.tensor([0, 4, 2, 7])
    expected = target.compute_logits(target(ids, kept))
    target.calls.clear()
    actual = SlimVerifier(target, SlimVerifierConfig(kept)).logits(ids, torch.tensor([0, 2]))
    assert target.calls == list(kept)
    assert actual.shape == (2, 11)
    torch.testing.assert_close(actual, expected[[0, 2]], rtol=1e-5, atol=1e-6)


def test_module_view_rebinds_bound_dispatch():
    class Dispatch(nn.Module):
        def __init__(self):
            super().__init__()
            self._forward_method = self.dispatch

        def dispatch(self):
            return self

        def forward(self):
            return self._forward_method()

    original = Dispatch()
    view = _module_view(original)
    assert view() is view
    assert original() is original


def test_view_hook_registration_does_not_mutate_target(target):
    slim = SlimVerifier(target, SlimVerifierConfig((0,)))
    calls = []
    handle = slim.layers[0].register_forward_pre_hook(lambda module, args: calls.append(0))
    target(torch.tensor([1, 2]))
    assert calls == []
    slim.hidden_states(torch.tensor([1, 2]))
    assert calls == [0]
    handle.remove()


def test_direct_config_always_executes_in_original_layer_order():
    assert SlimVerifierConfig((3, 0, 2)).keep_layers == (0, 2, 3)
    with pytest.raises(ValueError):
        SlimVerifierConfig((1, 1))


def _logits(predictions, vocab=8):
    result = torch.zeros(len(predictions), vocab)
    result[torch.arange(len(predictions)), predictions] = 3
    return result


def test_metrics_formulas_and_probability():
    m = SlimVerifierMetrics(top_k=2)
    d = torch.tensor([1, 2, 3, 4])
    t, s = _logits([1, 0, 3, 4]), _logits([1, 2, 0, 4])
    masks = m.record(d, t, s)
    assert masks.target_accept_mask.tolist() == [True, False, True, True]
    assert masks.slim_accept_mask.tolist() == [True, True, False, True]
    m.record(torch.tensor([5]), _logits([5]), _logits([0]))
    report = m.snapshot()
    assert report["blocks"] == 2 and report["draft_tokens"] == 5
    assert report["target_avg_accepted_length"] == report["slim_avg_accepted_length"] == 1
    assert report["target_prefix_acceptance_rate"] == report["slim_prefix_acceptance_rate"] == 0.4
    assert report["target_token_match_rate"] == 0.8
    assert report["slim_token_match_rate"] == 0.6
    assert report["target_full_block_acceptance_rate"] == 0.5
    assert report["slim_full_block_acceptance_rate"] == 0
    assert report["prediction_agreement_rate"] == report["decision_agreement_rate"] == 0.4
    assert report["slim_only_rate"] == 0.2
    assert report["target_only_rate"] == 0.4
    expected_lp = torch.cat((t.log_softmax(-1).gather(1, d[:, None]).flatten(), _logits([5]).log_softmax(-1)[:, 5]))
    assert report["target_avg_draft_logprob"] == pytest.approx(expected_lp.mean().item())
    assert report["mean_logprob_gap"] == pytest.approx(0.6, abs=1e-6)


def test_metrics_windows_empty_and_ties(caplog):
    m = SlimVerifierMetrics(log_interval=2, top_k=50)
    assert m.snapshot() == {"blocks": 0, "draft_tokens": 0}
    with pytest.raises(ValueError):
        m.record(torch.tensor([], dtype=torch.long), torch.empty(0, 3), torch.empty(0, 3))
    with caplog.at_level(logging.INFO):
        for _ in range(2):
            m.record(torch.tensor([0, 1]), torch.zeros(2, 3), torch.zeros(2, 3))
    assert m.window_blocks == 0 and m.blocks == 2
    assert "Top-1 Diagnostic" in caplog.text and "Cumulative" in caplog.text
    assert m.snapshot()["target_avg_accepted_length"] == 1
    assert m.snapshot()["target_avg_draft_rank"] == 1  # competition rank for ties
    assert m.snapshot()["top_k_overlap"] == 1
    m.record(torch.tensor([2]), _logits([0]), _logits([0]))
    assert m.snapshot(False)["blocks"] == 1 and m.snapshot()["blocks"] == 3
    m.flush()
    assert m.window_blocks == 0 and m.snapshot()["blocks"] == 3


def make_hook(target, kept=(0, 1, 2, 3)):
    hook = SlimVerifierEval.__new__(SlimVerifierEval)
    hook.config = SlimVerifierConfig(kept)
    hook.verifier = SlimVerifier(target, hook.config)
    hook.vocab_size = 11
    hook.metrics = SlimVerifierMetrics()
    hook.context = lambda n: nullcontext()
    hook.last_results = []
    return hook


def make_batch(target):
    # Request order is different from state-slot order. Ignore stale/rejected
    # tokens after each committed prefix and ignore graph padding at the tail.
    histories = [torch.tensor([5, 7, 3, 1, 2]), torch.tensor([4, 6, 8, 9, 0, 3])]
    hidden = torch.cat((target(histories[0])[2:], target(histories[1])[4:], torch.zeros(2, 8)))
    store = torch.tensor([[4, 6, 8, 9, 10, 10], [5, 7, 10, 10, 10, 10]])
    runner = SimpleNamespace(req_states=SimpleNamespace(all_token_ids=SimpleNamespace(gpu=store)))
    batch = SimpleNamespace(
        req_ids=["request-a", "request-b"],
        num_draft_tokens=3,
        num_reqs=2,
        query_start_loc_np=np.array([0, 3, 5]),
        positions=torch.tensor([2, 3, 4, 4, 5, 0, 0]),
        input_ids=torch.tensor([3, 1, 2, 0, 3, -1, -1]),
        idx_mapping_np=np.array([1, 0]),
        logits_indices=torch.tensor([0, 1, 2, 3, 4]),
        cu_num_logits_np=np.array([0, 3, 5]),
        num_computed_tokens_np=np.array([999, 999]),
    )
    return runner, hidden, batch, histories


def test_hook_history_alignment_bonus_padding_and_request_slots(target):
    hook = make_hook(target)
    runner, hidden, batch, histories = make_batch(target)
    original_logits = hook.verifier.logits
    observed = []

    def capture(ids, positions):
        observed.append((ids.clone(), positions.clone()))
        return original_logits(ids, positions)

    hook.verifier.logits = capture
    hook.evaluate(runner, hidden, batch)
    assert len(observed) == 2
    for index, history in enumerate(histories):
        torch.testing.assert_close(observed[index][0], history)
    assert observed[0][1].tolist() == [2, 3]
    assert observed[1][1].tolist() == [4]
    assert hook.last_results[0].draft_tokens.tolist() == [1, 2]
    assert hook.last_results[1].draft_tokens.tolist() == [3]
    assert hook.last_results[1].request_id == "request-b"
    assert hook.last_results[1].prediction_positions.tolist() == [4]
    assert hook.metrics.snapshot()["prediction_agreement_rate"] == 1
    assert hook.metrics.snapshot()["draft_tokens"] == 3  # NO bonus/padding
    batch.num_draft_tokens = 0
    hook.evaluate(runner, hidden, batch)
    assert hook.last_results == [] and hook.metrics.blocks == 2


def test_token_zero_is_a_valid_draft_and_request_slot_reuse_has_no_slim_state(target):
    hook = make_hook(target)
    runner, hidden, batch, _ = make_batch(target)
    batch.input_ids[4] = 0
    hook.evaluate(runner, hidden, batch)
    assert hook.last_results[1].draft_tokens.tolist() == [0]
    assert hook.metrics.tokens == 3
    # Reuse a state slot for a new request with a different committed prefix.
    runner.req_states.all_token_ids.gpu[1, :2] = torch.tensor([6, 6])
    batch.req_ids[0] = "new-request"
    hidden[:3] = target(torch.tensor([6, 6, 3, 1, 2]))[2:]
    hook.evaluate(runner, hidden, batch)
    assert hook.last_results[0].request_id == "new-request"
    assert hook.metrics.snapshot()["prediction_agreement_rate"] == 1


def test_worker_initialization_uses_actual_layers_and_rejects_unsupported_modes(target, monkeypatch, caplog):
    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, result)

    if "vllm" not in sys.modules:
        module("vllm", __path__=[])
    module("vllm.distributed", get_tensor_model_parallel_rank=lambda: 0)
    module("vllm.logger", logger=logging.getLogger("slim-test"))
    module("vllm_ascend.ascend_forward_context", set_ascend_forward_context=lambda *args, **kwargs: nullcontext())
    module("vllm_ascend.utils", enable_sp=lambda vc: False, lmhead_tp_enable=lambda: False)
    vc = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        speculative_config=SimpleNamespace(enable_adaptive_verification=False),
        lora_config=None,
        quant_config=None,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(enable_prompt_embeds=False, enable_sleep_mode=False),
        additional_config={
            "slim_verifier_eval": {
                "enabled": True,
                "keep_layers": [3, 0],
                "acceptance_strategy": "probability_ratio",
                "probability_ratio_threshold": 0.7,
            }
        },
    )
    runner = SimpleNamespace(vllm_config=vc, get_model=lambda: target)
    with caplog.at_level(logging.INFO):
        hook = SlimVerifierEval(runner)
    assert hook.config.keep_layers == (0, 3)
    assert hook.metrics.acceptance_strategy == "probability_ratio"
    assert hook.metrics.probability_ratio_threshold == 0.7
    assert "q(x)/q(max) > 0.7" in caplog.text
    assert "Target layers: 4" in caplog.text
    vc.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(NotImplementedError, match="pipeline_parallel_size"):
        SlimVerifierEval(runner)
    vc.parallel_config.pipeline_parallel_size = 1
    vc.speculative_config.enable_adaptive_verification = True
    with pytest.raises(NotImplementedError, match="adaptive verification"):
        SlimVerifierEval(runner)


def test_shadow_preserves_rng_state_and_input_tensors(target):
    hook = make_hook(target, (0, 2))
    runner, hidden, batch, _ = make_batch(target)
    before = (hidden.clone(), batch.input_ids.clone(), runner.req_states.all_token_ids.gpu.clone())
    original_logits = hook.verifier.logits

    def consume_rng(ids, positions):
        torch.rand(9)
        return original_logits(ids, positions)

    hook.verifier.logits = consume_rng
    state = torch.get_rng_state().clone()
    hook.evaluate(runner, hidden, batch)
    assert torch.equal(torch.get_rng_state(), state)
    for actual, expected in zip((hidden, batch.input_ids, runner.req_states.all_token_ids.gpu), before):
        torch.testing.assert_close(actual, expected)


def test_global_vocab_required_and_nonprimary_participates(target):
    hook = make_hook(target)
    runner, hidden, batch, _ = make_batch(target)
    native = target.compute_logits
    target.compute_logits = lambda h: native(h)[:, :5]
    with pytest.raises(RuntimeError, match="global-vocabulary"):
        hook.evaluate(runner, hidden, batch)
    # Nonprimary TP ranks may receive None, but must join both heads per block.
    calls = []
    target.compute_logits = lambda h: calls.append(h.shape[0])
    hook.metrics = None
    hook.evaluate(runner, hidden, batch)
    assert calls == [2, 2, 1, 1]


def test_runner_hook_keeps_parent_sampling_result():
    # Execute the actual changed method without importing the NPU-only runner.
    path = Path(__file__).parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    sample = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample")
    cls.body = [sample]
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    events = []
    sentinel = object()

    class Parent:
        def sample(self, *args):
            events.append("sample")
            return sentinel

    namespace = {"Parent": Parent, "lmhead_tp_enable": lambda: False}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    runner = namespace["NPUModelRunner"]()
    assert runner.sample(None, None, None) is sentinel
    runner.slim_verifier_eval = SimpleNamespace(evaluate=lambda *args: events.append("shadow"))
    assert runner.sample(None, None, None) is sentinel
    assert events == ["sample", "sample", "shadow"]


def test_deterministic_generated_sequence_unchanged_with_shadow(target):
    def generate(enabled):
        torch.manual_seed(121)
        ids = torch.tensor([1, 2, 3])
        hook = make_hook(target, (0, 2))
        result = []
        for _ in range(5):
            hidden = target(ids)
            token = target.compute_logits(hidden[-1:]).argmax(-1).item()
            if enabled:
                # Extra replay must not influence the chosen token or next RNG state.
                hook.verifier.logits(ids, torch.tensor([len(ids) - 1]))
            result.append(token)
            ids = torch.cat((ids, torch.tensor([token])))
        return result

    assert generate(False) == generate(True)
