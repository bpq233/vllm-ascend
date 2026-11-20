# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import logging
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from test_slim_verifier import Target, make_batch, make_hook

from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig
from vllm_ascend.spec_decode.slim_verifier_draft import DraftLogitsCapture
from vllm_ascend.spec_decode.slim_verifier_eval import SlimVerifierEval
from vllm_ascend.spec_decode.slim_verifier_kl import KL_NAMES, SlimVerifierKLMetrics


class Proposer:
    """Real tensor sampling with the MRV2 observer contract, no NPU imports."""

    max_num_reqs = 3
    num_speculative_steps = 3
    device = torch.device("cpu")
    use_local_argmax_reduction = False

    def compute_draft_logits(self, hidden_states, spec_step_idx=0):
        return hidden_states.clone()

    def sample_draft(self, hidden_states, positions, idx_mapping, temperature, seeds, draft_step, draft_logits):
        logits = self.compute_draft_logits(hidden_states)
        # Mimic a sampler mutating the supplied logits. Captured logits must
        # remain raw, and the observer must leave this exact sampling unchanged.
        logits.div_(temperature)
        return logits.argmax(-1) if draft_logits is None else torch.multinomial(logits.softmax(-1), 1).squeeze(1)

    def propose(self, input_batch, logits, positions, columns, probabilistic=False):
        return self.sample_draft(
            logits, positions, input_batch.idx_mapping, 0.7, None, columns, object() if probabilistic else None
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_direction_and_token_weighted_kl_against_torch(dtype, caplog):
    torch.manual_seed(42)
    logits = [torch.randn(5, 19).to(dtype) for _ in range(3)]  # D, S, T
    metric = SlimVerifierKLMetrics(2, logging.getLogger("kl-test"))
    with caplog.at_level(logging.INFO):
        metric.record(*(x[:1] for x in logits))
        metric.record(*(x[1:] for x in logits))
    result = metric.snapshot()
    for key, (p, q) in zip(KL_NAMES, ((0, 1), (0, 2), (1, 2))):
        expected = F.kl_div(logits[q].float().log_softmax(-1), logits[p].float().softmax(-1), reduction="batchmean")
        assert result[key] == pytest.approx(expected.item(), abs=2e-6)
    reverse = F.kl_div(logits[0].float().log_softmax(-1), logits[2].float().softmax(-1), reduction="batchmean")
    assert abs(result["kl_draft_to_target"] - reverse.item()) > 1e-3
    assert result["blocks"] == 2 and result["draft_tokens"] == 5
    assert "accepted_length" not in caplog.text and "KL Window" in caplog.text
    assert metric.snapshot(False) == {"blocks": 0, "draft_tokens": 0}
    metric.record(logits[0][:1], logits[0][:1], logits[0][:1])
    assert all(metric.snapshot(False)[name] == 0 for name in KL_NAMES)
    assert metric.snapshot()["draft_tokens"] == 6


def test_zero_mass_infinite_divergence_and_invalid_distribution():
    same = torch.tensor([[0.0, -torch.inf, 1.0]])
    metric = SlimVerifierKLMetrics()
    metric.record(same, same, same)
    assert all(metric.snapshot()[name] == 0 for name in KL_NAMES)
    metric = SlimVerifierKLMetrics()
    metric.record(torch.tensor([[0.0, -1000.0]]), torch.tensor([[0.0, -torch.inf]]), torch.zeros(1, 2))
    assert metric.snapshot()["kl_draft_to_slim"] == float("inf")
    metric = SlimVerifierKLMetrics()
    metric.record(torch.full((1, 2), -torch.inf), torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(RuntimeError, match="undefined KL"):
        metric.snapshot()


@pytest.mark.parametrize("probabilistic", [False, True])
def test_capture_preserves_tokens_rng_and_raw_logits(probabilistic):
    original, observed = Proposer(), Proposer()
    capture = DraftLogitsCapture(observed, 11)
    torch.manual_seed(91)
    logits = torch.randn(3, 11)
    batch = SimpleNamespace(idx_mapping=torch.tensor([2, 0, -1]))
    positions, columns = torch.tensor([7, 13, -1]), torch.zeros(3, dtype=torch.long)
    torch.manual_seed(17)
    expected = original.propose(batch, logits, positions, columns, probabilistic)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(17)
    actual = observed.propose(batch, logits, positions, columns, probabilistic)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for row, slot in ((0, 2), (1, 0)):
        raw, aligned = capture.read(slot, positions[row : row + 1], actual[row : row + 1])
        torch.testing.assert_close(raw, logits[row : row + 1], rtol=0, atol=0)
        assert aligned.all()
    assert capture.positions[1].eq(-1).all()  # Padding never writes a real slot.


def test_flattened_parallel_drafts_padding_and_slot_reuse():
    proposer = Proposer()
    capture = DraftLogitsCapture(proposer, 5)
    rows = torch.tensor([2, 2, 2, 0, 0, 0, -1, -1, -1])
    steps = torch.tensor([0, 1, 2] * 3)
    positions = torch.tensor([10, 11, 12, 20, 21, 22, -1, -1, -1])
    logits = torch.randn(9, 5)
    tokens = proposer.propose(SimpleNamespace(idx_mapping=rows), logits, positions, steps)
    raw, aligned = capture.read(0, positions[3:5], tokens[3:5])
    assert aligned.all()
    torch.testing.assert_close(raw, logits[3:5])
    # Reproposal invalidates the old tail, but not pending proposals in slot 2.
    new_logits = torch.randn(1, 5)
    new_tokens = proposer.propose(
        SimpleNamespace(idx_mapping=torch.tensor([0])), new_logits, torch.tensor([0]), torch.tensor([0])
    )
    assert capture.read(0, torch.tensor([0]), new_tokens)[1].all()
    assert not capture.read(0, positions[3:5], tokens[3:5])[1].any()
    assert capture.read(2, positions[:3], tokens[:3])[1].all()


def test_bad_alignment_cannot_publish_kl():
    metrics = SlimVerifierKLMetrics()
    logits = torch.zeros(2, 7)
    metrics.record(logits, logits, logits, torch.tensor([True, False]))
    with pytest.raises(RuntimeError, match="alignment failed"):
        metrics.flush()


def test_missing_or_sharded_draft_distributions_are_rejected():
    with pytest.raises(NotImplementedError, match="token-only"):
        DraftLogitsCapture(None, 11)
    proposer = Proposer()
    proposer.use_local_argmax_reduction = True
    with pytest.raises(NotImplementedError, match="use_local_argmax_reduction=False"):
        DraftLogitsCapture(proposer, 11)
    proposer.use_local_argmax_reduction = False
    capture = DraftLogitsCapture(proposer, 11)
    with pytest.raises(RuntimeError, match="global-vocabulary"):
        proposer.propose(
            SimpleNamespace(idx_mapping=torch.tensor([0])), torch.zeros(1, 5), torch.tensor([0]), torch.tensor([0])
        )
    assert capture._active is None
    proposer.speculative_config = SimpleNamespace(use_heterogeneous_vocab=True)
    with pytest.raises(NotImplementedError, match="same vocabulary"):
        DraftLogitsCapture.validate(proposer)


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
@pytest.mark.filterwarnings("ignore:.*torch.jit.trace.*:DeprecationWarning")
def test_traced_capture_uses_live_request_rows_positions_and_columns():
    # CPU tracing checks that copies and indexing live in the tensor graph,
    # not only in Python callbacks. Actual ACL graph replay still needs NPU ST.
    proposer = Proposer()
    capture = DraftLogitsCapture(proposer, 5)

    class TraceHarness(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Explicit state roots keep JIT from treating Python closure
            # buffers as dead constants (ACL graphs record kernel writes).
            self.register_buffer("logits", capture.logits)
            self.register_buffer("positions", capture.positions)
            self.register_buffer("tokens", capture.token_ids)

        def forward(self, logits, positions, rows, columns):
            return proposer.sample_draft(logits, positions, rows, 0.7, None, columns, None)

    traced = torch.jit.trace(
        TraceHarness(),
        (torch.randn(2, 5), torch.tensor([1, 4]), torch.tensor([0, 1]), torch.tensor([0, 0])),
        check_trace=False,
    )
    capture._head_calls = -1
    logits = torch.randn(2, 5)
    sampled = traced(logits, torch.tensor([8, -1]), torch.tensor([2, -1]), torch.tensor([1, 2]))
    assert capture._head_calls == -1  # Replay never invokes the Python observer.
    torch.testing.assert_close(capture.logits[2, 1], logits[0])
    assert capture.positions[2, 1] == 8 and capture.token_ids[2, 1] == sampled[0]
    assert capture.positions[1, 0] == 4  # Padding did not overwrite a live slot.


def test_kl_hook_aligns_by_state_slot_and_excludes_bonus_padding():
    torch.manual_seed(5)
    target = Target()
    hook = make_hook(target)
    hook.config = replace(hook.config, metrics_mode="kl")
    hook.metrics = SlimVerifierKLMetrics()
    proposer = Proposer()
    hook.draft_capture = DraftLogitsCapture(proposer, 11)
    runner, hidden, batch, _ = make_batch(target)
    raw = torch.randn(3, 11)
    tokens = batch.input_ids[torch.tensor([1, 2, 4])]
    raw[torch.arange(3), tokens] = 100  # Same proposed IDs, deliberately non-Target probabilities.
    proposer.propose(
        SimpleNamespace(idx_mapping=torch.tensor([1, 1, 0])), raw, torch.tensor([2, 3, 4]), torch.tensor([0, 1, 0])
    )
    hook.evaluate(runner, hidden, batch)
    result = hook.metrics.snapshot()
    assert result["blocks"] == 2 and result["draft_tokens"] == 3
    assert result["kl_slim_to_target"] < 1e-6
    assert result["kl_draft_to_slim"] == pytest.approx(result["kl_draft_to_target"], abs=1e-6)
    assert result["kl_draft_to_target"] > 0.1
    assert not hook.last_results
    batch.num_draft_tokens = 0
    hook.evaluate(runner, hidden, batch)
    assert hook.metrics.snapshot()["draft_tokens"] == 3
    # Other TP ranks still join Target and Slim heads, without a capture buffer.
    calls = []
    hook.metrics = hook.draft_capture = None
    batch.num_draft_tokens = 3
    target.compute_logits = lambda h: calls.append(h.shape[0])
    hook.evaluate(runner, hidden, batch)
    assert calls == [2, 2, 1, 1]


def test_kl_config_default_and_legacy_mode():
    raw = {"slim_verifier_eval": {"enabled": True, "skip_layers": []}}
    assert SlimVerifierConfig.from_additional_config(raw, 4).metrics_mode == "kl"
    raw["slim_verifier_eval"]["metrics_mode"] = "acceptance"
    assert SlimVerifierConfig.from_additional_config(raw, 4).metrics_mode == "acceptance"
    raw["slim_verifier_eval"]["metrics_mode"] = "unknown"
    with pytest.raises(ValueError, match="metrics_mode"):
        SlimVerifierConfig.from_additional_config(raw, 4)


@pytest.mark.parametrize("rank", [0, 1])
def test_kl_initialization_installs_capture_only_on_metrics_rank(rank, monkeypatch):
    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)

    if "vllm" not in sys.modules:
        module("vllm", __path__=[])
    module("vllm.distributed", get_tensor_model_parallel_rank=lambda: rank)
    module("vllm.logger", logger=logging.getLogger("kl-init-test"))
    module("vllm_ascend.ascend_forward_context", set_ascend_forward_context=lambda *a, **k: nullcontext())
    module("vllm_ascend.utils", enable_sp=lambda vc: False, lmhead_tp_enable=lambda: False)
    target, proposer = Target(), Proposer()
    original_sample = proposer.sample_draft
    vc = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        speculative_config=SimpleNamespace(),
        lora_config=None,
        quant_config=None,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(),
        additional_config={"slim_verifier_eval": {"enabled": True, "skip_layers": []}},
    )
    hook = SlimVerifierEval(SimpleNamespace(vllm_config=vc, get_model=lambda: target, speculator=proposer))
    assert hook.config.metrics_mode == "kl"
    assert (hook.metrics is not None) == (rank == 0)
    assert (hook.draft_capture is not None) == (rank == 0)
    assert (proposer.sample_draft == original_sample) == (rank != 0)
