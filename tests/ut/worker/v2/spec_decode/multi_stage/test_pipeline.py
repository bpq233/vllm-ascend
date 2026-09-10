# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("reject_at", [0, 2, None])
def test_final_prefix_and_bonus(core, k, reject_at):
    draft = torch.tensor([2, 3, 4, 5])
    logits = torch.full((5, 12), -20.0)
    for i, token in enumerate(draft):
        logits[i, token] = 10.0
    if reject_at is not None:
        logits[reject_at].zero_()
        logits[reject_at, draft[reject_at]] = -10
        logits[reject_at, 6:11] = 10
    samples = torch.tensor([6, 7, 8, 9, 10])
    result = core.acceptance.finalize_candidates(core.acceptance.TopKPolicy(k), logits, draft, samples)
    accepted = 4 if reject_at is None else reject_at
    assert result.tolist() == draft[:accepted].tolist() + [samples[accepted].item()]


def test_all_accept_and_empty_draft(core):
    result = core.acceptance.finalize_candidates(
        core.acceptance.AcceptAllPolicy(), torch.zeros(3, 8), torch.tensor([1, 2]), torch.tensor([6, 6, 7])
    )
    assert result.tolist() == [1, 2, 7]
    assert core.acceptance.finalize_candidates(
        core.acceptance.TopKPolicy(1), torch.zeros(1, 8), torch.tensor([], dtype=torch.int64), torch.tensor([3])
    ).tolist() == [3]


@pytest.mark.parametrize("method", ["topk", "all"])
def test_masked_and_invalid_tokens_never_accepted(core, method):
    policy = core.config.make_policy(core.config.VerificationConfig(method=method, top_k=100))
    logits = torch.zeros(4, 4)
    logits[0, 1] = -torch.inf
    logits[1, 2] = torch.nan
    assert not policy.accept(logits, torch.tensor([1, 2, -1, 4])).any()


class Backend:
    def __init__(self, core, limits=None):
        self.result = core.interfaces.VerificationResult
        self.limits = limits or {}
        self.verified = []
        self.proposed = []
        self.rollbacks = []

    def verify(self, states):
        self.verified.append([(s.request_id, s.round_id, s.context, tuple(s.current_draft)) for s in states])
        return {s.request_id: self.result(s.current_draft[: self.limits.get(s.request_id, 999)]) for s in states}

    def propose(self, states):
        self.proposed.append([(s.request_id, s.context) for s in states])
        return {s.request_id: [6, 7] for s in states}

    def rollback(self, request_id, committed_length):
        self.rollbacks.append((request_id, committed_length))


@pytest.mark.parametrize("rounds", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_rounds_batch_and_private_history(core, rounds, batch_size):
    backend = Backend(core)
    states = [core.state.SpeculativeState(str(i), (1, 2), 12) for i in range(batch_size)]
    result = core.pipeline.SpeculativePipeline(backend, backend, rounds).run(
        states, {s.request_id: [3, 4] for s in states}
    )
    assert all(v == [3, 4] + [6, 7] * (rounds - 1) for v in result.values())
    assert len(backend.verified) == rounds
    assert len(backend.proposed) == rounds - 1
    assert all(s.committed_tokens == (1, 2) for s in states)
    assert all(length == 2 for _, length in backend.rollbacks)
    if rounds > 1:
        assert backend.proposed[0][0][1] == (1, 2, 3, 4)


def test_batch_four_independent_stop_and_ragged(core):
    backend = Backend(core, {"reject": 0, "partial": 1})
    states = [
        core.state.SpeculativeState(name, (1,), cap, frozenset({9}))
        for name, cap in [("full", 10), ("partial", 2), ("reject", 10), ("eos", 10)]
    ]
    result = core.pipeline.SpeculativePipeline(backend, backend, 3).run(
        states, {"full": [3, 4], "partial": [3, 4], "reject": [3, 4], "eos": [3, 9, 4]}
    )
    assert result == {"full": [3, 4, 6, 7, 6, 7], "partial": [3, 6], "reject": [], "eos": [3, 9]}
    assert [entry[0] for entry in backend.verified[-1]] == ["full"]


def test_finished_zero_budget_empty_proposal(core):
    backend = Backend(core)
    states = [
        core.state.SpeculativeState("finished", (1,), 10, finished=True),
        core.state.SpeculativeState("budget", (1,), 0),
        core.state.SpeculativeState("empty", (1,), 10),
    ]
    result = core.pipeline.SpeculativePipeline(backend, backend, 3).run(states, {"empty": []})
    assert result == {"finished": [], "budget": [], "empty": []}
    assert not backend.verified


def test_bad_verifier_and_failure_roll_back_every_request(core):
    backend = Backend(core)
    backend.verify = lambda states: {s.request_id: backend.result([99]) for s in states}
    states = [core.state.SpeculativeState(str(i), (1, 2), 8) for i in range(4)]
    with pytest.raises(ValueError, match="prefix"):
        core.pipeline.SpeculativePipeline(backend, backend, 3).run(states, {s.request_id: [3] for s in states})
    assert {r for r, _ in backend.rollbacks} == {str(i) for i in range(4)}


def test_logging_and_metrics_switches(core, caplog):
    backend = Backend(core)
    pipeline = core.pipeline.SpeculativePipeline(backend, backend, 3, True, True)
    with caplog.at_level("DEBUG"):
        pipeline.run([core.state.SpeculativeState("r", (1,), 10)], {"r": [2]})
    for field in (
        "primary_draft_tokens",
        "intermediate_input_tokens",
        "intermediate_topk",
        "intermediate_accepted_length",
        "secondary_draft_tokens",
        "total_candidate_tokens",
    ):
        assert field in caplog.text
    assert pipeline.metrics.stages["intermediate_verifier"].calls == 3
    assert pipeline.metrics.stages["intermediate_verifier"].accepted_tokens == 5
    caplog.clear()
    disabled = core.pipeline.SpeculativePipeline(backend, backend, 1)
    with caplog.at_level("DEBUG"):
        disabled.run([core.state.SpeculativeState("r", (1,), 10)], {"r": [2]})
    assert not caplog.records
    assert disabled.metrics.stages == {}


@pytest.mark.parametrize(
    "raw",
    [
        {"enabled": "false"},
        {"num_intermediate_rounds": 0},
        {"num_intermediate_rounds": True},
        {"secondary_num_speculative_tokens": -1},
        {"unknown": 1},
        {"enabled": True},
        {"final_verification": {"top_k": 0}},
        {"final_verification": {"method": "exact"}},
        {"metrics_enabled": "false"},
    ],
)
def test_invalid_config(core, raw):
    with pytest.raises((ValueError, TypeError)):
        core.config.MultiStageConfig.from_dict(raw)


@pytest.mark.parametrize("change", ["async", "graph", "tp", "width", "prefix_cache", "method"])
def test_runtime_guards(core, change):
    config = core.config.MultiStageConfig(enabled=True, intermediate_model="a", secondary_model="b")
    runtime = SimpleNamespace(
        speculative_config=SimpleNamespace(
            use_dflash=lambda: change != "method", num_speculative_tokens=4 if change == "width" else 16
        ),
        scheduler_config=SimpleNamespace(async_scheduling=change == "async"),
        model_config=SimpleNamespace(enforce_eager=change != "graph"),
        parallel_config=SimpleNamespace(tensor_parallel_size=2 if change == "tp" else 1),
        cache_config=SimpleNamespace(enable_prefix_caching=change == "prefix_cache"),
    )
    with pytest.raises(ValueError):
        config.validate_runtime(runtime)
