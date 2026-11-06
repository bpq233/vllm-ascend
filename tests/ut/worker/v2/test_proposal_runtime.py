# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run the actual mixture pipeline, provenance and final sampler on CPU."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage"
PACKAGE = "vllm_ascend.worker.v2.spec_decode.multi_stage"


@pytest.fixture
def runtime(monkeypatch):
    def load(stem):
        name = f"{PACKAGE}.{stem}"
        spec = importlib.util.spec_from_file_location(name, SOURCE / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    modules = {
        name: load(name) for name in ("proposal_config", "proposal_math", "proposal_feedback", "proposal_sampling")
    }
    modules["proposal_pipeline"] = load("proposal_pipeline")
    load("acceptance")
    parent = ModuleType("vllm.v1.worker.gpu.spec_decode.rejection_sampler")

    class RejectionSampler:
        def __init__(self, sampler, spec_config, device):
            self.sampler = sampler
            self.num_speculative_steps = spec_config.num_speculative_tokens

    parent.RejectionSampler = RejectionSampler
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    load("final_verification")
    modules["proposal_verification"] = load("proposal_verification")
    return NS(**modules)


def sampling_params(**values):
    return NS(**{"temperature": 1.0, "top_k": -1, "top_p": 1.0, "min_p": 0.0, "seed": None, **values})


def make_context(runtime, *, weight=0.5, req_ids=("a",), params=None, feedback=False):
    context = runtime.proposal_sampling.ProposalSamplingContext(
        {"enabled": True, "mixture_lambda": weight, "feedback_enabled": feedback},
        NS(),
        NS(req_id_to_index={req_id: index * 3 + 2 for index, req_id in enumerate(req_ids)}),
        torch.device("cpu"),
        default_seed=53,
    )
    for req_id in req_ids:
        context.register(req_id, params or sampling_params())
    return context


class ToyBackend:
    def __init__(self, *, block=(1, 2), one_hot=None):
        self.block = list(block)
        self.one_hot = one_hot
        self.proposal_calls = []
        self.logit_calls = []

    def probabilities(self, prefix):
        if self.one_hot is not None:
            probabilities = torch.zeros(4)
            probabilities[self.one_hot] = 1
            return probabilities
        # The current prefix changes the next-token law, not only the token ID.
        return torch.tensor([0.1, 0.2, 0.3, 0.4]).roll(sum(prefix) % 4)

    def propose(self, contexts, req_ids):
        self.proposal_calls.extend((req_id, tuple(prefix)) for req_id, prefix in zip(req_ids, contexts))
        return [list(self.block) for _ in contexts]

    def proposal_logits(self, contexts, req_ids):
        self.logit_calls.extend((req_id, tuple(prefix)) for req_id, prefix in zip(req_ids, contexts))
        return [self.probabilities(prefix).log() for prefix in contexts]


def make_pipeline(runtime, context, backend=None, *, capacity=4, eos=None):
    return runtime.proposal_pipeline.ProposalPipeline(
        backend or ToyBackend(), NS(num_speculative_tokens=2), capacity, eos, context
    )


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("primary", [None, [[2, 3]]])
def test_pipeline_records_the_actual_conditional_mixture_at_every_position(runtime, weight, primary):
    context = make_context(runtime, weight=weight)
    backend = ToyBackend()
    pipeline = make_pipeline(runtime, context, backend, capacity=5)
    prefix = [3, 1]
    output = torch.full((1, 5), -1, dtype=torch.int64)
    tokens = pipeline.refine([prefix], primary, [5], ["a"], output=output)[0]
    assert len(tokens) == 5 and output[0].tolist() == tokens
    assert pipeline.device_output_ready
    entries = context.entries_for("a", len(prefix), tokens, prefix[-1])
    assert backend.logit_calls == [("a", tuple(prefix + tokens[:position])) for position in range(5)]
    for position, entry in enumerate(entries):
        p_m = backend.probabilities(prefix + tokens[:position])
        q_d = torch.zeros(4)
        q_d[entry.draft_token] = 1
        torch.testing.assert_close(entry.intermediate, p_m)
        torch.testing.assert_close(entry.mixture, (1 - weight) * p_m + weight * q_d)
        assert entry.token == tokens[position] and entry.weight == weight
        assert entry.source in ("primary_dflash", "intermediate_dflash")
    assert entries[0].source == ("primary_dflash" if primary is not None else "intermediate_dflash")
    # The caller's committed history is never mutated by speculative sampling.
    assert prefix == [3, 1]


def test_intermediate_correction_discards_all_old_primary_and_secondary_block_tails(runtime):
    context = make_context(runtime, weight=0.0)
    backend = ToyBackend(block=(3, 2), one_hot=0)
    pipeline = make_pipeline(runtime, context, backend, capacity=4)
    result = pipeline.refine([[2]], [[1, 2]], [4], ["a"])
    assert result == [[0, 0, 0, 0]]
    assert backend.proposal_calls == [("a", (2, 0)), ("a", (2, 0, 0)), ("a", (2, 0, 0, 0))]
    entries = context.ledgers["a"].entries
    assert [entry.draft_token for entry in entries] == [1, 3, 3, 3]
    assert [entry.spec_round for entry in entries] == [0, 1, 2, 3]
    assert [entry.intermediate_accept for entry in entries] == [False] * 4
    assert [entry.source for entry in entries] == ["primary_dflash"] + ["intermediate_dflash"] * 3


def test_all_accepted_block_is_reused_until_exhaustion_then_fresh_prefix_is_drafted(runtime):
    context = make_context(runtime, weight=1.0)
    backend = ToyBackend(block=(3, 0))
    pipeline = make_pipeline(runtime, context, backend, capacity=4)
    assert pipeline.refine([[2]], [[1, 2]], [4], ["a"]) == [[1, 2, 3, 0]]
    assert backend.proposal_calls == [("a", (2, 1, 2))]
    assert [entry.spec_round for entry in context.ledgers["a"].entries] == [0, 0, 1, 1]


@pytest.mark.parametrize("accept_uniform", [0.0, 0.99])
def test_token_budget_does_not_depend_on_intermediate_acceptance(runtime, accept_uniform):
    context = make_context(runtime, weight=0.5)
    context.uniforms = lambda req_id, stage, count: torch.tensor([accept_uniform, 0.0])
    backend = ToyBackend(block=(1, 2), one_hot=0)
    result = make_pipeline(runtime, context, backend, capacity=7).refine([[2]], None, [5], ["a"])
    assert len(result[0]) == len(context.ledgers["a"].entries) == 5


@pytest.mark.parametrize("ignore_eos,expected_length", [(False, 1), (True, 4)])
def test_eos_is_terminal_without_filtering_an_already_sampled_candidate(runtime, ignore_eos, expected_length):
    context = make_context(runtime, weight=0.0, params=sampling_params(ignore_eos=ignore_eos))
    backend = ToyBackend(one_hot=0)
    pipeline = make_pipeline(runtime, context, backend, eos=0)
    tokens = pipeline.refine([[2]], None, [4], ["a"])[0]
    assert tokens == [0] * expected_length
    assert [entry.token for entry in context.ledgers["a"].entries] == tokens


def test_zero_budget_or_missing_draft_leaves_next_sample_to_target(runtime):
    context = make_context(runtime)
    backend = ToyBackend(block=())
    pipeline = make_pipeline(runtime, context, backend)
    output = torch.full((1, 4), -1, dtype=torch.int64)
    assert pipeline.refine([[2]], None, [0], ["a"], output) == [[]]
    assert not backend.proposal_calls and not backend.logit_calls
    assert output.eq(0).all() and context.ledgers["a"].entries == []
    assert pipeline.refine([[2]], None, [4], ["a"], output) == [[]]
    assert backend.proposal_calls == [("a", (2,))] and not backend.logit_calls


def test_requests_are_independent_of_batch_order_and_other_stage_rng(runtime):
    def generate(order):
        context = make_context(runtime, req_ids=("a", "b"))
        pipeline = make_pipeline(runtime, context, capacity=6)
        prefixes = {"a": [2], "b": [3]}
        result = pipeline.refine([prefixes[req_id] for req_id in order], None, [6, 6], order)
        return dict(zip(order, result)), context

    first, first_context = generate(["a", "b"])
    second, second_context = generate(["b", "a"])
    assert first == second
    fresh = make_context(runtime, req_ids=("a", "b"))
    for req_id in ("a", "b"):
        torch.testing.assert_close(first_context.uniforms(req_id, "target", 8), fresh.uniforms(req_id, "target", 8))
        assert [entry.weight for entry in first_context.ledgers[req_id].entries] == [
            entry.weight for entry in second_context.ledgers[req_id].entries
        ]


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("temperature", 0),
        ("temperature", -1),
        ("temperature", float("nan")),
        ("temperature", True),
        ("top_k", 0),
        ("top_k", -2),
        ("top_k", 2.5),
        ("top_k", True),
        ("top_p", 0),
        ("top_p", 1.1),
        ("top_p", float("nan")),
        ("min_p", -0.1),
        ("min_p", 1.1),
        ("presence_penalty", 0.1),
        ("frequency_penalty", 0.1),
        ("repetition_penalty", 1.1),
        ("min_tokens", 1),
        ("bad_words", ["bad"]),
        ("bad_words_token_ids", [[1]]),
        ("logit_bias", {1: 2}),
        ("allowed_token_ids", [1]),
        ("structured_outputs", object()),
        ("guided_decoding", object()),
        ("logits_processors", [object()]),
        ("extra_args", {"thinking_budget": 4}),
        ("seed", 1.5),
    ],
)
def test_invalid_request_parameters_fail_before_rng_or_ledger_is_created(runtime, parameter, value):
    context = make_context(runtime, req_ids=())
    with pytest.raises(ValueError):
        context.register("bad", sampling_params(**{parameter: value}))
    assert not context.requests and not context.generators and not context.ledgers


def test_live_seed_or_sampling_parameters_cannot_change(runtime):
    context = make_context(runtime, params=sampling_params(seed=71))
    for values in ({"seed": 72}, {"seed": 71, "temperature": 2}):
        with pytest.raises(ValueError, match="Cannot change"):
            context.register("a", sampling_params(**values))
    context.register("a", sampling_params(seed=71))


def test_discard_drops_old_provenance_and_resets_only_its_own_rng(runtime):
    context = make_context(runtime, req_ids=("a", "b"))
    reference = make_context(runtime, req_ids=("a", "b"))
    first = context.uniforms("a", "intermediate", 4)
    context.uniforms("b", "intermediate", 4)
    reference.uniforms("b", "intermediate", 4)
    context.record("a", [1], [])
    context.discard("a")
    assert "a" not in context.ledgers and "a" not in context.generators
    context.register("a", sampling_params())
    torch.testing.assert_close(context.uniforms("a", "intermediate", 4), first, rtol=0, atol=0)
    torch.testing.assert_close(context.uniforms("b", "intermediate", 4), reference.uniforms("b", "intermediate", 4))


@pytest.mark.parametrize("mismatch", ["start", "token", "anchor", "unknown_slot"])
def test_provenance_mismatch_fails_closed(runtime, mismatch):
    context = make_context(runtime, weight=1.0)
    tokens = make_pipeline(runtime, context, capacity=2).refine([[3, 2]], [[1, 2]], [2], ["a"])[0]
    start, anchor = 2, 2
    if mismatch == "start":
        start = 1
    elif mismatch == "token":
        tokens[1] = 0
    elif mismatch == "anchor":
        anchor = 3
    with pytest.raises(ValueError):
        if mismatch == "unknown_slot":
            context.request_id(71)
        else:
            context.entries_for("a", start, tokens, anchor)


def test_recorded_suffix_uses_matching_previous_candidate_as_anchor(runtime):
    context = make_context(runtime, weight=1.0)
    make_pipeline(runtime, context, capacity=3).refine([[3, 2]], [[1, 2, 3]], [3], ["a"])
    entries = context.entries_for("a", 3, [2, 3], anchor=1)
    assert [entry.token for entry in entries] == [2, 3]
    with pytest.raises(ValueError, match="anchor"):
        context.entries_for("a", 3, [2, 3], anchor=2)


def test_intermediate_transform_matches_declared_temperature_topk_topp_and_minp(runtime):
    transform = runtime.proposal_sampling.intermediate_probabilities
    params_type = runtime.proposal_sampling.RequestSampling
    logits = torch.tensor([0.5, 0.3, 0.2]).log()
    result = transform(logits, params_type(1.0, -1, 0.6, 0, False))
    torch.testing.assert_close(result, torch.tensor([0.625, 0.375, 0.0]))
    result = transform(logits, params_type(1.0, 1, 1, 0, False))
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.0]))
    result = transform(logits, params_type(1.0, -1, 1, 0.7, False))
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.0]))
    result = transform(logits, params_type(2.0, -1, 1, 0, False))
    torch.testing.assert_close(result, torch.sqrt(logits.exp()) / torch.sqrt(logits.exp()).sum())


def make_verifier(runtime, context, *, capacity=5, processed=None):
    sampler = NS(
        trace_replay_state=None,
        apply_sampling_params=Mock(side_effect=lambda logits, *args: logits if processed is None else processed),
        sample=Mock(side_effect=AssertionError("The Target must use exact recorded proposal probabilities")),
    )
    verifier = runtime.proposal_verification.ProposalVerificationSampler(
        sampler, NS(num_speculative_tokens=capacity), torch.device("cpu"), context
    )
    return verifier, sampler


def verify(verifier, logits, tokens, positions, cumulative, indices, *, verification_tokens=None):
    indices = np.array(indices)
    expanded = torch.repeat_interleave(torch.from_numpy(indices), torch.diff(torch.tensor(cumulative)))
    local = torch.cat([torch.arange(end - start) for start, end in zip(cumulative, cumulative[1:])])
    return verifier._verify(
        logits,
        object(),
        torch.tensor(tokens),
        torch.tensor(positions),
        torch.tensor(cumulative),
        torch.from_numpy(indices),
        indices,
        expanded,
        local,
        np.full(len(indices), 100, dtype=np.int32),
        verify_draft_sampled=None if verification_tokens is None else torch.tensor(verification_tokens),
    )


def test_final_sampler_uses_processed_target_distribution_without_second_temperature(runtime):
    context = make_context(runtime, weight=1.0, params=sampling_params(temperature=2.0))
    make_pipeline(runtime, context, capacity=1).refine([[0, 2]], [[1]], [1], ["a"])
    processed = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]]).log()
    verifier, sampler = make_verifier(runtime, context, processed=processed)
    context.uniforms = lambda req_id, stage, count: torch.tensor([0.2, 0.4, 0.5])
    original = torch.randn(2, 4)
    output_logits, sampled, counts = verify(verifier, original, [2, 1], [1, 2], [0, 2], [2])
    assert output_logits is processed
    assert counts.tolist() == [1] and sampled[0].tolist() == [0, -1, -1, -1, -1, -1]
    assert sampler.apply_sampling_params.call_count == 1
    assert sampler.apply_sampling_params.call_args.args[0] is original
    sampler.sample.assert_not_called()


def test_exact_final_sampler_keeps_target_bonus_in_independent_chain(runtime):
    context = make_context(runtime, weight=1.0)
    make_pipeline(runtime, context, capacity=2).refine([[3, 0]], [[1, 2]], [2], ["a"])
    verifier, _ = make_verifier(runtime, context)
    context.uniforms = lambda req_id, stage, count: torch.tensor([0.8, 0.8, 0.0, 0.0, 0.3])
    logits = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.1, 0.2, 0.3, 0.4]]).log()
    _, sampled, counts = verify(verifier, logits, [0, 1, 2], [1, 2, 3], [0, 3], [2])
    assert verifier.emit_bonus_token
    assert counts.tolist() == [3]
    assert sampled[0, :3].tolist() == [1, 2, 2]


def test_final_verification_distinct_tokens_preserve_sampling_input_abi(runtime):
    context = make_context(runtime, weight=1.0)
    make_pipeline(runtime, context, capacity=1).refine([[3, 2]], [[1]], [1], ["a"])
    verifier, sampler = make_verifier(runtime, context)
    logits = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]).log()
    _, sampled, counts = verify(verifier, logits, [2, 0], [1, 2], [0, 2], [2], verification_tokens=[2, 1])
    assert sampled[0, :2].tolist() == [1, 2] and counts.tolist() == [2]
    assert sampler.apply_sampling_params.call_args.args[5].tolist() == [2, 0]


def test_final_chunking_and_reordering_do_not_mix_request_ledgers_or_random_streams(runtime):
    def prepare():
        context = make_context(runtime, weight=0.5, req_ids=("a", "b"))
        context.req_states.req_id_to_index = {"a": 5, "b": 2}
        pipeline = make_pipeline(runtime, context, capacity=2)
        rows = pipeline.refine([[0, 2], [3]], [[1, 2], [2, 1]], [2, 1], ["a", "b"])
        verifier, sampler = make_verifier(runtime, context)
        return context, verifier, sampler, rows

    context, verifier, sampler, rows = prepare()
    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).expand(5, -1)
    _, batched, batched_counts = verify(
        verifier, probabilities.log(), [2, *rows[0], 3, *rows[1]], [1, 2, 3, 0, 1], [0, 3, 5], [5, 2]
    )
    _, chunked_verifier, _, chunk_rows = prepare()
    _, chunk_b, counts_b = verify(chunked_verifier, probabilities[:2].log(), [3, *chunk_rows[1]], [0, 1], [0, 2], [2])
    _, chunk_a, counts_a = verify(
        chunked_verifier, probabilities[:3].log(), [2, *chunk_rows[0]], [1, 2, 3], [0, 3], [5]
    )
    torch.testing.assert_close(batched, torch.cat((chunk_a, chunk_b)), rtol=0, atol=0)
    torch.testing.assert_close(batched_counts, torch.cat((counts_a, counts_b)), rtol=0, atol=0)
    assert sampler.apply_sampling_params.call_args.args[-1].tolist() == [100, 100]
    assert context.metrics["request_verifications"] == 2


def test_final_empty_draft_uses_target_bonus_without_requiring_a_proposal_ledger(runtime):
    context = make_context(runtime)
    verifier, _ = make_verifier(runtime, context)
    context.uniforms = lambda req_id, stage, count: torch.tensor([0.75])
    _, sampled, counts = verify(verifier, torch.tensor([[0.5, 0.5, 0.0, 0.0]]).log(), [3], [1], [0, 1], [2])
    assert counts.tolist() == [1] and sampled[0].tolist() == [1, -1, -1, -1, -1, -1]


def test_forced_trace_replay_is_rejected_before_using_proposal_probabilities(runtime):
    context = make_context(runtime)
    verifier, sampler = make_verifier(runtime, context)
    sampler.trace_replay_state = object()
    with pytest.raises(ValueError, match="trace-replay"):
        verify(verifier, torch.ones((1, 4)), [3], [1], [0, 1], [2])
    sampler.apply_sampling_params.assert_not_called()


@pytest.mark.parametrize("reused", [False, True])
def test_new_request_registration_survives_mrv2_slot_removal(runtime, reused):
    context = make_context(runtime, req_ids=("a",) if reused else ())
    request = NS(req_id="a", sampling_params=sampling_params(seed=91))
    context.begin_step(NS(scheduled_new_reqs=[request], preempted_req_ids=set()))
    # Real add_requests removes a slot even for a never-before-seen ID.
    context.discard("a")
    assert "a" not in context.requests
    context.finish_step_registration()
    assert context.requests["a"].seed == 91
    assert make_pipeline(runtime, context, capacity=1).refine([[3]], [[1]], [1], ["a"])[0]


def test_runner_registers_requests_after_parent_slot_updates_before_sampling(runtime):
    context = make_context(runtime, req_ids=())
    request = NS(req_id="a", sampling_params=sampling_params(seed=91))
    scheduler = NS(scheduled_new_reqs=[request], preempted_req_ids=set())
    path = SOURCE.parents[1] / "model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    execute = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "execute_model")
    parent_work = next(
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.Try)
        and any(
            isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "output" for t in item.targets)
            for item in node.body
        )
    )
    execute.body = parent_work.body + [ast.Return(value=ast.Name(id="output", ctx=ast.Load()))]
    execute.decorator_list = []
    events = []
    output = object()

    class Parent:
        def execute_model(self, scheduler_output, **kwargs):
            context.begin_step(scheduler_output)
            context.discard("a")
            assert "a" not in context.requests
            events.append("parent_completed")
            return output

    finish = context.finish_step_registration

    def register():
        assert events == ["parent_completed"]
        finish()
        events.append("sampling_ready")

    context.finish_step_registration = register
    cls = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[execute], decorator_list=[]
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    namespace = {"Parent": Parent, "trace_verification": False, "profile_kwargs": {}}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    obj = namespace["Runner"]()
    obj.rejection_sampler = NS(proposal_context=context)
    assert obj.execute_model(scheduler) is output
    assert events == ["parent_completed", "sampling_ready"]
    assert context.requests["a"].seed == 91


def test_feedback_changes_only_future_mixtures_and_keeps_current_ledger(runtime):
    context = runtime.proposal_sampling.ProposalSamplingContext(
        {"enabled": True, "mode": "adaptive", "min_observations": 1, "lambda_candidates": [0.0, 0.5, 1.0]},
        NS(),
        NS(req_id_to_index={"a": 2}),
        torch.device("cpu"),
    )
    context.register("a", sampling_params(seed=7))
    pipeline = make_pipeline(runtime, context, capacity=1)
    pipeline.refine([[3]], [[1]], [1], ["a"])
    entry = context.ledgers["a"].entries[0]
    original = entry.mixture.clone()
    snapshot = context.feedback.snapshot()
    assert entry.weight == 0.5
    context.target_feedback("a", [entry], entry.intermediate[None], 1, 0.0, proposal_start=1, submitted_tokens=2)
    assert context.feedback.select(entry.state, snapshot=snapshot) == 0.5
    assert context.feedback.select(entry.state) == 0.0
    assert entry.weight == 0.5
    torch.testing.assert_close(entry.mixture, original, rtol=0, atol=0)
    pipeline.refine([[3]], [[1]], [1], ["a"])
    assert context.ledgers["a"].entries[0].weight == 0.0


def test_preemption_invalidates_candidates_without_rewinding_request_random_streams(runtime):
    context = make_context(runtime)
    reference = make_context(runtime)
    for stage in ("intermediate", "target"):
        torch.testing.assert_close(context.uniforms("a", stage, 7), reference.uniforms("a", stage, 7))
    make_pipeline(runtime, context, capacity=1).refine([[3]], [[1]], [1], ["a"])
    reference.uniforms("a", "intermediate", 2)
    assert context.metrics["ledger_bytes"] > 0
    context.begin_step(NS(preempted_req_ids={"a"}, finished_req_ids=set(), scheduled_new_reqs=[]))
    context.discard("a")
    context.finish_step_registration()
    assert not context.ledgers and context.metrics["ledger_bytes"] == 0
    # A later scheduler step resumes the request through the new-slot path.
    request = NS(req_id="a", sampling_params=sampling_params())
    context.begin_step(NS(preempted_req_ids=set(), finished_req_ids=set(), scheduled_new_reqs=[request]))
    context.discard("a")
    context.finish_step_registration()
    for stage in ("intermediate", "target"):
        torch.testing.assert_close(context.uniforms("a", stage, 13), reference.uniforms("a", stage, 13))
    context.begin_step(NS(preempted_req_ids={"a"}, finished_req_ids=set(), scheduled_new_reqs=[]))
    context.discard("a")
    context.begin_step(NS(preempted_req_ids=set(), finished_req_ids={"a"}, scheduled_new_reqs=[]))
    context.discard("a")
    assert not context.requests and not context.generators and not context._preempted_requests


def test_full_vocabulary_probabilities_stay_on_device_through_proposal_and_feedback(runtime, monkeypatch):
    vocabulary = 41
    original_cpu = torch.Tensor.cpu
    original_list = torch.Tensor.tolist

    def guarded_cpu(tensor, *args, **kwargs):
        assert not tensor.ndim or tensor.shape[-1] != vocabulary, "full-vocabulary D2H"
        return original_cpu(tensor, *args, **kwargs)

    def guarded_list(tensor, *args, **kwargs):
        assert not tensor.ndim or tensor.shape[-1] != vocabulary, "full-vocabulary host conversion"
        return original_list(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", guarded_cpu)
    monkeypatch.setattr(torch.Tensor, "tolist", guarded_list)
    context = make_context(runtime, feedback=True)
    backend = ToyBackend()
    backend.probabilities = lambda prefix: torch.arange(1, vocabulary + 1).float().softmax(0)
    tokens = make_pipeline(runtime, context, backend, capacity=2).refine([[3]], [[1, 2]], [2], ["a"])[0]
    verifier, _ = make_verifier(runtime, context)
    logits = torch.arange(vocabulary).float().repeat(3, 1)
    _, _, counts = verify(verifier, logits, [3, *tokens], [0, 1, 2], [0, 3], [2])
    assert counts.item() >= 1
    assert context.metrics["request_verifications"] == 1


def test_adapter_marks_resumed_request_before_resetting_its_old_slot(runtime):
    context = make_context(runtime)
    context.begin_step(NS(preempted_req_ids={"a"}, scheduled_new_reqs=[]))
    generator = context.generators["a"]
    tree = ast.parse((SOURCE / "adapter.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    begin_step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "begin_step")
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[begin_step], type_ignores=[])), "adapter.py", "exec"),
        namespace,
    )
    handler = NS(proposal_context=context, reset_requests=lambda ids: [context.discard(req_id) for req_id in ids])
    request = NS(req_id="a", sampling_params=sampling_params())
    namespace["begin_step"](handler, NS(scheduled_new_reqs=[request], finished_req_ids=set(), preempted_req_ids=set()))
    context.finish_step_registration()
    assert context.generators["a"] is generator


def test_benchmark_ignores_warmup_and_deduplicates_multi_token_cycles(tmp_path):
    path = SOURCE.parents[4] / "examples/offline_inference/target_aligned_proposal_benchmark.py"
    spec = importlib.util.spec_from_file_location("_proposal_model_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    first = {
        "event": "target_verified",
        "request_id": "a",
        "target_cycle_id": "a:2",
        "candidate_length": 2,
        "target_accepted_length": 1,
        "submitted_tokens": 2,
    }
    second = {
        **first,
        "target_cycle_id": "a:4",
        "candidate_length": 3,
        "target_accepted_length": 3,
        "submitted_tokens": 4,
    }
    proposal = {
        "event": "proposal_generated",
        "request_id": "a",
        "target_cycle_id": "a:2",
        "intermediate_accepted_tokens": 2,
        "intermediate_blocks": 1,
        "sampling_overhead_ms": 1,
        "retained_probability_bytes": 64,
    }
    later_proposal = {
        **proposal,
        "target_cycle_id": "a:4",
        "intermediate_accepted_tokens": 3,
        "intermediate_blocks": 2,
        "sampling_overhead_ms": 3,
        "retained_probability_bytes": 96,
    }
    record = lambda value: "INFO proposal_sampling " + json.dumps(value) + "\n"
    log_path = tmp_path / "benchmark.log"
    log_path.write_text(
        record(first)
        + benchmark.MEASURE_START
        + "\n"
        + "".join(record(value) for value in (proposal, first, first, later_proposal, second, second))
        + benchmark.MEASURE_END
        + "\n"
        + record(second),
        encoding="utf-8",
    )
    result = benchmark.collect_logs(log_path, tmp_path / "measured.jsonl")
    assert result["target_request_verification_cycles"] == 2
    assert result["target_mean_accepted_length"] == 2
    assert result["target_acceptance_rate"] == 0.8
    assert result["mean_sampler_output_tokens_per_cycle"] == 3
    assert result["intermediate_mean_accepted_per_draft_block"] == 5 / 3
    assert result["mean_proposal_sampling_host_wall_ms"] == 2
    assert result["max_logged_retained_probability_bytes"] == 96
