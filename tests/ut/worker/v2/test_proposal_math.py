# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Probability identities and CPU sampling checks, independent of model quality."""

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture
def math_module(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage/proposal_math.py"
    spec = importlib.util.spec_from_file_location("_proposal_math_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def intervals(probabilities):
    cumulative = probabilities.double().cumsum(-1)
    cumulative /= cumulative[-1].clone()
    previous = 0.0
    for token, boundary in enumerate(cumulative.tolist()):
        if boundary > previous:
            yield token, boundary - previous, (previous + boundary) / 2
        previous = boundary


def rejection_law(module, target, proposal):
    """Enumerate acceptance and every positive residual interval of real code."""
    target = module.normalize_probabilities(target)
    proposal = module.normalize_probabilities(proposal)
    output = torch.zeros_like(target, dtype=torch.float64)
    for token, q_mass, _ in intervals(proposal):
        acceptance = min(1.0, float(target[token].double() / proposal[token].double()))
        if acceptance > 0:
            result = module.rejection_step(
                target,
                proposal,
                torch.tensor(token),
                accept_uniform=torch.tensor(acceptance / 2),
                sample_uniform=torch.tensor(0.5),
            )
            assert result.accepted
            output[result.tokens] += q_mass * acceptance
        if acceptance < 1:
            residual = module.residual_probabilities(target, proposal)
            for replacement, mass, uniform in intervals(residual):
                result = module.rejection_step(
                    target,
                    proposal,
                    torch.tensor(token),
                    accept_uniform=torch.tensor((1 + acceptance) / 2),
                    sample_uniform=torch.tensor(uniform),
                )
                assert not result.accepted
                assert result.tokens == replacement
                output[replacement] += q_mass * (1 - acceptance) * mass
    return output


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("drafter", [[0.0, 1.0, 0.0], [0.5, 0.125, 0.375]])
def test_exhaustive_intermediate_output_law_is_mixture(math_module, weight, drafter):
    intermediate = torch.tensor([0.125, 0.25, 0.625])
    drafter = torch.tensor(drafter)
    mixture = math_module.mix_probabilities(intermediate, drafter, weight)
    torch.testing.assert_close(mixture, (1 - weight) * intermediate + weight * drafter, rtol=0, atol=0)
    actual = rejection_law(math_module, mixture, drafter)
    torch.testing.assert_close(actual, mixture.double(), rtol=0, atol=2e-7)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("target", [[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.125, 0.125, 0.75]])
def test_exhaustive_final_output_law_is_target(math_module, weight, target):
    mixture = math_module.mix_probabilities(torch.tensor([0.25, 0.25, 0.5]), torch.tensor([0.0, 1.0, 0.0]), weight)
    actual = rejection_law(math_module, torch.tensor(target), mixture)
    torch.testing.assert_close(actual, torch.tensor(target, dtype=torch.float64), rtol=0, atol=2e-7)


def test_greedy_proposal_is_delta_not_raw_draft_softmax(math_module):
    raw_logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 5.0, 1.0]])
    actual = math_module.delta_probabilities(raw_logits.argmax(-1), raw_logits.shape[-1])
    torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), rtol=0, atol=0)
    assert not torch.equal(actual, raw_logits.softmax(-1))


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_two_stage_monte_carlo_frequencies_match_both_laws(math_module, weight):
    size = 60000
    intermediate = torch.tensor([0.15, 0.25, 0.6])
    drafter = torch.tensor([0.0, 1.0, 0.0])
    target = torch.tensor([0.4, 0.1, 0.5])
    mixture = math_module.mix_probabilities(intermediate, drafter, weight)
    generator = torch.Generator().manual_seed(49)
    first = math_module.rejection_step(
        mixture.expand(size, -1), drafter.expand(size, -1), torch.ones(size, dtype=torch.int64), generator=generator
    )
    first_frequency = torch.bincount(first.tokens, minlength=3) / size
    torch.testing.assert_close(first_frequency, mixture, rtol=0, atol=0.008)
    final = math_module.rejection_step(
        target.expand(size, -1), mixture.expand(size, -1), first.tokens, generator=generator
    )
    frequency = torch.bincount(final.tokens, minlength=3) / size
    torch.testing.assert_close(frequency, target, rtol=0, atol=0.008)


def test_probability_mixing_and_rejection_use_fp32(math_module):
    intermediate = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)
    drafter = torch.tensor([0.5, 0.5], dtype=torch.float16)
    mixture = math_module.mix_probabilities(intermediate, drafter, 0.5)
    result = math_module.rejection_step(mixture, drafter, torch.tensor(1))
    assert mixture.dtype == result.acceptance_probability.dtype == result.residual_mass.dtype == torch.float32


def test_tiny_nonzero_residual_is_normalized_without_epsilon(math_module):
    target = torch.tensor([0.5, 0.5])
    proposal = torch.tensor([0.50000006, 0.49999994])
    residual = math_module.residual_probabilities(target, proposal)
    torch.testing.assert_close(residual, torch.tensor([0.0, 1.0]), rtol=0, atol=0)
    result = math_module.rejection_step(
        target,
        proposal,
        torch.tensor(0),
        accept_uniform=torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)),
        sample_uniform=torch.tensor(0.0),
    )
    assert not result.accepted
    assert result.tokens == 1
    assert 0 < result.residual_mass < 1e-6


def test_equal_distributions_never_use_undefined_residual(math_module):
    probabilities = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    result = math_module.rejection_step(
        probabilities,
        probabilities,
        torch.tensor([0, 1]),
        accept_uniform=torch.nextafter(torch.ones(2), torch.zeros(2)),
        sample_uniform=torch.zeros(2),
    )
    assert result.accepted.all()
    assert result.residual_mass.eq(0).all()
    torch.testing.assert_close(result.tokens, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="positive finite mass"):
        math_module.residual_probabilities(probabilities, probabilities)


def test_zero_recorded_probability_fails_instead_of_clamping(math_module):
    with pytest.raises(ValueError, match="zero probability"):
        math_module.rejection_step(torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0]), torch.tensor(1))


@pytest.mark.parametrize("probabilities", [[0.0, 0.0], [0.5, -0.1], [float("nan"), 0.0], [float("inf"), 1.0]])
def test_invalid_probability_rows_fail_closed(math_module, probabilities):
    with pytest.raises(ValueError, match="finite"):
        math_module.sample_categorical(torch.tensor(probabilities))


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_weights_fail_closed(math_module, weight):
    with pytest.raises(ValueError, match="weights"):
        math_module.mix_probabilities(torch.ones(2), torch.ones(2), weight)


@pytest.mark.parametrize("uniform", [-0.1, 1.0, float("nan")])
def test_invalid_acceptance_and_sampling_uniforms_fail_closed(math_module, uniform):
    probabilities = torch.tensor([0.5, 0.5])
    with pytest.raises(ValueError, match="uniforms"):
        math_module.rejection_step(probabilities, probabilities, torch.tensor(0), accept_uniform=torch.tensor(uniform))
    with pytest.raises(ValueError, match="uniforms"):
        math_module.rejection_step(probabilities, probabilities, torch.tensor(0), sample_uniform=torch.tensor(uniform))


def test_categorical_zero_mass_intervals_and_last_float_below_one(math_module):
    probabilities = torch.tensor([[0.0, 0.3, 0.0, 0.7, 0.0], [0.0, 0.3, 0.0, 0.7, 0.0]])
    uniforms = torch.tensor([0.0, 1.0])
    uniforms[1] = torch.nextafter(uniforms[1], torch.tensor(0.0))
    result = math_module.sample_categorical(probabilities, uniform=uniforms)
    torch.testing.assert_close(result, torch.tensor([1, 3]))


def test_verify_sequence_stops_at_first_rejection_and_discards_tail(math_module):
    target = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.3, 0.7]])
    proposal = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    result = math_module.verify_sequence(
        target,
        proposal,
        torch.tensor([0, 0, 1]),
        accept_uniforms=torch.tensor([0.0, 0.9, 0.0]),
        sample_uniforms=torch.zeros(3),
        bonus_uniform=torch.tensor(0.0),
    )
    assert result.accepted_length == 1 and result.num_sampled == 2
    assert result.sampled_tokens.tolist() == [0, 1, -1, -1]


def test_full_acceptance_samples_real_target_bonus(math_module):
    target = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.2, 0.8]])
    result = math_module.verify_sequence(
        target,
        target[:-1].clone(),
        torch.tensor([1, 0]),
        accept_uniforms=torch.tensor([0.99, 0.99]),
        sample_uniforms=torch.zeros(2),
        bonus_uniform=torch.tensor(0.3),
    )
    assert result.accepted_length == 2 and result.num_sampled == 3
    assert result.sampled_tokens.tolist() == [1, 0, 1]


def test_empty_drafts_sample_only_target(math_module):
    result = math_module.verify_sequence(
        torch.tensor([[0.2, 0.8]]),
        torch.empty((0, 2)),
        torch.empty(0, dtype=torch.int64),
        bonus_uniform=torch.tensor(0.4),
    )
    assert result.accepted_length == 0 and result.num_sampled == 1
    assert result.sampled_tokens.tolist() == [1]


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_two_token_joint_law_uses_correct_conditional_prefix(math_module, weight):
    # Independent teacher-forced conditionals differ sharply after token 0/1.
    target = [torch.tensor([0.6, 0.4]), torch.tensor([0.2, 0.8]), torch.tensor([0.9, 0.1])]
    intermediate = [torch.tensor([0.25, 0.75]), torch.tensor([0.8, 0.2]), torch.tensor([0.1, 0.9])]
    greedy = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])]
    proposals = [math_module.mix_probabilities(p, q, weight) for p, q in zip(intermediate, greedy)]
    joint = torch.zeros((2, 2), dtype=torch.float64)
    for first, second in itertools.product(range(2), repeat=2):
        q0, q1 = proposals[0], proposals[first + 1]
        path_mass = float(q0[first].double() * q1[second].double())
        if path_mass == 0:
            continue
        a0 = min(1.0, float(target[0][first].double() / q0[first].double()))
        a1 = min(1.0, float(target[first + 1][second].double() / q1[second].double()))
        probabilities = torch.stack((target[0], target[first + 1], torch.tensor([0.5, 0.5])))
        candidate_q = torch.stack((q0, q1))
        tokens = torch.tensor([first, second])
        if a0 < 1:
            for corrected, mass, uniform in intervals(math_module.residual_probabilities(target[0], q0)):
                result = math_module.verify_sequence(
                    probabilities,
                    candidate_q,
                    tokens,
                    accept_uniforms=torch.tensor([(1 + a0) / 2, 0.0]),
                    sample_uniforms=torch.tensor([uniform, 0.0]),
                    bonus_uniform=torch.tensor(0.0),
                )
                assert result.num_sampled == 1 and result.sampled_tokens[0] == corrected
                # Next decode after correction uses its new prefix, never stale q1.
                joint[corrected] += path_mass * (1 - a0) * mass * target[corrected + 1].double()
        if a0 > 0 and a1 < 1:
            for corrected, mass, uniform in intervals(math_module.residual_probabilities(target[first + 1], q1)):
                result = math_module.verify_sequence(
                    probabilities,
                    candidate_q,
                    tokens,
                    accept_uniforms=torch.tensor([a0 / 2, (1 + a1) / 2]),
                    sample_uniforms=torch.tensor([0.0, uniform]),
                    bonus_uniform=torch.tensor(0.0),
                )
                assert result.num_sampled == 2 and result.sampled_tokens[:2].tolist() == [first, corrected]
                joint[first, corrected] += path_mass * a0 * (1 - a1) * mass
        if a0 > 0 and a1 > 0:
            result = math_module.verify_sequence(
                probabilities,
                candidate_q,
                tokens,
                accept_uniforms=torch.tensor([a0 / 2, a1 / 2]),
                sample_uniforms=torch.zeros(2),
                bonus_uniform=torch.tensor(0.0),
            )
            assert result.sampled_tokens[:2].tolist() == [first, second]
            joint[first, second] += path_mass * a0 * a1
    expected = target[0].double()[:, None] * torch.stack(target[1:]).double()
    torch.testing.assert_close(joint, expected, rtol=0, atol=2e-7)


def test_per_request_generators_do_not_depend_on_request_order(math_module):
    def draws(order):
        generators = {name: torch.Generator().manual_seed(seed) for name, seed in (("a", 27), ("b", 39))}
        return {
            name: math_module.sample_categorical(torch.tensor([0.3, 0.7]).expand(100, -1), generator=generators[name])
            for name in order
        }

    first, second = draws(["a", "b"]), draws(["b", "a"])
    for name in ("a", "b"):
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
