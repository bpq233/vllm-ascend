# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exhaustively check Target sequence laws with acceptance-dependent round ends.

Each strict round keeps the first intermediate residual correction and then
stops, or stops after accepting its whole two-token draft block. Candidate
length therefore depends on intermediate outcomes. Enumerating only the first
Target output would miss potential stopping/continuation bias: these checks
enumerate three output tokens, starting additional Target cycles as required.
"""

import importlib.util
import itertools
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pytest
import torch


@pytest.fixture
def math_module(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage/proposal_math.py"
    spec = importlib.util.spec_from_file_location("_proposal_round_law_math", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def target(prefix):
    return torch.tensor([0.625, 0.375] if not prefix or prefix[-1] == 0 else [0.375, 0.625])


def intermediate(prefix):
    return torch.tensor([0.25, 0.75] if not prefix or prefix[-1] == 0 else [0.75, 0.25])


def draft_block(prefix, primary):
    last = prefix[-1] if prefix else 0
    return [1 - last, last] if primary else [last, last]


def intervals(probabilities):
    left = 0.0
    for token, mass in enumerate(probabilities.double().tolist()):
        if mass > 0:
            yield token, mass, left + mass / 2
        left += mass


def step_branches(module, p, q, draft):
    """Enumerate the actual primitive's acceptance and residual draw intervals."""
    acceptance = min(1.0, float(p[draft]) / float(q[draft]))
    if acceptance > 0:
        result = module.rejection_step(
            p, q, torch.tensor(draft), accept_uniform=torch.tensor(acceptance / 2), sample_uniform=torch.tensor(0.5)
        )
        assert result.accepted and int(result.tokens) == draft
        yield acceptance, draft, True
    if acceptance < 1:
        residual = module.residual_probabilities(p, q)
        for token, mass, uniform in intervals(residual):
            result = module.rejection_step(
                p,
                q,
                torch.tensor(draft),
                accept_uniform=torch.tensor((1 + acceptance) / 2),
                sample_uniform=torch.tensor(uniform),
            )
            assert not result.accepted and int(result.tokens) == token
            yield (1 - acceptance) * mass, token, False


def proposal_paths(module, prefix, rounds, weight):
    """All strict-round paths, including short correction-terminated blocks."""

    def walk(tokens, rows, round_id, block, offset, probability):
        history = prefix + tokens
        if round_id == rounds:
            yield tokens, rows, probability
            return
        draft = block[offset]
        q_d = module.delta_probabilities(torch.tensor(draft), 2)
        mixture = module.mix_probabilities(intermediate(history), q_d, weight)
        for mass, token, accepted in step_branches(module, mixture, q_d, draft):
            next_tokens, next_rows = (*tokens, token), (*rows, mixture)
            if accepted and offset + 1 < len(block):
                yield from walk(next_tokens, next_rows, round_id, block, offset + 1, probability * mass)
            else:
                # A round boundary is decided after including the current
                # sample. No candidate is selected by its future suffix.
                next_block = draft_block(prefix + next_tokens, primary=False)
                yield from walk(next_tokens, next_rows, round_id + 1, next_block, 0, probability * mass)

    return walk((), (), 0, draft_block(prefix, primary=True), 0, 1.0)


def final_paths(module, prefix, tokens, proposal_rows):
    """Enumerate the real final verifier's first rejection or full-accept bonus."""
    count = len(tokens)
    p = torch.stack([target(prefix + tokens[:i]) for i in range(count + 1)])
    q = torch.stack(proposal_rows)
    acceptances = [min(1.0, float(p[i, token]) / float(q[i, token])) for i, token in enumerate(tokens)]
    passing = 1.0
    accept_uniforms = torch.tensor([acceptance / 2 for acceptance in acceptances])
    for position, acceptance in enumerate(acceptances):
        if passing > 0 and acceptance < 1:
            residual = module.residual_probabilities(p[position], q[position])
            for replacement, mass, uniform in intervals(residual):
                accept_draws, sample_draws = accept_uniforms.clone(), torch.full((count,), 0.5)
                accept_draws[position] = (1 + acceptance) / 2
                sample_draws[position] = uniform
                result = module.verify_sequence(
                    p,
                    q,
                    torch.tensor(tokens),
                    accept_uniforms=accept_draws,
                    sample_uniforms=sample_draws,
                    bonus_uniform=torch.tensor(0.5),
                )
                actual = tuple(result.sampled_tokens[: int(result.num_sampled)].tolist())
                assert actual == (*tokens[:position], replacement)
                yield actual, passing * (1 - acceptance) * mass
        passing *= acceptance
    if passing > 0:
        for bonus, mass, uniform in intervals(p[-1]):
            result = module.verify_sequence(
                p,
                q,
                torch.tensor(tokens),
                accept_uniforms=accept_uniforms,
                sample_uniforms=torch.full((count,), 0.5),
                bonus_uniform=torch.tensor(uniform),
            )
            actual = tuple(result.sampled_tokens[: int(result.num_sampled)].tolist())
            assert actual == (*tokens, bonus)
            yield actual, passing * mass


def output_law(module, initial_prefix, rounds, weight, output_count=3):
    @lru_cache(None)
    def cycle(prefix):
        outcomes = defaultdict(float)
        for tokens, rows, proposal_mass in proposal_paths(module, prefix, rounds, weight):
            for output, final_mass in final_paths(module, prefix, tokens, rows):
                outcomes[output] += proposal_mass * final_mass
        assert sum(outcomes.values()) == pytest.approx(1, abs=2e-7)
        return tuple(outcomes.items())

    result = defaultdict(float)

    def append(history, probability):
        produced = history[len(initial_prefix) :]
        if len(produced) >= output_count:
            result[produced[:output_count]] += probability
            return
        for output, mass in cycle(history):
            append(history + output, probability * mass)

    append(initial_prefix, 1.0)
    return result


@pytest.mark.parametrize("rounds", [1, 2])
@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("initial_prefix", [(), (1,)])
def test_strict_round_stopping_preserves_three_token_target_law(math_module, rounds, weight, initial_prefix):
    actual = output_law(math_module, initial_prefix, rounds, weight)
    assert sum(actual.values()) == pytest.approx(1, abs=2e-7)
    for output in itertools.product(range(2), repeat=3):
        expected = 1.0
        for position, token in enumerate(output):
            expected *= float(target(initial_prefix + output[:position])[token])
        assert actual[output] == pytest.approx(expected, abs=2e-7), (rounds, weight, initial_prefix, output)


@pytest.mark.parametrize("rounds", [1, 2])
def test_round_end_is_outcome_dependent_and_retains_every_correction(math_module, rounds):
    paths = list(proposal_paths(math_module, (), rounds, weight=0.5))
    lengths = {len(tokens) for tokens, _, _ in paths}
    assert lengths == set(range(rounds, 2 * rounds + 1))
    assert sum(mass for _, _, mass in paths) == pytest.approx(1, abs=2e-7)
    assert all(len(tokens) == len(rows) for tokens, rows, _ in paths)
