# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Probability-space primitives for exact speculative rejection sampling.

Inputs describe the *actual* sampling distributions at the same conditional
prefix. In particular, an argmax DFlash proposal is a point mass, not its raw
softmax. Exactness here means the standard rejection law up to FP32 arithmetic;
there is no epsilon, probability floor, or uniform residual fallback.

Validation reduces device predicates to host scalars. Callers may disable it
after establishing these invariants at their own boundary; the arithmetic and
returned samples otherwise stay on the input device. Graph compatibility and
NPU operator support must be checked separately from the CPU math tests.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RejectionResult:
    tokens: torch.Tensor
    accepted: torch.Tensor
    acceptance_probability: torch.Tensor
    residual_mass: torch.Tensor


@dataclass(frozen=True)
class VerificationResult:
    sampled_tokens: torch.Tensor
    accepted_length: torch.Tensor
    num_sampled: torch.Tensor
    acceptance_probabilities: torch.Tensor


def normalize_probabilities(probabilities: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
    """Normalize nonnegative probability weights in FP32, without repairing them."""
    if probabilities.ndim == 0 or probabilities.shape[-1] == 0:
        raise ValueError("Probability tensors must have a nonempty vocabulary dimension.")
    probabilities = probabilities.float()
    mass = probabilities.sum(dim=-1, keepdim=True)
    if validate and not bool(
        (torch.isfinite(probabilities) & (probabilities >= 0)).all() & (torch.isfinite(mass) & (mass > 0)).all()
    ):
        raise ValueError("Probability rows must be finite, nonnegative and have positive finite mass.")
    return probabilities / mass


def delta_probabilities(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Return the actual probability law of deterministic, greedy token IDs."""
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError("vocab_size must be a positive integer.")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Greedy token IDs must be integer tensors.")
    # scatter reports invalid IDs rather than assigning them artificial mass.
    result = torch.zeros((*token_ids.shape, vocab_size), dtype=torch.float32, device=token_ids.device)
    return result.scatter_(-1, token_ids.long().unsqueeze(-1), 1.0)


def mix_probabilities(
    intermediate: torch.Tensor, drafter: torch.Tensor, weight: float | torch.Tensor, *, validate: bool = True
) -> torch.Tensor:
    """Mix normalized probabilities, using one preselected weight per row."""
    if intermediate.shape != drafter.shape or intermediate.device != drafter.device:
        raise ValueError("Mixture components must have identical shapes and devices.")
    intermediate = normalize_probabilities(intermediate, validate=validate)
    drafter = normalize_probabilities(drafter, validate=validate)
    weight = torch.as_tensor(weight, dtype=torch.float32, device=intermediate.device)
    weight = torch.broadcast_to(weight, intermediate.shape[:-1]).unsqueeze(-1)
    if validate and not bool((torch.isfinite(weight) & (weight >= 0) & (weight <= 1)).all()):
        raise ValueError("Mixture weights must be finite and lie in [0, 1].")
    return normalize_probabilities((1 - weight) * intermediate + weight * drafter, validate=validate)


def _uniform(
    shape: torch.Size | tuple,
    device: torch.device,
    uniform: torch.Tensor | None,
    generator: torch.Generator | None,
    validate: bool,
) -> torch.Tensor:
    if uniform is None:
        return torch.rand(shape, dtype=torch.float32, device=device, generator=generator)
    uniform = torch.as_tensor(uniform, dtype=torch.float32, device=device)
    uniform = torch.broadcast_to(uniform, shape)
    if validate and not bool((torch.isfinite(uniform) & (uniform >= 0) & (uniform < 1)).all()):
        raise ValueError("Sampling uniforms must lie in [0, 1).")
    return uniform


def sample_categorical(
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    uniform: torch.Tensor | None = None,
    validate: bool = True,
) -> torch.Tensor:
    """Inverse-CDF categorical sampling, including zero-probability intervals."""
    probabilities = normalize_probabilities(probabilities, validate=validate)
    uniform = _uniform(probabilities.shape[:-1], probabilities.device, uniform, generator, validate)
    cumulative = probabilities.cumsum(dim=-1)
    # Dividing by the computed total makes the last CDF entry exactly one.
    # It avoids a rounding-induced out-of-vocabulary sample near uniform=1.
    cumulative = cumulative / cumulative[..., -1:]
    return (cumulative <= uniform.unsqueeze(-1)).sum(dim=-1)


def residual_probabilities(target: torch.Tensor, proposal: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
    """Return normalized positive residual; equal laws have no rejection law."""
    if target.shape != proposal.shape or target.device != proposal.device:
        raise ValueError("Target and proposal probabilities must have identical shapes and devices.")
    target = normalize_probabilities(target, validate=validate)
    proposal = normalize_probabilities(proposal, validate=validate)
    return normalize_probabilities((target - proposal).clamp_min(0), validate=validate)


def rejection_step(
    target: torch.Tensor,
    proposal: torch.Tensor,
    proposed_tokens: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    accept_uniform: torch.Tensor | None = None,
    sample_uniform: torch.Tensor | None = None,
    validate: bool = True,
) -> RejectionResult:
    """Convert a draw from proposal into a draw from target by exact rejection.

    The caller must provide proposed_tokens sampled from these very proposal
    rows. For accepted rows, a categorical draw from target is computed but
    discarded; this avoids constructing an undefined zero-mass residual when
    the two laws coincide. A rejection with no positive residual is an error.
    """
    if target.shape != proposal.shape or target.device != proposal.device:
        raise ValueError("Target and proposal probabilities must have identical shapes and devices.")
    if proposed_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Proposed token IDs must be integer tensors.")
    target = normalize_probabilities(target, validate=validate)
    proposal = normalize_probabilities(proposal, validate=validate)
    tokens = torch.broadcast_to(proposed_tokens.to(device=proposal.device), proposal.shape[:-1]).long()
    target_mass = target.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    proposal_mass = proposal.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    if validate and not bool((proposal_mass > 0).all()):
        raise ValueError("The proposed token has zero probability in its recorded proposal distribution.")
    acceptance = (target_mass / proposal_mass).clamp_max(1)
    uniform = _uniform(tokens.shape, target.device, accept_uniform, generator, validate)
    accepted = uniform < acceptance
    residual = (target - proposal).clamp_min(0)
    residual_mass = residual.sum(dim=-1)
    if validate and bool((~accepted & (residual_mass <= 0)).any()):
        raise ValueError("A rejected row has no positive residual probability mass.")
    # One is only a denominator for unreachable, accepted zero-residual rows;
    # it is not epsilon smoothing or a fallback distribution for rejection.
    denominator = torch.where(residual_mass > 0, residual_mass, torch.ones_like(residual_mass))
    replacement_probabilities = torch.where(accepted.unsqueeze(-1), target, residual / denominator.unsqueeze(-1))
    sample_uniform = _uniform(tokens.shape, target.device, sample_uniform, generator, validate)
    replacement = sample_categorical(replacement_probabilities, uniform=sample_uniform, validate=False)
    return RejectionResult(torch.where(accepted, tokens, replacement), accepted, acceptance, residual_mass)


def verify_sequence(
    target_probabilities: torch.Tensor,
    proposal_probabilities: torch.Tensor,
    candidate_tokens: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    accept_uniforms: torch.Tensor | None = None,
    sample_uniforms: torch.Tensor | None = None,
    bonus_uniform: torch.Tensor | None = None,
    validate: bool = True,
) -> VerificationResult:
    """Verify one teacher-forced candidate sequence; emit prefix plus correction/bonus.

    target_probabilities contains K+1 rows; proposal_probabilities and tokens
    contain K. Row i describes sampling after candidate_tokens[:i]. Candidate
    length and each row's distribution must be fixed without inspecting that
    row's future sampling randomness. Per-request generators prevent unrelated
    batch composition from changing a request's random stream.
    """
    count = candidate_tokens.numel()
    if candidate_tokens.ndim != 1 or target_probabilities.ndim != 2 or proposal_probabilities.ndim != 2:
        raise ValueError("Sequence probabilities must be matrices and candidate tokens a vector.")
    if target_probabilities.shape != (count + 1, proposal_probabilities.shape[-1]):
        raise ValueError("Target probabilities require one row per candidate plus a bonus row.")
    if proposal_probabilities.shape[0] != count:
        raise ValueError("Every candidate requires its own proposal probability row.")
    if (
        target_probabilities.device != proposal_probabilities.device
        or candidate_tokens.device != target_probabilities.device
    ):
        raise ValueError("Sequence probabilities and tokens must be on the same device.")
    bonus = sample_categorical(target_probabilities[-1], generator=generator, uniform=bonus_uniform, validate=validate)
    if count == 0:
        zero = torch.zeros((), dtype=torch.int32, device=bonus.device)
        return VerificationResult(bonus.reshape(1), zero, zero + 1, target_probabilities.new_empty(0))
    result = rejection_step(
        target_probabilities[:-1],
        proposal_probabilities,
        candidate_tokens,
        generator=generator,
        accept_uniform=accept_uniforms,
        sample_uniform=sample_uniforms,
        validate=validate,
    )
    steps = torch.arange(count, device=candidate_tokens.device)
    stop = torch.where(result.accepted, count, steps).amin()
    replacements = torch.cat((result.tokens, bonus.reshape(1)))
    positions = torch.arange(count + 1, device=candidate_tokens.device)
    sampled = torch.where(positions <= stop, replacements, -1)
    accepted_length = stop.to(torch.int32)
    return VerificationResult(sampled, accepted_length, accepted_length + 1, result.acceptance_probability)
