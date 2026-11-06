# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Small conditional-model experiment, executable without vLLM or an NPU.

Example: python benchmarks/proposal_sampling_cpu.py --tokens 2000 --output result.json

These timings describe this Python/CPU toy workload only. They do not estimate
real DFlash throughput, NPU graph overhead, TTFT, TPOT, or a model speedup.
Approximate policies and strict greedy are reported in separate comparison
groups from exact, temperature-one sampling.
"""

import argparse
import importlib.util
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch


def load_local_module(filename):
    path = Path(__file__).resolve().parents[1] / "vllm_ascend/worker/v2/spec_decode/multi_stage" / filename
    name = "_benchmark_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ToyModels:
    """Three-token Markov distributions; every row depends on the real prefix."""

    vocab_size = 3

    def __init__(self):
        # The last row represents the empty-prefix state.
        self.target = torch.tensor([[0.15, 0.65, 0.20], [0.60, 0.10, 0.30], [0.25, 0.25, 0.50], [0.45, 0.35, 0.20]])
        self.intermediate = torch.tensor(
            [[0.25, 0.45, 0.30], [0.30, 0.15, 0.55], [0.45, 0.20, 0.35], [0.25, 0.50, 0.25]]
        )
        # Fixed model outputs, not online access to Target validation feedback.
        self.primary_ids = torch.tensor([1, 0, 2, 0])
        self.secondary_ids = torch.tensor([1, 2, 0, 1])

    def state(self, prefix):
        return prefix[-1] if prefix else self.vocab_size

    def draft(self, prefix, source):
        return (self.primary_ids if source == "primary" else self.secondary_ids)[self.state(prefix)]

    def target_rows(self, prefix, candidates):
        history = list(prefix)
        rows = []
        for token in candidates:
            rows.append(self.target[self.state(history)])
            history.append(token)
        rows.append(self.target[self.state(history)])
        return torch.stack(rows)


def normalized_frequencies(tokens, size):
    return torch.bincount(torch.tensor(tokens), minlength=size).float().div(len(tokens)).tolist()


def sequence_frequencies(tokens):
    pairs = Counter(zip(tokens, tokens[1:]))
    return {f"{left},{right}": count / max(len(tokens) - 1, 1) for (left, right), count in sorted(pairs.items())}


def run_reference(models, math_ops, count, seed):
    generator = torch.Generator().manual_seed(seed)
    tokens = []
    started = perf_counter()
    for _ in range(count):
        tokens.append(int(math_ops.sample_categorical(models.target[models.state(tokens)], generator=generator)))
    elapsed = perf_counter() - started
    return {
        "mode": "target_sampling_reference",
        "comparison_group": "exact_temperature_1",
        "temperature": 1.0,
        "target_calls": count,
        "committed_tokens": count,
        "decode_latency_ms": elapsed * 1000,
        "tokens_per_second": count / elapsed,
        "token_frequencies": normalized_frequencies(tokens, models.vocab_size),
        "pair_frequencies": sequence_frequencies(tokens),
    }


def generate_exact(models, math_ops, feedback_ops, feedback, snapshot, prefix, capacity, generator):
    history = list(prefix)
    candidates, proposals, records = [], [], []
    for position in range(capacity):
        source = "primary" if position < 2 else "secondary"
        spec_round = position // 2
        started = perf_counter()
        draft = models.draft(history, source)
        draft_ms = (perf_counter() - started) * 1000
        intermediate = models.intermediate[models.state(history)]
        features = feedback_ops.device_features(intermediate[None], draft.reshape(1))[0].tolist()
        state = feedback_ops.make_state(features, spec_round, position, source)
        weight = feedback.select(state, snapshot)
        delta = math_ops.delta_probabilities(draft, models.vocab_size)
        sampling_started = perf_counter()
        proposal = math_ops.mix_probabilities(intermediate, delta, weight)
        result = math_ops.rejection_step(proposal, delta, draft, generator=generator)
        sampling_ms = (perf_counter() - sampling_started) * 1000
        token = int(result.tokens)
        candidates.append(token)
        proposals.append(proposal)
        records.append(
            {
                "state": state,
                "pM": intermediate,
                "qD_token": draft,
                "lambda": weight,
                "spec_round": spec_round,
                "token_position": position,
                "draft_source": source,
                "draft_token": token,
                "intermediate_accept": bool(result.accepted),
                "pM_draft_token": float(intermediate[token]),
                "qD_draft_token": float(delta[token]),
                "qMix_draft_token": float(proposal[token]),
                "draft_time_ms": draft_ms,
                "sampling_overhead_ms": sampling_ms,
            }
        )
        history.append(token)
    return candidates, torch.stack(proposals), records


def run_exact(models, math_ops, feedback_ops, count, capacity, seed, mode, weight, debug):
    feedback = feedback_ops.ProposalFeedback(
        {"mode": mode, "mixture_lambda": weight, "min_observations": 4, "debug": debug}
    )
    proposal_rng = torch.Generator().manual_seed(seed + 1)
    target_rng = torch.Generator().manual_seed(seed + 2)
    tokens = []
    target_calls = proposed = accepted = intermediate_accepted = 0
    target_ms = sampling_ms = 0.0
    overlaps = torch.zeros(3)
    overlap_count = 0
    lambda_counts = Counter()
    max_probability_bytes = 0
    started = perf_counter()
    while len(tokens) < count:
        candidates, proposals, records = generate_exact(
            models, math_ops, feedback_ops, feedback, feedback.snapshot(), tokens, capacity, proposal_rng
        )
        target = models.target_rows(tokens, candidates)
        verify_started = perf_counter()
        result = math_ops.verify_sequence(target, proposals, torch.tensor(candidates), generator=target_rng)
        cycle_ms = (perf_counter() - verify_started) * 1000
        accepted_length = int(result.accepted_length)
        target_ms += cycle_ms
        accepted += accepted_length
        proposed += len(candidates)
        target_calls += 1
        committed = result.sampled_tokens[: int(result.num_sampled)].tolist()
        tokens.extend(committed[: count - len(tokens)])
        max_probability_bytes = max(max_probability_bytes, (target.numel() + proposals.numel()) * target.element_size())
        # Only the accepted prefix and first rejected context are on the actual
        # Target path. All update packets come from completed verification.
        for position, record in enumerate(records[: min(accepted_length + 1, len(records))]):
            scores = feedback_ops.overlap_scores(
                target[position : position + 1], record["pM"][None], record["qD_token"].reshape(1), feedback.candidates
            )[0].tolist()
            feedback.update(record["state"], scores[2:], target_verified=True)
            selected = 1 - (target[position] - proposals[position]).abs().sum() * 0.5
            overlaps += torch.tensor([scores[0], scores[1], float(selected)])
            overlap_count += 1
            oracle = feedback_ops.offline_oracle_lambda(scores[2:], feedback.candidates)
            payload = {key: value for key, value in record.items() if key not in ("state", "pM", "qD_token")}
            payload.update(
                request_id="toy-request",
                target_accept=position < accepted_length,
                intermediate_accepted_length=sum(int(item["intermediate_accept"]) for item in records),
                target_accepted_length=accepted_length,
                pT_draft_token=float(target[position, candidates[position]]),
                tv_target_intermediate=1 - scores[0],
                tv_target_dflash=1 - scores[1],
                tv_target_mixture=1 - float(selected),
                target_verify_time_ms=cycle_ms,
                offline_oracle_lambda=oracle,
            )
            feedback.emit_record(payload)
        intermediate_accepted += sum(int(record["intermediate_accept"]) for record in records)
        sampling_ms += sum(record["sampling_overhead_ms"] for record in records)
        lambda_counts.update(str(record["lambda"]) for record in records)
    elapsed = perf_counter() - started
    return {
        "mode": "adaptive" if mode == "adaptive" else f"fixed_{weight:g}",
        "comparison_group": "exact_temperature_1",
        "temperature": 1.0,
        "target_calls": target_calls,
        "target_expected_accepted_length": accepted / target_calls,
        "target_acceptance_rate": accepted / proposed,
        "intermediate_mean_accepted_per_proposal_batch": intermediate_accepted / target_calls,
        "committed_per_target_call": len(tokens) / target_calls,
        "committed_tokens": len(tokens),
        "decode_latency_ms": elapsed * 1000,
        "tokens_per_second": len(tokens) / elapsed,
        "target_verify_time_ms": target_ms,
        "sampling_overhead_ms": sampling_ms,
        "mean_overlap_target_intermediate": float(overlaps[0] / max(overlap_count, 1)),
        "mean_overlap_target_dflash": float(overlaps[1] / max(overlap_count, 1)),
        "mean_overlap_target_mixture": float(overlaps[2] / max(overlap_count, 1)),
        "lambda_counts": dict(sorted(lambda_counts.items())),
        "retained_target_and_proposal_tensor_bytes": max_probability_bytes,
        "token_frequencies": normalized_frequencies(tokens, models.vocab_size),
        "pair_frequencies": sequence_frequencies(tokens),
    }


def run_legacy(models, math_ops, count, capacity, seed, method):
    generator = torch.Generator().manual_seed(seed)
    tokens = []
    accepted = proposed = calls = intermediate_accepted = 0
    started = perf_counter()
    while len(tokens) < count:
        history = list(tokens)
        candidates = []
        for position in range(capacity):
            source = "primary" if position < 2 else "secondary"
            draft = models.draft(history, source)
            intermediate = models.intermediate[models.state(history)]
            intermediate_ok = bool((intermediate.topk(2).indices == draft).any())
            token = int(draft if intermediate_ok else intermediate.argmax())
            intermediate_accepted += intermediate_ok
            candidates.append(token)
            history.append(token)
        target = models.target_rows(tokens, candidates)
        output = []
        accepted_here = 0
        for position, token in enumerate(candidates):
            if method == "strict_greedy":
                keep = token == int(target[position].argmax())
            elif method == "topk":
                keep = bool((target[position].topk(2).indices == token).any())
            else:
                keep = float(target[position, token] / target[position].max()) > 0.5
            if not keep:
                correction = (
                    target[position].argmax()
                    if method == "strict_greedy"
                    else math_ops.sample_categorical(target[position], generator=generator)
                )
                output.append(int(correction))
                break
            output.append(token)
            accepted_here += 1
        else:
            bonus = (
                target[-1].argmax()
                if method == "strict_greedy"
                else math_ops.sample_categorical(target[-1], generator=generator)
            )
            output.append(int(bonus))
        tokens.extend(output[: count - len(tokens)])
        calls += 1
        accepted += accepted_here
        proposed += capacity
    elapsed = perf_counter() - started
    return {
        "mode": method,
        "comparison_group": "strict_temperature_0" if method == "strict_greedy" else "approximate_temperature_1",
        "temperature": 0.0 if method == "strict_greedy" else 1.0,
        "target_calls": calls,
        "target_expected_accepted_length": accepted / calls,
        "target_acceptance_rate": accepted / proposed,
        "intermediate_mean_accepted_per_proposal_batch": intermediate_accepted / calls,
        "committed_per_target_call": len(tokens) / calls,
        "committed_tokens": len(tokens),
        "decode_latency_ms": elapsed * 1000,
        "tokens_per_second": len(tokens) / elapsed,
        "token_frequencies": normalized_frequencies(tokens, models.vocab_size),
        "pair_frequencies": sequence_frequencies(tokens),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2000)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug", action="store_true", help="Emit per-token JSON experiment records to stderr.")
    args = parser.parse_args(argv)
    if args.tokens < 2 or args.capacity < 1 or args.seed < 0:
        parser.error("tokens must be >= 2, capacity >= 1, and seed >= 0")
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO if args.debug else logging.WARNING)
    math_ops = load_local_module("proposal_math.py")
    feedback_ops = load_local_module("proposal_feedback.py")
    models = ToyModels()
    results = [run_reference(models, math_ops, args.tokens, args.seed)]
    results.extend(
        run_legacy(models, math_ops, args.tokens, args.capacity, args.seed, mode)
        for mode in ("strict_greedy", "topk", "prob_ratio")
    )
    results.extend(
        run_exact(models, math_ops, feedback_ops, args.tokens, args.capacity, args.seed, "fixed", weight, args.debug)
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    results.append(
        run_exact(models, math_ops, feedback_ops, args.tokens, args.capacity, args.seed, "adaptive", 0.5, args.debug)
    )
    report = {
        "scope": "CPU toy conditional distributions; not an NPU or model speed benchmark",
        "exactness": "FP32 rejection identities; approximate modes do not preserve the Target distribution",
        "comparison_warning": "Do not compare greedy temperature-zero and random temperature-one timings as a speedup",
        "memory_scope": "Retained Target/proposal tensor sizes only; excludes shared toy tables and temporary buffers",
        "tokens": args.tokens,
        "capacity": args.capacity,
        "seed": args.seed,
        "results": results,
    }
    output = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
