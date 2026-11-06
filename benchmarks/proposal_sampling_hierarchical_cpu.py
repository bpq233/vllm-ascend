# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run the production proposal pipeline with deterministic CPU model substitutes.

This exercises real block reuse, correction invalidation, provenance and EMA,
but does not run Qwen weights, DFlash networks, KV caches, graphs or NPU kernels.
The toy drafter builds a deterministic block along its own predicted prefix.
"""

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch


def load_modules():
    root = Path(__file__).resolve().parents[1]
    source = root / "vllm_ascend/worker/v2/spec_decode/multi_stage"
    modules = {}
    for stem in (
        "proposal_config",
        "proposal_math",
        "proposal_feedback",
        "proposal_sampling",
        "proposal_pipeline",
        "config",
    ):
        name = f"vllm_ascend.worker.v2.spec_decode.multi_stage.{stem}"
        spec = importlib.util.spec_from_file_location(name, source / f"{stem}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[stem] = module
    spec = importlib.util.spec_from_file_location("_proposal_cpu_toy", root / "benchmarks/proposal_sampling_cpu.py")
    toy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(toy)
    modules["toy"] = toy
    return SimpleNamespace(**modules)


class ToyBackend:
    def __init__(self, models, width):
        self.models = models
        self.width = width
        self.proposal_calls = 0
        self.verifier_calls = 0

    def block(self, prefix, source, width):
        history, output = list(prefix), []
        for _ in range(width):
            token = int(self.models.draft(history, source))
            output.append(token)
            history.append(token)
        return output

    def propose(self, prefixes, req_ids):
        self.proposal_calls += len(prefixes)
        return [self.block(prefix, "secondary", self.width) for prefix in prefixes]

    def proposal_logits(self, prefixes, req_ids):
        self.verifier_calls += len(prefixes)
        return [self.models.intermediate[self.models.state(prefix)].log() for prefix in prefixes]

    def proposal_block_logits(self, prefixes, drafts, req_ids):
        self.verifier_calls += len(prefixes)
        return [
            torch.stack(
                [
                    self.models.intermediate[self.models.state(prefix + draft[:position])].log()
                    for position in range(len(draft))
                ]
            )
            for prefix, draft in zip(prefixes, drafts)
        ]


def run_case(modules, models, args, capacity, mode, weight):
    options = {
        "enabled": True,
        "mode": mode,
        "mixture_lambda": weight,
        "feedback_enabled": True,
        "min_observations": 4,
    }
    context = modules.proposal_sampling.ProposalSamplingContext(
        options, None, SimpleNamespace(req_id_to_index={"toy": 0}), torch.device("cpu")
    )
    context.register(
        "toy", SimpleNamespace(temperature=1.0, top_k=-1, top_p=1.0, min_p=0.0, ignore_eos=True, seed=args.seed)
    )
    backend = ToyBackend(models, args.draft_tokens)
    pipeline = modules.proposal_pipeline.ProposalPipeline(
        backend,
        SimpleNamespace(
            num_speculative_tokens=args.draft_tokens,
            num_rounds=args.rounds,
            max_generated_tokens=args.max_generated_tokens,
        ),
        capacity,
        None,
        context,
    )
    output = torch.empty((1, capacity), dtype=torch.int64)
    tokens, lambda_counts = [], Counter()
    calls = accepted = proposed = intermediate_accepted = blocks = 0
    sampling_ms = proposal_ms = target_ms = 0.0
    probability_peak = 0
    started = perf_counter()
    while len(tokens) < args.tokens:
        proposal_started = perf_counter()
        primary = backend.block(tokens, "primary", args.primary_tokens)
        # Leave room for the Target correction/bonus, as the runner does.
        budget = min(capacity, max(0, args.tokens - len(tokens) - 1))
        candidates = pipeline.refine([tokens], [primary], [budget], ["toy"], output=output)[0]
        proposal_ms += (perf_counter() - proposal_started) * 1000
        ledger = context.ledgers["toy"]
        entries = ledger.entries
        q_mix = torch.stack([entry.mixture for entry in entries]) if entries else torch.empty((0, models.vocab_size))
        target = models.target_rows(tokens, candidates)
        uniforms = context.uniforms("toy", "target", 2 * len(entries) + 1)
        verify_started = perf_counter()
        result = modules.proposal_math.verify_sequence(
            target,
            q_mix,
            torch.tensor(candidates, dtype=torch.int64),
            accept_uniforms=uniforms[: len(entries)],
            sample_uniforms=uniforms[len(entries) : 2 * len(entries)],
            bonus_uniform=uniforms[-1],
        )
        accepted_here, submitted = int(result.accepted_length), int(result.num_sampled)
        context.target_feedback(
            "toy",
            entries,
            target[:-1],
            accepted_here,
            verify_started,
            proposal_start=len(tokens),
            submitted_tokens=submitted,
        )
        target_ms += (perf_counter() - verify_started) * 1000
        tokens.extend(result.sampled_tokens[:submitted].tolist())
        calls += 1
        accepted += accepted_here
        proposed += len(entries)
        intermediate_accepted += sum(entry.intermediate_accept for entry in entries)
        blocks += len({entry.spec_round for entry in entries})
        sampling_ms += sum(entry.sampling_overhead_ms for entry in entries)
        probability_peak = max(probability_peak, context.metrics["ledger_bytes"])
        lambda_counts.update(str(entry.weight) for entry in entries)
    elapsed = perf_counter() - started
    return {
        "mode": "adaptive" if mode == "adaptive" else f"fixed_{weight:g}",
        "comparison_group": "exact_temperature_1",
        "target_calls": calls,
        "target_expected_accepted_length": accepted / calls,
        "target_acceptance_rate": accepted / proposed,
        "proposed_tokens": proposed,
        "mean_candidate_length": proposed / calls,
        "committed_tokens": len(tokens),
        "committed_per_target_call": len(tokens) / calls,
        "intermediate_mean_accepted_per_draft_block": intermediate_accepted / blocks,
        "mean_intermediate_draft_blocks_per_target_cycle": blocks / calls,
        "intermediate_verifier_calls": backend.verifier_calls,
        "secondary_draft_calls": backend.proposal_calls,
        "decode_latency_ms": elapsed * 1000,
        "tokens_per_second": len(tokens) / elapsed,
        "proposal_host_wall_ms": proposal_ms,
        "target_sampling_and_feedback_host_wall_ms": target_ms,
        "intermediate_sampling_host_wall_ms": sampling_ms,
        "peak_retained_ledger_bytes": probability_peak,
        "lambda_counts": dict(sorted(lambda_counts.items())),
        "token_frequencies": modules.toy.normalized_frequencies(tokens, models.vocab_size),
        "pair_frequencies": modules.toy.sequence_frequencies(tokens),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-tokens", type=int, default=15)
    parser.add_argument("--draft-tokens", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-generated-tokens", type=int, help="Hard cap on candidates across all strict rounds")
    parser.add_argument("--tokens", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.primary_tokens <= 15 or not 1 <= args.draft_tokens <= 15 or args.rounds < 1 or args.tokens < 2:
        parser.error("Draft widths must be in [1,15], rounds >= 1 and tokens >= 2.")
    if args.max_generated_tokens is not None and args.max_generated_tokens < 1:
        parser.error("max-generated-tokens must be a positive integer")
    torch.set_num_threads(1)
    modules = load_modules()
    models = modules.toy.ToyModels()
    capacity = modules.config.candidate_capacity(
        args.primary_tokens,
        SimpleNamespace(
            num_rounds=args.rounds,
            num_speculative_tokens=args.draft_tokens,
            max_generated_tokens=args.max_generated_tokens,
        ),
        include_bonus=False,
    )
    results = [modules.toy.run_reference(models, modules.proposal_math, args.tokens, args.seed)]
    for mode, weight in [("fixed", value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)] + [("adaptive", 0.5)]:
        result = run_case(modules, models, args, capacity, mode, weight)
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    report = {
        "scope": "Production ProposalPipeline with CPU toy model substitutes; NOT Qwen/DFlash/NPU performance",
        "primary_draft_tokens": args.primary_tokens,
        "intermediate_draft_tokens": args.draft_tokens,
        "intermediate_num_rounds": args.rounds,
        "max_generated_tokens": args.max_generated_tokens,
        "candidate_capacity": capacity,
        "round_semantics": "One block per round; primary is round one; first correction ends its round.",
        "tokens": args.tokens,
        "seed": args.seed,
        "feedback_min_observations": 4,
        "feedback_enabled_for_all_cases": True,
        "rng_note": "Uses production request-local RNGs; differs from the original arithmetic-only toy benchmark.",
        "memory_scope": "Probability ledger only; excludes Target tensors, temporary tensors, KV and model weights.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
