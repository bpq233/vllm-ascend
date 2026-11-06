# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""NPU model experiment; each case runs in its own process to release all KV.

Use --help without vLLM installed. Timing includes prefill unless request metrics
provide a separate decode interval. Missing acceptance/timing data stays null.
The independent CPU probability experiment is benchmarks/proposal_sampling_cpu.py.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

CASES = (
    "target_random",
    "strict_greedy",
    "topk",
    "prob_ratio",
    "standard",
    "lambda_025",
    "lambda_05",
    "lambda_075",
    "adaptive",
)
MEASURE_START = "PROPOSAL_BENCH_MEASURE_START"
MEASURE_END = "PROPOSAL_BENCH_MEASURE_END"


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True, help="Target checkpoint path or model ID")
    cli.add_argument("--primary-drafter", required=True, help="DFlash checkpoint matching Target")
    cli.add_argument("--intermediate-model", required=True)
    cli.add_argument("--intermediate-drafter", required=True)
    cli.add_argument("--use-primary-drafter", action="store_true")
    cli.add_argument("--prompts", type=Path, help="JSON list of strings, or one prompt per nonempty text line")
    cli.add_argument("--output-dir", type=Path, default=Path("proposal_model_results"))
    cli.add_argument("--log-dir", type=Path, help="Defaults to OUTPUT_DIR/logs")
    cli.add_argument("--cases", default=",".join(CASES))
    cli.add_argument("--temperature", type=float, default=0.8)
    cli.add_argument("--top-p", type=float, default=1.0)
    cli.add_argument("--top-k", type=int, default=-1)
    cli.add_argument("--seed", type=int, default=17)
    cli.add_argument("--max-tokens", type=positive_int, default=128)
    cli.add_argument("--repetitions", type=positive_int, default=3)
    cli.add_argument("--tp", type=positive_int, default=1)
    cli.add_argument("--rounds", type=positive_int, default=6)
    cli.add_argument("--draft-tokens", type=positive_int, default=4)
    cli.add_argument("--max-num-seqs", type=positive_int, default=4)
    cli.add_argument("--max-model-len", type=positive_int, default=2048)
    cli.add_argument("--max-num-batched-tokens", type=positive_int, default=4096)
    cli.add_argument("--graphs", action="store_true", help="Opt into FULL model graphs; eager is the default")
    cli.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    return cli


def prompts_from_args(args):
    if args.prompts is None:
        return [
            "Explain how a compiler translates a program into machine instructions.",
            "Describe a practical way to compare two probability distributions.",
            "Write a short explanation of the water cycle.",
            "What are the main differences between a stack and a queue?",
        ]
    text = args.prompts.read_text(encoding="utf-8")
    prompts = json.loads(text) if args.prompts.suffix.lower() == ".json" else text.splitlines()
    if not isinstance(prompts, list) or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("Prompts must be a list of strings.")
    prompts = [prompt for prompt in prompts if prompt.strip()]
    if not prompts:
        raise ValueError("At least one nonempty prompt is required.")
    return prompts


def model_kwargs(args, case):
    kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=not args.graphs,
        async_scheduling=False,
    )
    if args.graphs:
        kwargs["compilation_config"] = {"cudagraph_mode": "FULL"}
    if case == "target_random":
        return kwargs
    options = {
        "use_primary_drafter": args.use_primary_drafter,
        "intermediate": {
            "verifier": {"model": args.intermediate_model},
            "drafter": {"model": args.intermediate_drafter},
            "num_rounds": args.rounds,
            "num_speculative_tokens": args.draft_tokens,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
        },
    }
    if args.use_primary_drafter:
        options["primary_num_speculative_tokens"] = args.draft_tokens
    if case in ("strict_greedy", "topk", "prob_ratio"):
        policy = (
            {"method": "prob_ratio", "threshold": 0.5}
            if case == "prob_ratio"
            else {"method": "topk", "top_k": 1 if case == "strict_greedy" else 5}
        )
        options["intermediate"]["verification"] = policy.copy()
        options["final_verification"] = policy.copy()
    else:
        weights = {"standard": 0.0, "lambda_025": 0.25, "lambda_05": 0.5, "lambda_075": 0.75, "adaptive": 0.5}
        options["proposal_sampling"] = {
            "enabled": True,
            "mode": "adaptive" if case == "adaptive" else "fixed",
            "mixture_lambda": weights[case],
            "feedback_enabled": True,
            "debug": True,
        }
    kwargs["speculative_config"] = {
        "model": args.primary_drafter,
        "method": "dflash",
        "num_speculative_tokens": args.draft_tokens,
        "draft_sample_method": "greedy",
        "rejection_sample_method": "standard",
    }
    kwargs["additional_config"] = {"multi_stage_speculative": options}
    return kwargs


def run_worker(args):
    # Import after argparse so help and log analysis work on a CPU-only host.
    from vllm import LLM, SamplingParams

    case = args.worker_case
    temperature = 0.0 if case == "strict_greedy" else args.temperature
    sampling = SamplingParams(
        temperature=temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    prompts = prompts_from_args(args)
    llm = LLM(**model_kwargs(args, case))
    llm.generate(prompts, sampling, use_tqdm=False)
    elapsed, decoded, token_count = [], [], 0
    print(MEASURE_START, flush=True)
    for _ in range(args.repetitions):
        start = perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed.append(perf_counter() - start)
        for request in outputs:
            token_count += sum(len(completion.token_ids) for completion in request.outputs)
            metrics = getattr(request, "metrics", None)
            first = getattr(metrics, "first_token_time", None)
            last = getattr(metrics, "last_token_time", None)
            if first is not None and last is not None:
                decoded.append((last - first) * 1000)
    print(MEASURE_END, flush=True)
    group = "strict_greedy" if case == "strict_greedy" else "approximate" if case in ("topk", "prob_ratio") else "exact"
    result = {
        "case": case,
        "comparison_group": group,
        "sampling": {"temperature": temperature, "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed},
        "model_configuration": model_kwargs(args, case),
        "warmup_batches": 1,
        "generated_tokens": token_count,
        "generation_latency_ms": mean(elapsed) * 1000,
        "decode_latency_ms_per_request": mean(decoded) if decoded else None,
        "tokens_per_second_including_prefill": token_count / sum(elapsed),
        "timing": "host_wall_generation_including_prefill; decode_interval_from_request_metrics_when_available",
        "target_forward_calls": None,
        "npu_peak_memory_bytes": None,
        "note": "Acceptance rows are aggregated from measured-window logs by the parent; no NPU speedup is assumed.",
    }
    (args.output_dir / f"{case}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def collect_logs(log_path, jsonl_path):
    measured, records = False, []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if MEASURE_START in line:
            measured = True
        elif MEASURE_END in line:
            measured = False
        elif measured and "proposal_sampling " in line:
            try:
                value, _ = json.JSONDecoder().raw_decode(line.split("proposal_sampling ", 1)[1])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    jsonl_path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records), encoding="utf-8")
    cycles = {}
    for record in records:
        if record.get("event") != "target_verified" or record.get("target_cycle_id") is None:
            continue
        key = (record.get("request_id"), record["target_cycle_id"])
        cycles[key] = record
    usable = [
        record
        for record in cycles.values()
        if type(record.get("target_accepted_length")) is int and type(record.get("candidate_length")) is int
    ]
    accepted = sum(record["target_accepted_length"] for record in usable)
    proposed = sum(record["candidate_length"] for record in usable)
    oracle_overlaps, chosen_overlaps, oracle_weights = [], [], {}
    for record in records:
        if record.get("event") != "target_verified":
            continue
        grid = record.get("lambda_candidates")
        scores = record.get("lambda_candidate_overlaps")
        distance = record.get("tv_target_mixture")
        if not isinstance(grid, list) or not isinstance(scores, list) or not grid or len(grid) != len(scores):
            continue
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in [*grid, *scores]):
            continue
        if type(distance) not in (int, float) or not math.isfinite(distance):
            continue
        best = max(range(len(grid)), key=scores.__getitem__)
        oracle_overlaps.append(scores[best])
        chosen_overlaps.append(1 - distance)
        label = str(grid[best])
        oracle_weights[label] = oracle_weights.get(label, 0) + 1
    proposals = {}
    for record in records:
        if record.get("event") == "proposal_generated" and record.get("target_cycle_id") is not None:
            proposals[(record.get("request_id"), record["target_cycle_id"])] = record
    proposal_rows = list(proposals.values())
    blocks = sum(record.get("intermediate_blocks", 0) for record in proposal_rows)
    accepted_intermediate = sum(record.get("intermediate_accepted_tokens", 0) for record in proposal_rows)

    def proposal_mean(name):
        values = [record[name] for record in proposal_rows if type(record.get(name)) in (int, float)]
        return mean(values) if values else None

    probability_bytes = [
        record["retained_probability_bytes"]
        for record in proposal_rows
        if type(record.get("retained_probability_bytes")) is int
    ]
    return {
        "experiment_json_records": len(records),
        "target_request_verification_cycles": len(usable) if usable else None,
        "target_mean_accepted_length": accepted / len(usable) if usable else None,
        "target_acceptance_rate": accepted / proposed if proposed else None,
        "mean_sampler_output_tokens_per_cycle": (
            mean(record["submitted_tokens"] for record in usable)
            if usable and all(type(record.get("submitted_tokens")) is int for record in usable)
            else None
        ),
        "experiment_log": str(jsonl_path),
        "offline_oracle_mean_overlap": mean(oracle_overlaps) if oracle_overlaps else None,
        "actual_mixture_mean_overlap": mean(chosen_overlaps) if chosen_overlaps else None,
        "offline_oracle_mean_overlap_gain": (
            mean(oracle - chosen for oracle, chosen in zip(oracle_overlaps, chosen_overlaps))
            if oracle_overlaps
            else None
        ),
        "offline_oracle_lambda_histogram": oracle_weights or None,
        "intermediate_mean_accepted_per_draft_block": accepted_intermediate / blocks if blocks else None,
        "mean_proposal_sampling_host_wall_ms": proposal_mean("sampling_overhead_ms"),
        "mean_proposal_intermediate_verify_host_wall_ms": proposal_mean("intermediate_verify_time_ms"),
        "max_logged_retained_probability_bytes": max(probability_bytes) if probability_bytes else None,
    }


def main():
    args = parser().parse_args()
    if (
        not math.isfinite(args.temperature)
        or args.temperature <= 0
        or not 0 < args.top_p <= 1
        or args.top_k == 0
        or args.top_k < -1
    ):
        raise ValueError("Use temperature > 0, top_p in (0,1], and top_k=-1 or a positive integer.")
    if args.draft_tokens > 15:
        raise ValueError("DFlash per-round draft tokens must be <= 15.")
    cases = args.cases.split(",")
    if not cases or len(set(cases)) != len(cases) or any(case not in CASES for case in cases):
        raise ValueError("Select distinct comma-separated cases: " + ",".join(CASES))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_case:
        run_worker(args)
        return
    log_dir = args.log_dir or args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="1", PYTHONUNBUFFERED="1")
    results = []
    for case in cases:
        log_path = log_dir / f"{case}.log"
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker-case", case]
        with log_path.open("w", encoding="utf-8") as output:
            completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=environment)
        if completed.returncode:
            raise RuntimeError(f"Case {case} failed with exit {completed.returncode}; see {log_path}")
        result_path = args.output_dir / f"{case}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.update(collect_logs(log_path, log_dir / f"{case}.jsonl"))
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
