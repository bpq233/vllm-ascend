# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

r"""Benchmark original DFlash and all eight sparse strategies on a real NPU.

Every configuration runs in a separate process, so graph capture and private
KV allocations cannot leak between cases. Initialization/capture is measured
separately from warmed inference. No CPU model or synthetic throughput is used.

Example::

    python benchmarks/dflash_sparse_verify.py --model /home/data/Qwen3-8B \
        --draft-model /home/data/Qwen3-8B-DFlash-b16 \
        --layers 12 all --rounds 1 2 4 --batch-sizes 1 4 \
        --graph-modes NONE FULL FULL_DECODE_ONLY --output results/sparse.json
"""

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/offline_inference/dflash_sparse_verify.py"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--model", required=True)
    result.add_argument("--draft-model", required=True)
    result.add_argument("--layers", nargs="+", default=["12"], help="Sparse M values; 'all' tests M=L")
    result.add_argument("--rounds", nargs="+", type=int, default=[4])
    result.add_argument("--draft-lengths", nargs="+", type=int, default=[7])
    result.add_argument("--batch-sizes", nargs="+", type=int, default=[4])
    result.add_argument("--high-thresholds", nargs="+", type=float, default=[0.9])
    result.add_argument("--low-thresholds", nargs="+", type=float, default=[0.5])
    result.add_argument("--modes", nargs="+", choices=("fixed", "confidence"), default=["fixed", "confidence"])
    result.add_argument(
        "--final-methods",
        nargs="+",
        choices=("strict", "topk", "confidence", "accept_all"),
        default=["strict", "topk", "confidence", "accept_all"],
    )
    result.add_argument(
        "--graph-modes",
        nargs="+",
        choices=("NONE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "PIECEWISE"),
        default=["NONE", "FULL"],
    )
    result.add_argument("--max-tokens", type=int, default=128)
    result.add_argument("--max-model-len", type=int, default=4096)
    result.add_argument("--max-num-batched-tokens", type=int, default=4096)
    result.add_argument("--tensor-parallel-size", type=int, default=1)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--warmup-runs", type=int, default=1)
    result.add_argument("--prompts-file", type=Path)
    result.add_argument("--capture-sizes", type=int, nargs="+")
    result.add_argument("--max-capture-size", type=int)
    result.add_argument("--timeout", type=float, default=3600, help="Per-process timeout in seconds")
    result.add_argument("--require-graph-replays", action="store_true")
    result.add_argument("--require-min-final-proposal", type=int, default=0)
    result.add_argument("--continue-on-error", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    return result


def token_agreement(actual: list[list[int]], reference: list[list[int]]) -> dict[str, Any]:
    matched = total = 0
    prefixes = []
    for generated, baseline in itertools.zip_longest(actual, reference, fillvalue=[]):
        matched += sum(a == b for a, b in zip(generated, baseline))
        total += max(len(generated), len(baseline))
        prefix = 0
        for a, b in zip(generated, baseline):
            if a != b:
                break
            prefix += 1
        prefixes.append(prefix)
    return {
        "exact_match": actual == reference,
        "position_match_fraction": matched / total if total else 1.0,
        "matching_prefix_lengths": prefixes,
    }


def run_case(args, case: dict[str, Any], directory: Path, index: int) -> dict[str, Any]:
    stem = f"{index:04d}_{case['graph_mode']}_{case['variant']}_{case.get('final_method', 'none')}"
    output = directory / f"{stem}.json"
    stdout = directory / f"{stem}.stdout.log"
    stderr = directory / f"{stem}.stderr.log"
    command = [
        sys.executable,
        str(EXAMPLE),
        "--model",
        args.model,
        "--draft-model",
        args.draft_model,
        "--output",
        str(output),
    ]
    values = {
        **case,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(args.batch_sizes),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "repetitions": args.repetitions,
        "warmup_runs": args.warmup_runs,
    }
    for key, value in values.items():
        command.extend(("--" + key.replace("_", "-"), str(value)))
    if args.prompts_file:
        command.extend(("--prompts-file", str(args.prompts_file.resolve())))
    if args.capture_sizes:
        command.append("--capture-sizes")
        command.extend(map(str, args.capture_sizes))
    if args.max_capture_size:
        command.extend(("--max-capture-size", str(args.max_capture_size)))
    if case["variant"] in ("fixed", "confidence"):
        if args.require_graph_replays and case["graph_mode"] in ("FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"):
            command.append("--require-graph-replays")
        if args.require_min_final_proposal:
            command.extend(("--require-min-final-proposal", str(args.require_min_final_proposal)))
    print(f"Running {stem}: {case}", flush=True)
    entry: dict[str, Any] = {
        "case": case,
        "command": command,
        "result_file": str(output),
        "stdout_file": str(stdout),
        "stderr_file": str(stderr),
    }
    try:
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            process = subprocess.Popen(
                command,
                cwd=EXAMPLE.parents[2],
                env={**os.environ, "VLLM_USE_V2_MODEL_RUNNER": "1"},
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                # Scope cleanup to the process group started for this one
                # benchmark case, including its vLLM worker descendants.
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                raise
        if returncode != 0:
            entry["error"] = f"Worker exited {returncode}; inspect log files"
        else:
            entry["result"] = json.loads(output.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        entry["error"] = f"Worker timed out after {args.timeout} seconds"
    return entry


def main() -> None:
    args = parser().parse_args()
    if any(k < 1 or k > 15 for k in args.draft_lengths):
        raise ValueError("DFlash block widths must be in [1, 15]; long candidates use multiple rounds")
    if any(n < 1 for n in args.rounds) or any(b < 1 for b in args.batch_sizes):
        raise ValueError("rounds and batch sizes must be positive")
    if any(not 0 <= low < high <= 1 for high, low in itertools.product(args.high_thresholds, args.low_thresholds)):
        raise ValueError("Every threshold combination must satisfy 0 <= low < high <= 1")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.parent / f"{args.output.stem}_cases"
    directory.mkdir(parents=True, exist_ok=True)
    cases = []
    for graph, batch in itertools.product(args.graph_modes, args.batch_sizes):
        cases.append({"graph_mode": graph, "batch_size": batch, "variant": "target"})
        for k in args.draft_lengths:
            cases.append({"graph_mode": graph, "batch_size": batch, "variant": "original", "draft_tokens": k})
            for layers, rounds, mode, final in itertools.product(
                args.layers, args.rounds, args.modes, args.final_methods
            ):
                thresholds = (
                    itertools.product(args.high_thresholds, args.low_thresholds)
                    if mode == "confidence"
                    else [(0.9, 0.5)]
                )
                for high, low in thresholds:
                    cases.append(
                        {
                            "graph_mode": graph,
                            "batch_size": batch,
                            "variant": mode,
                            "final_method": final,
                            "draft_tokens": k,
                            "intermediate_layers": layers,
                            "num_rounds": rounds,
                            "threshold_high": high,
                            "threshold_low": low,
                        }
                    )
    report: dict[str, Any] = {
        "model": args.model,
        "draft_model": args.draft_model,
        "notes": [
            "Only strict final verification and original DFlash are expected to match Target greedy tokens.",
            "Approximate output agreement is a measured quality indicator, not a proof of losslessness.",
            "Raw case JSON and worker logs include actual graph counters, selected layers and candidate lengths.",
            "Missing kernel-level timing or memory evidence remains unavailable; no estimate is substituted.",
        ],
        "cases": [],
    }
    references: dict[tuple[str, int], dict[str, Any]] = {}
    eager: dict[str, list[list[int]]] = {}
    failed = False
    for index, case in enumerate(cases):
        entry = run_case(args, case, directory, index)
        result = entry.get("result")
        if result is None:
            failed = True
        else:
            key = (case["graph_mode"], case["batch_size"])
            actual = result["runs"][0]["token_ids"]
            if case["variant"] == "target":
                references[key] = result
            baseline = references.get(key)
            if baseline:
                entry["target_output_agreement"] = token_agreement(actual, baseline["runs"][0]["token_ids"])
                entry["throughput_relative_to_target"] = (
                    result["median_tokens_per_second"] / baseline["median_tokens_per_second"]
                )
                if result["strict_greedy_equivalence_expected"] and not entry["target_output_agreement"]["exact_match"]:
                    entry["correctness_error"] = "Strict/original tokens differ from the Target baseline"
                    failed = True
            eager_key = json.dumps({key: value for key, value in case.items() if key != "graph_mode"}, sort_keys=True)
            if case["graph_mode"] == "NONE":
                eager[eager_key] = actual
            elif eager_key in eager:
                entry["eager_output_agreement"] = token_agreement(actual, eager[eager_key])
        report["cases"].append(entry)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if "error" in entry and not args.continue_on_error:
            break
    report["failed"] = failed
    report["completed_cases"] = len(report["cases"])
    report["planned_cases"] = len(cases)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved observed results to {args.output}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
