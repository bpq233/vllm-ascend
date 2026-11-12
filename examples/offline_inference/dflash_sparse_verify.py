# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

r"""Run original or sparse multi-round DFlash and save measured NPU evidence.

Example::

    python examples/offline_inference/dflash_sparse_verify.py \
        --model /home/data/Qwen3-8B \
        --draft-model /home/data/Qwen3-8B-DFlash-b16 \
        --variant confidence --final-method strict --graph-mode FULL \
        --num-rounds 4 --draft-tokens 7 --output sparse-result.json

``--intermediate-layers all`` is the M=L correctness reference. The CLI uses
only upstream CompilationConfig options for graph configuration. Approximate
final strategies (topk, confidence, accept_all) can change generated tokens.
"""

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--model", required=True)
    result.add_argument("--draft-model")
    result.add_argument("--variant", choices=("target", "original", "fixed", "confidence"), default="fixed")
    result.add_argument("--final-method", choices=("strict", "topk", "confidence", "accept_all"), default="strict")
    result.add_argument("--intermediate-layers", default="12", help="Positive layer count, or 'all' for M=L")
    result.add_argument("--num-rounds", type=int, default=4)
    result.add_argument("--draft-tokens", type=int, default=7, help="Original DFlash per-round block width K")
    result.add_argument("--threshold-high", type=float, default=0.9)
    result.add_argument("--threshold-low", type=float, default=0.5)
    result.add_argument("--target-threshold", type=float, default=0.1)
    result.add_argument("--topk", type=int, default=2)
    result.add_argument("--feature-source", choices=("sparse", "target"), default="sparse")
    result.add_argument(
        "--graph-mode",
        choices=("default", "NONE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "PIECEWISE"),
        default="default",
        help="Upstream CompilationConfig.cudagraph_mode; default preserves upstream behavior",
    )
    result.add_argument("--capture-sizes", type=int, nargs="+", help="Upstream cudagraph_capture_sizes (total tokens)")
    result.add_argument("--max-capture-size", type=int, help="Upstream max_cudagraph_capture_size")
    result.add_argument("--tensor-parallel-size", type=int, default=1)
    result.add_argument("--max-model-len", type=int, default=4096)
    result.add_argument("--max-num-seqs", type=int, default=16)
    result.add_argument("--max-num-batched-tokens", type=int, default=4096)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--max-tokens", type=int, default=128)
    result.add_argument("--ignore-eos", action="store_true")
    result.add_argument("--prompt", action="append", help="Repeat for several prompts")
    result.add_argument("--prompts-file", type=Path, help="JSON array of strings, or one prompt per line")
    result.add_argument("--warmup-runs", type=int, default=1)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--output", type=Path)
    result.add_argument(
        "--require-graph-replays", action="store_true", help="Fail unless all three models replay graphs"
    )
    result.add_argument(
        "--require-min-final-proposal",
        type=int,
        default=0,
        help="Fail if this workload never verifies this many candidates",
    )
    return result


def read_prompts(args: argparse.Namespace) -> list[str]:
    prompts = args.prompt or []
    if args.prompts_file is not None:
        source = args.prompts_file.read_text(encoding="utf-8")
        if args.prompts_file.suffix.lower() == ".json":
            source_prompts = json.loads(source)
            if not isinstance(source_prompts, list) or not all(isinstance(p, str) for p in source_prompts):
                raise ValueError("--prompts-file JSON must contain an array of strings")
        else:
            source_prompts = [line for line in source.splitlines() if line.strip()]
        prompts = prompts + source_prompts
    if not prompts:
        prompts = [
            "Explain why the sky is blue in simple language.",
            "Write a Python function that returns the Fibonacci sequence.",
            "List the first ten prime numbers and explain what makes a number prime.",
            "The capital of France is",
        ]
    return [prompts[i % len(prompts)] for i in range(args.batch_size)]


def worker_evidence(worker) -> dict[str, Any]:
    """Executed by LLM.collective_rpc in the actual NPU worker process."""
    runner = worker.model_runner
    speculator = getattr(runner, "speculator", None)
    metrics = getattr(speculator, "sparse_metrics", None)
    sparse_model = getattr(speculator, "sparse_model", None)
    return {
        "metrics": metrics.snapshot() if metrics is not None else {},
        "selected_layers": list(getattr(sparse_model, "selected_layers", ())),
        "aux_layer_ids": list(getattr(sparse_model, "aux_layer_ids", ())),
        "effective_cudagraph_mode": str(runner.compilation_config.cudagraph_mode),
        "device": str(runner.device),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Keep runtime imports in this worker entry point: --help and benchmark
    # orchestration do not require loading vLLM, CANN, or a device context.
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    if not torch.npu.is_available():
        raise RuntimeError("This example requires a real Ascend NPU; CPU mocks are not benchmark evidence")
    sparse = args.variant in ("fixed", "confidence")
    if args.variant != "target" and not args.draft_model:
        raise ValueError("--draft-model is required for original and sparse DFlash")
    if args.batch_size < 1 or args.batch_size > args.max_num_seqs:
        raise ValueError("--batch-size must be in [1, --max-num-seqs]")
    if args.repetitions < 1 or args.warmup_runs < 0 or args.max_tokens < 1:
        raise ValueError("repetitions and max-tokens must be positive; warmup-runs must be nonnegative")
    model_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "seed": args.seed,
    }
    compilation: dict[str, Any] = {}
    if args.graph_mode != "default":
        compilation["cudagraph_mode"] = args.graph_mode
        if args.graph_mode == "NONE":
            model_kwargs["enforce_eager"] = True
    if args.capture_sizes is not None:
        compilation["cudagraph_capture_sizes"] = args.capture_sizes
    if args.max_capture_size is not None:
        compilation["max_cudagraph_capture_size"] = args.max_capture_size
    if compilation:
        model_kwargs["compilation_config"] = CompilationConfig(**compilation)
    if args.variant != "target":
        model_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": args.draft_tokens,
        }
    intermediate_layers = None
    if sparse:
        if args.intermediate_layers == "all":
            intermediate_layers = AutoConfig.from_pretrained(args.model).get_text_config().num_hidden_layers
        else:
            intermediate_layers = int(args.intermediate_layers)
        model_kwargs["additional_config"] = {
            "dflash_sparse_verify": {
                "enabled": True,
                "mode": args.variant,
                "num_intermediate_layers": intermediate_layers,
                "num_rounds": args.num_rounds,
                "threshold_high": args.threshold_high,
                "threshold_low": args.threshold_low,
                "feature_source": args.feature_source,
                "enable_metrics": True,
                "final_verification": {
                    "method": args.final_method,
                    "topk": args.topk,
                    "target_threshold": args.target_threshold,
                },
            }
        }
    prompts = read_prompts(args)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=args.ignore_eos)
    initialization_start = time.perf_counter()
    llm = LLM(**model_kwargs)
    initialization_seconds = time.perf_counter() - initialization_start
    for _ in range(args.warmup_runs):
        llm.generate(prompts, sampling, use_tqdm=False)
    runs = []
    for _ in range(args.repetitions):
        # generate() waits for completed requests. The extra driver sync makes
        # this an explicit wall-clock measurement, not kernel submission time.
        torch.npu.synchronize()
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        torch.npu.synchronize()
        seconds = time.perf_counter() - started
        ids = [list(output.outputs[0].token_ids) for output in outputs]
        generated = sum(map(len, ids))
        request_timings = []
        for output in outputs:
            metrics = output.metrics
            first = getattr(metrics, "first_token_time", None)
            last = getattr(metrics, "last_token_time", None)
            tokens = len(output.outputs[0].token_ids)
            request_timings.append(
                (last - first) / (tokens - 1) if first is not None and last is not None and tokens > 1 else None
            )
        runs.append(
            {
                "seconds": seconds,
                "generated_tokens": generated,
                "tokens_per_second": generated / seconds,
                "request_tpot_seconds": request_timings,
                "token_ids": ids,
                "texts": [output.outputs[0].text for output in outputs],
            }
        )
    evidence = llm.collective_rpc(worker_evidence)
    observed_max = max(
        (record["proposed"] for worker in evidence for record in worker["metrics"].get("finals", [])),
        default=0,
    )
    validation_errors = []
    if args.require_min_final_proposal and observed_max < args.require_min_final_proposal:
        validation_errors.append(
            f"Workload exercised at most {observed_max} final candidates; required "
            f"{args.require_min_final_proposal}. Increase generation budget or use a suitable prompt/checkpoint."
        )
    if args.require_graph_replays:
        if not sparse:
            validation_errors.append("--require-graph-replays requires a sparse variant with three-model evidence")
        for worker in evidence:
            counters = worker["metrics"].get("graph_calls", {})
            missing = [name for name in ("dflash", "sparse", "target") if counters.get(f"{name}.replay", 0) < 1]
            if missing:
                validation_errors.append(f"No observed graph replay for {missing}; inspect worker_evidence")
    tpot = [value for run in runs for value in run["request_tpot_seconds"] if value is not None]
    return {
        "variant": args.variant,
        "final_method": args.final_method if sparse else None,
        "strict_greedy_equivalence_expected": args.variant in ("target", "original") or args.final_method == "strict",
        "configuration": {
            **{key: value for key, value in vars(args).items() if key not in ("output", "prompts_file")},
            "num_intermediate_layers": intermediate_layers,
            "max_final_candidates": args.draft_tokens * args.num_rounds
            if sparse
            else args.draft_tokens
            if args.variant == "original"
            else 0,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "npu": torch.npu.get_device_name(0),
        },
        "initialization_seconds_including_capture": initialization_seconds,
        "runs": runs,
        "median_tokens_per_second": statistics.median(run["tokens_per_second"] for run in runs),
        "mean_request_tpot_seconds": statistics.mean(tpot) if tpot else None,
        "observed_max_final_candidates": observed_max,
        "worker_evidence": evidence,
        "validation_errors": validation_errors,
        "deterministic_across_repetitions": all(run["token_ids"] == runs[0]["token_ids"] for run in runs),
        "measurement_notes": [
            "Initialization includes weight loading and graph capture; it is excluded from steady-state runs.",
            "Throughput includes prefill and generation; request TPOT uses available request timestamps.",
            "Graph configuration alone is not proof of replay; inspect worker_evidence.metrics.graph_calls.",
            "Approximate final strategies do not guarantee Target greedy equivalence.",
        ],
    }


def main() -> None:
    args = parser().parse_args()
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    result = run(args)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if result["validation_errors"]:
        raise SystemExit("; ".join(result["validation_errors"]))


if __name__ == "__main__":
    main()
