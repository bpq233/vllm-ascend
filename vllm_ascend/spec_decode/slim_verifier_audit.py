# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Offline Target-KV layer/sublayer ablations and budgeted mask search.

python -m vllm_ascend.spec_decode.slim_verifier_audit --model /path/to/Qwen3-8B

Uses one target and teacher-forced calibration texts. Does not change serving
configuration, sampling, production KV, or apply a searched mask automatically.
"""

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from vllm_ascend.spec_decode.slim_verifier_calibration import audit_target


class SlimVerifierAuditWorker:
    def audit_slim_layers(self, token_batches, **options):
        # Worker-only imports keep the reference/search code CPU-testable.
        from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.utils import enable_sp, lmhead_tp_enable

        runner = self.model_runner
        vc = runner.vllm_config
        for name in (
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(vc.parallel_config, name, 1) != 1:
                raise NotImplementedError(f"Audit requires {name}=1")
        if vc.quant_config is not None or vc.lora_config or enable_sp(vc) or lmhead_tp_enable():
            raise NotImplementedError("Audit requires unquantized dense Qwen2/Qwen3 without LoRA/SP/lmhead TP")
        target = runner.get_model()
        return audit_target(
            target,
            token_batches,
            context=lambda n: set_ascend_forward_context(
                None, vc, num_tokens=n, num_actual_tokens=n, model_instance=target, skip_compiled=True
            ),
            primary=get_tensor_model_parallel_rank() == 0,
            sync=lambda result: get_tp_group().broadcast_object(result, src=0),
            **options,
        )


def _read_prompts(path, parser):
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p for p in prompts):
        parser.error(f"{path} must contain a nonempty JSON list of calibration texts")
    return prompts


def main():
    # Initializing a vLLM worker is deliberately outside the importable tests.
    from vllm import LLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompts-json", type=Path, help="Calibration text list; no chat template is added")
    parser.add_argument("--validation-prompts-json", type=Path, help="Separate held-out texts, never used for search")
    parser.add_argument("--mask-json", type=Path, help="slim_verifier_eval options or an additional_config JSON object")
    parser.add_argument("--kv-source", choices=["target", "replay"], default="target")
    parser.add_argument("--draft-length", type=int, default=15)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--search-skip-count", type=int, default=0, help="0 disables search; units follow search-scope")
    parser.add_argument("--search-scope", choices=["layer", "attention", "mlp", "sublayer"], default="layer")
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--max-evaluations", type=int, default=256)
    parser.add_argument("--allow-edge-skips", action="store_true")
    parser.add_argument("--skip-single-ablations", action="store_true")
    parser.add_argument(
        "--compare-kv-refresh", action="store_true", help="Offline diagnostic only, for configured/best mask"
    )
    parser.add_argument("--acceptance-strategy", choices=["top1", "probability_ratio"], default="probability_ratio")
    parser.add_argument("--probability-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.1)
    args = parser.parse_args()
    if any(not math.isfinite(t) or t < 0 for t in (args.rtol, args.atol)):
        parser.error("Reference tolerances must be finite and nonnegative")
    if (
        min(args.draft_length, args.beam_width, args.max_evaluations, args.max_total_tokens) < 1
        or args.max_model_len < 2
    ):
        parser.error("Lengths and budgets must be positive; max-model-len must be at least 2")
    if args.search_skip_count < 0:
        parser.error("search-skip-count must be nonnegative")
    if not 0 <= args.probability_ratio_threshold <= 1:
        parser.error("probability-ratio-threshold must be finite and in [0, 1]")
    if args.compare_kv_refresh and not args.mask_json and not args.search_skip_count:
        parser.error("compare-kv-refresh requires mask-json or search-skip-count")
    prompts = (
        _read_prompts(args.prompts_json, parser)
        if args.prompts_json
        else [
            "The sum of the first five positive integers is 1 + 2 + 3 + 4 + 5 = 15. Explain the next step.",
            "A robot walks into a garden. It sees red flowers and green leaves, then turns toward a small tree.",
        ]
    )
    validation = _read_prompts(args.validation_prompts_json, parser) if args.validation_prompts_json else []
    mask = json.loads(args.mask_json.read_text(encoding="utf-8")) if args.mask_json else None
    if mask is not None:
        if not isinstance(mask, dict):
            parser.error("mask-json must contain an options object")
        mask = mask.get("slim_verifier_eval", mask)
        if not isinstance(mask, dict):
            parser.error("slim_verifier_eval must be an options object")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.6,
        worker_extension_cls="vllm_ascend.spec_decode.slim_verifier_audit.SlimVerifierAuditWorker",
    )
    tokenizer = llm.get_tokenizer()
    token_batches = [tokenizer.encode(p, add_special_tokens=True)[: args.max_model_len] for p in prompts]
    validation_tokens = [tokenizer.encode(p, add_special_tokens=True)[: args.max_model_len] for p in validation]
    options = {
        key: getattr(args, key)
        for key in (
            "draft_length",
            "kv_source",
            "rtol",
            "atol",
            "acceptance_strategy",
            "probability_ratio_threshold",
            "search_skip_count",
            "search_scope",
            "beam_width",
            "max_evaluations",
            "compare_kv_refresh",
            "max_total_tokens",
        )
    }
    options.update(
        mask=mask,
        ablations=not args.skip_single_ablations,
        keep_edges=not args.allow_edge_skips,
        validation_token_batches=validation_tokens,
    )
    results = llm.collective_rpc("audit_slim_layers", kwargs={"token_batches": token_batches, **options}, timeout=3600)
    report = next(r for r in results if r is not None)
    report.update(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        token_batches=token_batches,
        validation_token_batches=validation_tokens,
        used_demo_prompts=args.prompts_json is None,
        max_history_tokens=args.max_model_len,
    )
    output = args.output or Path("slim_verifier_audit_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + ".json")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Audit saved to {output}; reference mismatches: {report['reference_mismatch_count']}")
    if "search" in report:
        print(json.dumps({k: v for k, v in report["search"].items() if k != "trace"}, ensure_ascii=False))
    print("Masks are calibration candidates; inspect held-out results before choosing a serving configuration.")


if __name__ == "__main__":
    main()
