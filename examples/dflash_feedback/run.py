# SPDX-License-Identifier: Apache-2.0
"""Single fresh benchmark process. Input JSONL: {id, prompt, dataset}."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["baseline", "target_only", "off", "observe", "calibrate"], required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 4], default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    from vllm import LLM, SamplingParams

    records = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) < args.warmup + args.limit or args.warmup < 1 or args.limit < 1:
        raise ValueError("provide disjoint warmup + evaluation samples; warmup and limit must be positive")
    records = records[: args.warmup + args.limit]
    if len({str(r["id"]) for r in records}) != len(records):
        raise ValueError("prompt IDs must be unique")
    kwargs = dict(
        model=args.model,
        enforce_eager=True,
        max_num_seqs=args.batch_size,
        max_model_len=1024,
        max_num_batched_tokens=4096,
        seed=42,
        async_scheduling=False,
        worker_extension_cls="worker_metrics.FeedbackMetricsExtension",
    )
    if args.mode != "target_only":
        kwargs["speculative_config"] = dict(
            model=args.draft,
            method="dflash",
            num_speculative_tokens=7,
            enforce_eager=True,
            draft_sample_method="greedy",
            rejection_sample_method="standard",
        )
    # The untouched baseline checkout does not know this new config key.
    if args.mode not in ("baseline", "target_only"):
        kwargs["additional_config"] = dict(
            dflash_calib_config=dict(
                mode=args.mode,
                output=str(root / "feedback"),
            )
        )
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=42)

    def generate(rows):
        outputs = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            outputs.extend(llm.generate([r["prompt"] for r in batch], params, use_tqdm=False))
        return outputs

    # Warmup always uses the same disjoint samples in every mode.
    generate(records[: args.warmup])
    llm.collective_rpc("feedback_clear_pending")
    llm.collective_rpc("feedback_start_metrics")
    start = perf_counter()
    outputs = generate(records[args.warmup :])
    elapsed = perf_counter() - start
    metrics = llm.collective_rpc("feedback_get_metrics")
    rows = [
        dict(id=r["id"], dataset=r.get("dataset"), request_id=o.request_id, token_ids=list(o.outputs[0].token_ids))
        for r, o in zip(records[args.warmup :], outputs)
    ]
    tokens = sum(len(r["token_ids"]) for r in rows)
    manifest = vars(args) | dict(
        input_sha256=hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        model_kwargs=kwargs,
        elapsed_s=elapsed,
        generated_tokens=tokens,
        tokens_per_s=tokens / elapsed,
        ms_per_token=elapsed * 1000 / tokens if tokens else None,
        worker_metrics=metrics,
    )
    (root / "result.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "outputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
