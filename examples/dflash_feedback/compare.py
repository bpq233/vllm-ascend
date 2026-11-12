# SPDX-License-Identifier: Apache-2.0
"""Run independent processes; fail immediately on greedy output differences."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="feedback_results")
    parser.add_argument("--baseline-root", help="untouched vllm-ascend checkout; required for pristine-baseline test")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4], choices=[1, 4])
    args = parser.parse_args()
    script = Path(__file__).resolve().parent
    root = Path(args.output).resolve()
    for batch in args.batch_sizes:
        outputs = {}
        modes = (["baseline"] if args.baseline_root else []) + ["target_only", "off", "observe", "calibrate"]
        for mode in modes:
            dest = root / f"bs{batch}" / mode
            env = os.environ.copy()
            checkout = Path(args.baseline_root).resolve() if mode == "baseline" else script.parents[1]
            env["PYTHONPATH"] = os.pathsep.join([str(checkout), str(script)])
            command = [
                sys.executable,
                str(script / "run.py"),
                "--model",
                args.model,
                "--draft",
                args.draft,
                "--prompts",
                str(Path(args.prompts).resolve()),
                "--output",
                str(dest),
                "--mode",
                mode,
                "--batch-size",
                str(batch),
                "--warmup",
                str(args.warmup),
                "--limit",
                str(args.limit),
            ]
            subprocess.run(command, env=env, cwd=checkout, check=True)
            outputs[mode] = {
                r["id"]: r["token_ids"] for r in map(json.loads, (dest / "outputs.jsonl").read_text().splitlines())
            }
            if mode != "target_only" and "target_only" in outputs and outputs[mode] != outputs["target_only"]:
                raise RuntimeError(f"greedy output mismatch: bs={batch} {mode} vs target_only")
        if "baseline" in outputs and outputs["baseline"] != outputs["off"]:
            raise RuntimeError("pristine baseline differs from off")

        # Observe must preserve the entire scheduled acceptance trace too.
        def trace(mode, batch=batch):
            data = json.loads((root / f"bs{batch}" / mode / "result.json").read_text())
            return [(r["proposed"], r["accepted"], r["candidates"]) for r in data["worker_metrics"][0]["rounds"]]

        if trace("off") != trace("observe"):
            raise RuntimeError("observe changed the draft acceptance trace")
    subprocess.run([sys.executable, str(script / "analyze.py"), str(root)], check=True)


if __name__ == "__main__":
    main()
