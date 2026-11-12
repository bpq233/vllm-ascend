# SPDX-License-Identifier: Apache-2.0
"""Download public prompts once; never execute generated HumanEval programs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-dataset", type=int, default=120)
    args = parser.parse_args()
    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    human = load_dataset("openai/openai_humaneval", split="test")
    if not 1 <= args.per_dataset <= min(len(gsm), len(human)):
        raise ValueError("per-dataset exceeds available samples")
    rows = []
    for i in range(args.per_dataset):
        rows.extend(
            [
                dict(id=f"gsm8k-{i}", dataset="gsm8k", prompt=gsm[i]["question"] + "\nAnswer:"),
                dict(id=f"humaneval-{i}", dataset="humaneval", prompt=human[i]["prompt"]),
            ]
        )
    Path(args.output).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
