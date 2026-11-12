# SPDX-License-Identifier: Apache-2.0
"""Analyze completed real runs only. Missing metrics remain blank, never zero-filled."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summaries, histograms, paired = [], {}, []
    for result in sorted(args.root.glob("bs*/*/result.json")):
        data = json.loads(result.read_text())
        rounds = data["worker_metrics"][0]["rounds"]
        events = []
        for path in sorted(result.parent.glob("feedback.*.jsonl")):
            events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        feedback = [e for e in events if e["type"] == "verification"]
        positions = [row for e in feedback for row in e["rows"]]
        proposed = sum(r["proposed"] for r in rounds)
        accepted = sum(r["accepted"] for r in rounds)
        label = f"bs{data['batch_size']}/{data['mode']}"
        histograms[label] = Counter(str(r["accepted"]) if r["accepted"] < r["proposed"] else "all" for r in rounds)
        summaries.append(
            dict(
                label=label,
                mean_accepted=accepted / len(rounds) if rounds else None,
                acceptance_ratio=accepted / proposed if proposed else None,
                top1_agreement=sum(r["draft_argmax"] == r["target_argmax"] for r in positions) / len(positions)
                if positions
                else None,
                mean_target_rank=sum(r["target_rank_in_draft"] for r in positions) / len(positions)
                if positions
                else None,
                union_probability_l1=sum(r["union_probability_l1"] for r in positions) / len(positions)
                if positions
                else None,
                changed_ratio=sum(e["changed"] for e in feedback) / proposed if feedback and proposed else None,
                target_forward_calls=data["worker_metrics"][0]["target_forward_calls"],
                generated_tokens=data["generated_tokens"],
                tokens_per_s=data["tokens_per_s"],
                ms_per_token=data["ms_per_token"],
                adjust_ms=sum(e["adjust_ms"] for e in events if e["type"] == "proposal") if events else None,
                feedback_ms=sum(e["feedback_ms"] for e in events if e["type"] == "target_forward") if events else None,
                jsonl_ms=sum(e["jsonl_ms"] for e in events if e["type"] == "target_forward") if events else None,
                peak_pending_logits_bytes=max((e.get("pending_logits_bytes", 0) for e in events), default=None),
                table_bytes=max((e["table_bytes"] for e in feedback), default=None),
            )
        )
        # Only pair actual online verifications at a shared request + anchor.
        # Later round numbers do not align when acceptance lengths differ.
        if data["mode"] == "calibrate":
            off = result.parent.parent / "observe"
            files = list(off.glob("feedback.*.jsonl"))
            output_map = {
                r["request_id"]: r["id"]
                for r in map(json.loads, (result.parent / "outputs.jsonl").read_text().splitlines())
            }
            observe_map = {
                r["request_id"]: r["id"] for r in map(json.loads, (off / "outputs.jsonl").read_text().splitlines())
            }
            reference = {}
            for path in files:
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    if event["type"] == "verification":
                        reference[(observe_map[event["request_id"]], event["anchor_position"])] = event
            for event in feedback:
                key = output_map[event["request_id"]], event["anchor_position"]
                if key in reference:
                    old = reference[key]
                    paired.append(
                        dict(
                            batch_size=data["batch_size"],
                            id=key[0],
                            anchor=key[1],
                            changed=event["changed"],
                            accepted_delta=event["accepted"] - old["accepted"],
                            proposed=event["proposed"],
                            same_budget=event["proposed"] == old["proposed"],
                        )
                    )
    if not summaries:
        raise SystemExit("No completed runs found; no plots or fabricated results were produced")

    def write_csv(name, rows):
        if rows:
            with (args.root / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    write_csv("summary.csv", summaries)
    write_csv("paired_online.csv", paired)
    (args.root / "rejection_histogram.json").write_text(json.dumps(histograms, indent=2))
    eligible = [p for p in paired if p["changed"] and p["same_budget"]]
    (args.root / "paired_counts.json").write_text(
        json.dumps(
            dict(
                increased=sum(p["accepted_delta"] > 0 for p in eligible),
                decreased=sum(p["accepted_delta"] < 0 for p in eligible),
                unchanged=sum(p["accepted_delta"] == 0 for p in eligible),
                note=(
                    "Online shared-anchor comparisons only; unpaired anchors excluded. "
                    "Observational, not causal attribution."
                ),
            ),
            indent=2,
        )
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, filename in [("mean_accepted", "acceptance.png"), ("tokens_per_s", "throughput.png")]:
        rows = [r for r in summaries if r[metric] is not None]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([r["label"] for r in rows], [r[metric] for r in rows])
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(args.root / filename)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = [str(i) for i in range(7)] + ["all"]
    for label, histogram in histograms.items():
        total = sum(histogram.values())
        if total:
            ax.plot(keys, [histogram[k] / total for k in keys], marker="o", label=label)
    ax.set_xlabel("First rejection index (zero based)")
    ax.set_ylabel("Fraction of verification rounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.root / "first_rejection.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    if paired:
        ax.scatter([p["changed"] for p in paired], [p["accepted_delta"] for p in paired], alpha=0.4)
    else:
        ax.text(0.5, 0.5, "No shared-anchor online pairs", ha="center")
    ax.set_xlabel("Changed candidates")
    ax.set_ylabel("Accepted length delta at shared anchor")
    fig.tight_layout()
    fig.savefig(args.root / "bias_acceptance.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
