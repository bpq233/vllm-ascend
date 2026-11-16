# SPDX-License-Identifier: Apache-2.0
"""Summarize existing feedback JSONL from any user script; no vLLM required.

Only report verifier-observed valid prefixes. In particular, an original
candidate after the first changed position has no validated counterfactual
target distribution. Such comparisons are excluded from correction counts.
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def summarize(events):
    rounds = proposed = accepted = changed = valid = raw_matches = 0
    prefix_changed = changed_after_rejection = 0
    first_change_correct = first_change_wrong = excluded_later_changes = 0
    first_change_at_first_rejection = 0
    target_in_draft_topk = 0
    all_accepted = 0
    rejections = Counter()
    ranks = Counter()
    eligible = []
    observed = []
    for event in events:
        if event["type"] == "proposal":
            if event.get("eligible_tokens") is not None:
                eligible.append(event["eligible_tokens"])
            if event.get("observed_tokens") is not None:
                observed.append(event["observed_tokens"])
        if event["type"] != "verification":
            continue
        rounds += 1
        proposed += event["proposed"]
        accepted += event["accepted"]
        changed += event["changed"]
        first_rejection = event["first_rejection"]
        if first_rejection is None:
            all_accepted += 1
        else:
            rejections[first_rejection] += 1
        first_change_seen = False
        changed_here = 0
        for row in event["rows"]:
            valid += 1
            raw_matches += row["draft_argmax"] == row["target_argmax"]
            target_in_draft_topk += row["target_argmax"] in row["draft_top_ids"]
            ranks[row["target_rank_in_draft"]] += 1
            if row["candidate"] == row["draft_argmax"]:
                continue
            changed_here += 1
            # After a changed candidate, original and calibrated prefixes
            # diverge. Never call later original->target matches "rescues".
            if first_change_seen:
                excluded_later_changes += 1
                continue
            first_change_seen = True
            if row["candidate"] == row["target_argmax"]:
                first_change_correct += 1
            else:
                first_change_wrong += 1
                first_change_at_first_rejection += not row["accepted"]
        prefix_changed += changed_here
        changed_after_rejection += event["changed"] - changed_here
    fraction = lambda count, total: count / total if total else None
    return dict(
        verification_rounds=rounds,
        scheduled_candidates=proposed,
        accepted_draft_tokens=accepted,
        mean_accepted_length=fraction(accepted, rounds),
        acceptance_ratio=fraction(accepted, proposed),
        mean_valid_feedback_positions=fraction(valid, rounds),
        valid_feedback_positions=valid,
        feedback_coverage=fraction(valid, proposed),
        changed_candidates=changed,
        changed_candidate_ratio=fraction(changed, proposed),
        changed_within_valid_prefix=prefix_changed,
        changed_after_first_rejection=changed_after_rejection,
        first_changed_candidate_correct=first_change_correct,
        first_changed_candidate_wrong=first_change_wrong,
        first_changed_candidate_is_first_rejection=first_change_at_first_rejection,
        later_changes_excluded_from_counterfactual_comparison=excluded_later_changes,
        target_in_draft_topk_ratio=fraction(target_in_draft_topk, valid),
        # Descriptive agreement along the ACTUAL calibrated prefix. This is
        # not the acceptance of an independently executed unadjusted baseline.
        raw_argmax_agreement_on_actual_prefix=fraction(raw_matches, valid),
        target_rank_histogram=dict(sorted(ranks.items())),
        first_rejection_histogram=dict(sorted(rejections.items())),
        all_accepted_rounds=all_accepted,
        mean_eligible_tokens=fraction(sum(eligible), len(eligible)),
        max_observed_tokens=max(observed, default=None),
        notes=[
            "Only accepted positions plus first rejection contribute feedback.",
            "First-change correctness is along the verified prefix, not a measured acceptance-length gain.",
            "Baseline comparison and end-to-end throughput still require actual separate runs.",
            "Eligibility fields are unavailable in older logs; missing values remain null.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="JSONL paths or quoted glob patterns; one result per file")
    args = parser.parse_args()
    paths = sorted({Path(match).resolve() for pattern in args.logs for match in glob.glob(pattern)})
    if not paths:
        parser.error("no feedback JSONL files matched")
    for path in paths:

        def events(path=path):
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)

        print(json.dumps(dict(file=str(path), **summarize(events())), indent=2))


if __name__ == "__main__":
    main()
