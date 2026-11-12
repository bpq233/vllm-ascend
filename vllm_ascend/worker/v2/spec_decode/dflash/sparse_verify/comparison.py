# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pair prefix rejection diagnostics with the later full Target prediction."""

from .decisions import rejection_logits


def compare_rejections(records, logits, candidates, positions, *, bonus_only=False):
    # Only first rejected rows were recorded: their earlier context is the
    # retained prefix. Never compare arbitrary discarded suffix rows, whose
    # teacher-forcing context can differ from the final submitted sequence.
    row_by_position = {position: row for row, position in enumerate(positions.cpu().tolist())}
    token_ids = candidates.cpu().tolist()
    results = []
    for record in records:
        prefix = record["rejection"]
        position = prefix["candidate_position"]
        result = {
            "round": record["round"],
            "position": position,
            "candidate_token_id": prefix["candidate_token_id"],
            "M": prefix,
            "L": None,
        }
        row = row_by_position.get(position)
        if row is None:
            result["status"] = "not_submitted_to_target"
        elif token_ids[row] != prefix["candidate_token_id"]:
            result["status"] = "candidate_changed"
        elif bonus_only:
            result["status"] = "target_candidate_logits_not_computed_accept_all"
        else:
            target = rejection_logits(logits[row], token_ids[row])
            result.update(
                status="compared",
                L=target,
                candidate_logit_M_minus_L=prefix["candidate_logit"] - target["candidate_logit"],
                max_token_match=prefix["max_token_id"] == target["max_token_id"],
            )
        results.append(result)
    return results
