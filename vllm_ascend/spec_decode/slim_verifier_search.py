# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import math

from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig


def config_for_actions(num_layers, actions, **options):
    removed = {i for kind, i in actions if kind == "layer"}
    return SlimVerifierConfig(
        keep_layers=tuple(i for i in range(num_layers) if i not in removed),
        skip_attention_layers=tuple(i for kind, i in actions if kind == "attention"),
        skip_mlp_layers=tuple(i for kind, i in actions if kind == "mlp"),
        **options,
    )


def config_dict(config):
    return {
        "enabled": True,
        "keep_layers": list(config.keep_layers),
        "skip_attention_layers": list(config.skip_attention_layers),
        "skip_mlp_layers": list(config.skip_mlp_layers),
        "kv_source": config.kv_source,
        "acceptance_strategy": config.acceptance_strategy,
        "probability_ratio_threshold": config.probability_ratio_threshold,
    }


def search_masks(
    num_layers, skip_count, evaluate, *, scope="layer", beam_width=1, max_evaluations=256, keep_edges=True
):
    """Budgeted deterministic greedy/beam search, rescoring JOINT masks.

    evaluate(actions) returns token-weighted KL(Target || Slim); lower is better.
    No assumptions of additive single-layer effects, and no production mutation.
    Every TP rank must receive the same score from evaluate before branching.
    """
    if scope not in ("layer", "attention", "mlp", "sublayer"):
        raise ValueError("Unknown mask search scope")
    if beam_width < 1 or max_evaluations < 1 or skip_count < 0:
        raise ValueError("Invalid search budget, beam width or skip count")
    indices = range(1, num_layers - 1) if keep_edges else range(num_layers)
    kinds = ("attention", "mlp") if scope == "sublayer" else (scope,)
    actions = tuple((kind, i) for kind in kinds for i in indices)
    if skip_count > len(actions):
        raise ValueError("Requested skip count exceeds the search space (first/last layers protected by default)")
    scores = {}
    trace = []

    def score(mask):
        value = float(evaluate(mask))
        if not math.isfinite(value):
            raise ValueError("Nonfinite calibration objective; search aborted without recommending a mask")
        scores[mask] = value
        trace.append({"actions": list(mask), "mean_kl_target_to_slim": value})
        return value

    score(())
    beam = [()]
    exhausted = False
    for _ in range(skip_count):
        candidates = sorted(
            {tuple(sorted((*mask, action))) for mask in beam for action in actions if action not in mask}
        )
        evaluated = []
        for mask in candidates:
            if mask not in scores:
                if len(scores) >= max_evaluations:
                    exhausted = True
                    break
                score(mask)
            evaluated.append(mask)
        if not evaluated:
            break
        beam = sorted(evaluated, key=lambda m: (scores[m], m))[:beam_width]
        if exhausted:
            break
    best = beam[0]
    return {
        "scope": scope,
        "requested_skip_count": skip_count,
        "achieved_skip_count": len(best),
        "completed": len(best) == skip_count,
        "budget_exhausted": exhausted,
        "beam_width": beam_width,
        "evaluations": len(scores),
        "keep_first_last": keep_edges,
        "best_actions": list(best),
        "best_mean_kl_target_to_slim": scores[best],
        "trace": trace,
        "objective": "token_weighted_KL_target_to_slim",
        "optimality": "heuristic_not_global_optimum",
    }
