# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from vllm_ascend.spec_decode.slim_verifier import SlimVerifier
from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig
from vllm_ascend.spec_decode.slim_verifier_metrics import SlimVerifierMetrics
from vllm_ascend.spec_decode.slim_verifier_reference import compare_tensors, reference_hidden_states
from vllm_ascend.spec_decode.slim_verifier_search import config_dict, config_for_actions, search_masks


@dataclass
class CalibrationBlock:
    ids: torch.Tensor
    prefix_length: int
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]]
    target_logits: torch.Tensor | None

    def prefix(self, length):
        return lambda index: tuple(t[:length] for t in self.kv[index])


@torch.inference_mode()
def prepare_blocks(target, token_batches, context, draft_length):
    """Full-weight, all-layer prefill creates Target-history K/V once.

    This offline reference uses SDPA, not serving paged Attention. Captured K/V
    are post Q/K-normalization and RoPE, with original layer/head numbering.
    No production KV, request state, or token-generation process is modified.
    """
    full = SlimVerifier(target, SlimVerifierConfig(tuple(range(len(target.model.layers)))))
    blocks = []
    device = target.model.embed_tokens.weight.device
    for tokens in token_batches:
        if len(tokens) < 2:
            raise ValueError("Each calibration history needs at least two tokens")
        ids = torch.tensor(tokens, dtype=torch.long, device=device)
        prefix_length = max(0, ids.numel() - draft_length - 1)
        captured = {}
        handles = []
        for index, layer in enumerate(full.layers):
            attention = layer.self_attn.attn

            def capture(module, args, i=index, saved=captured):
                saved[i] = tuple(t.reshape(-1, module.num_kv_heads, module.head_dim).clone() for t in args[1:])

            handles.append(attention.register_forward_pre_hook(capture))
        try:
            with context(ids.numel()):
                hidden = full.hidden_states(ids)
                logits = target.compute_logits(hidden[prefix_length:-1])
        finally:
            for handle in handles:
                handle.remove()
        blocks.append(CalibrationBlock(ids, prefix_length, captured, logits.clone() if logits is not None else None))
    return blocks


def _run_block(target, verifier, block, context, mode, validate, primary):
    p, ids = block.prefix_length, block.ids
    if mode == "target_per_token":
        calls = [
            (ids[i : i + 1], torch.zeros(1, dtype=torch.long, device=ids.device), i) for i in range(p, len(ids) - 1)
        ]
    elif mode == "target":
        calls = [(ids[p:], torch.arange(len(ids) - p - 1, device=ids.device), p)]
    elif mode == "replay":
        calls = [(ids, torch.arange(p, len(ids) - 1, device=ids.device), 0)]
    else:
        raise ValueError("Unknown audit KV mode")
    actual, reference = [], []
    for inputs, positions, offset in calls:
        kwargs = {} if mode == "replay" else {"position_offset": offset, "prefix_kv_provider": block.prefix(offset)}
        with context(inputs.numel()):
            logits = verifier.logits(inputs, positions, **kwargs)
            expected = None
            if validate:
                hidden = reference_hidden_states(target, inputs, verifier.config, **kwargs)
                expected = target.compute_logits(hidden[positions])
        if primary:
            if logits is None or (validate and expected is None):
                raise RuntimeError("Primary TP rank did not receive global audit logits")
            actual.append(logits)
            if validate:
                reference.append(expected)
    return (torch.cat(actual), torch.cat(reference) if validate else None) if primary else (None, None)


@torch.inference_mode()
def evaluate_mask(
    target,
    config,
    blocks,
    context,
    *,
    primary=True,
    sync=lambda value: value,
    mode=None,
    validate=True,
    rtol=0.03,
    atol=0.1,
):
    verifier = SlimVerifier(target, config)
    metrics = (
        SlimVerifierMetrics(
            log_interval=2**31 - 1,
            acceptance_strategy=config.acceptance_strategy,
            probability_ratio_threshold=config.probability_ratio_threshold,
        )
        if primary
        else None
    )
    checks = []
    full_checks = []
    is_full = len(config.keep_layers) == len(target.model.layers) and not (
        config.skip_attention_layers or config.skip_mlp_layers
    )
    for block in blocks:
        logits, expected = _run_block(target, verifier, block, context, mode or config.kv_source, validate, primary)
        if primary:
            if block.target_logits is None:
                raise RuntimeError("Missing full-model audit baseline")
            if logits.shape != block.target_logits.shape or logits.shape[-1] != target.config.vocab_size:
                raise RuntimeError("Audit requires complete global-vocabulary logits, not a TP shard")
            metrics.record(block.ids[block.prefix_length + 1 :], block.target_logits, logits)
            if is_full:
                full_checks.append(compare_tensors(logits, block.target_logits, rtol, atol))
            if validate:
                checks.append(compare_tensors(logits, expected, rtol, atol))
    report = None
    if primary:
        report = {
            "config": config_dict(config),
            "kv_mode": mode or config.kv_source,
            "metrics": metrics.snapshot(),
            "reference_checks": checks,
            "reference_checks_passed": all(c["within_tolerance"] for c in checks) if validate else None,
            "full_layer_target_checks": full_checks,
        }
    # Synchronize before search decisions so non-primary TP ranks execute the
    # exact same candidate forwards and LM-head collectives as the primary.
    report = sync(report)
    if report is None:
        raise ValueError("Non-primary audit ranks require a score/report broadcast callback")
    if not math.isfinite(report["metrics"]["mean_kl_target_to_slim"]):
        raise ValueError("Nonfinite audit logits/distribution; no mask recommendation produced")
    return report


@torch.inference_mode()
def audit_target(
    target,
    token_batches,
    context=None,
    primary=True,
    *,
    sync=lambda value: value,
    draft_length=15,
    kv_source="target",
    rtol=0.03,
    atol=0.1,
    acceptance_strategy="probability_ratio",
    probability_ratio_threshold=0.5,
    mask=None,
    ablations=True,
    search_skip_count=0,
    search_scope="layer",
    beam_width=1,
    max_evaluations=256,
    keep_edges=True,
    validation_token_batches=None,
    compare_kv_refresh=False,
    max_total_tokens=4096,
):
    """Calibrate block-wise Target-KV masks, with optional search and held-out data.

    Histories are teacher-forced calibration text, not actual draft-model output.
    Search scores use distribution KL; acceptance metrics remain diagnostics.
    """
    if draft_length < 1 or not token_batches:
        raise ValueError("Provide calibration histories and a positive draft length")
    if any(not math.isfinite(x) or x < 0 for x in (rtol, atol)):
        raise ValueError("Reference tolerances must be finite and nonnegative")
    validation_token_batches = validation_token_batches or []
    if sum(map(len, token_batches + validation_token_batches)) > max_total_tokens:
        raise ValueError("Calibration exceeds max_total_tokens; reduce data or explicitly increase this memory budget")
    if compare_kv_refresh and mask is None and not search_skip_count:
        raise ValueError("KV refresh comparison requires a configured mask or mask search")
    context = context or (lambda n: nullcontext())
    num_layers = len(target.model.layers)
    options = dict(
        kv_source=kv_source,
        acceptance_strategy=acceptance_strategy,
        probability_ratio_threshold=probability_ratio_threshold,
    )
    full_config = config_for_actions(num_layers, (), **options)
    blocks = prepare_blocks(target, token_batches, context, draft_length)
    held_out = prepare_blocks(target, validation_token_batches, context, draft_length)
    common = dict(primary=primary, sync=sync, rtol=rtol, atol=atol)
    evaluated = {}

    def evaluate(config, validate=True):
        key = (config, validate)
        if key not in evaluated:
            evaluated[key] = evaluate_mask(target, config, blocks, context, validate=validate, **common)
        return evaluated[key]

    baseline = evaluate(full_config)
    if not baseline["reference_checks_passed"] or not all(
        c["within_tolerance"] for c in baseline["full_layer_target_checks"]
    ):
        raise RuntimeError("Full-layer audit reference failed; resolve numerical differences before mask search")
    validation_baseline = None
    if held_out:
        validation_baseline = evaluate_mask(target, full_config, held_out, context, **common)
        if not validation_baseline["reference_checks_passed"] or not all(
            c["within_tolerance"] for c in validation_baseline["full_layer_target_checks"]
        ):
            raise RuntimeError("Held-out full-layer control failed; resolve numerical differences before mask search")
    single = []
    if ablations:
        for kind in ("layer", "attention", "mlp"):
            for index in range(num_layers):
                result = evaluate(config_for_actions(num_layers, ((kind, index),), **options))
                single.append({"kind": kind, "layer": index, **result})
    report = {
        "schema_version": 2,
        "num_layers": num_layers,
        "torch_version": torch.__version__,
        "device": str(target.model.embed_tokens.weight.device),
        "dtype": str(target.model.embed_tokens.weight.dtype),
        "kv_source": kv_source,
        "draft_length": draft_length,
        "baseline": "full_weights_all_layers_SDPA_teacher_forced",
        "full_layer_control": baseline,
        "validation_full_layer_control": validation_baseline,
        "rtol": rtol,
        "atol": atol,
        "single_ablation_effects_least_kl_first": sorted(single, key=lambda r: r["metrics"]["mean_kl_target_to_slim"]),
        "validation_histories": len(held_out),
        "selected_masks": {},
        "scope": "Offline calibration; full-model KV computed with SDPA, not serving paged attention. "
        "Text continuations are teacher-forced, not actual drafter proposals. No production mask is applied.",
    }
    selected = {}
    if mask is not None:
        selected["configured"] = SlimVerifierConfig.from_additional_config(
            {"slim_verifier_eval": {**mask, "enabled": True, **options}}, num_layers
        )
    if search_skip_count:

        def objective(actions):
            config = config_for_actions(num_layers, actions, **options)
            cached = evaluated.get((config, True))
            result = cached or evaluate(config, validate=False)
            if result["reference_checks_passed"] is False:
                raise RuntimeError("Candidate failed its independent numerical reference; mask search aborted")
            return result["metrics"]["mean_kl_target_to_slim"]

        search = search_masks(
            num_layers,
            search_skip_count,
            objective,
            scope=search_scope,
            beam_width=beam_width,
            max_evaluations=max_evaluations,
            keep_edges=keep_edges,
        )
        report["search"] = search
        if search["completed"]:
            selected["search_best"] = config_for_actions(num_layers, search["best_actions"], **options)
    extra_checks = []
    for name, config in selected.items():
        result = {"calibration": evaluate(config)}
        if held_out:
            result["validation"] = evaluate_mask(target, config, held_out, context, **common)
            extra_checks.append(result["validation"])
        if compare_kv_refresh:
            result["target_kv_refreshed_each_position"] = evaluate_mask(
                target, config, held_out or blocks, context, mode="target_per_token", **common
            )
            result["refresh_comparison_dataset"] = "validation" if held_out else "calibration"
            extra_checks.append(result["target_kv_refreshed_each_position"])
        passed = all(
            result[key]["reference_checks_passed"]
            for key in ("calibration", "validation", "target_kv_refreshed_each_position")
            if key in result
        )
        result["candidate_additional_config"] = {"slim_verifier_eval": config_dict(config)} if passed else None
        result["held_out_evaluated"] = bool(held_out)
        result["numerical_reference_checks_passed"] = passed
        report["selected_masks"][name] = result
    report["reference_mismatch_count"] = sum(
        r["reference_checks_passed"] is False for r in [*evaluated.values(), *extra_checks]
    )
    return report if primary else None
