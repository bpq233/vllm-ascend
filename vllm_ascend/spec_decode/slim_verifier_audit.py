# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Offline single-layer ablations and an independent Qwen replay reference.

Run with an installed vLLM-Ascend environment:
python -m vllm_ascend.spec_decode.slim_verifier_audit --model /path/to/Qwen3-8B

This loads one target, without a draft model, and never changes sampling or
automatically chooses a production layer mask. All TP ranks join every forward.
"""

import argparse
import json
import math
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import torch

from vllm_ascend.spec_decode.slim_verifier import SlimVerifier
from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig


def _linear(module, hidden):
    output = module(hidden)
    return output[0] if isinstance(output, tuple) else output


@torch.inference_mode()
def reference_hidden_states(target, input_ids, keep_layers, trace_positions=None):
    """No copied modules, DecoderLayer.forward, fused residual, or SDPA.

    Reuse only the original projection/norm/RoPE/MLP/embedding/head operations.
    Explicit FP32 grouped attention and explicit residual additions provide a
    separate numerical reference. This is quadratic in history length and is
    intended for SHORT diagnostic inputs, never serving.
    """
    length = input_ids.numel()
    positions = torch.arange(length, device=input_ids.device, dtype=torch.int64)
    future = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).triu(1)
    hidden = target.model.embed_tokens(input_ids)
    trace = {}
    for index in keep_layers:
        layer = target.model.layers[index]
        attn = layer.self_attn
        qkv = _linear(attn.qkv_proj, layer.input_layernorm(hidden))
        q, k, v = qkv.split((attn.q_size, attn.kv_size, attn.kv_size), dim=-1)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q.reshape(length, attn.num_heads, attn.head_dim)).reshape(length, attn.q_size)
            k = attn.k_norm(k.reshape(length, attn.num_kv_heads, attn.head_dim)).reshape(length, attn.kv_size)
        q, k = attn.rotary_emb(positions, q, k)
        # Group query heads instead of expanding K/V as ReplayAttention does.
        q = q.float().reshape(length, attn.num_kv_heads, attn.num_heads // attn.num_kv_heads, attn.head_dim)
        q = q.permute(1, 2, 0, 3)
        k = k.float().reshape(length, attn.num_kv_heads, attn.head_dim).permute(1, 0, 2).unsqueeze(1)
        values = v.float().reshape(length, attn.num_kv_heads, attn.head_dim).permute(1, 0, 2).unsqueeze(1)
        scores = (q @ k.transpose(-1, -2)) * attn.scaling
        output = scores.masked_fill(future, -torch.inf).softmax(-1) @ values
        output = output.permute(2, 0, 1, 3).reshape(length, attn.q_size).to(v.dtype)
        hidden = hidden + _linear(attn.o_proj, output)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        if trace_positions is not None:
            trace[index] = hidden[trace_positions].clone()
    return target.model.norm(hidden), trace


def compare_tensors(actual, reference, rtol, atol):
    """Host synchronization is intentional in this offline diagnostic."""
    actual, reference = actual.float(), reference.float()
    finite = bool((torch.isfinite(actual).all() & torch.isfinite(reference).all()).item())
    if not finite:
        return {"finite": False, "within_tolerance": False, "mean_abs_diff": None, "max_abs_diff": None}
    diff = (actual - reference).abs()
    return {
        "finite": True,
        "within_tolerance": bool(torch.all(diff <= atol + rtol * reference.abs()).item()),
        "mean_abs_diff": diff.mean().item(),
        "max_abs_diff": diff.max().item(),
    }


def compare_distributions(full_logits, slim_logits, threshold):
    full, slim = full_logits.float(), slim_logits.float()
    if not bool((torch.isfinite(full).all() & torch.isfinite(slim).all()).item()):
        return {"finite": False}
    full_logprob, slim_logprob = full.log_softmax(-1), slim.log_softmax(-1)
    full_top1 = full.argmax(-1)
    log_ratio = slim.gather(1, full_top1[:, None]).squeeze(1) - slim.max(-1).values
    return {
        "finite": True,
        "prediction_agreement_rate": full_top1.eq(slim.argmax(-1)).float().mean().item(),
        "mean_kl_full_to_slim": (full_logprob.exp() * (full_logprob - slim_logprob)).sum(-1).mean().item(),
        # This probes support for the FULL MODEL'S top1, not actual drafter
        # tokens or speculative prefix acceptance. Keep its name explicit.
        "full_top1_ratio_acceptance_rate": (log_ratio > math.log(threshold)).float().mean().item(),
    }


@torch.inference_mode()
def audit_target(target, token_batches, context=None, primary=True, positions_per_prompt=16, rtol=0.03, atol=0.1):
    """Compare all layers and EVERY single-layer deletion on identical inputs."""
    if positions_per_prompt <= 0 or not token_batches:
        raise ValueError("Provide nonempty token batches and positive positions_per_prompt")
    if any(not math.isfinite(tolerance) or tolerance < 0 for tolerance in (rtol, atol)):
        raise ValueError("Reference tolerances must be finite and nonnegative")
    context = context or (lambda n: nullcontext())
    total_layers = len(target.model.layers)
    device = target.model.embed_tokens.weight.device
    rows = []
    for prompt_index, tokens in enumerate(token_batches):
        if not tokens or len(tokens) > 256:
            raise ValueError("Audit histories must contain 1 to 256 tokens")
        ids = torch.tensor(tokens, device=device, dtype=torch.int64)
        positions = torch.arange(max(0, ids.numel() - positions_per_prompt), ids.numel(), device=device)
        full_logits = None
        for skipped in (None, *range(total_layers)):
            kept = tuple(i for i in range(total_layers) if i != skipped)
            verifier = SlimVerifier(target, SlimVerifierConfig(kept))
            actual_trace = {}
            handles = []
            if primary:
                for index, layer in zip(kept, verifier.layers):

                    def capture(module, args, output, index=index, trace=actual_trace, positions=positions):
                        h, residual = output
                        trace[index] = (h[positions] + residual[positions]).clone()

                    handles.append(layer.register_forward_hook(capture))
            try:
                with context(ids.numel()):
                    actual_hidden = verifier.hidden_states(ids)
                    actual_logits = target.compute_logits(actual_hidden[positions])
                    ref_hidden, ref_trace = reference_hidden_states(target, ids, kept, positions if primary else None)
                    ref_logits = target.compute_logits(ref_hidden[positions])
            finally:
                for handle in handles:
                    handle.remove()
            if not primary:
                continue
            if actual_logits is None or ref_logits is None or actual_logits.shape[-1] != target.config.vocab_size:
                raise RuntimeError("Audit requires global-vocabulary logits on TP rank zero")
            if skipped is None:
                full_logits = actual_logits.clone()
            trace_checks = {index: compare_tensors(actual_trace[index], ref_trace[index], rtol, atol) for index in kept}
            first_mismatch = next((index for index in kept if not trace_checks[index]["within_tolerance"]), None)
            hidden_check = compare_tensors(actual_hidden[positions], ref_hidden[positions], rtol, atol)
            logits_check = compare_tensors(actual_logits, ref_logits, rtol, atol)
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "history_tokens": len(tokens),
                    "prediction_positions": positions.cpu().tolist(),
                    "skipped_layer": skipped,
                    "keep_layers": list(kept),
                    "hidden_reference_check": hidden_check,
                    "logits_reference_check": logits_check,
                    "first_residual_mismatch_layer": first_mismatch,
                    "layer_reference_checks": trace_checks,
                    "vs_full_replay": compare_distributions(full_logits, actual_logits, 0.5),
                }
            )
    if not primary:
        return None
    failed = [
        {"prompt_index": r["prompt_index"], "skipped_layer": r["skipped_layer"]}
        for r in rows
        if r["first_residual_mismatch_layer"] is not None
        or not r["hidden_reference_check"]["within_tolerance"]
        or not r["logits_reference_check"]["within_tolerance"]
    ]
    ranked = []
    for index in range(total_layers):
        candidates = [r for r in rows if r["skipped_layer"] == index]
        count = sum(len(r["prediction_positions"]) for r in candidates)
        entry = {"skipped_layer": index, "positions_checked": count}
        entry["reference_checks_passed"] = all(
            r["first_residual_mismatch_layer"] is None
            and r["hidden_reference_check"]["within_tolerance"]
            and r["logits_reference_check"]["within_tolerance"]
            for r in candidates
        )
        for name in ("prediction_agreement_rate", "mean_kl_full_to_slim", "full_top1_ratio_acceptance_rate"):
            entry[name] = (
                sum(r["vs_full_replay"][name] * len(r["prediction_positions"]) for r in candidates) / count
                if all(r["vs_full_replay"]["finite"] for r in candidates)
                else None
            )
        ranked.append(entry)
    ranked.sort(key=lambda r: r["prediction_agreement_rate"] if r["prediction_agreement_rate"] is not None else -1)
    return {
        "num_layers": total_layers,
        "torch_version": torch.__version__,
        "device": str(device),
        "dtype": str(target.model.embed_tokens.weight.dtype),
        "baseline": "all_layers_full_history_replay",
        "reference": "original_modules_explicit_residual_fp32_grouped_attention",
        "rtol": rtol,
        "atol": atol,
        "reference_mismatches": failed,
        "single_layer_effects_most_changed_first": ranked,
        "rows": rows,
        "scope": "Short teacher-forced histories; no production mask selection or actual draft acceptance measurement.",
    }


class SlimVerifierAuditWorker:
    def audit_slim_layers(self, token_batches, rtol=0.03, atol=0.1):
        # Worker-only imports: the numerical reference is CPU-testable.
        from vllm.distributed import get_tensor_model_parallel_rank

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
            rtol=rtol,
            atol=atol,
        )


def main():
    # Keep vLLM initialization out of the importable numerical reference.
    from vllm import LLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--prompts-json", type=Path, help="Optional JSON list of calibration texts; no chat template is added"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.1)
    args = parser.parse_args()
    if any(not math.isfinite(tolerance) or tolerance < 0 for tolerance in (args.rtol, args.atol)):
        parser.error("Reference tolerances must be finite and nonnegative")
    prompts = (
        json.loads(args.prompts_json.read_text(encoding="utf-8"))
        if args.prompts_json
        else [
            "The sum of the first five positive integers is 1 + 2 + 3 + 4 + 5 = 15. Explain the next step.",
            "A robot walks into a garden. It sees red flowers and green leaves, then turns toward a small tree.",
        ]
    )
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p for p in prompts):
        parser.error("prompts-json must contain a nonempty list of nonempty strings")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.6,
        worker_extension_cls="vllm_ascend.spec_decode.slim_verifier_audit.SlimVerifierAuditWorker",
    )
    token_batches = [llm.get_tokenizer().encode(p, add_special_tokens=True)[:256] for p in prompts]
    results = llm.collective_rpc(
        "audit_slim_layers", kwargs={"token_batches": token_batches, "rtol": args.rtol, "atol": args.atol}, timeout=3600
    )
    report = next(r for r in results if r is not None)
    report["model"] = args.model
    report["tensor_parallel_size"] = args.tensor_parallel_size
    report["token_batches"] = token_batches
    output = args.output or Path("slim_verifier_audit_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + ".json")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Audit saved to {output}; reference mismatches: {len(report['reference_mismatches'])}")
    print("Single-layer effects (largest Top-1 change first; these are calibration metrics, not draft acceptance):")
    for result in report["single_layer_effects_most_changed_first"]:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
