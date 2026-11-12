# SPDX-License-Identifier: Apache-2.0
"""Eager-only experimental feedback. No dependency on vLLM (CPU-testable).

Full logits live on device for at most one outstanding proposal per request;
only top-k unions, normalizers and scalar metrics cross to the host.
"""

import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import torch

from vllm_ascend.dflash_config import DFlashCalibrationConfig as FeedbackConfig


class FeedbackCalibrator:
    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.bias: dict[int, tuple[float, int]] = {}
        self.pending: dict[str, dict] = {}
        self.steps: dict[str, int] = {}
        self.target_forward_calls = 0
        self.output = Path(f"{config.output}.{os.getpid()}.jsonl")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # One object per worker/benchmark process; never load old feedback.
        self.output.write_text("", encoding="utf-8")

    def prune(self, live_ids):
        live = set(live_ids)
        self.pending = {k: v for k, v in self.pending.items() if k in live}
        self.steps = {k: v for k, v in self.steps.items() if k in live}

    def table_bytes(self):
        return sys.getsizeof(self.bias) + sum(
            sys.getsizeof(k) + sys.getsizeof(v) + sum(sys.getsizeof(x) for x in v) for k, v in self.bias.items()
        )

    @torch.inference_mode()
    def propose(self, logits, req_ids, positions):
        """Rows are request-major mask predictions; positions exclude anchor."""
        start = perf_counter()
        if logits.ndim != 3 or positions.shape != logits.shape[:2] or len(req_ids) != logits.shape[0]:
            raise ValueError("proposal shape mismatch")
        raw = logits.detach().clone()
        original = raw.argmax(-1)
        adjusted = raw
        if self.config.mode == "calibrate":
            eligible = [(v, b) for v, (b, n) in self.bias.items() if n >= self.config.min_observations]
            if eligible:
                ids, values = zip(*sorted(eligible))
                index = torch.tensor(ids, device=raw.device, dtype=torch.long)
                if max(ids) >= raw.shape[-1]:
                    raise ValueError("feedback vocabulary changed")
                adjusted = raw.clone()
                adjusted[..., index] += torch.tensor(values, device=raw.device, dtype=raw.dtype) * self.config.strength
        tokens = adjusted.argmax(-1)
        # These transfers include device completion in the measured duration.
        original_cpu, tokens_cpu, pos_cpu = original.cpu().tolist(), tokens.cpu().tolist(), positions.cpu().tolist()
        elapsed = (perf_counter() - start) * 1000
        for i, req in enumerate(req_ids):
            self.pending[req] = dict(
                logits=raw[i],
                positions=pos_cpu[i],
                original=original_cpu[i],
                tokens=tokens_cpu[i],
                adjust_ms=elapsed / len(req_ids),
            )
        self._write(
            dict(
                type="proposal",
                requests=list(req_ids),
                adjust_ms=elapsed,
                proposed=tokens.numel(),
                changed=int((original != tokens).sum().cpu()),
                pending_logits_bytes=sum(
                    p["logits"].numel() * p["logits"].element_size() for p in self.pending.values()
                ),
            )
        )
        return tokens

    def _write(self, event):
        # No persistent file handles, including at engine shutdown.
        with self.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    @torch.inference_mode()
    def verify(self, logits, batch, output, all_token_ids):
        """Called AFTER strict rejection sampling, BEFORE next propose.

        Target row j predicts token at input position[row j]+1. Last row is
        bonus-only. Identity, position and actual scheduled draft IDs must all
        match the stored proposal, including when scheduler truncates a block.
        """
        start = perf_counter()
        self.target_forward_calls += 1
        self.steps = {k: v for k, v in self.steps.items() if k in set(batch.req_ids) | set(self.pending)}
        sampled = output.num_sampled.cpu().tolist()
        rejected = output.num_rejected.cpu().tolist()
        indices = batch.logits_indices.cpu().tolist()
        positions = batch.positions.cpu().tolist()
        inputs = batch.input_ids.cpu().tolist()
        cu = batch.cu_num_logits_np.tolist()
        events = []
        for r, req in enumerate(batch.req_ids):
            step = self.steps.get(req, 0)
            self.steps[req] = step + 1
            lo, hi = int(cu[r]), int(cu[r + 1])
            n = hi - lo - 1
            pending = self.pending.pop(req, None)
            if n <= 0:
                continue
            if pending is None:
                raise RuntimeError(f"missing DFlash proposal for {req}")
            expected_pos = [positions[indices[j]] + 1 for j in range(lo, hi - 1)]
            actual_ids = [inputs[indices[j + 1]] for j in range(lo, hi - 1)]
            if expected_pos != pending["positions"][:n] or actual_ids != pending["tokens"][:n]:
                raise RuntimeError(f"DFlash/Target alignment mismatch for {req}")
            accepted = int(sampled[r]) - 1
            if not 0 <= accepted <= n or int(rejected[r]) != n - accepted:
                raise RuntimeError("unexpected strict verification counts")
            # First rejection still has a valid conditioning prefix.
            valid = min(n, accepted + 1)
            draft = pending["logits"][:valid].float()
            target = logits[lo : lo + valid].float()
            if draft.shape != target.shape:
                raise RuntimeError("draft/target vocabulary mismatch")
            target_ids = target.argmax(-1)
            target_cpu = target_ids.cpu().tolist()
            matches = [actual_ids[i] == target_cpu[i] for i in range(valid)]
            if matches != [True] * accepted + ([False] if accepted < n else []):
                raise RuntimeError("raw greedy logits disagree with strict verifier; unsupported sampling modifiers")
            k = min(self.config.top_k, draft.shape[-1])
            dk, tk = draft.topk(k, -1), target.topk(k, -1)
            # Full-vocabulary normalizers; NEVER softmax the truncated top-k.
            dz, tz = draft.logsumexp(-1), target.logsumexp(-1)
            union = torch.cat((dk.indices, tk.indices), -1)
            dl = draft.gather(-1, union) - dz[:, None]
            tl = target.gather(-1, union) - tz[:, None]
            rank = (draft > draft.gather(-1, target_ids[:, None])).sum(-1) + 1
            union_cpu = union.cpu().tolist()
            packed = torch.stack((dl, tl), -1).cpu().tolist()
            dki, tki = dk.indices.cpu().tolist(), tk.indices.cpu().tolist()
            dkv, tkv = dk.values.cpu().tolist(), tk.values.cpu().tolist()
            ranks = rank.cpu().tolist()
            rows = []
            for i in range(valid):
                pos = expected_pos[i]
                state_idx = int(batch.idx_mapping_np[r])
                # Request history is already committed before this target pass;
                # within-block tokens before first rejection are valid as well.
                context = []
                for p in range(max(0, pos - 2), pos):
                    local = next((j for j in range(lo, hi) if positions[indices[j]] == p), None)
                    context.append(inputs[indices[local]] if local is not None else int(all_token_ids[state_idx, p]))
                seen = set()
                sparse = []
                for v, (dlogp, tlogp) in zip(union_cpu[i], packed[i]):
                    if v in seen:
                        continue
                    seen.add(v)
                    delta = tlogp - dlogp
                    if not math.isfinite(delta):
                        raise RuntimeError("non-finite feedback logits")
                    value, count = self.bias.get(v, (0.0, 0))
                    clipped = max(-self.config.clip_value, min(self.config.clip_value, delta))
                    self.bias[v] = ((1 - self.config.alpha) * value + self.config.alpha * clipped, count + 1)
                    sparse.append([v, dlogp, tlogp])
                rows.append(
                    dict(
                        position=pos,
                        context=context,
                        draft_top_ids=dki[i],
                        draft_top_logits=dkv[i],
                        target_top_ids=tki[i],
                        target_top_logits=tkv[i],
                        draft_argmax=pending["original"][i],
                        candidate=actual_ids[i],
                        target_argmax=target_cpu[i],
                        target_rank_in_draft=ranks[i],
                        accepted=i < accepted,
                        union_logprobs=sparse,
                        # A partial L1 distance on the union; not a full KL estimate.
                        union_probability_l1=sum(abs(math.exp(d) - math.exp(t)) for _, d, t in sparse),
                    )
                )
            events.append(
                dict(
                    type="verification",
                    mode=self.config.mode,
                    request_id=req,
                    decode_step=step,
                    anchor_position=expected_pos[0] - 1,
                    proposed=n,
                    accepted=accepted,
                    acceptance_ratio=accepted / n,
                    first_rejection=accepted if accepted < n else None,
                    rows=rows,
                    changed=sum(a != b for a, b in zip(pending["original"][:n], actual_ids)),
                    adjust_ms=pending["adjust_ms"],
                    target_forward_call=self.target_forward_calls,
                    table_bytes=self.table_bytes(),
                )
            )
        elapsed = (perf_counter() - start) * 1000
        io_start = perf_counter()
        for event in events:
            event["feedback_ms"] = elapsed / len(events)
            self._write(event)
        self._write(
            dict(
                type="target_forward",
                call=self.target_forward_calls,
                requests=list(batch.req_ids),
                feedback_ms=elapsed,
                jsonl_ms=(perf_counter() - io_start) * 1000,
                pending_logits_bytes=sum(
                    p["logits"].numel() * p["logits"].element_size() for p in self.pending.values()
                ),
            )
        )


def sample_with_feedback(runner, hidden_states, batch, grammar_output, legacy_rejection=False):
    """Minimal target dispatch for the explicitly restricted experiment path."""
    if grammar_output is not None or batch.has_structured_output_reqs:
        raise ValueError("DFlash calibration does not support structured outputs")
    if getattr(runner, "batch_sharder", None) is not None:
        raise ValueError("DFlash calibration requires unsharded sampling")
    states = runner.sampler.sampling_states
    if bool((states.temperature.gpu[batch.idx_mapping] != 0).any()):
        raise ValueError("DFlash calibration requires temperature=0")
    states_model = runner.model_state
    sample_hidden = hidden_states[batch.logits_indices]
    compute = getattr(states_model, "compute_sample_logits", runner.model.compute_logits)
    logits = compute(sample_hidden)
    raw = logits.detach().clone()
    if batch.num_draft_tokens == 0 or runner.rejection_sampler is None:
        result = runner.sampler(logits, batch)
    elif legacy_rejection:
        result = runner.rejection_sampler(logits, batch, runner.speculator.draft_logits, None)
    else:
        result = runner.rejection_sampler(logits, batch, runner.speculator.draft_logits)
    calibrator = runner.speculator.feedback_calibrator
    calibrator.prune(runner.req_states.req_id_to_index)
    calibrator.verify(raw, batch, result, runner.req_states.all_token_ids.gpu)
    return result, result.num_sampled, result.num_rejected
