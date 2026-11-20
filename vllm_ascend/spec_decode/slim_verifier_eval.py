# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch

from vllm_ascend.spec_decode.slim_verifier import SlimVerifier
from vllm_ascend.spec_decode.slim_verifier_cache import TargetPrefixKV
from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig
from vllm_ascend.spec_decode.slim_verifier_draft import DraftLogitsCapture
from vllm_ascend.spec_decode.slim_verifier_kl import SlimVerifierKLMetrics
from vllm_ascend.spec_decode.slim_verifier_metrics import SlimVerifierMetrics


class SlimVerifierEval:
    """MRV2 shadow hook. Reads executed inputs; never returns sampling decisions.

    By default, read Target KV strictly before the current executed query and
    run only that query through Slim. Full-history replay remains an A/B option.
    """

    def __init__(self, runner):
        # Worker-only imports keep config/forward/metric components CPU-testable.
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.logger import logger

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.utils import enable_sp, lmhead_tp_enable

        vc = runner.vllm_config
        parallel = vc.parallel_config
        for name in (
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            if getattr(parallel, name, 1) != 1:
                raise NotImplementedError(f"Slim Verifier currently requires {name}=1 (ordinary TP is supported)")
        if vc.speculative_config is None:
            raise ValueError("Slim Verifier requires speculative decoding to be configured")
        if getattr(vc.speculative_config, "enable_adaptive_verification", False):
            raise NotImplementedError("Slim Verifier does not yet support adaptive verification's compacted layout")
        if vc.lora_config or enable_sp(vc) or lmhead_tp_enable():
            raise NotImplementedError("Slim Verifier does not yet support LoRA, sequence parallel, or lmhead TP")
        if vc.quant_config is not None or vc.cache_config.cache_dtype != "auto":
            raise NotImplementedError("Slim Verifier currently requires unquantized weights and auto KV cache dtype")
        if getattr(vc.model_config, "enable_prompt_embeds", False):
            raise NotImplementedError("Slim Verifier requires token-ID prompts, not prompt embeddings")
        if getattr(vc.model_config, "enable_sleep_mode", False):
            raise NotImplementedError("Slim Verifier module views do not yet support sleep/wake parameter replacement")
        target = runner.get_model()
        if not hasattr(target, "model") or not hasattr(target.model, "layers"):
            raise NotImplementedError("Slim Verifier requires a supported dense text decoder target")
        self.config = SlimVerifierConfig.from_additional_config(vc.additional_config, len(target.model.layers))
        self.verifier = SlimVerifier(target, self.config)
        self.vocab_size = target.config.vocab_size
        self.primary_rank = get_tensor_model_parallel_rank() == 0
        self.draft_capture = None
        self.metrics = None
        if self.config.metrics_mode == "kl":
            DraftLogitsCapture.validate(getattr(runner, "speculator", None))
            if self.primary_rank:
                self.draft_capture = DraftLogitsCapture(runner.speculator, self.vocab_size)
                self.metrics = SlimVerifierKLMetrics(self.config.log_interval, logger)
        elif self.primary_rank:
            self.metrics = SlimVerifierMetrics(
                self.config.log_interval,
                self.config.top_k,
                logger,
                acceptance_strategy=self.config.acceptance_strategy,
                probability_ratio_threshold=self.config.probability_ratio_threshold,
            )
        self.last_results = []
        self.context = lambda n: set_ascend_forward_context(
            None, vc, num_tokens=n, num_actual_tokens=n, model_instance=target, skip_compiled=True
        )
        if self.primary_rank:
            count = len(target.model.layers)
            kept = self.config.keep_layers
            logger.info("[SlimVerifier] Enabled (kv_source=%s; shadow diagnostics only)", self.config.kv_source)
            logger.info("[SlimVerifier] Metrics mode: %s", self.config.metrics_mode)
            if self.config.metrics_mode == "kl":
                logger.info(
                    "[SlimVerifier] KL(Draft || Slim), KL(Draft || Target), KL(Slim || Target); "
                    "raw logits, temperature=1, vocabulary sum then token mean, units=nats"
                )
                logger.info(
                    "[SlimVerifier] Private FP32 draft-logit capture: %.1f MiB on TP rank 0",
                    self.draft_capture.logits.numel() * self.draft_capture.logits.element_size() / (1024**2),
                )
            else:
                logger.info("[SlimVerifier] Target acceptance: Top-1 Diagnostic")
                logger.info("[SlimVerifier] Slim acceptance strategy: %s", self.config.acceptance_strategy)
            if self.config.metrics_mode == "acceptance" and self.config.acceptance_strategy == "probability_ratio":
                logger.info(
                    "[SlimVerifier] Strict acceptance: q(x)/q(max) > %s",
                    self.config.probability_ratio_threshold,
                )
            logger.info("[SlimVerifier] Target layers: %d", count)
            logger.info("[SlimVerifier] Kept layers: %s", list(kept))
            logger.info("[SlimVerifier] Skipped layers: %s", [i for i in range(count) if i not in kept])
            logger.info("[SlimVerifier] Whole-layer keep ratio: %.4f", len(kept) / count)
            logger.info("[SlimVerifier] Skipped attention branches: %s", list(self.config.skip_attention_layers))
            logger.info("[SlimVerifier] Skipped MLP branches: %s", list(self.config.skip_mlp_layers))
            logger.info(
                "[SlimVerifier] Executed attention branches: %d; MLP branches: %d",
                len(kept) - len(self.config.skip_attention_layers),
                len(kept) - len(self.config.skip_mlp_layers),
            )

    @torch.inference_mode()
    def evaluate(self, runner, hidden_states, batch) -> None:
        self.last_results = []
        if batch.num_draft_tokens == 0 or getattr(batch, "is_dummy", False):
            return
        # The parent sampler has already run. Preserve its CPU/device RNG state
        # even if a future implementation of a reused operator consumes RNG.
        device = hidden_states.device
        devices = (
            [device.index if device.index is not None else torch.npu.current_device()] if device.type == "npu" else []
        )
        with torch.random.fork_rng(devices=devices, device_type=device.type):
            self._evaluate_blocks(runner, hidden_states, batch)

    def _evaluate_blocks(self, runner, hidden_states, batch) -> None:
        # One small host transfer for all actual query starts. CPU counters can
        # be optimistic following rejection and are unsafe for replay slicing.
        starts = batch.query_start_loc_np[: batch.num_reqs]
        start_indices = torch.as_tensor(starts, device=batch.positions.device, dtype=torch.int64)
        prefix_lengths = batch.positions[start_indices].cpu().tolist()
        cu = batch.cu_num_logits_np
        for req in range(batch.num_reqs):
            # Each verification block has n prediction rows plus ONE bonus row.
            n = int(cu[req + 1] - cu[req]) - 1
            if n <= 0:
                continue
            query_start, query_end = map(int, batch.query_start_loc_np[req : req + 2])
            if n >= query_end - query_start:
                raise ValueError("Verification query must contain the conditioning token and all draft tokens")
            prefix_len = prefix_lengths[req]
            indices = batch.logits_indices[int(cu[req]) : int(cu[req]) + n].long()
            positions = batch.positions[indices].long()
            draft_tokens = batch.input_ids[indices + 1].long()
            # The Target has already written this query's K/V. Only read cache
            # positions strictly BEFORE its start; all query K/V must be Slim's.
            query = batch.input_ids[query_start:query_end].long()
            if self.config.kv_source == "target":
                inputs = query
                prediction_positions = positions - prefix_len
                forward_kwargs = {
                    "position_offset": prefix_len,
                    "prefix_kv_provider": TargetPrefixKV(runner, self.verifier.target, req, prefix_len),
                }
            else:
                state_index = int(batch.idx_mapping_np[req])
                committed = runner.req_states.all_token_ids.gpu[state_index, :prefix_len]
                inputs = torch.cat((committed, query)).long()
                prediction_positions = positions
                forward_kwargs = {}
            with self.context(inputs.numel()):
                target_logits = self.verifier.target.compute_logits(hidden_states[indices])
                start_event = end_event = None
                if self.config.profile_latency and self.config.metrics_mode == "acceptance":
                    if hidden_states.device.type != "npu":
                        raise ValueError("profile_latency requires NPU Events on an NPU device")
                    start_event = torch.npu.Event(enable_timing=True)
                    end_event = torch.npu.Event(enable_timing=True)
                    start_event.record()
                slim_logits = self.verifier.logits(inputs, prediction_positions, **forward_kwargs)
                slim_ms = None
                if end_event is not None:
                    end_event.record()
                    end_event.synchronize()
                    slim_ms = start_event.elapsed_time(end_event)
            # compute_logits is the model's global-vocabulary interface. Some
            # platforms gather only to TP rank zero; other ranks still participate
            # in EVERY forward and head collective, in exactly the same order.
            if self.metrics is not None:
                if target_logits is None or slim_logits is None:
                    raise RuntimeError("TP primary rank did not receive global model logits")
                if target_logits.shape != (n, self.vocab_size) or slim_logits.shape != target_logits.shape:
                    raise RuntimeError("Slim Verifier requires complete global-vocabulary logits")
                if self.config.metrics_mode == "kl":
                    state_index = int(batch.idx_mapping_np[req])
                    draft_logits, aligned = self.draft_capture.read(state_index, positions, draft_tokens)
                    self.metrics.record(draft_logits, slim_logits, target_logits, aligned)
                else:
                    result = self.metrics.record(draft_tokens, target_logits, slim_logits, slim_ms)
                    result.request_id = batch.req_ids[req]
                    result.prediction_positions = positions
                    self.last_results.append(result)

    def flush(self) -> None:
        if self.metrics is not None:
            self.metrics.flush()
