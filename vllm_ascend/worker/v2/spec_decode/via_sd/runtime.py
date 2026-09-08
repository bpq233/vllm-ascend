"""Eager MRV2 hierarchy with independent target KV progress."""

from __future__ import annotations

import copy
import inspect
import time

import numpy as np
import torch

from .coordinator import ViaSdBatchResult, ViaSdExecutionCoordinator, ViaSdTargetDecision
from .routing import ViaSdRoutePlan


class ViaSdRuntime:
    def __init__(self, runner):
        self.runner = runner
        config = runner._via_sd_config()
        self.coordinator = ViaSdExecutionCoordinator(config.accept_ratio, config.escalate_ratio)
        runner._via_sd_coordinator = self.coordinator
        self.pending = None
        self.target_logits = {}
        self.target_features = {}
        self._target_backend = None
        self.stats = {}

    def validate(self, scheduled):
        r = self.runner
        if r.vllm_config.scheduler_config.async_scheduling:
            raise NotImplementedError("VIA-SD requires synchronous scheduling")
        if getattr(r.model_config, "logits_processors", None):
            raise NotImplementedError("VIA-SD does not support custom logits processors")
        if not r.model_config.enforce_eager:
            raise NotImplementedError("VIA-SD currently requires eager execution")
        p = r.vllm_config.parallel_config
        if any(
            getattr(p, name, 1) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        ):
            raise NotImplementedError("VIA-SD requires one device")
        if r.speculator is None or r.via_sd_verifier is None:
            raise RuntimeError("VIA-SD requires initialized qprime and existing drafter")
        if r.lora_config or r.supports_mm_inputs or scheduled.has_structured_output_requests:
            raise NotImplementedError("VIA-SD requires plain text without LoRA or grammar")
        if getattr(r.vllm_config, "kv_transfer_config", None):
            raise NotImplementedError("VIA-SD does not support external KV transfer")
        if r.vllm_config.cache_config.enable_prefix_caching:
            raise NotImplementedError("VIA-SD requires prefix caching disabled")
        if getattr(r.rejection_sampler, "use_block_verification", False):
            raise NotImplementedError("VIA-SD requires tokenwise rejection")
        for req in scheduled.scheduled_new_reqs:
            p = req.sampling_params
            if (
                p.prompt_logprobs is not None
                or p.logprobs is not None
                or p.presence_penalty
                or p.frequency_penalty
                or p.repetition_penalty != 1
                or getattr(p, "logprob_token_ids", None)
                or getattr(p, "thinking_token_budget", None) is not None
                or getattr(p, "bad_words", None)
                or getattr(p, "logits_processors", None)
            ):
                raise NotImplementedError("VIA-SD requires no logprobs or token penalties")
        for group in r.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if type(spec).__name__ != "FullAttentionSpec":
                raise NotImplementedError("VIA-SD catch-up requires full-attention KV")

    @torch.inference_mode()
    def execute(self, scheduled):
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

        r = self.runner
        self.validate(scheduled)
        self.stats = dict(
            qprime_calls=0, qprime_tokens=0, qprime_ms=0.0, target_calls=0, target_tokens=0, target_ms=0.0
        )
        if self.pending is not None:
            raise RuntimeError("Consume pending VIA-SD sample before execute")
        r._discard_via_sd_requests(scheduled)
        self.target_features.clear()
        for req in scheduled.scheduled_new_reqs:
            self.coordinator.discard([req.req_id])
            self.target_logits.pop(req.req_id, None)
            r.via_sd_verifier.discard([req.req_id])
        for rid in set(scheduled.finished_req_ids) | set(scheduled.preempted_req_ids or ()):
            self.target_logits.pop(rid, None)
        r.update_pp_decode_requests()
        r.finish_requests(scheduled)
        r.free_states(scheduled)
        r.add_requests(scheduled)
        r.update_requests(scheduled)
        r.block_tables.apply_staged_writes()
        if not scheduled.total_num_scheduled_tokens:
            return r.kv_connector.no_forward(scheduled)
        desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=scheduled.total_num_scheduled_tokens,
            num_reqs=len(scheduled.num_scheduled_tokens),
        )
        if 'batch_req_state' in inspect.signature(r.prepare_inputs).parameters:
            # Ascend's newer override accepts this upstream argument but does
            # not read it; request state has already been updated above.
            batch = r.prepare_inputs(scheduled, batch_req_state=None, batch_desc=desc)
        else:
            batch = r.prepare_inputs(scheduled, desc)
        tables, slots = r.prepare_attn(batch)
        backend = r.via_sd_verifier.backend
        backend.set_request_block_tables(tables, batch.idx_mapping_np)
        hidden_rows, aux_rows, prefixes, drafts = [], [], [], []
        for row, rid in enumerate(batch.req_ids):
            index = int(batch.idx_mapping_np[row])
            lo, hi = map(int, batch.query_start_loc_np[row : row + 2])
            start = int(batch.positions[lo].item())
            old = r.req_states.all_token_ids.gpu[index, :start].tolist()
            query = batch.input_ids[lo:hi].tolist()
            tokens = old + query
            count = len(scheduled.scheduled_spec_decode_tokens.get(rid, ()))
            prefix = tokens[:-count] if count else tokens
            state = self.coordinator.state_for(rid, prefix)
            begin = min(state.qprime_computed_len, start) if backend.cache_enabled else 0
            hs, aux = self._features(tokens, begin, row)
            hidden_rows.append(hs[start - begin :])
            aux_rows.append([value[start - begin :] for value in aux] if aux else [])
            state.record_qprime_forward(len(tokens))
            drafts.append(query[-count:] if count else [])
            prefixes.append(prefix)
        hidden = torch.cat(hidden_rows)
        aux = [torch.cat([row[i] for row in aux_rows]) for i in range(len(aux_rows[0]))] or None
        metadata = r.model_state.prepare_attn(
            batch, CUDAGraphMode.NONE, tables, slots, r.attn_groups, r.kv_cache_config
        )
        self.pending = (
            batch,
            hidden,
            aux,
            metadata,
            build_slot_mappings_by_layer(slots, r.kv_cache_config),
            prefixes,
            drafts,
            scheduled,
        )
        return None

    def _features(self, tokens, begin, row):
        backend = self.runner.via_sd_verifier.backend
        limit = self.runner.max_num_tokens if backend.cache_enabled else self.runner.max_model_len
        hidden, auxiliary = [], None
        for start in range(begin, len(tokens), limit):
            started = self._clock()
            hs, aux = backend.forward_features([tokens[start : start + limit]], [start], [row])
            self._record("qprime", min(limit, len(tokens) - start), started)
            hidden.append(hs)
            if aux:
                if auxiliary is None:
                    auxiliary = [[] for _ in aux]
                for values, value in zip(auxiliary, aux):
                    values.append(value)
        return torch.cat(hidden), [torch.cat(values) for values in auxiliary] if auxiliary else None

    def _clock(self):
        if not self.runner._via_sd_timing_enabled():
            return None
        torch.npu.synchronize()
        return time.perf_counter()

    def _record(self, stage, count, started):
        self.stats[stage + "_calls"] += 1
        self.stats[stage + "_tokens"] += count
        if started is not None:
            torch.npu.synchronize()
            self.stats[stage + "_ms"] += (time.perf_counter() - started) * 1000

    def _single_batch(self, batch, row, position):
        result = copy.copy(batch)
        device = batch.input_ids.device
        result.req_ids = [batch.req_ids[row]]
        result.num_reqs = result.num_reqs_after_padding = 1
        result.idx_mapping = batch.idx_mapping[row : row + 1]
        result.idx_mapping_np = batch.idx_mapping_np[row : row + 1]
        result.expanded_idx_mapping = result.idx_mapping
        result.expanded_local_pos = torch.zeros(1, dtype=torch.int32, device=device)
        result.positions = torch.tensor([position], dtype=torch.int64, device=device)
        lo, hi = map(int, batch.query_start_loc_np[row : row + 2])
        relative = position - int(batch.positions[lo].item())
        if not 0 <= relative < hi - lo:
            raise ValueError("Sampling position is outside the prepared request")
        result.input_ids = batch.input_ids[lo + relative : lo + relative + 1]
        lo, hi = map(int, batch.query_start_loc_np[row : row + 2])
        local_position = position - int(batch.positions[lo].item())
        result.input_ids = batch.input_ids[lo + local_position : lo + local_position + 1]
        if result.input_ids.numel() != 1:
            raise RuntimeError("Sampler position is outside the scheduled request query")
        result.logits_indices = torch.zeros(1, dtype=torch.int64, device=device)
        result.seq_lens = torch.tensor([position + 1], dtype=torch.int32, device=device)
        result.cu_num_logits_np = np.array([0, 1], dtype=np.int32)
        result.cu_num_logits = torch.tensor([0, 1], dtype=torch.int32, device=device)
        result.num_draft_tokens = 0
        return result

    def _sample(self, logits, batch, row, position):
        output = self.runner.sampler(logits.reshape(1, -1).clone(), self._single_batch(batch, row, position))
        return int(output.sampled_token_ids[0, 0].item())

    def _catchup(self, event):
        r = self.runner
        if self._target_backend is None:
            backend = copy.copy(r.via_sd_verifier.backend)
            excluded = set(r.via_sd_model.attention_layer_names) | set(r.speculator.draft_attn_layer_names)
            backend.groups = []
            for groups in r.attn_groups:
                selected = []
                for group in groups:
                    clone = copy.copy(group)
                    clone.layer_names = [name for name in group.layer_names if name not in excluded]
                    if clone.layer_names:
                        clone.create_metadata_builders(
                            vllm_config=r.vllm_config,
                            device=r.device,
                            kernel_block_size=r.kernel_block_sizes[int(clone.kv_cache_group_id)],
                            num_metadata_builders=max(1, len(group.metadata_builders)),
                        )
                        selected.append(clone)
                backend.groups.append(selected)

            class TargetModel:
                def __call__(self, **kwargs):
                    from vllm_ascend.ops.rotary_embedding import update_cos_sin

                    update_cos_sin(kwargs["positions"])
                    value = r.model(**kwargs)
                    self.last_hidden_states = value[0] if isinstance(value, tuple) else value
                    self.last_aux_hidden_states = value[1] if isinstance(value, tuple) else None
                    return value[0] if isinstance(value, tuple) else value

                def compute_logits(self, hidden):
                    return r.model.compute_logits(hidden)

            backend.model = TargetModel()
            backend.cache_enabled = True
            self._target_backend = backend
        backend = self._target_backend
        qbackend = r.via_sd_verifier.backend
        backend.set_request_block_tables(qbackend._request_block_tables, qbackend._request_state_indices)
        row = self.pending[0].req_ids.index(event.request_id)
        for start in range(event.target_computed_len, event.committed_len, r.max_num_tokens):
            chunk = event.prefix_tokens[start : min(start + r.max_num_tokens, event.committed_len)]
            started = self._clock()
            logits = backend.forward(chunk, start, row)
            self._save_target_features(event.request_id, start, backend)
            self._record("target", len(chunk), started)
            self.target_logits[event.request_id] = logits[-1].clone()
            self.coordinator.states[event.request_id].record_target_forward(start + len(chunk))
        return event.committed_len

    def _save_target_features(self, request_id, start, backend):
        model = getattr(backend, "model", None)
        hidden = getattr(model, "last_hidden_states", None)
        if hidden is None:
            return
        batch = self.pending[0]
        row = batch.req_ids.index(request_id)
        lo, hi = map(int, batch.query_start_loc_np[row : row + 2])
        query_start = int(batch.positions[lo].item())
        begin = max(start, query_start)
        end = min(start + hidden.shape[0], query_start + hi - lo)
        if begin >= end:
            return
        local = slice(begin - start, end - start)
        aux = getattr(model, "last_aux_hidden_states", None)
        self.target_features.setdefault(request_id, []).append(
            (begin, hidden[local].clone(), [value[local].clone() for value in aux] if aux else None)
        )

    def _draft_features(self, batch, hidden, aux):
        # Only replace features inside the valid causal prefix after routing.
        selected_hidden = hidden.clone() if self.target_features else hidden
        selected_aux = [value.clone() for value in aux] if aux and self.target_features else aux
        for row, request_id in enumerate(batch.req_ids):
            state = self.coordinator.states[request_id]
            lo, hi = map(int, batch.query_start_loc_np[row : row + 2])
            query_start = int(batch.positions[lo].item())
            valid_end = min(
                state.target_computed_len, state.qprime_computed_len, int(state.committed_len), query_start + hi - lo
            )
            for start, target_hidden, target_aux in self.target_features.get(request_id, ()):
                begin = max(start, query_start)
                end = min(start + target_hidden.shape[0], valid_end)
                if begin >= end:
                    continue
                source = slice(begin - start, end - start)
                destination = slice(lo + begin - query_start, lo + end - query_start)
                selected_hidden[destination] = target_hidden[source]
                if selected_aux and target_aux:
                    if len(selected_aux) != len(target_aux):
                        raise ValueError("Target and qprime auxiliary feature boundaries differ")
                    for output, value in zip(selected_aux, target_aux):
                        output[destination] = value[source]
        return selected_hidden, selected_aux

    def _verify_target(self, event):
        r = self.runner
        batch = self.pending[0]
        row = batch.req_ids.index(event.request_id)
        logits = self.target_logits[event.request_id]
        device = logits.device
        view = self._single_batch(batch, row, len(event.prefix_tokens) - 1)
        # The extra row invokes the upstream one-token rejection kernel. Its
        # bonus is ignored because qprime owns the next routing decision.
        pos = torch.tensor([len(event.prefix_tokens) - 1, len(event.prefix_tokens)], device=device)
        draft_sampled = torch.tensor([event.prefix_tokens[-1], event.draft_token], device=device)
        local = torch.tensor([0, 1], dtype=torch.int32, device=device)
        cu = torch.tensor([0, 2], dtype=torch.int32, device=device)
        draft_logits = r.speculator.draft_logits
        if draft_logits is not None:
            draft_logits = draft_logits.clone()
            index = int(view.idx_mapping_np[0])
            draft_logits[index, 0] = r.speculator.draft_logits[index, event.position]
        _, sampled, count = r.rejection_sampler._verify(
            torch.stack([logits, logits]),
            draft_logits,
            draft_sampled,
            pos,
            cu,
            view.idx_mapping,
            view.idx_mapping_np,
            view.idx_mapping.repeat(2),
            local,
        )
        return ViaSdTargetDecision(int(count[0].item()) > 1, int(sampled[0, 0].item()), len(event.prefix_tokens))

    @torch.inference_mode()
    def sample(self, grammar_output):
        from vllm.logger import logger
        from vllm.v1.outputs import ModelRunnerOutput

        if self.pending is None:
            return None
        if grammar_output is not None:
            raise NotImplementedError("VIA-SD does not support grammar")
        r = self.runner
        batch, hidden, aux, metadata, slots, prefixes, drafts, scheduled = self.pending
        logits = r.via_sd_model.compute_logits(hidden)
        outputs = []
        row_results = []
        fallback_events = []
        plans = []
        for row, rid in enumerate(batch.req_ids):
            lo, hi = map(int, batch.query_start_loc_np[row : row + 2])
            draft = drafts[row]
            if draft:
                score = logits[hi - len(draft) - 1 : hi - 1]
                result = self.coordinator.run(
                    [rid],
                    [prefixes[row]],
                    [draft],
                    score.unsqueeze(0),
                    batch_rows=[row],
                    qprime_computed_lengths=[len(prefixes[row]) + len(draft)],
                    target_catchup=self._catchup,
                    target_verify=self._verify_target,
                    qprime_sampler=lambda req, pos, values, params: self._sample(
                        values, batch, batch.req_ids.index(req), len(prefixes[batch.req_ids.index(req)]) + pos - 1
                    ),
                )
                row_results.extend(result.results)
                fallback_events.extend(result.compact_fallbacks)
                plans.append(result.plan)
                tokens = list(result.results[0].scheduler_tokens)
                state = self.coordinator.states[rid]
                state.commit(prefixes[row] + tokens, qprime_computed_len=len(prefixes[row]) + len(tokens) - 1)
            else:
                state = self.coordinator.state_for(rid, prefixes[row])
                complete = int(batch.seq_lens[row].item()) >= int(batch.prefill_len_np[row])
                tokens = [self._sample(logits[hi - 1], batch, row, len(prefixes[row]) - 1)] if complete else []
                state.commit(prefixes[row] + tokens, qprime_computed_len=len(prefixes[row]))
            r.via_sd_verifier.cache.truncate(rid, state.qprime_computed_len)
            outputs.append(tokens)
        if row_results:
            plan = ViaSdRoutePlan(
                request_ids=tuple(value for part in plans for value in part.request_ids),
                draft_tokens=tuple(value for part in plans for value in part.draft_tokens),
                scores=tuple(value for part in plans for value in part.scores),
                decisions=tuple(value for part in plans for value in part.decisions),
                fallbacks=tuple(value for part in plans for value in part.fallbacks),
                logits=tuple(value for part in plans for value in part.logits),
                batch_rows=tuple(value for part in plans for value in part.batch_rows),
            )
            r._via_sd_route_plan = plan
            r._via_sd_execution_result = ViaSdBatchResult(
                plan,
                tuple(row_results),
                tuple(fallback_events),
                target_calls=sum(len(result.target_fallback_positions) for result in row_results),
            )
        sampled = torch.full((batch.num_reqs, r.num_speculative_steps + 1), -1, dtype=torch.int64, device=r.device)
        for row, tokens in enumerate(outputs):
            sampled[row, : len(tokens)] = torch.tensor(tokens, device=r.device)
        num_sampled = torch.tensor([len(row) for row in outputs], dtype=torch.int32, device=r.device)
        num_rejected = torch.tensor(
            [len(draft) + 1 - len(tokens) if tokens else 0 for draft, tokens in zip(drafts, outputs)],
            dtype=torch.int32,
            device=r.device,
        )
        r.postprocess_sampled(batch.idx_mapping, sampled, num_sampled, num_rejected, batch.query_start_loc)
        draft_hidden, draft_aux = self._draft_features(batch, hidden, aux)
        proposals = r.speculator.propose(
            batch,
            metadata,
            slots,
            draft_hidden,
            draft_aux,
            num_sampled,
            num_rejected,
            r.req_states.last_sampled_tokens,
            r.req_states.next_prefill_tokens,
            r.sampler.sampling_states.temperature.gpu,
            r.sampler.sampling_states.seeds.gpu,
        )
        r.req_states.draft_tokens[batch.idx_mapping] = proposals
        r.draft_tokens_handler.set_draft_tokens(batch, proposals)
        output = ModelRunnerOutput(
            req_ids=batch.req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(batch.req_ids)},
            sampled_token_ids=outputs,
            prompt_logprobs_dict={},
            kv_connector_output=r.kv_connector.post_forward(scheduled.finished_req_ids),
        )
        if r._via_sd_logging_enabled():
            logger.info(
                "[VIA-SD] requests=%d committed=%s target_computed=%s qprime_computed=%s work=%s",
                batch.num_reqs,
                [len(tokens) for tokens in outputs],
                [self.coordinator.states[rid].target_computed_len for rid in batch.req_ids],
                [self.coordinator.states[rid].qprime_computed_len for rid in batch.req_ids],
                self.stats,
            )
        self.pending = None
        self.target_features.clear()
        if self._target_backend is not None:
            model = getattr(self._target_backend, 'model', None)
            if model is not None:
                model.last_hidden_states = None
                model.last_aux_hidden_states = None
        return output
