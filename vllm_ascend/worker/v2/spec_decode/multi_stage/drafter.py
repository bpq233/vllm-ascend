# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Secondary-only entry for context KV already populated by the verifier."""

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator


class IntermediateDFlashSpeculator(AscendDFlashSpeculator):
    def propose_precomputed(self, batch, anchors, ones, zeros, temperature, seeds):
        """Reuse DFlash input preparation, graph dispatch and sampling.

        The backend must have written ALL context KV before calling this entry,
        including on a cache miss. Query KV can overwrite the speculative tail;
        the backend truncates its valid prefix before this proposal.
        Primary/dummy proposals continue to use the inherited implementation.
        """
        self.input_batch = batch
        n = batch.num_reqs
        query_tokens = n * self.num_query_per_req
        self.draft_max_seq_len = min(int(batch.seq_lens_np.max()) + self.num_query_per_req, self.max_model_len)
        cp_args = (
            {}
            if vllm_version_is("0.27.1")
            else dict(
                cp_rank=self.block_tables.cp_rank,
                cp_size=self.block_tables.cp_size,
                cp_interleave=self.block_tables.cp_interleave,
            )
        )
        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            prepare_dflash_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self.context_positions,
                self._context_slot_mappings[i],
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                self.temperature,
                self.seeds,
                batch,
                ones,
                zeros,
                anchors,
                anchors,
                temperature,
                seeds,
                self.block_tables.input_block_tables[gid],
                self.block_tables.kernel_block_sizes[gid],
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                num_query_per_req=self.num_query_per_req,
                num_speculative_steps=self.num_speculative_steps,
                max_num_reqs=self.max_num_reqs,
                max_num_tokens=self.max_num_tokens,
                max_model_len=self.max_model_len,
                sample_from_anchor=self.sample_from_anchor,
                **cp_args,
            )
        desc, dp_tokens = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            n,
            query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=False,
        )
        self._prepare_eplb_forward(query_tokens)
        if desc.cg_mode == CUDAGraphMode.FULL:
            # Ascend's manager builds and updates draft metadata on replay.
            # Upstream propose also builds it before this call, redundantly.
            self.query_cudagraph_manager.run_fullgraph(desc)
        else:
            metadata = self.build_draft_attn_metadatas(desc.num_reqs or n, batch.seq_lens_cpu_upper_bound)[0]
            slots = build_slot_mappings_by_layer(
                self.block_tables.slot_mappings[:, : desc.num_tokens], self.kv_cache_config
            )
            self._generate_draft(n, desc.num_tokens, metadata, slots, dp_tokens, desc.cg_mode)
        return self.draft_tokens[:n]
