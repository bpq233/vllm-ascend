# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Resident KV capacity stays independent of compute batch and graph inputs."""

from contextlib import nullcontext
from copy import copy, deepcopy
from math import ceil
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from test_intermediate import modules as modules
from test_intermediate_backend import backend as backend


@pytest.mark.parametrize("capacity", [None, 2, 6])
def test_capacity_configuration_preserves_compute_size(backend, modules, capacity):
    obj, _, _ = backend
    config, _ = modules
    options = config.IntermediateConfig.from_dict(
        dict(
            verifier={"model": "v"},
            drafter={"model": "d"},
            max_num_seqs=2,
            max_model_len=8,
            num_speculative_tokens=2,
            cache_max_num_seqs=capacity,
        )
    )
    namespace = obj.__init__.__globals__
    namespace.update(
        copy=copy,
        deepcopy=deepcopy,
        logger=Mock(),
        IntermediateGraphState=lambda: NS(context=nullcontext),
        IntermediateKVCache=type(obj.cache),
        CompilationConfig=NS,
        CompilationMode=NS(NONE=0),
        primary_draft_width=config.primary_draft_width,
    )
    parent = NS(
        model_config=NS(enforce_eager=True, max_model_len=8),
        compilation_config=NS(cudagraph_mode=obj.CUDAGraphMode.NONE, custom_ops=[]),
        speculative_config=NS(num_speculative_tokens=2),
        scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=12),
        cache_config=NS(),
        additional_config={},
    )
    obj.__init__(parent, options, torch.device("cpu"))
    assert obj.cache_capacity == obj.cache.capacity == (capacity or 2)
    assert len(obj._context_rows) == obj.cache_capacity
    assert obj.max_num_reqs == obj.vllm_config.scheduler_config.max_num_seqs == 2
    assert obj.max_num_tokens == obj.vllm_config.scheduler_config.max_num_batched_tokens == 12
    assert parent.scheduler_config.max_num_seqs == 16


@pytest.mark.parametrize("capacity", [0, -1, 1, True, 2.0, "6"])
def test_invalid_cache_capacity_is_rejected(modules, capacity):
    config, _ = modules
    with pytest.raises(ValueError, match="cache_max_num_seqs"):
        config.IntermediateConfig.from_dict(
            dict(verifier={"model": "v"}, drafter={"model": "d"}, max_num_seqs=2, cache_max_num_seqs=capacity)
        )


@pytest.fixture
def resident_backend(backend):
    obj, metadata, _ = backend
    obj.cache_capacity = 6
    obj.cache = type(obj.cache)(obj.cache_capacity)
    obj._context_rows = [None] * obj.cache_capacity

    class FullAttentionSpec(NS):
        pass

    specs = {
        "verifier": FullAttentionSpec(block_size=4, page_size_bytes=64),
        "draft": FullAttentionSpec(block_size=8, page_size_bytes=128),
    }

    def block_tables(**kwargs):
        return NS(
            input_block_tables=[
                torch.zeros((kwargs["max_num_reqs"], count * block // kernel), dtype=torch.int32)
                for count, block, kernel in zip(
                    kwargs["max_num_blocks_per_group"], kwargs["block_sizes"], kwargs["kernel_block_sizes"]
                )
            ],
            kernel_block_sizes=kwargs["kernel_block_sizes"],
            slot_mappings=torch.zeros(len(specs), kwargs["max_num_batched_tokens"], dtype=torch.int64),
        )

    def init_attn(cache_config, *args):
        return [[NS(layer_names=g.layer_names)] for g in cache_config.kv_cache_groups], None, [4, 4]

    obj._init_scratch.__globals__.update(
        ceil=ceil,
        FullAttentionSpec=FullAttentionSpec,
        get_kv_cache_spec=lambda cfg: specs,
        KVCacheGroupSpec=lambda names, spec, **kw: NS(layer_names=names, kv_cache_spec=spec, **kw),
        KVCacheConfig=NS,
        KVCacheTensor=lambda size, names: NS(size=size, shared_by=names),
        init_attn_backend=init_attn,
        BlockTables=block_tables,
        init_asecnd_model_state=lambda *args: NS(),
        init_secondary_graphs=Mock(),
        init_kv_cache=Mock(),
    )
    obj.drafter.draft_attn_layer_names = ["draft"]
    obj.drafter.set_attn = Mock()
    obj.vllm_config.compilation_config.static_forward_context = {}
    obj.vllm_config.cache_config = NS(cache_dtype="auto")
    obj._init_scratch()
    return obj, metadata


def test_resident_pages_include_kernel_block_splits(resident_backend):
    obj, _ = resident_backend
    # 6 resident requests * 2 allocator blocks plus reserved block zero.
    assert obj.kv_cache_config.num_blocks == 13
    assert [t.size for t in obj.kv_cache_config.kv_cache_tensors] == [13 * 64, 13 * 128]
    assert [t.shape for t in obj.block_tables.input_block_tables] == [(2, 2), (2, 2)]
    assert [t.shape for t in obj.cache_block_tables] == [(6, 2), (6, 2)]
    assert obj.cache_block_tables[0][-1].tolist() == [11, 12]
    # The draft allocator uses block size 8, split into kernel blocks of 4.
    assert obj.cache_block_tables[1][0].tolist() == [2, 3]
    assert obj.cache_block_tables[1][-1].tolist() == [12, 13]
    for table, cached in zip(obj.block_tables.input_block_tables, obj.cache_block_tables):
        assert torch.equal(table, cached[:2])
        assert table.data_ptr() != cached.data_ptr()


def test_all_microbatches_reuse_resident_kv_and_reorder_pages(resident_backend):
    obj, metadata = resident_backend
    contexts = [[i, i + 10] for i in range(6)]
    ids = [str(i) for i in range(6)]
    list(obj.verify(contexts, [[20]] * 6, req_ids=ids))
    assert len(obj.cache.slots) == 6
    assert all(row is not None for row in obj._context_rows)
    assert obj.forward_tokens == 18
    pointers = [t.data_ptr() for t in obj.block_tables.input_block_tables]
    before = obj.forward_tokens
    # Reverse request order across microbatches, including resident slot 5.
    list(obj.verify(list(reversed(contexts)), [[21]] * 6, req_ids=list(reversed(ids))))
    assert obj.forward_tokens - before == 12  # predictor + draft, prefix reused
    assert obj.reused_tokens == 6
    assert pointers == [t.data_ptr() for t in obj.block_tables.input_block_tables]
    assert obj.block_tables.input_block_tables[0].tolist() == [[3, 4], [1, 2]]
    assert obj.block_tables.input_block_tables[1].tolist() == [[4, 5], [2, 3]]
    assert metadata.call_args.kwargs["slot_mappings"][0, :4].tolist() == [13, 14, 5, 6]
    assert obj.cache.slots["5"] == 5
    # A resident slot beyond the compute table's row count still addresses
    # its own physical pages when selected as the first execution row.
    list(obj.verify([contexts[5], contexts[0]], [[22], [22]], req_ids=["5", "0"]))
    assert obj.block_tables.input_block_tables[0].tolist() == [[11, 12], [1, 2]]
    assert obj.block_tables.input_block_tables[1].tolist() == [[12, 13], [2, 3]]
    assert metadata.call_args.kwargs["slot_mappings"][0, :4].tolist() == [45, 46, 5, 6]


def test_resident_lru_evicts_only_after_capacity_is_exhausted(resident_backend):
    obj, _ = resident_backend
    ids = [str(i) for i in range(6)]
    contexts = [[i, 10] for i in range(6)]
    list(obj.verify(contexts, [[20]] * 6, req_ids=ids))
    list(obj.verify([contexts[0]], [[21]], req_ids=["0"]))
    list(obj.verify([[99]], [[22]], req_ids=["new"]))
    assert "0" in obj.cache.slots and "1" not in obj.cache.slots
    assert obj.cache.slots["new"] == 1
    assert obj.cache.tokens[1] == [99, 22]
    assert obj._context_rows[1][0] == "new"
    obj.cache.retain(["0", "new"])
    assert set(obj.cache.slots) == {"0", "new"}
    assert obj.cache.query_start("1", contexts[1], 1) == 0
