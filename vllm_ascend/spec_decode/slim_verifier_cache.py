# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch


def gather_prefix_kv(
    cache,
    block_table: torch.Tensor,
    prefix_length: int,
    *,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    head_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather [prefix, local KV heads, head dim] without modifying paged KV.

    The block table is the executed batch's kernel-block table, whose physical
    block IDs have already been validated/used by the Target attention backend.
    Never gather the current query, even if Target has already cached it.
    """
    if isinstance(cache, torch.Tensor):
        if cache.ndim != 5 or cache.shape[0] != 2:
            raise NotImplementedError("Slim target KV requires a K/V pair or a [2, ...] 5-D cache")
        key, value = cache.unbind(0)
    elif isinstance(cache, (tuple, list)) and len(cache) == 2:
        key, value = cache
    else:
        raise NotImplementedError("Slim target KV requires a bound K/V cache pair")
    expected = (num_kv_heads, block_size, head_dim) if head_first else (block_size, num_kv_heads, head_dim)
    if key.ndim != 4 or key.shape != value.shape or key.shape[1:] != expected:
        raise NotImplementedError("Slim target KV supports only standard NHD/HND/LBHNC caches, not NZ or packed KV")
    if key.dtype not in (torch.float16, torch.bfloat16, torch.float32) or key.dtype != value.dtype:
        raise NotImplementedError("Slim target KV requires matching unquantized K/V dtypes")
    if key.device != value.device or block_table.device != key.device:
        raise ValueError("Target K/V and block table must be on the same device")
    if prefix_length < 0 or block_table.ndim != 1 or prefix_length > block_table.numel() * block_size:
        raise ValueError("Prefix length is outside the request's block table")
    positions = torch.arange(prefix_length, device=key.device, dtype=torch.int64)
    physical_blocks = block_table[positions // block_size].long()
    offsets = positions % block_size
    # Advanced indexing allocates private tensors. No alias can write through
    # to Target, including a prefix ending in the middle of a shared block.
    if head_first:
        return key[physical_blocks, :, offsets, :], value[physical_blocks, :, offsets, :]
    return key[physical_blocks, offsets], value[physical_blocks, offsets]


class TargetPrefixKV:
    """Resolve original layer names and gather each selected layer on demand.

    Construct at evaluation time: caches are bound AFTER model loading, and
    input_block_tables rows follow executed request order, not state-slot order.
    """

    def __init__(self, runner, target, request_index: int, prefix_length: int):
        self.runner = runner
        self.target = target
        self.request_index = request_index
        self.prefix_length = prefix_length
        self.groups = {
            name: index
            for index, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            for name in group.layer_names
        }

    def __call__(self, original_layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self_attention = self.target.model.layers[original_layer_index].self_attn
        attention = self_attention.attn
        if getattr(attention, "kv_sharing_target_layer_name", None):
            raise NotImplementedError("Slim target KV does not support cross-layer KV sharing")
        name = attention.layer_name
        if name not in self.groups:
            raise ValueError(f"Target attention layer {name!r} has no KV cache group")
        group = self.groups[name]
        block_tables = self.runner.block_tables
        return gather_prefix_kv(
            attention.kv_cache,
            block_tables.input_block_tables[group][self.request_index],
            self.prefix_length,
            block_size=block_tables.kernel_block_sizes[group],
            num_kv_heads=self_attention.num_kv_heads,
            head_dim=self_attention.head_dim,
            head_first=attention.impl.use_bnsd_kv_cache,
        )
