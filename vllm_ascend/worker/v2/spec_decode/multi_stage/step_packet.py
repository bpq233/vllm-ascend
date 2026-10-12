# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pack committed progress and token tails before the adapter's existing D2H."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _pack_step_kernel(
    lengths_ptr,
    computed_ptr,
    tokens_ptr,
    indices_ptr,
    primary_ptr,
    output_ptr,
    lengths_stride,
    computed_stride,
    tokens_row_stride,
    tokens_col_stride,
    indices_stride,
    primary_row_stride,
    primary_col_stride,
    NUM_REQS: tl.constexpr,
    PRIMARY_WIDTH: tl.constexpr,
    HISTORY_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    valid_row = row < NUM_REQS
    index = tl.load(indices_ptr + row * indices_stride, mask=valid_row, other=0).to(tl.int64)
    length = tl.load(lengths_ptr + index * lengths_stride, mask=valid_row, other=0)
    computed = tl.load(computed_ptr + index * computed_stride, mask=valid_row, other=0)
    output_row = output_ptr + row * (2 + PRIMARY_WIDTH + HISTORY_WIDTH)
    tl.store(output_row, length, mask=valid_row)
    tl.store(output_row + 1, computed, mask=valid_row)
    offsets = tl.arange(0, BLOCK_SIZE)
    # Independent mode supplies a zero-width view, whose pointer must not be read.
    if PRIMARY_WIDTH > 0:
        primary = tl.load(
            primary_ptr + row * primary_row_stride + offsets * primary_col_stride,
            mask=valid_row & (offsets < PRIMARY_WIDTH),
            other=0,
        )
        tl.store(output_row + 2 + offsets, primary, mask=valid_row & (offsets < PRIMARY_WIDTH))
    if HISTORY_WIDTH > 0:
        # Match the existing short-history behavior: repeat position zero on
        # the left. The CPU consumer only appends the actual history delta.
        positions = tl.maximum(length - HISTORY_WIDTH + offsets, 0)
        tokens = tl.load(
            tokens_ptr + index * tokens_row_stride + positions * tokens_col_stride,
            mask=valid_row & (offsets < HISTORY_WIDTH),
            other=0,
        )
        tl.store(output_row + 2 + PRIMARY_WIDTH + offsets, tokens, mask=valid_row & (offsets < HISTORY_WIDTH))


@torch.inference_mode()
def pack_step(req_states, input_batch, primary, width, out=None):
    """Return int64 rows ``[total_len, computed, primary..., history_tail...]``.

    Inputs belong to the same device and contain valid live request indices
    and lengths within the token table. ``primary`` is in batch order; request
    state is gathered through ``idx_mapping``. Strided inputs are supported.
    Reuse ``out`` only after its previous consumer has finished (the adapter's
    blocking D2H provides that boundary). This helper adds no synchronization.
    """
    source = req_states.all_token_ids.gpu
    n = input_batch.num_reqs
    if primary.ndim != 2 or primary.shape[0] != n:
        raise ValueError("Primary rows must match the input batch.")
    if not 0 <= width <= source.shape[1]:
        raise ValueError("History width must fit the token table.")
    primary_width = primary.shape[1]
    shape = (n, 2 + primary_width + width)
    if (
        out is None
        or out.shape != shape
        or out.dtype != torch.int64
        or out.device != primary.device
        or not out.is_contiguous()
    ):
        out = torch.empty(shape, dtype=torch.int64, device=primary.device)
    if n == 0:
        return out
    indices = input_batch.idx_mapping[:n]
    lengths = req_states.total_len.gpu
    computed = req_states.num_computed_tokens.gpu
    if primary.device.type != "npu":
        batch_lengths = lengths[indices]
        positions = (batch_lengths[:, None] - width + torch.arange(width, device=primary.device)).clamp(min=0)
        out[:, 0] = batch_lengths
        out[:, 1] = computed[indices]
        out[:, 2 : 2 + primary_width] = primary
        out[:, 2 + primary_width :] = source[indices[:, None], positions.long()]
        return out
    _pack_step_kernel[(n,)](
        lengths,
        computed,
        source,
        indices,
        primary,
        out,
        lengths.stride(0),
        computed.stride(0),
        source.stride(0),
        source.stride(1),
        indices.stride(0),
        primary.stride(0),
        primary.stride(1),
        NUM_REQS=n,
        PRIMARY_WIDTH=primary_width,
        HISTORY_WIDTH=width,
        BLOCK_SIZE=triton.next_power_of_2(max(primary_width, width, 1)),
    )
    return out
