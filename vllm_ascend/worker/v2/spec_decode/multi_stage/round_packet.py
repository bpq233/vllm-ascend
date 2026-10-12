# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pack the existing decision transfer and keep accepted candidates on device."""

import torch
from vllm.triton_utils import tl, triton

REQUESTS_PER_PACKET_LAUNCH = 4


@triton.jit
def _pack_round_kernel(
    decisions,
    draft0,
    draft1,
    draft2,
    draft3,
    packet,
    candidates,
    counts,
    eos_ids,
    limit0,
    limit1,
    limit2,
    limit3,
    row0,
    row1,
    row2,
    row3,
    BASE_ROW: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    DECISION_STRIDE: tl.constexpr,
    DECISION_COL_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    WIDTH0: tl.constexpr,
    WIDTH1: tl.constexpr,
    WIDTH2: tl.constexpr,
    WIDTH3: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    STRIDE3: tl.constexpr,
    OFFSET0: tl.constexpr,
    OFFSET1: tl.constexpr,
    OFFSET2: tl.constexpr,
    OFFSET3: tl.constexpr,
    NUM_EOS: tl.constexpr,
    EOS_BLOCK: tl.constexpr,
    INITIALIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    steps = tl.arange(0, BLOCK)
    # Older Triton-Ascend binders read .dtype on every non-scalar runtime
    # argument, and cannot index constexpr tuples. All arguments are tensors
    # or scalars, including compile-time widths, strides and packet offsets.
    for row in tl.static_range(NUM_ROWS):
        if request == row:
            accepted = tl.load(decisions + (BASE_ROW + row) * DECISION_STRIDE)
            replacement = tl.load(decisions + (BASE_ROW + row) * DECISION_STRIDE + DECISION_COL_STRIDE)
            if row == 0:
                tokens = tl.load(draft0 + steps * STRIDE0, steps < WIDTH0, other=0)
                output_row = row0
                limit = limit0
                width = WIDTH0
                offset = OFFSET0
            elif row == 1:
                tokens = tl.load(draft1 + steps * STRIDE1, steps < WIDTH1, other=0)
                output_row = row1
                limit = limit1
                width = WIDTH1
                offset = OFFSET1
            elif row == 2:
                tokens = tl.load(draft2 + steps * STRIDE2, steps < WIDTH2, other=0)
                output_row = row2
                limit = limit2
                width = WIDTH2
                offset = OFFSET2
            else:
                tokens = tl.load(draft3 + steps * STRIDE3, steps < WIDTH3, other=0)
                output_row = row3
                limit = limit3
                width = WIDTH3
                offset = OFFSET3
            tl.store(packet + 2 * (BASE_ROW + row), accepted)
            tl.store(packet + 2 * (BASE_ROW + row) + 1, replacement)
            tl.store(packet + offset + steps, tokens, steps < width)
            start = 0 if INITIALIZE else tl.load(counts + output_row)
            room = limit - start
            size = tl.minimum(accepted + 1, tl.minimum(room, CAPACITY - start))
            additions = tl.where(steps < accepted, tokens, replacement)
            is_eos = tl.full((BLOCK,), False, tl.int1)
            if NUM_EOS > 0:
                eos_steps = tl.arange(0, EOS_BLOCK)
                eos = tl.load(eos_ids + eos_steps, eos_steps < NUM_EOS, other=0)
                matches = (additions[:, None] == eos[None, :]) & (eos_steps[None, :] < NUM_EOS)
                is_eos = tl.sum(tl.where(matches, 1, 0), 1) > 0
            first_eos = tl.min(tl.where((steps < size) & is_eos, steps + 1, BLOCK), 0)
            size = tl.minimum(size, first_eos)
            tl.store(candidates + output_row * CAPACITY + start + steps, additions, steps < size)
            tl.store(counts + output_row, start + size)


def pack_round(decisions, drafts, candidates, counts, limits, row_ids, eos_ids, initialize=False):
    """Replace the round's cat packet; append the same accepted prefix on device.

    The CPU controller still reads this packet once for KV identities and
    scheduling. Publication reuses candidates directly, with no candidate H2D.
    Widths depend on bounded microbatch shapes, never token contents. Output
    row IDs and CPU length limits are separate scalar launch arguments, avoiding
    runtime tuple binding and metadata H2D. A launch handles up to four requests;
    larger groups share the packet and issue bounded chunks without an extra D2H.
    """
    widths = tuple(len(t) for t in drafts)
    if candidates.device.type == "cpu":
        packet = torch.cat((decisions.reshape(-1), *drafts))
        for row, index in enumerate(row_ids):
            start = 0 if initialize else int(counts[index])
            accepted, replacement = decisions[row].tolist()
            additions = drafts[row][:accepted].tolist() + [replacement]
            additions = additions[: min(int(limits[index]) - start, candidates.shape[1] - start)]
            for position, token in enumerate(additions):
                if token in eos_ids:
                    additions = additions[: position + 1]
                    break
            candidates[index, start : start + len(additions)] = torch.tensor(additions, dtype=candidates.dtype)
            counts[index] = start + len(additions)
        return packet
    packet = torch.empty(2 * len(drafts) + sum(widths), dtype=torch.int64, device=candidates.device)
    if not isinstance(eos_ids, torch.Tensor):
        eos_ids = torch.tensor(eos_ids, dtype=torch.int64, device=candidates.device)
    elif eos_ids.device != candidates.device:
        eos_ids = eos_ids.to(candidates.device)
    offsets, offset = [], 2 * len(drafts)
    for width in widths:
        offsets.append(offset)
        offset += width
    for begin in range(0, len(drafts), REQUESTS_PER_PACKET_LAUNCH):
        end = min(begin + REQUESTS_PER_PACKET_LAUNCH, len(drafts))
        rows = row_ids[begin:end]
        chunk = list(drafts[begin:end])
        # Unused pointer/scalar arguments are never read by NUM_ROWS' loop.
        chunk.extend([packet] * (REQUESTS_PER_PACKET_LAUNCH - len(chunk)))
        chunk_limits = [int(limits[index]) for index in rows]
        chunk_limits.extend([0] * (REQUESTS_PER_PACKET_LAUNCH - len(rows)))
        chunk_rows = list(rows) + [0] * (REQUESTS_PER_PACKET_LAUNCH - len(rows))
        chunk_widths = list(widths[begin:end]) + [0] * (REQUESTS_PER_PACKET_LAUNCH - len(rows))
        chunk_strides = [t.stride(0) for t in drafts[begin:end]] + [1] * (REQUESTS_PER_PACKET_LAUNCH - len(rows))
        chunk_offsets = list(offsets[begin:end]) + [0] * (REQUESTS_PER_PACKET_LAUNCH - len(rows))
        _pack_round_kernel[(end - begin,)](
            decisions,
            *chunk,
            packet,
            candidates,
            counts,
            eos_ids,
            *chunk_limits,
            *chunk_rows,
            BASE_ROW=begin,
            NUM_ROWS=end - begin,
            DECISION_STRIDE=decisions.stride(0),
            DECISION_COL_STRIDE=decisions.stride(1),
            CAPACITY=candidates.shape[1],
            WIDTH0=chunk_widths[0],
            WIDTH1=chunk_widths[1],
            WIDTH2=chunk_widths[2],
            WIDTH3=chunk_widths[3],
            STRIDE0=chunk_strides[0],
            STRIDE1=chunk_strides[1],
            STRIDE2=chunk_strides[2],
            STRIDE3=chunk_strides[3],
            OFFSET0=chunk_offsets[0],
            OFFSET1=chunk_offsets[1],
            OFFSET2=chunk_offsets[2],
            OFFSET3=chunk_offsets[3],
            NUM_EOS=eos_ids.numel(),
            EOS_BLOCK=triton.next_power_of_2(max(1, eos_ids.numel())),
            INITIALIZE=initialize,
            BLOCK=triton.next_power_of_2(max(widths[begin:end]) + 1),
        )
    return packet
