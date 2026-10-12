# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pack the existing decision transfer and keep accepted candidates on device."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _pack_round_kernel(
    decisions,
    drafts,
    packet,
    candidates,
    counts,
    limits,
    DECISION_STRIDE: tl.constexpr,
    DECISION_COL_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    row_ids,
    WIDTHS: tl.constexpr,
    STRIDES: tl.constexpr,
    OFFSETS: tl.constexpr,
    EOS_IDS: tl.constexpr,
    INITIALIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    steps = tl.arange(0, BLOCK)
    # Tensor tuples expand to kernel arguments, without a pointer-table upload.
    for row in tl.static_range(len(drafts)):
        if request == row:
            accepted = tl.load(decisions + row * DECISION_STRIDE)
            replacement = tl.load(decisions + row * DECISION_STRIDE + DECISION_COL_STRIDE)
            tokens = tl.load(drafts[row] + steps * STRIDES[row], steps < WIDTHS[row], other=0)
            tl.store(packet + 2 * row, accepted)
            tl.store(packet + 2 * row + 1, replacement)
            tl.store(packet + OFFSETS[row] + steps, tokens, steps < WIDTHS[row])
            output_row = row_ids[row]
            start = 0 if INITIALIZE else tl.load(counts + output_row)
            room = limits[row] - start
            size = tl.minimum(accepted + 1, tl.minimum(room, CAPACITY - start))
            additions = tl.where(steps < accepted, tokens, replacement)
            is_eos = tl.full((BLOCK,), False, tl.int1)
            for eos in tl.static_range(len(EOS_IDS)):
                is_eos = is_eos | (additions == EOS_IDS[eos])
            first_eos = tl.min(tl.where((steps < size) & is_eos, steps + 1, BLOCK), 0)
            size = tl.minimum(size, first_eos)
            tl.store(candidates + output_row * CAPACITY + start + steps, additions, steps < size)
            tl.store(counts + output_row, start + size)


def pack_round(decisions, drafts, candidates, counts, limits, row_ids, eos_ids, initialize=False):
    """Replace the round's cat packet; append the same accepted prefix on device.

    The CPU controller still reads this packet once for KV identities and
    scheduling. Publication reuses candidates directly, with no candidate H2D.
    Widths depend on bounded microbatch shapes, never token contents. Output
    row IDs and CPU length limits are scalar launch arguments, avoiding a
    metadata H2D. Each group's first round initializes its device counts.
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
    offsets, offset = [], 2 * len(drafts)
    for width in widths:
        offsets.append(offset)
        offset += width
    _pack_round_kernel[(len(drafts),)](
        decisions,
        tuple(drafts),
        packet,
        candidates,
        counts,
        tuple(int(limits[index]) for index in row_ids),
        decisions.stride(0),
        decisions.stride(1),
        candidates.shape[1],
        tuple(row_ids),
        widths,
        tuple(t.stride(0) for t in drafts),
        tuple(offsets),
        tuple(eos_ids),
        initialize,
        triton.next_power_of_2(max(widths, default=0) + 1),
    )
    return packet
