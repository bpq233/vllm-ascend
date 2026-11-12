# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Select a fixed, uniformly spaced subset without dropping feature layers."""

from collections.abc import Iterable


def _layer_ids(values: Iterable[int], total_layers: int, name: str) -> tuple[int, ...]:
    values = tuple(values)
    if any(type(value) is not int for value in values):
        raise TypeError(f"{name} must contain integers")
    if any(not 0 <= value < total_layers for value in values):
        raise ValueError(f"{name} must be in [0, {total_layers})")
    return values


def required_layers_from_aux_capture(total_layers: int, capture_layers: Iterable[int]) -> tuple[int, ...]:
    """Convert Qwen pre-layer capture indices to required post-layer indices.

    The initialized target's ``aux_hidden_state_layers`` is authoritative.
    Upstream adds one to DFlash checkpoint ``target_layer_ids``. Qwen captures
    the residual-combined stream *before* layer ``idx``; capture ``c > 0`` thus
    requires execution of original layer ``c - 1``. Capture zero is embedding
    and does not require a Transformer layer. Preserve checkpoint feature
    ordering separately; this return value is the selection constraint only.
    """
    captures = tuple(capture_layers)
    if any(type(value) is not int for value in captures):
        raise TypeError("capture_layers must contain integers")
    if len(set(captures)) != len(captures):
        raise ValueError("capture_layers must not contain duplicates")
    if any(not 0 <= value <= total_layers for value in captures):
        raise ValueError(f"capture_layers must be in [0, {total_layers}]")
    return tuple(sorted(value - 1 for value in captures if value > 0))


def select_sparse_layers(total_layers: int, num_layers: int, required_layers: Iterable[int] = ()) -> tuple[int, ...]:
    """Minimize layer spacing while retaining all required original indices.

    Dynamic programming minimizes the sum of squared gaps, including the two
    model boundaries. The squared cost favors balanced intervals rather than
    repeatedly bisecting one interval. Mandatory layers cannot be crossed by
    a transition. Ties choose the lexicographically smaller selection, making
    the result deterministic and independently reproducible.
    """
    if type(total_layers) is not int or type(num_layers) is not int:
        raise TypeError("total_layers and num_layers must be integers")
    if not 1 <= num_layers <= total_layers:
        raise ValueError("require 1 <= num_layers <= total_layers")
    required = set(_layer_ids(required_layers, total_layers, "required_layers"))
    if len(required) > num_layers:
        raise ValueError(f"num_layers={num_layers} is smaller than {len(required)} required feature layers")
    if num_layers == total_layers:
        return tuple(range(total_layers))
    # Prefix counts make testing whether a transition skipped a mandatory
    # layer constant time. Virtual boundaries are -1 and total_layers.
    required_prefix = [0]
    for idx in range(total_layers):
        required_prefix.append(required_prefix[-1] + (idx in required))
    states: dict[int, tuple[int, tuple[int, ...]]] = {-1: (0, ())}
    for count in range(1, num_layers + 1):
        next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
        for current in range(count - 1, total_layers - (num_layers - count)):
            options = []
            for previous, (cost, selected) in states.items():
                if previous >= current:
                    continue
                if required_prefix[current] != required_prefix[previous + 1]:
                    continue
                options.append((cost + (current - previous) ** 2, (*selected, current)))
            if options:
                next_states[current] = min(options)
        states = next_states
    endings = [
        (cost + (total_layers - last) ** 2, selected)
        for last, (cost, selected) in states.items()
        if required_prefix[total_layers] == required_prefix[last + 1]
    ]
    return min(endings)[1]
