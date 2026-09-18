# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Isolate the worker-local verifier's ACL graph capture and replay state."""

from contextlib import contextmanager

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.compilation import acl_graph


class IntermediateGraphState:
    """Own graph parameter buckets without replacing the main runner's buckets.

    ACL attention stores capture handles in module globals. The intermediate
    verifier and secondary draft have independent weights and KV, so sharing
    those handles with the target/primary draft would update the wrong graphs.
    Like the worker's forward context, this scope is used on its execution
    thread. Reentrant entry must retain any buckets initialized by the caller.
    """

    _names = ("_graph_params", "_draft_graph_params", "_draft_graph_prefill_params")

    def __init__(self):
        self._params = dict.fromkeys(self._names)
        self._depth = 0

    @contextmanager
    def context(self):
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        saved = {name: getattr(acl_graph, name) for name in self._names}
        for name, value in self._params.items():
            setattr(acl_graph, name, value)
        self._depth = 1
        try:
            yield
        finally:
            self._params = {name: getattr(acl_graph, name) for name in self._names}
            for name, value in saved.items():
                setattr(acl_graph, name, value)
            self._depth = 0


def init_secondary_graphs(drafter, mode, device):
    """Initialize after set_attn, inside the intermediate graph-state scope."""
    enabled = mode != CUDAGraphMode.NONE
    drafter.update_stream = torch.npu.Stream(device=device) if enabled else None
    # DFlash's parallel query is uniform, including when verifier queries are
    # ragged. Request its full decode graph separately from verifier piecewise.
    drafter.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY if enabled else CUDAGraphMode.NONE)
    if enabled and not drafter.query_cudagraph_manager.needs_capture():
        raise ValueError("Secondary DFlash requires full graph attention support and nonempty capture sizes.")


def capture_secondary_graphs(drafter):
    """Capture only after both models' KV tensors have been initialized."""
    if drafter.query_cudagraph_manager.needs_capture():
        drafter.capture()
