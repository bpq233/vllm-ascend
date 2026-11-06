# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise the MRV2 draft handler contract on both hierarchical paths."""

from collections import OrderedDict
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from test_intermediate import modules as modules
from test_intermediate import settings, vllm_config
from test_intermediate_backend import load_class


@pytest.fixture(params=["primary", "independent"])
def handler(request):
    primary = load_class(
        "adapter.py",
        "MultiStageDFlashSpeculator",
        dict(
            torch=torch,
            AscendDFlashSpeculator=type("Primary", (), {}),
            DraftTokenIds=lambda ids, tokens: NS(req_ids=ids, draft_token_ids=tokens),
            logger=Mock(),
        ),
    )
    cls = primary
    if request.param == "independent":
        cls = load_class(
            "intermediate_speculator.py",
            "IntermediateOnlySpeculator",
            dict(torch=torch, BaseSpeculator=object, MultiStageDFlashSpeculator=primary),
        )
    cache_cls = load_class("backend.py", "IntermediateKVCache", dict(OrderedDict=OrderedDict))
    cache = cache_cls(2)
    cache.slots.update((req_id, slot) for slot, req_id in enumerate(("a", "b")))
    cache.tokens = [[1, 2], [3, 4]]
    obj = cls.__new__(cls)
    obj.final_capacity = 8
    obj.req_ids = []
    obj.candidates = []
    obj._host_histories = {"a": [1, 2], "b": [3, 4]}
    obj._draft_tokens_by_req = {}
    obj._deferred_req_ids = None
    obj.pipeline = NS(backend=NS(cache=cache, _context_rows=[object(), object()]))
    return obj


def step(*, finished=(), new=(), deferred=None):
    return NS(
        finished_req_ids=set(finished),
        scheduled_new_reqs=[NS(req_id=req_id) for req_id in new],
        scheduled_spec_decode_tokens=dict.fromkeys(deferred or (), []),
        pending_structured_output_tokens=deferred is not None,
    )


def publish(handler, req_ids, candidates, width=8):
    handler.req_ids = list(req_ids)
    handler.candidates = [list(tokens) for tokens in candidates]
    handler.set_draft_tokens(NS(req_ids=req_ids), torch.empty((len(req_ids), width), dtype=torch.int64))


def drafts(handler):
    result = handler.get_draft_tokens()
    return dict(zip(result.req_ids, result.draft_token_ids))


def test_live_unscheduled_drafts_survive_steps_and_repeated_reads(handler):
    handler.begin_step(step())
    publish(handler, ["a", "b"], [[11, 12], [21]])
    assert drafts(handler) == {"a": [11, 12], "b": [21]}
    handler.begin_step(step())
    publish(handler, ["b"], [[22, 23]])
    assert drafts(handler) == {"a": [11, 12], "b": [22, 23]}
    assert drafts(handler) == {"a": [11, 12], "b": [22, 23]}
    handler.begin_step(step())
    assert drafts(handler) == {"a": [11, 12], "b": [22, 23]}


def test_discard_clears_pending_drafts_history_and_private_cache(handler):
    publish(handler, ["a", "b"], [[11], [21]])
    handler.discard("a")
    handler.discard("a")
    handler.discard("missing")
    assert drafts(handler) == {"b": [21]}
    assert handler._host_histories == {"b": [3, 4]}
    assert dict(handler.pipeline.backend.cache.slots) == {"b": 1}
    assert handler.pipeline.backend.cache.tokens == [[], [3, 4]]
    assert handler.pipeline.backend._context_rows[0] is None
    assert handler.pipeline.backend._context_rows[1] is not None


@pytest.mark.parametrize("finished", [(), ("a",)])
def test_new_request_id_cannot_reuse_old_prompt_of_same_length(handler, finished):
    publish(handler, ["a", "b"], [[11], [21]])
    handler.begin_step(step(finished=finished, new=("a",)))
    assert drafts(handler) == {"b": [21]}
    assert "a" not in handler._host_histories
    assert "a" not in handler.pipeline.backend.cache.slots
    publish(handler, ["a"], [[31]])
    assert drafts(handler) == {"b": [21], "a": [31]}


def test_deferred_structured_output_fetches_only_requested_live_rows(handler):
    publish(handler, ["a", "b"], [[11], [21]])
    handler.begin_step(step(deferred=["b", "missing", "a"]))
    result = handler.get_draft_tokens()
    assert result.req_ids == ["b", "a"]
    assert result.draft_token_ids == [[21], [11]]
    handler.discard("b")
    assert drafts(handler) == {"a": [11]}
    handler.begin_step(step(deferred=[]))
    assert drafts(handler) == {}
    handler.begin_step(step())
    assert drafts(handler) == {"a": [11]}


@pytest.mark.parametrize("width", [0, 1, 3])
def test_runner_budget_updates_only_current_request_without_d2h(handler, width):
    publish(handler, ["a", "b"], [[11, 12], [21, 22, 23, 24]])
    handler.begin_step(step())
    publish(handler, ["b"], [[31, 32, 33, 34]], width=width)
    assert drafts(handler) == {"a": [11, 12], "b": [31, 32, 33, 34][:width]}


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("additional_config", {"finegrained_tp_config": {"lmhead_tensor_parallel_size": 2}}, "lmhead"),
        ("kv_transfer_config", object(), "KV transfer"),
    ],
)
def test_private_verifier_rejects_incompatible_new_runtime_modes(modules, field, value, match):
    config, _ = modules
    cfg = vllm_config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=match):
        config.validate_multi_stage(cfg, settings())
