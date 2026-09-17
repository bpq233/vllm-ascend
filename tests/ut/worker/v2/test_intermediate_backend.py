# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise packed inputs and lifecycle with CPU stand-ins for NPU kernels."""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode"


def load_class(filename, name, namespace):
    tree = ast.parse((SOURCE / filename).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    exec(compile(tree, filename, "exec"), namespace)
    return namespace[name]


@pytest.fixture
def backend():
    metadata = Mock(side_effect=lambda **kw: kw)
    rope = NS(update_cos_sin=Mock(), _cos=object())
    namespace = dict(
        torch=torch,
        np=np,
        contextmanager=contextmanager,
        rope=rope,
        vllm_version_is=lambda version: False,
        set_current_vllm_config=lambda *a: nullcontext(),
        set_forward_context=lambda *a, **kw: nullcontext(),
        AscendInputBatch=NS,
        AscendAttentionState=NS(PrefillNoCache="prefill"),
        CUDAGraphMode=NS(NONE=None),
        build_attn_metadata=metadata,
        build_slot_mappings_by_layer=lambda slots, config: {"layer": slots[0]},
    )
    cls = load_class("intermediate_backend.py", "IntermediateBackend", namespace)
    obj = cls.__new__(cls)
    obj.config = NS(num_speculative_tokens=2)
    obj.device = torch.device("cpu")
    obj.max_num_reqs, obj.max_num_tokens, obj.max_model_len = 2, 12, 8
    obj.vllm_config, obj.kv_cache_config, obj.attn_groups = NS(), NS(), []
    obj._rope_state = {"_cos": object()}
    obj.block_tables = NS(
        input_block_tables=[torch.tensor([[1, 2], [3, 4]])],
        kernel_block_sizes=[4],
        slot_mappings=torch.zeros(1, 12, dtype=torch.long),
    )
    # Encode each input's position and ID, making logit alignment observable.
    obj.model = Mock(side_effect=lambda input_ids, positions: torch.stack((input_ids, positions), dim=-1))
    obj.model.compute_logits = lambda hidden: hidden.float()
    obj.drafter = NS(propose=Mock(return_value=torch.tensor([[9, 10], [11, 12]])))
    return obj, metadata, rope


def test_verify_alignment_and_request_isolation(backend):
    obj, metadata, _ = backend
    result = list(obj.verify([[1, 2], [5]], [[3, 4], [6]]))
    assert [x.tolist() for x in result] == [[[2, 1], [3, 2], [4, 3]], [[5, 0], [6, 1]]]
    assert metadata.call_args.kwargs["causal"] is True
    assert metadata.call_args.kwargs["query_start_loc_cpu"].tolist() == [0, 4, 6]
    assert obj.block_tables.slot_mappings[0, :6].tolist() == [4, 5, 6, 7, 12, 13]
    # A shorter, unrelated request reuses scratch slot 0 without a stale prefix.
    assert list(obj.verify([[7]], [[8]]))[0].tolist() == [[7, 0], [8, 1]]
    assert metadata.call_args.kwargs["seq_lens"].tolist() == [2]


def test_secondary_uses_last_token_as_anchor_and_excludes_it_from_context(backend):
    obj, _, _ = backend
    result = obj.propose([[1, 2, 3], [4, 5]])
    assert result == [[9, 10], [11, 12]]
    args = obj.drafter.propose.call_args.args
    assert args[0].input_ids.tolist() == [1, 2, 4]
    assert args[7].tolist() == [3, 5]
    assert args[5].tolist() == [1, 1]
    assert args[6].tolist() == [0, 0]


def test_boundary_falls_back_to_verifier_only(backend):
    obj, _, _ = backend
    obj.drafter.propose.return_value = torch.tensor([[9, 10]])
    assert obj.propose([[1] * 7, [2, 3], [4]]) == [[], [9, 10], []]
    assert obj.drafter.propose.call_args.args[7][0] == 3


def test_microbatch_limits_and_rope_restored_on_error(backend):
    obj, _, rope = backend
    assert [(i, len(rows)) for i, rows in obj._batches([[1] * 8, [2] * 6, [3], [4]])] == [(0, 1), (1, 2), (3, 1)]
    main_rope = rope._cos
    with pytest.raises(RuntimeError), obj._context():
        assert rope._cos is obj._rope_state["_cos"]
        raise RuntimeError("forward failed")
    assert rope._cos is main_rope


def test_adapter_replaces_only_candidate_buffer_and_publishes_real_lengths():
    primary = torch.tensor([[7, 8], [9, 10]])

    class OriginalDFlash:
        def propose(self, batch, *args, **kwargs):
            return primary

    cls = load_class(
        "multi_stage.py",
        "MultiStageDFlashSpeculator",
        dict(torch=torch, AscendDFlashSpeculator=OriginalDFlash, DraftTokenIds=lambda ids, tokens: (ids, tokens)),
    )
    obj = cls.__new__(cls)
    obj.final_capacity, obj.max_model_len, obj.device = 5, 16, torch.device("cpu")
    history = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 0, 0, 0]])
    obj.req_states = NS(
        total_len=NS(gpu=torch.tensor([4, 2])),
        all_token_ids=NS(gpu=history),
        max_seq_len=np.array([16, 16]),
        prefill_len=NS(np=np.array([3, 2])),
    )
    original_history = history.clone()
    obj.pipeline = NS(backend=NS(max_model_len=16, profile=Mock()), refine=Mock(return_value=[[7, 11, 12], []]))
    batch = NS(num_reqs=2, idx_mapping=torch.tensor([0, 1]), idx_mapping_np=np.array([0, 1]), req_ids=["a", "b"])
    result = obj.propose(batch)
    assert result.tolist() == [[7, 11, 12, 0, 0], [0, 0, 0, 0, 0]]
    assert obj.pipeline.refine.call_args.args == ([[1, 2, 3, 4], [5, 6]], [[7, 8], [9, 10]], [5, 0])
    assert obj.get_draft_tokens() == (["a", "b"], [[7, 11, 12], []])
    assert torch.equal(history, original_history)
    assert primary.tolist() == [[7, 8], [9, 10]]
    obj.propose(batch, dummy_run=True, is_profile=True)
    obj.pipeline.backend.profile.assert_called_once()
    assert obj.pipeline.refine.call_count == 1
