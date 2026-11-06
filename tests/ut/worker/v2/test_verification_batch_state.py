# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU checks for the current BatchReqState verification-order contract."""

import importlib.util
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/attention/spec_decode.py"
SPEC = importlib.util.spec_from_file_location("_verification_batch_routing", SOURCE)
routing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(routing)


@pytest.fixture
def batch_state():
    # Include an extra row array: upstream may add fields to BatchReqState.
    state_type = namedtuple(
        "BatchReqState",
        (
            "req_ids num_scheduled_tokens idx_mapping_np num_computed_tokens "
            "prefill_len future_row_array num_tokens has_structured_output_requests scalar_array"
        ),
    )

    def make(lengths):
        count = len(lengths)
        return state_type(
            req_ids=[f"request-{index}" for index in range(count)],
            num_scheduled_tokens=np.asarray(lengths, dtype=np.int32),
            idx_mapping_np=np.arange(count, dtype=np.int32) + 20,
            num_computed_tokens=np.arange(count, dtype=np.int64) + 100,
            prefill_len=np.arange(count, dtype=np.int32) + 50,
            future_row_array=np.arange(count * 2, dtype=np.int64).reshape(count, 2),
            num_tokens=sum(lengths),
            has_structured_output_requests=True,
            scalar_array=np.asarray(9),
        )

    return make


@pytest.mark.parametrize(
    "lengths,order",
    [
        ([33, 1, 16, 17, 5], [1, 2, 4, 0, 3]),
        ([17, 16], [1, 0]),
        ([33, 17, 1, 5], [2, 3, 0, 1]),
    ],
)
def test_short_queries_precede_long_queries_with_all_request_rows_aligned(batch_state, lengths, order):
    original = batch_state(lengths)

    result = routing.prepare_verification_batch_req_state(original)

    assert result.req_ids == [original.req_ids[index] for index in order]
    for field in original._fields:
        expected = getattr(original, field)
        if isinstance(expected, np.ndarray) and expected.ndim:
            np.testing.assert_array_equal(getattr(result, field), expected[order])
        elif field != "req_ids":
            assert getattr(result, field) is expected
    np.testing.assert_array_equal(original.num_scheduled_tokens, lengths)
    assert result.num_scheduled_tokens.dtype == np.int32
    assert result.num_computed_tokens.dtype == np.int64


@pytest.mark.parametrize("lengths", [[], [1], [16], [17], [17, 33], [1, 5, 16], [1, 16, 17, 33]])
def test_already_ordered_batch_preserves_identity_and_avoids_array_copies(batch_state, lengths):
    original = batch_state(lengths)

    result = routing.prepare_verification_batch_req_state(original)

    assert result is original
