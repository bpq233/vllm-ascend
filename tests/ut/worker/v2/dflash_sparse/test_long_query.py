# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU proof of each virtual row's exact causal attention semantics."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify/long_query.py"
spec = importlib.util.spec_from_file_location("_sparse_long_query_test", SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
build_long_query_plan = module.build_long_query_plan


def causal_attention(query, key, value, prefix):
    """Bottom-right causal reference with an explicit historical prefix."""
    mask = torch.arange(key.shape[0])[None, :] <= prefix + torch.arange(query.shape[0])[:, None]
    scores = query @ key.T / math.sqrt(query.shape[-1])
    return scores.masked_fill(~mask, -torch.inf).softmax(dim=-1) @ value


class TestLongQueryPlan(unittest.TestCase):
    def test_all_requested_lengths_and_every_output_row(self):
        generator = torch.Generator().manual_seed(4127)
        for length in (1, 7, 15, 16, 17, 24, 28, 32, 48, 64):
            for historical_prefix in (0, 1, 19, 127):
                with self.subTest(length=length, historical_prefix=historical_prefix):
                    query = torch.randn(length, 8, generator=generator, dtype=torch.float64)
                    key = torch.randn(historical_prefix + length, 8, generator=generator, dtype=torch.float64)
                    value = torch.randn(historical_prefix + length, 5, generator=generator, dtype=torch.float64)
                    reference = causal_attention(query, key, value, historical_prefix)
                    plan = build_long_query_plan([0, length], [historical_prefix + length])
                    actual = torch.cat(
                        [
                            causal_attention(
                                query[chunk.query_begin : chunk.query_end],
                                key[: chunk.seq_len],
                                value[: chunk.seq_len],
                                historical_prefix + chunk.logical_begin,
                            )
                            for chunk in plan.chunks
                        ]
                    )
                    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
                    self.assertEqual(plan.query_start_loc[-1], length)
                    self.assertTrue(all(1 <= size <= 16 for size in plan.query_lens))
                    flattened_rows = [row for chunk in plan.chunks for row in range(chunk.query_begin, chunk.query_end)]
                    self.assertEqual(flattened_rows, list(range(length)))

    def test_variable_batch_and_shared_paged_rows(self):
        sizes = [1, 28, 0, 17, 64, 7]
        prefixes = [5, 37, 81, 0, 51, 3]
        starts = [0]
        for size in sizes:
            starts.append(starts[-1] + size)
        lengths = [prefix + size for prefix, size in zip(prefixes, sizes)]
        plan = build_long_query_plan(starts, lengths)
        self.assertEqual(plan.num_tokens, sum(sizes))
        self.assertEqual(plan.logical_last_chunk[2], -1)
        table = torch.arange(len(sizes) * 12).reshape(len(sizes), 12)
        expanded_table = plan.expand_block_table(table)
        expanded_lengths = plan.expand_seq_lens(torch.tensor(lengths, dtype=torch.int32))
        self.assertEqual(expanded_lengths.tolist(), list(plan.seq_lens))
        for idx, chunk in enumerate(plan.chunks):
            torch.testing.assert_close(expanded_table[idx], table[chunk.source_request])
            self.assertEqual(chunk.seq_len, prefixes[chunk.source_request] + chunk.logical_end)
        self.assertEqual(plan.source_rows, (0, 1, 1, 3, 3, 4, 4, 4, 4, 5))
        self.assertEqual(plan.query_start_tensor("cpu").tolist(), list(plan.query_start_loc))
        self.assertEqual(plan.seq_lens_tensor("cpu").tolist(), list(plan.seq_lens))

    def test_variable_batch_attention(self):
        generator = torch.Generator().manual_seed(73)
        sizes = [7, 28, 0, 17, 64]
        prefixes = [8, 21, 12, 3, 7]
        starts = [0]
        for size in sizes:
            starts.append(starts[-1] + size)
        lengths = [prefix + size for prefix, size in zip(prefixes, sizes)]
        plan = build_long_query_plan(starts, lengths)
        queries = [torch.randn(size, 8, generator=generator, dtype=torch.float64) for size in sizes]
        keys = [torch.randn(length, 8, generator=generator, dtype=torch.float64) for length in lengths]
        values = [torch.randn(length, 4, generator=generator, dtype=torch.float64) for length in lengths]
        reference = torch.cat(
            [
                causal_attention(query, key, value, prefix)
                for query, key, value, prefix in zip(queries, keys, values, prefixes)
            ]
        )
        flat_query = torch.cat(queries)
        actual = torch.cat(
            [
                causal_attention(
                    flat_query[chunk.query_begin : chunk.query_end],
                    keys[chunk.source_request][: chunk.seq_len],
                    values[chunk.source_request][: chunk.seq_len],
                    prefixes[chunk.source_request] + chunk.logical_begin,
                )
                for chunk in plan.chunks
            ]
        )
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)

    def test_device_corrected_lengths_stay_authoritative(self):
        plan = build_long_query_plan([0, 28, 45], [100, 200])
        corrected = torch.tensor([93, 191], dtype=torch.int32)
        # Each correction is propagated to every virtual row; the exact
        # device state is never overwritten by the CPU upper bound.
        self.assertEqual(plan.expand_seq_lens(corrected).tolist(), [81, 93, 190, 191])

    def test_rejected_suffix_cannot_change_accepted_prefix(self):
        generator = torch.Generator().manual_seed(91)
        prefix, length, accepted = 11, 48, 19
        query = torch.randn(length, 8, generator=generator, dtype=torch.float64)
        key = torch.randn(prefix + length, 8, generator=generator, dtype=torch.float64)
        value = torch.randn(prefix + length, 6, generator=generator, dtype=torch.float64)
        reference = causal_attention(query, key, value, prefix)
        # Physically cached rejected slots may remain. Sequence-length
        # rollback makes them invisible, and replacement overwrites them.
        key[prefix + accepted :] = 1000
        value[prefix + accepted :] = -1000
        shortened = causal_attention(query[:accepted], key[: prefix + accepted], value[: prefix + accepted], prefix)
        torch.testing.assert_close(shortened, reference[:accepted], atol=1e-12, rtol=1e-12)
        plan = build_long_query_plan([0, accepted], [prefix + accepted])
        self.assertEqual(plan.seq_lens[-1], prefix + accepted)
        self.assertEqual(plan.num_tokens, accepted)

    def test_zero_rows_and_empty_batch(self):
        plan = build_long_query_plan([0, 0, 0], [9, 0])
        self.assertEqual(plan.source_rows, ())
        self.assertEqual(plan.query_start_loc, (0,))
        self.assertEqual(plan.logical_last_chunk, (-1, -1))
        self.assertEqual(plan.expand_block_table(torch.ones(2, 4)).shape, (0, 4))
        self.assertEqual(build_long_query_plan([0], []).num_tokens, 0)

    def test_invalid_metadata(self):
        for starts, lengths, limit in (
            ([1, 2], [3], 16),
            ([0, 2, 1], [4, 4], 16),
            ([0, 3], [2], 16),
            ([0, 3], [3], 0),
            ([0, 3], [-1], 16),
            ([0, 1.5], [3], 16),
            ([0], [1], 16),
        ):
            with self.subTest(starts=starts, lengths=lengths, limit=limit), self.assertRaises((TypeError, ValueError)):
                build_long_query_plan(starts, lengths, limit)


if __name__ == "__main__":
    unittest.main()
