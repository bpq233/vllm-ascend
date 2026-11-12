# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Execute the real FIA metadata builder on CPU with device-only hooks stubbed.

The production build method and long-query planner execute unchanged. These
tests establish metadata routing/causality, not NPU kernel or ACL graph results.
"""

import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[5]
LONG_QUERY_NAME = "vllm_ascend.worker.v2.spec_decode.dflash.sparse_verify.long_query"


def load_builder_method():
    source = ROOT / "vllm_ascend/attention/attention_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    builder = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionMetadataBuilder"
    )
    method = next(node for node in builder.body if isinstance(node, ast.FunctionDef) and node.name == "build")
    extracted = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                method,
            ],
            type_ignores=[],
        )
    )
    states = SimpleNamespace(DecodeOnly="decode", SpecDecoding="spec", ChunkedPrefill="chunked")
    namespace = {
        "torch": torch,
        "uses_pcp_sharded_layout": lambda metadata, use_pcp: False,
        "_select_seq_lens": lambda metadata, **kwargs: metadata.seq_lens_cpu,
        "CrossAttentionSpec": type("CrossAttentionSpec", (), {}),
        "AscendAttentionState": states,
    }
    exec(compile(extracted, str(source), "exec"), namespace)
    return namespace["build"], states


def load_long_query():
    source = ROOT / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify/long_query.py"
    spec = importlib.util.spec_from_file_location("_actual_long_query_for_builder_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestAttentionMetadataIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build, cls.states = load_builder_method()
        cls.long_query = load_long_query()

    def metadata(self, sizes, prefixes, *, enabled=True, input_padding=0, causal=True):
        starts = [0]
        for size in sizes:
            starts.append(starts[-1] + size)
        total = starts[-1]
        lengths = torch.tensor([size + prefix for size, prefix in zip(sizes, prefixes)], dtype=torch.int32)
        block_table = torch.arange(len(sizes) * 8, dtype=torch.int32).reshape(len(sizes), 8) + 1
        slots = torch.arange(total + input_padding, dtype=torch.int32)
        if input_padding:
            slots[total:] = -1
        common = SimpleNamespace(
            num_reqs=len(sizes),
            num_actual_tokens=total,
            num_input_tokens=total + input_padding,
            query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
            query_start_loc=torch.tensor(starts, dtype=torch.int32),
            seq_lens_cpu=lengths,
            seq_lens=lengths.clone(),
            block_table_tensor=block_table,
            slot_mapping=slots,
            attn_state=self.states.ChunkedPrefill,
            causal=causal,
            is_prefilling=torch.zeros(len(sizes), dtype=torch.bool),
            max_query_len=max(sizes),
        )
        builder = SimpleNamespace(
            use_pcp=False,
            kv_cache_spec=object(),
            speculative_config=object(),
            vllm_config=SimpleNamespace(use_v2_model_runner=True),
            model_config=SimpleNamespace(runner_type="generate"),
            device="cpu",
            attn_mask_builder=SimpleNamespace(get_attention_mask=lambda *args: "causal mask"),
            metadata_cls=SimpleNamespace,
            sparse_verify_enabled=enabled,
            _split_decodes_and_prefills=lambda metadata: (0, len(sizes), 0, total),
            _build_backend_metadata=lambda *args, **kwargs: {},
        )
        with (
            patch.dict(sys.modules, {LONG_QUERY_NAME: self.long_query}),
            patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
        ):
            result = type(self).build(builder, 0, common)
        return result, common

    def assert_causal_endpoints(self, result, sizes, prefixes, expected_sources):
        boundaries = [0, *result.actual_seq_lengths_q]
        per_source_seen = [0] * len(sizes)
        for row, source in enumerate(expected_sources):
            length = boundaries[row + 1] - boundaries[row]
            self.assertTrue(1 <= length <= 16)
            local_start = per_source_seen[source]
            local_end = local_start + length
            self.assertEqual(result.seq_lens_list[row], prefixes[source] + local_end)
            # Bottom-right causal visibility must match the logical request
            # at every output row, including the first row after a chunk edge.
            keys = torch.arange(prefixes[source] + sizes[source])
            for offset in range(length):
                actual_visible = keys <= result.seq_lens_list[row] - length + offset
                expected_visible = keys <= prefixes[source] + local_start + offset
                torch.testing.assert_close(actual_visible, expected_visible)
            per_source_seen[source] = local_end
        self.assertEqual(per_source_seen, sizes)

    def test_actual_builder_splits_long_queries_with_exact_causal_endpoints(self):
        for size, prefix, expected_boundaries, expected_lengths in (
            (17, 13, [16, 17], [29, 30]),
            (28, 31, [16, 28], [47, 59]),
            (64, 9, [16, 32, 48, 64], [25, 41, 57, 73]),
        ):
            with self.subTest(size=size):
                result, common = self.metadata([size], [prefix])
                self.assertEqual(result.actual_seq_lengths_q, expected_boundaries)
                self.assertEqual(result.seq_lens_list, expected_lengths)
                self.assertLessEqual(result.max_query_len, 16)
                self.assertEqual(result.actual_seq_lengths_q[-1], common.num_input_tokens)
                self.assertEqual(result.query_start_loc_gpu.tolist(), [0, *expected_boundaries])
                self.assertEqual(result.seq_lens_gpu.tolist(), expected_lengths)
                self.assertEqual(result.attn_state, self.states.ChunkedPrefill)
                self.assertEqual(result.num_decodes, 0)
                self.assertEqual(result.num_prefills, len(expected_boundaries))
                for table_row in result.block_tables:
                    torch.testing.assert_close(table_row, common.block_table_tensor[0])
                torch.testing.assert_close(result.slot_mapping, common.slot_mapping)
                self.assert_causal_endpoints(result, [size], [prefix], [0] * len(expected_boundaries))

    def test_variable_batch_omits_zero_query_padding_and_repeats_source_rows(self):
        sizes = [17, 0, 28, 7, 0, 64]
        prefixes = [13, 0, 31, 2, 0, 9]
        sources = [0, 0, 2, 2, 3, 5, 5, 5, 5]
        result, common = self.metadata(sizes, prefixes)
        self.assertEqual(result.actual_seq_lengths_q, [16, 17, 33, 45, 52, 68, 84, 100, 116])
        self.assertEqual(result.seq_lens_list, [29, 30, 47, 59, 9, 25, 41, 57, 73])
        self.assertEqual(result.actual_seq_lengths_q[-1], sum(sizes))
        self.assertEqual(result.block_tables.shape[0], len(sources))
        for index, source in enumerate(sources):
            torch.testing.assert_close(result.block_tables[index], common.block_table_tensor[source])
        torch.testing.assert_close(result.slot_mapping, common.slot_mapping)
        self.assert_causal_endpoints(result, sizes, prefixes, sources)

    def test_zero_query_rows_are_removed_even_when_no_query_exceeds_limit(self):
        result, common = self.metadata([7, 0, 0, 3], [19, 0, 0, 5])
        self.assertEqual(result.actual_seq_lengths_q, [7, 10])
        self.assertEqual(result.seq_lens_list, [26, 8])
        self.assertEqual(result.block_tables.shape[0], 2)
        torch.testing.assert_close(result.block_tables[0], common.block_table_tensor[0])
        torch.testing.assert_close(result.block_tables[1], common.block_table_tensor[3])
        self.assertEqual(result.query_start_loc_gpu.tolist(), [0, 7, 10])
        self.assert_causal_endpoints(result, [7, 0, 0, 3], [19, 0, 0, 5], [0, 3])

    def test_native_synthetic_padding_row_keeps_all_input_tokens_without_cache_writes(self):
        result, common = self.metadata([28], [11], input_padding=4)
        self.assertEqual(result.actual_seq_lengths_q, [16, 28, 32])
        self.assertEqual(result.seq_lens_list, [27, 39, 4])
        self.assertEqual(result.num_actual_tokens, 28)
        self.assertEqual(result.slot_mapping.numel(), 28)
        torch.testing.assert_close(result.slot_mapping, common.slot_mapping[:28])
        self.assertTrue(bool((result.block_tables[-1] == 0).all()))
        self.assertEqual(result.actual_seq_lengths_q[-1], common.num_input_tokens)

    def test_disabled_experiment_preserves_original_metadata(self):
        result, common = self.metadata([28, 0, 7], [19, 0, 5], enabled=False)
        self.assertEqual(result.actual_seq_lengths_q, [28, 28, 35])
        self.assertEqual(result.seq_lens_list, [47, 0, 12])
        self.assertEqual(result.max_query_len, 28)
        self.assertEqual(result.block_tables.data_ptr(), common.block_table_tensor.data_ptr())
        self.assertEqual(result.query_start_loc_gpu.data_ptr(), common.query_start_loc.data_ptr())
        torch.testing.assert_close(result.slot_mapping, common.slot_mapping)

    def test_noncausal_draft_is_not_rewritten(self):
        result, common = self.metadata([28], [19], causal=False)
        self.assertEqual(result.actual_seq_lengths_q, [28])
        self.assertEqual(result.max_query_len, 28)
        self.assertEqual(result.block_tables.data_ptr(), common.block_table_tensor.data_ptr())


if __name__ == "__main__":
    unittest.main()
