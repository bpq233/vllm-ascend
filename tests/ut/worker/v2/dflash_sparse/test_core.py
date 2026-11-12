# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU reference tests, including machines without the vLLM/NPU packages.

Run with ``pytest --confcutdir=tests/ut/worker/v2/dflash_sparse
tests/ut/worker/v2/dflash_sparse/test_core.py`` to avoid unrelated NPU fixtures.
"""

import importlib
import itertools
import math
import sys
import types
import unittest
from pathlib import Path

import torch

SOURCE = Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify"
PACKAGE_NAME = "_dflash_sparse_core_tests"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE_NAME] = package
config = importlib.import_module(f"{PACKAGE_NAME}.config")
selection = importlib.import_module(f"{PACKAGE_NAME}.selection")
decisions = importlib.import_module(f"{PACKAGE_NAME}.decisions")
metrics = importlib.import_module(f"{PACKAGE_NAME}.metrics")


def ratio_logits(ratios):
    """Candidate ID 1 has the requested probability ratio to ID 0."""
    return torch.tensor([[0.0, math.log(ratio)] for ratio in ratios], dtype=torch.float32)


class TestSparseConfig(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(config.SparseVerifyConfig.from_additional_config(None).enabled)
        self.assertFalse(config.SparseVerifyConfig.from_additional_config({}).enabled)

    def test_parses_nested_config_and_budget(self):
        value = config.SparseVerifyConfig.from_additional_config(
            {
                "unrelated": True,
                "dflash_sparse_verify": {
                    "enabled": True,
                    "mode": "confidence",
                    "num_rounds": 4,
                    "final_verification": {"method": "topk", "topk": 3},
                },
            }
        )
        self.assertEqual(value.total_token_budget(7), 28)
        self.assertEqual(value.final_verification.topk, 3)
        self.assertEqual(value.feature_source, "sparse")

    def test_bad_types_and_unknown_options(self):
        for values in (
            {"enabled": "true"},
            {"num_rounds": True},
            {"num_intermediate_layers": 1.5},
            {"threshold_high": "0.9"},
            {"graph_mode": "FULL"},
            {"feature_source": "old"},
            {"num_rounds": 0},
            {"mode": "unknown"},
            {"enable_metrics": 1},
        ):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                config.SparseVerifyConfig.from_dict(values)

    def test_threshold_validation(self):
        for high, low in ((0.5, 0.5), (1.1, 0.5), (0.9, -0.1), (float("nan"), 0.5), (float("inf"), 0.5)):
            with self.subTest(high=high, low=low), self.assertRaises(ValueError):
                config.SparseVerifyConfig(mode="confidence", threshold_high=high, threshold_low=low)
        config.SparseVerifyConfig(mode="confidence", threshold_high=1, threshold_low=0)
        config.SparseVerifyConfig(mode="fixed", threshold_high=-1, threshold_low=2)

    def test_final_config_validation(self):
        for values in (
            {"method": "other"},
            {"method": "topk", "topk": 0},
            {"method": "confidence", "target_threshold": 0},
            {"method": "confidence", "target_threshold": 1.1},
            {"method": "confidence", "target_threshold": float("nan")},
            {"topk": True},
        ):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                config.FinalVerificationConfig.from_dict(values)
        config.FinalVerificationConfig(method="strict", topk=0, target_threshold=-1)

    def test_temperature_guard(self):
        value = config.SparseVerifyConfig(enabled=True)
        value.validate_temperature(0)
        with self.assertRaisesRegex(ValueError, "temperature=0"):
            value.validate_temperature(0.1)


class TestLayerSelection(unittest.TestCase):
    def test_selection_invariants(self):
        for total, count in itertools.product(range(1, 18), range(1, 18)):
            if count > total:
                continue
            mandatory = [total // 2]
            chosen = selection.select_sparse_layers(total, count, mandatory)
            self.assertEqual(len(chosen), count)
            self.assertEqual(tuple(sorted(set(chosen))), chosen)
            self.assertIn(mandatory[0], chosen)
            self.assertTrue(all(0 <= layer < total for layer in chosen))

    def test_full_model(self):
        self.assertEqual(selection.select_sparse_layers(32, 32, [1, 15, 30]), tuple(range(32)))

    def test_uniform_gaps(self):
        for count in range(1, 16):
            selected = selection.select_sparse_layers(32, count)
            boundaries = (-1, *selected, 32)
            gaps = [right - left for left, right in zip(boundaries, boundaries[1:])]
            self.assertLessEqual(max(gaps) - min(gaps), 1)

    def test_required_layers_and_invalid_ids(self):
        with self.assertRaisesRegex(ValueError, "required"):
            selection.select_sparse_layers(32, 2, [1, 15, 30])
        for args in ((0, 0, []), (4, 5, []), (4, 2, [-1]), (4, 2, [4]), (4, 2, [True])):
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                selection.select_sparse_layers(*args)

    def test_post_layer_conversion(self):
        # Official upstream converts checkpoint target_layer_ids + 1 to the
        # target's pre-layer capture convention. The embedding capture is 0.
        self.assertEqual(selection.required_layers_from_aux_capture(32, [0, 2, 16, 31]), (1, 15, 30))
        self.assertEqual(selection.required_layers_from_aux_capture(4, [4]), (3,))
        for values in ([2, 2], [-1], [33], [1.0]):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                selection.required_layers_from_aux_capture(32, values)


class TestIntermediateDecision(unittest.TestCase):
    def decision(self, ratios, high=0.9, low=0.5):
        return decisions.ConfidenceDecision("confidence", high, low).evaluate(
            ratio_logits(ratios), torch.ones(len(ratios), dtype=torch.long)
        )

    def test_all_high(self):
        result = self.decision([0.98, 0.95, 1.0])
        self.assertEqual(result.accepted_length.item(), 3)
        self.assertEqual(result.first_rejection.item(), -1)
        self.assertFalse(result.stop)
        self.assertFalse(result.retry)

    def test_first_and_middle_retry(self):
        for ratios, accepted in (([0.72, 0.95], 0), ([0.98, 0.95, 0.72, 0.99], 2)):
            result = self.decision(ratios)
            self.assertEqual(result.accepted_length.item(), accepted)
            self.assertTrue(result.retry)
            self.assertFalse(result.stop)

    def test_first_and_middle_low(self):
        for ratios, accepted in (([0.32, 0.99], 0), ([0.98, 0.95, 0.32, 0.99], 2)):
            result = self.decision(ratios)
            self.assertEqual(result.first_rejection.item(), accepted)
            self.assertTrue(result.stop)
            self.assertFalse(result.retry)

    def test_later_low_cannot_override_earlier_middle(self):
        result = self.decision([0.98, 0.72, 0.1])
        self.assertTrue(result.retry)
        self.assertFalse(result.stop)
        self.assertEqual(result.accepted_mask.tolist(), [True, False, False])

    def test_threshold_equalities(self):
        result = self.decision([1.0, 0.5], high=1.0, low=0.5)
        self.assertEqual(result.accepted_length.item(), 1)
        self.assertFalse(result.stop)
        self.assertTrue(result.retry)
        self.assertEqual(self.decision([0.5], high=0.5, low=0.0).accepted_length.item(), 1)

    def test_nan_and_infinity_stop(self):
        for row in ([float("nan"), 0], [float("inf"), 0], [-float("inf"), -float("inf")]):
            with self.subTest(row=row):
                result = decisions.ConfidenceDecision("confidence").evaluate(torch.tensor([row]), torch.tensor([1]))
                self.assertTrue(result.stop)
                self.assertEqual(result.accepted_length.item(), 0)

    def test_fixed_uses_argmax_and_consumes_mismatch(self):
        result = decisions.ConfidenceDecision("fixed").evaluate(
            torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]), torch.tensor([1, 1, 1])
        )
        self.assertEqual(result.accepted_length.item(), 1)
        self.assertTrue(result.retry)
        self.assertFalse(result.stop)

    def test_batched_lengths_and_empty(self):
        logits = torch.tensor([[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
        result = decisions.ConfidenceDecision().evaluate(
            logits, torch.ones((2, 2), dtype=torch.long), torch.tensor([1, 2])
        )
        self.assertEqual(result.accepted_length.tolist(), [1, 1])
        self.assertEqual(result.first_rejection.tolist(), [-1, 1])
        empty = decisions.ConfidenceDecision().evaluate(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        self.assertEqual(empty.accepted_length.item(), 0)
        self.assertFalse(empty.stop)


class TestFinalVerifier(unittest.TestCase):
    def verify(self, method, logits, candidates, **kwargs):
        return decisions.FinalTargetVerifier(config.FinalVerificationConfig(method=method, **kwargs)).verify(
            torch.tensor(logits, dtype=torch.float32), torch.tensor(candidates, dtype=torch.long)
        )

    def test_strict_correction_and_bonus(self):
        result = self.verify("strict", [[0, 2, 1], [2, 0, 1], [0, 0, 3]], [1, 1])
        self.assertEqual(result.accepted_length.item(), 1)
        self.assertEqual(result.next_token.item(), 0)
        self.assertEqual(result.first_rejection.item(), 1)
        result = self.verify("strict", [[0, 2, 1], [2, 0, 1], [0, 0, 3]], [1, 0])
        self.assertEqual(result.accepted_length.item(), 2)
        self.assertEqual(result.next_token.item(), 2)

    def test_topk_accepts_nonargmax(self):
        result = self.verify("topk", [[0, 2, 1], [2, 0, 1], [0, 0, 3]], [2, 2])
        self.assertEqual(result.accepted_length.item(), 2)
        result = self.verify("topk", [[0, 2, 1], [2, 0, 1], [0, 0, 3]], [2, 1])
        self.assertEqual(result.accepted_length.item(), 1)
        self.assertEqual(result.next_token.item(), 0)

    def test_confidence_and_ties(self):
        logits = [[0, 0], [0, 1]]
        strict = self.verify("strict", logits, [1])
        confidence = self.verify("confidence", logits, [1], target_threshold=1)
        self.assertEqual(strict.accepted_length.item(), 0)
        self.assertEqual(confidence.accepted_length.item(), 1)
        result = self.verify("confidence", [[0, math.log(0.5)], [0, 1]], [1], target_threshold=0.5)
        self.assertEqual(result.accepted_length.item(), 1)

    def test_accept_all_only_bonus_logits(self):
        result = self.verify("accept_all", [[0, 1, 2]], [2, 1, 0, 2])
        self.assertEqual(result.accepted_length.item(), 4)
        self.assertEqual(result.next_token.item(), 2)
        self.assertEqual(result.first_rejection.item(), -1)

    def test_batched_bonus_and_correction(self):
        logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
        candidates = torch.tensor([[1, 0], [0, 1]])
        result = decisions.FinalTargetVerifier().verify(logits, candidates, lengths=torch.tensor([1, 2]))
        self.assertEqual(result.accepted_length.tolist(), [1, 0])
        self.assertEqual(result.next_token.tolist(), [0, 1])

    def test_bad_target_numbers_fail_explicitly(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "Target"):
                self.verify("strict", [[bad, bad], [0, 1]], [1])
        with self.assertRaisesRegex(ValueError, "bonus"):
            self.verify("accept_all", [[float("nan"), 0]], [1])

    def test_negative_infinity_mask_allowed(self):
        result = self.verify("strict", [[-float("inf"), 1], [0, 1]], [1])
        self.assertEqual(result.accepted_length.item(), 1)

    def test_topk_and_shapes_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.verify("topk", [[0, 1], [0, 1]], [1], topk=3)
        with self.assertRaisesRegex(ValueError, "bonus"):
            self.verify("strict", [[0, 1]], [1])

    def test_all_requested_candidate_lengths(self):
        for length in (1, 7, 15, 16, 17, 24, 28, 32, 48, 64):
            for method in ("strict", "topk", "confidence", "accept_all"):
                with self.subTest(length=length, method=method):
                    logits = torch.tensor([[0.0, 1.0, -1.0]]).repeat(length + 1, 1)
                    candidates = torch.ones(length, dtype=torch.long)
                    result = decisions.FinalTargetVerifier(config.FinalVerificationConfig(method=method)).verify(
                        logits, candidates
                    )
                    self.assertEqual(result.accepted_length.item(), length)
                    self.assertEqual(result.next_token.item(), 1)


class TestMetrics(unittest.TestCase):
    def test_disabled_has_no_work(self):
        value = metrics.SparseVerificationMetrics(False)
        value.record_graph("invalid", "invalid")
        self.assertEqual(value.snapshot(), {})

    def test_observed_graph_evidence_and_clear(self):
        value = metrics.SparseVerificationMetrics(True)
        value.record_graph("sparse", "capture")
        value.record_graph("sparse", "replay")
        self.assertEqual(value.snapshot()["graph_calls"], {"sparse.capture": 1, "sparse.replay": 1})
        value.record_final(metrics.FinalMetrics("r", "strict", 5, 3, 3, 0.01, 0.001, 4))
        self.assertEqual(value.snapshot(0.5)["emitted_tokens_per_second"], 8)
        self.assertEqual(value.snapshot(0.5)["mean_target_accepted_length"], 3)
        value.clear()
        self.assertEqual(value.snapshot()["graph_calls"], {})


if __name__ == "__main__":
    unittest.main()
