# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Compare only identical candidate positions with identical retained context."""

import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType

import torch

directory = Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify"
package = ModuleType("_prefix_comparison_tests")
package.__path__ = [str(directory)]
sys.modules[package.__name__] = package
compare = importlib.import_module(f"{package.__name__}.comparison").compare_rejections


def record(position=101, token=2):
    return {
        "round": 0,
        "rejection": {
            "candidate_position": position,
            "candidate_token_id": token,
            "candidate_logit": -6.0,
            "max_token_id": 7,
            "max_logit": 10.0,
            "logit_gap": 16.0,
            "confidence": 1.125351747e-7,
        },
    }


class TestComparison(unittest.TestCase):
    def test_position_alignment_and_comparison_after_target_accepts(self):
        logits = torch.tensor([[0.0, 1.0, 12.0, 2.0], [0.0, 1.0, 2.0, 13.0], [14.0, 1.0, 2.0, 3.0]])
        result = compare([record()], logits, torch.tensor([2, 3]), torch.tensor([101, 102]))[0]
        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["M"]["candidate_logit"], -6.0)
        self.assertEqual(result["L"]["candidate_logit"], 12.0)
        self.assertEqual(result["L"]["max_token_id"], 2)
        self.assertEqual(result["L"]["confidence"], 1.0)
        self.assertEqual(result["candidate_logit_M_minus_L"], -18.0)
        self.assertFalse(result["max_token_match"])

    def test_discarded_candidate_and_changed_token_are_not_paired(self):
        results = compare([record(100), record(101, 3)], torch.zeros(2, 4), torch.tensor([2]), torch.tensor([101]))
        self.assertEqual([r["status"] for r in results], ["not_submitted_to_target", "candidate_changed"])
        self.assertTrue(all(r["L"] is None for r in results))

    def test_accept_all_does_not_invent_missing_candidate_logits(self):
        result = compare([record()], torch.zeros(1, 4), torch.tensor([2]), torch.tensor([101]), bonus_only=True)[0]
        self.assertEqual(result["status"], "target_candidate_logits_not_computed_accept_all")
        self.assertIsNone(result["L"])


if __name__ == "__main__":
    unittest.main()
