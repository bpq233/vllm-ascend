# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Execute the actual sampling adapter against CPU logits and scheduler rows."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch


def _load_modules():
    directory = Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify"
    name = "_dflash_sparse_sampling_tests"
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    fake_modules = {}
    for module_name in (
        "vllm",
        "vllm.logger",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.sample",
        "vllm.v1.worker.gpu.sample.output",
    ):
        module = ModuleType(module_name)
        module.__path__ = []
        fake_modules[module_name] = module
    fake_modules["vllm.logger"].logger = MagicMock()
    fake_modules["vllm.v1.worker.gpu.sample.output"].SamplerOutput = SimpleNamespace
    modules = []
    with patch.dict(sys.modules, fake_modules):
        for leaf in ("config", "decisions", "metrics", "sampling"):
            spec = importlib.util.spec_from_file_location(f"{name}.{leaf}", directory / f"{leaf}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules.append(module)
    return modules


config_module, decisions, metrics, sampling = _load_modules()


class FakeTarget:
    def __init__(self, logits):
        self.logits = logits
        self.lm_head_inputs = []

    def compute_logits(self, hidden_states):
        self.lm_head_inputs.append(hidden_states.clone())
        return self.logits[hidden_states[:, 0].long()]


def make_runner(method, logits, *, enable_metrics=True, **final_options):
    config = config_module.SparseVerifyConfig(
        enabled=True,
        enable_metrics=enable_metrics,
        final_verification=config_module.FinalVerificationConfig(method=method, **final_options),
    )
    return SimpleNamespace(
        model=FakeTarget(logits),
        speculator=SimpleNamespace(
            sparse_config=config,
            total_candidates=64,
            sparse_metrics=metrics.SparseVerificationMetrics(enable_metrics),
        ),
    )


def make_batch(ids, boundaries, indices=None):
    indices = list(range(len(ids))) if indices is None else indices
    return SimpleNamespace(
        input_ids=torch.tensor(ids, dtype=torch.int32),
        logits_indices=torch.tensor(indices, dtype=torch.long),
        cu_num_logits_np=np.array(boundaries, dtype=np.int32),
        num_reqs=len(boundaries) - 1,
        req_ids=[f"r{i}" for i in range(len(boundaries) - 1)],
    )


def greedy_logits(tokens, vocabulary=16):
    logits = torch.full((len(tokens), vocabulary), -10.0)
    logits.scatter_(1, torch.tensor(tokens)[:, None], 10)
    return logits


class TestSamplingAdapter(unittest.TestCase):
    def run_sample(self, method, ids, predictions, *, boundaries=None, indices=None, **options):
        logits = predictions if isinstance(predictions, torch.Tensor) else greedy_logits(predictions)
        runner = make_runner(method, logits, **options)
        batch = make_batch(ids, [0, len(ids)] if boundaries is None else boundaries, indices)
        hidden = torch.arange(logits.shape[0], dtype=torch.float32)[:, None]
        result = sampling.sample_sparse_target(runner, hidden, batch, None)
        return runner, result

    def test_strict_logit_shift_and_all_accept_bonus(self):
        runner, (output, sampled, rejected) = self.run_sample("strict", [9, 2, 3], [2, 3, 4])
        self.assertEqual(output.sampled_token_ids[0, :3].tolist(), [2, 3, 4])
        self.assertEqual(sampled.tolist(), [3])
        self.assertEqual(rejected.tolist(), [0])
        self.assertIs(output.num_sampled, sampled)
        self.assertIs(output.num_rejected, rejected)
        self.assertEqual(runner.model.lm_head_inputs[0][:, 0].tolist(), [0, 1, 2])

    def test_reject_first_and_middle_keeps_correction_only_once(self):
        for predictions, expected, accepted in (([8, 3, 4], [8], 0), ([2, 8, 4], [2, 8], 1)):
            with self.subTest(predictions=predictions):
                _, (output, sampled, rejected) = self.run_sample("strict", [9, 2, 3], predictions)
                self.assertEqual(output.sampled_token_ids[0, : accepted + 1].tolist(), expected)
                self.assertTrue((output.sampled_token_ids[0, accepted + 1 :] == -1).all())
                self.assertEqual(sampled.tolist(), [accepted + 1])
                self.assertEqual(rejected.tolist(), [2 - accepted])
                # Upstream postprocess subtracts rejections from the three
                # computed query inputs. The emitted correction stays uncached.
                self.assertEqual(3 - int(rejected[0]), accepted + 1)

    def test_all_four_policies_and_nonargmax_accepts(self):
        logits = torch.full((3, 16), -10.0)
        logits[0, 1], logits[0, 2] = 1.0, 0.5
        logits[1, 1], logits[1, 3] = 1.0, 0.5
        logits[2, 4] = 2.0
        for method in ("strict", "topk", "confidence", "accept_all"):
            with self.subTest(method=method):
                runner, (output, sampled, rejected) = self.run_sample(method, [9, 2, 3], logits)
                if method == "strict":
                    self.assertEqual(output.sampled_token_ids[0, 0], 1)
                    self.assertEqual(sampled.tolist(), [1])
                    self.assertEqual(rejected.tolist(), [2])
                else:
                    self.assertEqual(output.sampled_token_ids[0, :3].tolist(), [2, 3, 4])
                    self.assertEqual(sampled.tolist(), [3])
                    self.assertEqual(rejected.tolist(), [0])
                if method == "accept_all":
                    self.assertEqual(runner.model.lm_head_inputs[0][:, 0].tolist(), [2])

    def test_noncontiguous_input_rows_follow_logit_indices(self):
        logits = greedy_logits([0, 0, 2, 0, 3, 0, 4])
        _, (output, sampled, rejected) = self.run_sample(
            "strict",
            [0, 0, 9, 0, 2, 0, 3],
            logits,
            boundaries=[0, 3],
            indices=[2, 4, 6],
        )
        self.assertEqual(output.sampled_token_ids[0, :3].tolist(), [2, 3, 4])
        self.assertEqual((sampled.tolist(), rejected.tolist()), ([3], [0]))

    def test_mixed_batch_zero_prefill_and_different_candidate_lengths(self):
        # Request zero is an unsampled prefill; request one has one candidate;
        # request two has three. Selected indices belong to requests one/two.
        ids = [0, 0, 0, 9, 2, 9, 3, 4, 5]
        logits = greedy_logits([0, 0, 0, 2, 6, 3, 8, 5, 7])
        for method in ("strict", "accept_all"):
            with self.subTest(method=method):
                runner, (output, sampled, rejected) = self.run_sample(
                    method,
                    ids,
                    logits,
                    boundaries=[0, 0, 2, 6],
                    indices=[3, 4, 5, 6, 7, 8],
                )
                self.assertTrue((output.sampled_token_ids[0] == -1).all())
                if method == "strict":
                    self.assertEqual(output.sampled_token_ids[1, :2].tolist(), [2, 6])
                    self.assertEqual(output.sampled_token_ids[2, :2].tolist(), [3, 8])
                    self.assertEqual(sampled.tolist(), [0, 2, 2])
                    self.assertEqual(rejected.tolist(), [0, 0, 2])
                else:
                    self.assertEqual(output.sampled_token_ids[2, :4].tolist(), [3, 4, 5, 7])
                    self.assertEqual(sampled.tolist(), [0, 2, 4])
                    self.assertEqual(rejected.tolist(), [0, 0, 0])
                    self.assertEqual(runner.model.lm_head_inputs[0][:, 0].tolist(), [4, 8])

    def test_all_unsampled_prefill_rows_skip_lm_head(self):
        for method in ("strict", "topk", "confidence", "accept_all"):
            with self.subTest(method=method):
                runner, (output, sampled, rejected) = self.run_sample(
                    method,
                    [1, 2, 3],
                    greedy_logits([0, 0, 0]),
                    boundaries=[0, 0, 0],
                    indices=[],
                )
                self.assertEqual(runner.model.lm_head_inputs, [])
                self.assertTrue((output.sampled_token_ids == -1).all())
                self.assertEqual(sampled.tolist(), [0, 0])
                self.assertEqual(rejected.tolist(), [0, 0])

    def test_no_candidates_still_samples_target_next_token(self):
        for method in ("strict", "topk", "confidence", "accept_all"):
            with self.subTest(method=method):
                _, (output, sampled, rejected) = self.run_sample(method, [9], [4])
                self.assertEqual(output.sampled_token_ids[0, 0], 4)
                self.assertEqual(sampled.tolist(), [1])
                self.assertEqual(rejected.tolist(), [0])

    def test_all_long_candidate_lengths_keep_complete_sequence(self):
        for length in (1, 7, 15, 16, 17, 24, 28, 32, 48, 64):
            for method in ("strict", "topk", "confidence", "accept_all"):
                with self.subTest(length=length, method=method):
                    _, (output, sampled, rejected) = self.run_sample(
                        method,
                        [9] + [2] * length,
                        [2] * length + [4],
                    )
                    self.assertEqual(output.sampled_token_ids[0, : length + 1].tolist(), [2] * length + [4])
                    self.assertEqual(sampled.tolist(), [length + 1])
                    self.assertEqual(rejected.tolist(), [0])

    def test_grammar_rejected_before_lm_head(self):
        runner = make_runner("strict", greedy_logits([2]))
        batch = make_batch([9], [0, 1])
        with self.assertRaisesRegex(ValueError, "structured outputs"):
            sampling.sample_sparse_target(runner, torch.zeros(1, 1), batch, object())
        self.assertEqual(runner.model.lm_head_inputs, [])

    def test_invalid_ids_and_numeric_failures_are_distinguishable(self):
        with self.assertRaisesRegex(ValueError, r"req=r0.*invalid_id_values.*-1"):
            self.run_sample("strict", [9, -1], [2, 3])
        logits = greedy_logits([2, 3])
        logits[0, 2] = float("nan")
        with self.assertRaisesRegex(ValueError, r"req=r0.*invalid_id_values.*\[\].*nan_rows.*\[\[0\]\]"):
            self.run_sample("strict", [9, 2], logits)

    def test_target_rejection_logs_actual_rejected_prediction_row(self):
        with patch.object(sampling, "logger") as logger:
            self.run_sample("strict", [9, 2, 3], [2, 8, 4])
        calls = [call.args for call in logger.info.call_args_list if call.args[0].startswith("DFlash Target rejection")]
        self.assertEqual(len(calls), 1)
        _, request_id, rejection_index, details = calls[0]
        self.assertEqual((request_id, rejection_index), ("r0", 1))
        self.assertEqual(details["candidate_token_id"], 3)
        self.assertEqual(details["candidate_logit"], -10.0)
        self.assertEqual(details["max_token_id"], 8)
        self.assertEqual(details["max_logit"], 10.0)
        self.assertEqual(details["logit_gap"], 20.0)

    def test_m_l_comparison_uses_same_token_position_and_retires_pending_record(self):
        runner = make_runner("strict", greedy_logits([2, 3, 4]))
        runner.speculator.sparse_model = SimpleNamespace(selected_layers=tuple(range(12)))
        runner.speculator.target_num_layers = 32
        runner.speculator.pending_rejection_comparisons = {
            "r0": [
                {
                    "round": 0,
                    "rejection": {
                        "candidate_position": 51,
                        "candidate_token_id": 2,
                        "candidate_logit": -6.0,
                        "max_token_id": 7,
                        "max_logit": 10.0,
                        "logit_gap": 16.0,
                        "confidence": 1.125e-7,
                    },
                }
            ]
        }
        batch = make_batch([9, 2, 3], [0, 3])
        batch.positions = torch.tensor([50, 51, 52])
        with patch.object(sampling, "logger") as logger:
            output, _, _ = sampling.sample_sparse_target(runner, torch.arange(3).float()[:, None], batch, None)
        calls = [c.args for c in logger.info.call_args_list if c.args[0].startswith("DFlash M/L comparison")]
        self.assertEqual(len(calls), 1)
        _, req, m, total, details = calls[0]
        self.assertEqual((req, m, total), ("r0", 12, 32))
        self.assertEqual(details["L"]["candidate_logit"], 10.0)
        self.assertEqual(details["L"]["confidence"], 1.0)
        self.assertEqual(output.sampled_token_ids[0, :3].tolist(), [2, 3, 4])
        self.assertEqual(runner.speculator.pending_rejection_comparisons, {})


class TestBoundedMetrics(unittest.TestCase):
    def test_detail_retention_is_bounded_but_aggregate_totals_are_lifetime(self):
        value = metrics.SparseVerificationMetrics(True)
        count = metrics.MAX_DETAIL_RECORDS + 13
        for index in range(count):
            value.record_round(
                metrics.RoundMetrics(
                    f"r{index}",
                    0,
                    7,
                    3,
                    3,
                    (1.0, 1.0, 1.0, 0.7),
                    "retry",
                    "sparse",
                    (7, 2),
                    0,
                    7,
                )
            )
            value.record_final(metrics.FinalMetrics(f"r{index}", "topk", 7, 4, 4, 0.1, 0.01, 5, 2, 2))
        snapshot = value.snapshot(1.0)
        self.assertEqual(len(value.rounds), metrics.MAX_DETAIL_RECORDS)
        self.assertEqual(len(value.finals), metrics.MAX_DETAIL_RECORDS)
        self.assertEqual(snapshot["total_round_records"], count)
        self.assertEqual(snapshot["total_final_records"], count)
        self.assertEqual(snapshot["temporary_accepted"], count * 3)
        self.assertEqual(snapshot["final_accepted"], count * 4)
        self.assertEqual(snapshot["mean_target_accepted_length"], 4)
        self.assertEqual(snapshot["approx_extra_accepted"], count * 2)
        self.assertEqual(snapshot["emitted_tokens_per_second"], count * 5)
        self.assertEqual(snapshot["detail_scope"], "most_recent_records")
        self.assertEqual(snapshot["rounds_per_request_scope"], "retained_round_records")
        self.assertNotIn("r0", snapshot["rounds_per_request"])
        self.assertEqual(snapshot["rounds"][0]["request_id"], "r13")
        value.clear()
        self.assertEqual(value.snapshot()["total_final_records"], 0)
        self.assertEqual(value.snapshot()["approx_extra_accepted"], 0)

    def test_accept_all_does_not_invent_strict_reference_metrics(self):
        value = metrics.SparseVerificationMetrics(True)
        value.record_final(metrics.FinalMetrics("r", "accept_all", 7, 7, -1, 0.1, 0.01, 8))
        snapshot = value.snapshot()
        self.assertEqual(snapshot["accept_all_passed"], 7)
        self.assertEqual(snapshot["strict_reference_records"], 0)
        self.assertEqual(snapshot["approx_reference_records"], 0)
        self.assertIsNone(snapshot["finals"][0]["strict_accepted_length"])


if __name__ == "__main__":
    unittest.main()
