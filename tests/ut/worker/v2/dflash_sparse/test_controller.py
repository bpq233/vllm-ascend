# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""CPU contract tests, runnable without vLLM, CANN, or NPU bindings."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType

import torch


def _load_modules():
    directory = Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify"
    name = "_dflash_sparse_controller_tests"
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    modules = []
    for leaf in ("config", "decisions", "controller"):
        spec = importlib.util.spec_from_file_location(f"{name}.{leaf}", directory / f"{leaf}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


config_module, decision_module, controller_module = _load_modules()
SparseVerifyConfig = config_module.SparseVerifyConfig
FinalVerificationConfig = config_module.FinalVerificationConfig
MultiRoundDFlashController = controller_module.MultiRoundDFlashController
FeatureBundle = controller_module.FeatureBundle
SparseForwardOutput = controller_module.SparseForwardOutput


class FakeCallbacks:
    def __init__(self, decisions=None):
        self.decisions = decisions
        self.draft_calls = []
        self.verify_calls = []
        self.retained = []

    def draft(self, request):
        self.draft_calls.append(request)
        # Consume the feature values so a later assertion can establish that
        # the second draft really received new sparse conditioning features.
        base = int(request.features.values[0][-1, 0].item())
        return torch.arange(base + 1, base + 1 + request.max_tokens) % 31

    def verify(self, request):
        self.verify_calls.append(request)
        tokens = request.input_ids[1:]
        logits = torch.full((tokens.numel(), 32), -50.0)
        logits.scatter_(1, tokens[:, None], 0.0)
        if self.decisions is not None:
            ratios = self.decisions[min(request.round_index, len(self.decisions) - 1)]
            for index, ratio in enumerate(ratios[: tokens.numel()]):
                logits[index, (int(tokens[index]) + 1) % 32] = 0.0
                logits[index, tokens[index]] = math.log(ratio) if ratio else -100.0
        positions = request.positions.to(torch.float32)
        values = torch.stack((positions + 10, positions + 20), dim=-1)
        features = FeatureBundle((values, values + 100), (1, 4), int(request.positions[0]), "sparse", request.input_ids)
        return SparseForwardOutput(logits, features, self.retained.append)


def initial_features():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    return FeatureBundle((values, values + 100), (1, 4), 0, "target")


class TestMultiRoundController(unittest.TestCase):
    def controller(self, mode="fixed", rounds=4, length=7, **kwargs):
        config = SparseVerifyConfig(enabled=True, mode=mode, num_rounds=rounds, **kwargs)
        return MultiRoundDFlashController(config, length, rounds * length)

    def run_controller(self, callbacks, mode="fixed", rounds=4, length=7, budget=None, **kwargs):
        controller = self.controller(mode, rounds, length, **kwargs)
        return controller.run_request(
            "request-a",
            7,
            3,
            initial_features(),
            callbacks.draft,
            callbacks.verify,
            token_budget=rounds * length if budget is None else budget,
        )

    def test_fixed_round_counts_and_updated_features(self):
        for rounds in (1, 2, 4):
            with self.subTest(rounds=rounds):
                callbacks = FakeCallbacks()
                result = self.run_controller(callbacks, rounds=rounds)
                self.assertEqual(len(result.rounds), rounds)
                self.assertEqual(result.tokens.numel(), rounds * 7)
                self.assertEqual(result.temporary_accepted_length, rounds * 7)
                self.assertEqual(result.rounds[0].feature_source, "target")
                for index, request in enumerate(callbacks.draft_calls[1:], 1):
                    self.assertEqual(request.features.source, "sparse")
                    self.assertEqual(request.anchor_position, 3 + index * 7)
                    self.assertEqual(request.features.end_position, request.anchor_position)
                    self.assertEqual(request.features.start_position, 3 + (index - 1) * 7)
                    self.assertEqual(request.features.values[0][-1, 0], request.anchor_position - 1 + 10)
                self.assertEqual(callbacks.retained, [3 + (index + 1) * 7 for index in range(rounds)])

    def test_zero_accept_consumes_fixed_and_adaptive_rounds(self):
        for mode in ("fixed", "confidence"):
            callbacks = FakeCallbacks([[0.7] * 7])
            result = self.run_controller(callbacks, mode=mode)
            self.assertEqual(len(result.rounds), 4)
            self.assertEqual(result.tokens.numel(), 0)
            self.assertEqual(callbacks.retained, [3] * 4)
            self.assertTrue(all(request.anchor_position == 3 for request in callbacks.draft_calls))
            self.assertTrue(
                all(request.features is callbacks.draft_calls[0].features for request in callbacks.draft_calls)
            )

    def test_middle_reject_cannot_keep_discrete_later_accepts(self):
        callbacks = FakeCallbacks([[1, 1, 0.7, 1, 1, 1, 1]])
        result = self.run_controller(callbacks, mode="confidence", rounds=2)
        self.assertEqual(result.tokens.numel(), 4)
        self.assertEqual([record.temporarily_accepted_length for record in result.rounds], [2, 2])
        self.assertEqual([record.first_rejection for record in result.rounds], [2, 2])
        second_features = callbacks.draft_calls[1].features
        self.assertEqual(second_features.end_position, 5)
        self.assertEqual(second_features.values[0].shape[0], 2)
        self.assertTrue(torch.equal(second_features.token_ids, callbacks.verify_calls[0].input_ids[:2]))
        self.assertEqual(callbacks.draft_calls[1].anchor_token, int(result.tokens[1]))

    def test_rejected_row_reports_candidate_and_maximum_logits(self):
        callbacks = FakeCallbacks([[1, 1, 0.7, 1, 1, 1, 1]])
        result = self.run_controller(callbacks, mode="confidence", rounds=1)
        details = result.rounds[0].rejection
        candidate = int(callbacks.verify_calls[0].input_ids[3])
        self.assertEqual(details["candidate_token_id"], candidate)
        self.assertEqual(details["candidate_position"], 6)
        self.assertEqual(details["max_token_id"], (candidate + 1) % 32)
        self.assertAlmostEqual(details["candidate_logit"], math.log(0.7), places=6)
        self.assertEqual(details["max_logit"], 0.0)
        self.assertAlmostEqual(math.exp(-details["logit_gap"]), details["confidence"], places=6)

    def test_no_rejection_details_for_all_accept_or_disabled_metrics(self):
        accepted = self.run_controller(FakeCallbacks(), rounds=1)
        self.assertIsNone(accepted.rounds[0].rejection)
        rejected = self.run_controller(FakeCallbacks([[0.1] * 7]), mode="confidence", rounds=1, enable_metrics=False)
        self.assertEqual(rejected.rounds[0].first_rejection, 0)
        self.assertIsNone(rejected.rounds[0].rejection)

    def test_low_threshold_hands_entire_pending_block_to_target(self):
        callbacks = FakeCallbacks([[1] * 7, [1, 1, 0.2, 1, 1, 1, 1]])
        result = self.run_controller(callbacks, mode="confidence")
        self.assertEqual(result.stop_reason, "low_confidence")
        self.assertEqual(len(result.rounds), 2)
        self.assertEqual(result.tokens.numel(), 14)
        self.assertEqual(result.temporary_accepted_length, 9)
        self.assertEqual(result.rounds[-1].first_rejection, 2)
        self.assertEqual(callbacks.retained, [10, 12])

    def test_first_low_threshold_preserves_pending_block(self):
        callbacks = FakeCallbacks([[0.2] * 7])
        result = self.run_controller(callbacks, mode="confidence")
        self.assertEqual(result.tokens.numel(), 7)
        self.assertEqual(result.temporary_accepted_length, 0)
        self.assertEqual(len(result.rounds), 1)

    def test_request_budget_shortens_final_block(self):
        callbacks = FakeCallbacks()
        result = self.run_controller(callbacks, budget=10)
        self.assertEqual([request.max_tokens for request in callbacks.draft_calls], [7, 3])
        self.assertEqual(result.tokens.numel(), 10)
        self.assertEqual(result.stop_reason, "token_budget")

    def test_zero_budget_does_not_invoke_models(self):
        callbacks = FakeCallbacks()
        result = self.run_controller(callbacks, budget=0)
        self.assertEqual(result.tokens.numel(), 0)
        self.assertEqual(len(callbacks.draft_calls), 0)

    def test_long_candidates_and_all_eight_final_modes(self):
        for mode in ("fixed", "confidence"):
            for final_method in ("strict", "topk", "confidence", "accept_all"):
                for length in (1, 7, 15, 16, 17, 24, 28, 32, 48, 64):
                    with self.subTest(mode=mode, final=final_method, length=length):
                        rounds = (length + 6) // 7
                        callbacks = FakeCallbacks()
                        final = FinalVerificationConfig(method=final_method)
                        result = self.run_controller(
                            callbacks,
                            mode=mode,
                            rounds=rounds,
                            budget=length,
                            final_verification=final,
                        )
                        self.assertEqual(result.tokens.numel(), length)
                        self.assertEqual(
                            result.tokens.tolist(),
                            [int(token) for request in callbacks.verify_calls for token in request.input_ids[1:]],
                        )
                        logits = torch.full((length + 1, 32), -10.0)
                        logits[:-1].scatter_(1, result.tokens[:, None], 0.0)
                        logits[-1, 31] = 10
                        verifier = decision_module.FinalTargetVerifier(final)
                        decision = verifier.verify(
                            logits[-1:] if final_method == "accept_all" else logits, result.tokens
                        )
                        self.assertEqual(int(decision.accepted_length), length)
                        self.assertEqual(int(decision.next_token), 31)

    def test_eos_accepted_trims_following_candidates(self):
        callbacks = FakeCallbacks()
        controller = self.controller()
        result = controller.run_request(
            "eos",
            7,
            3,
            initial_features(),
            callbacks.draft,
            callbacks.verify,
            token_budget=28,
            eos_token_ids=(8,),
        )
        self.assertEqual(result.tokens.tolist(), [6, 7, 8])
        self.assertEqual(result.stop_reason, "eos")

    def test_target_features_never_reuses_stale_context(self):
        callbacks = FakeCallbacks()
        result = self.run_controller(callbacks, feature_source="target")
        self.assertEqual(result.stop_reason, "target_features_unavailable")
        self.assertEqual(len(result.rounds), 1)
        self.assertEqual(result.tokens.numel(), 7)

    def test_requests_have_independent_rounds_and_prefixes(self):
        controller = self.controller(mode="confidence")
        low = FakeCallbacks([[0.1] * 7])
        good = FakeCallbacks()
        first = controller.run_request("low", 7, 3, initial_features(), low.draft, low.verify, token_budget=28)
        second = controller.run_request("good", 7, 3, initial_features(), good.draft, good.verify, token_budget=28)
        self.assertEqual((len(first.rounds), len(second.rounds)), (1, 4))
        self.assertEqual((first.tokens.numel(), second.tokens.numel()), (7, 28))

    def test_bad_feature_positions_order_width_roll_back(self):
        for kind in ("position", "order", "width", "source"):
            callbacks = FakeCallbacks()

            def wrong(request, callbacks=callbacks, kind=kind):
                output = callbacks.verify(request)
                features = output.features
                if kind == "position":
                    features = FeatureBundle(features.values, features.layer_ids, features.start_position + 1, "sparse")
                if kind == "order":
                    features = FeatureBundle(features.values, (4, 1), features.start_position, "sparse")
                if kind == "width":
                    features = FeatureBundle(
                        tuple(value[:, :1] for value in features.values),
                        features.layer_ids,
                        features.start_position,
                        "sparse",
                    )
                if kind == "source":
                    features = FeatureBundle(features.values, features.layer_ids, features.start_position, "target")
                return SparseForwardOutput(output.candidate_logits, features, callbacks.retained.append)

            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.controller().run_request("bad", 7, 3, initial_features(), callbacks.draft, wrong, token_budget=28)
            self.assertEqual(callbacks.retained[-1], 3)

    def test_initial_features_must_cover_anchor(self):
        callbacks = FakeCallbacks()
        with self.assertRaisesRegex(ValueError, "stale or rejected"):
            self.controller().run_request(
                "bad", 7, 4, initial_features(), callbacks.draft, callbacks.verify, token_budget=28
            )

    def test_capacity_must_cover_all_rounds(self):
        with self.assertRaisesRegex(ValueError, "draft_length \\* num_rounds"):
            MultiRoundDFlashController(SparseVerifyConfig(enabled=True), 7, 16)

    def test_target_coverage_requires_matching_actual_token_identity(self):
        for matching in (False, True):
            callbacks = FakeCallbacks()
            # First draft is [6, 7, 8, 9, 10, 11, 12]; next anchor position is
            # 10 and true feature rows through position 9 cover [7,6,..,11].
            ids = torch.tensor([0, 0, 0, 7, 6, 7, 8, 9, 10, 11])
            if not matching:
                ids[7] = 30
            values = torch.zeros(10, 2)
            known = FeatureBundle((values, values + 100), (1, 4), 0, "target", ids)
            result = self.controller(rounds=2, feature_source="target").run_request(
                "known",
                7,
                3,
                initial_features(),
                callbacks.draft,
                callbacks.verify,
                token_budget=14,
                target_features=known,
            )
            self.assertEqual(len(result.rounds), 2 if matching else 1)
            if matching:
                self.assertEqual(callbacks.draft_calls[1].features.source, "target")

    def test_sparse_tokens_after_rejection_never_become_features(self):
        callbacks = FakeCallbacks([[1, 1, 0.7, 1, 1, 1, 1], [1] * 7])
        result = self.run_controller(callbacks, rounds=2)
        # Retained sparse input rows are the previous anchor and first accepted
        # draft. The second accepted draft is the new anchor, not a context row.
        second = callbacks.draft_calls[1]
        self.assertEqual(second.features.token_ids.tolist(), [7, 6])
        self.assertEqual(second.anchor_token, 7)
        self.assertEqual(second.anchor_position, 5)
        self.assertEqual(result.tokens.numel(), 9)


if __name__ == "__main__":
    unittest.main()
