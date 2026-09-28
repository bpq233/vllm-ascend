# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU tests: python -m unittest discover -s tests/ut/worker/v2 -p test_final_verification.py."""

import importlib.util
import math
import sys
import unittest
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(module)
    return module


acceptance = load_module("_test_acceptance", "acceptance.py")
AcceptancePolicy = acceptance.AcceptancePolicy
assemble = acceptance.assemble_verified_tokens


class TestAcceptancePolicy(unittest.TestCase):
    def test_probability_ratio_matches_softmax(self):
        generator = torch.Generator().manual_seed(19)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            logits = torch.randn(64, 47, generator=generator).to(dtype)
            tokens = torch.randint(47, (64,), generator=generator, dtype=torch.int32)
            probabilities = logits.double().softmax(-1)
            ratios = probabilities.gather(1, tokens.long()[:, None]).squeeze(1) / probabilities.amax(-1)
            for threshold in (0, 0.1, 0.5, 0.9, 1):
                with self.subTest(dtype=dtype, threshold=threshold):
                    policy = AcceptancePolicy("prob_ratio", threshold=threshold)
                    torch.testing.assert_close(policy.accept(logits, tokens), ratios > threshold)

    def test_probability_ratio_strict_boundary_ties_and_masks(self):
        logits = torch.tensor([[0.0, -0.5], [0.0, -1.0], [0.0, -2.0], [0.0, 0.0], [0.0, -torch.inf]])
        tokens = torch.ones(5, dtype=torch.long)
        policy = AcceptancePolicy("prob_ratio", threshold=math.exp(-1))
        self.assertEqual(policy.accept(logits, tokens).tolist(), [True, False, False, True, False])
        self.assertEqual(AcceptancePolicy("prob_ratio", threshold=1).accept(logits, tokens).tolist(), [False] * 5)
        invalid = torch.tensor([[-torch.inf, -torch.inf], [torch.nan, 0], [torch.inf, 0], [0, -torch.inf]])
        self.assertEqual(
            AcceptancePolicy("prob_ratio", threshold=0).accept(invalid, torch.ones(4, dtype=torch.long)).tolist(),
            [False] * 4,
        )
        # A zero threshold accepts even very small positive ratios without exp underflow.
        self.assertTrue(AcceptancePolicy("prob_ratio", threshold=0).accept(torch.tensor([[0.0, -1000]]), tokens[:1]))

    def test_probability_ratio_rejects_invalid_thresholds(self):
        for threshold in (-0.1, 1.1, math.nan, math.inf, -math.inf, True, "0.5", None):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "threshold"):
                AcceptancePolicy("prob_ratio", threshold=threshold)

    def test_probability_ratio_long_ragged_prefix_and_target_bonus(self):
        widths = [0, 3, 17, 32]
        boundaries = torch.tensor([0, *np.cumsum([width + 1 for width in widths])])
        size = int(boundaries[-1])
        drafts = torch.arange(size) % 47
        logits = torch.full((size, 49), -10.0)
        logits[:, 48] = 0.0
        logits[torch.arange(size), drafts.roll(-1)] = -0.25
        # Ratios exp(-0.25) pass 0.5. The selected failure exp(-1) does not;
        # later passing tokens must not be returned after the first failure.
        target = torch.full((size,), 47)
        for failure in (0, 15, 16, 31, 32):
            scores = logits.clone()
            last_start = int(boundaries[-2])
            if failure < 32:
                row = last_start + failure
                scores[row, drafts[row + 1]] = -1.0
            sampled, counts = assemble(
                scores, drafts, target, boundaries, 32, AcceptancePolicy("prob_ratio", threshold=0.5)
            )
            self.assertEqual(counts.tolist(), [1, 4, 18, failure + 1])
            for req, accepted in enumerate([0, 3, 17, failure]):
                start = int(boundaries[req])
                self.assertEqual(sampled[req, :accepted].tolist(), drafts[start + 1 : start + accepted + 1].tolist())
                self.assertEqual(sampled[req, accepted], 47)
                self.assertTrue((sampled[req, accepted + 1 :] == -1).all())

    def test_topk_membership_and_mask(self):
        logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [1.0, -torch.inf, -torch.inf]])
        tokens = torch.tensor([1, 0, 1])
        self.assertEqual(AcceptancePolicy(top_k=2).accept(logits, tokens).tolist(), [True, False, False])
        self.assertEqual(AcceptancePolicy(top_k=99).accept(logits, tokens).tolist(), [True, True, False])
        self.assertEqual(AcceptancePolicy("all").accept(logits, tokens).tolist(), [True, True, True])

    def test_invalid_policy(self):
        for kwargs in ({"method": "other"}, {"top_k": 0}, {"top_k": 1.5}, {"top_k": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AcceptancePolicy(**kwargs)

    def test_first_rejection_bonus_and_no_draft_in_one_batch(self):
        # First request accepts 1, rejects 2, then must ignore a passing 3.
        # Second accepts both drafts; third has no draft.
        logits = torch.full((8, 5), -10.0)
        logits[torch.arange(8), torch.tensor([1, 4, 3, 0, 2, 3, 1, 4])] = 10.0
        drafts = torch.tensor([0, 1, 2, 3, 0, 2, 3, 0])
        targets = logits.argmax(-1)
        sampled, lengths = assemble(logits, drafts, targets, torch.tensor([0, 4, 7, 8]), 3, AcceptancePolicy(top_k=1))
        self.assertEqual(sampled.tolist(), [[1, 4, -1, -1], [2, 3, 1, -1], [4, -1, -1, -1]])
        self.assertEqual(lengths.tolist(), [2, 3, 1])
        self.assertEqual(lengths.dtype, torch.int32)

    def test_all_skips_acceptance_with_nonfinite_logits_and_ragged_bonus(self):
        # Empty, two-token and one-token drafts share the same padded output.
        # All-policy must ignore even nonfinite logits and retain the supplied
        # sampled bonus rather than accidentally reading the next request.
        logits = torch.tensor([[float("nan"), -torch.inf, torch.inf]]).expand(6, -1)
        drafts = torch.tensor([0, 0, 1, 2, 0, 2], dtype=torch.int32)
        targets = torch.tensor([2, 1, 0, 1, 0, 1], dtype=torch.int32)
        with patch.object(AcceptancePolicy, "accept", side_effect=AssertionError("unnecessary acceptance")):
            sampled, counts = assemble(logits, drafts, targets, torch.tensor([0, 1, 4, 6]), 3, AcceptancePolicy("all"))
        self.assertEqual(sampled.tolist(), [[2, -1, -1, -1], [1, 2, 1, -1], [2, 1, -1, -1]])
        self.assertEqual(counts.tolist(), [1, 3, 2])
        self.assertEqual(sampled.dtype, torch.int64)
        self.assertEqual(counts.dtype, torch.int32)

    def test_all_zero_capacity_returns_each_target_bonus(self):
        sampled, counts = assemble(
            torch.zeros(2, 4),
            torch.tensor([1, 3]),
            torch.tensor([2, 0]),
            torch.tensor([0, 1, 2]),
            0,
            AcceptancePolicy("all"),
        )
        self.assertEqual(sampled.tolist(), [[2], [0]])
        self.assertEqual(counts.tolist(), [1, 1])

    def test_batched_prefix_matches_sequential_reference(self):
        rng = torch.Generator().manual_seed(71)
        lengths = [1, 5, 3, 2, 4]
        boundaries = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32)
        logits = torch.randn(sum(lengths), 7, generator=rng)
        drafts = torch.randint(7, (sum(lengths),), generator=rng)
        # Distinct supplied samples verify replacement uses target sampling,
        # rather than unconditionally taking the argmax.
        target = torch.randint(7, (sum(lengths),), generator=rng)
        for method in ("topk", "all"):
            for k in (1, 3, 7):
                sampled, counts = assemble(logits, drafts, target, boundaries, 4, AcceptancePolicy(method, k))
                for req, (start, end) in enumerate(zip(boundaries[:-1].tolist(), boundaries[1:].tolist())):
                    expected = []
                    for row in range(start, end - 1):
                        if method != "all" and drafts[row + 1] not in logits[row].topk(k).indices:
                            break
                        expected.append(drafts[row + 1].item())
                    expected.append(target[start + len(expected)].item())
                    self.assertEqual(sampled[req, : counts[req]].tolist(), expected)

    def test_long_ragged_batch_rejection_boundary_and_bonus(self):
        widths = [0, 15, 16, 32]
        boundaries = torch.tensor([0, *np.cumsum([w + 1 for w in widths])])
        size = int(boundaries[-1])
        drafts = torch.arange(size) % 47
        logits = torch.full((size, 49), -10.0)
        next_tokens = drafts.roll(-1)
        logits[torch.arange(size), next_tokens] = 10.0
        target = torch.full((size,), 48)  # Distinct supplied target samples.
        for fail in (0, 15, 16, 31, 32):
            scores = logits.clone()
            start = int(boundaries[-2])
            if fail < 32:
                scores[start + fail, :] = -10.0
                scores[start + fail, 47] = 10.0
            sampled, counts = assemble(scores, drafts, target, boundaries, 32, AcceptancePolicy(top_k=1))
            self.assertEqual(counts.tolist(), [1, 16, 17, fail + 1])
            for i, width in enumerate([0, 15, 16, fail]):
                begin = int(boundaries[i])
                self.assertEqual(sampled[i, :width].tolist(), drafts[begin + 1 : begin + width + 1].tolist())
                self.assertEqual(sampled[i, width], 48)
                self.assertTrue((sampled[i, width + 1 :] == -1).all())
            _, all_counts = assemble(scores, drafts, target, boundaries, 32, AcceptancePolicy("all"))
            self.assertEqual(all_counts.tolist(), [1, 16, 17, 33])


class TestFinalVerificationSampler(unittest.TestCase):
    def test_sampling_metadata_forwarded_and_draft_probabilities_ignored(self):
        parent = ModuleType("vllm.v1.worker.gpu.spec_decode.rejection_sampler")

        class RejectionSampler:
            def __init__(self, sampler, spec_config, device):
                self.sampler = sampler
                self.num_speculative_steps = spec_config.num_speculative_tokens

        parent.RejectionSampler = RejectionSampler
        utils = ModuleType("vllm_ascend.utils")
        utils.vllm_version_is = MagicMock(return_value=False)
        with patch.dict(
            sys.modules,
            {
                parent.__name__: parent,
                utils.__name__: utils,
                "vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance": acceptance,
            },
        ):
            module = load_module("_test_final_verification", "final_verification.py")
        logits = torch.tensor([[3.0, 2.0, 1.0], [2.0, 1.0, 3.0]])
        processed = torch.tensor([[1.0, 3.0, 2.0], [3.0, 1.0, 2.0]])
        sampler = MagicMock()
        sampler.sample.return_value = (torch.tensor([2, 0]), processed)
        verifier = module.FinalVerificationSampler(
            sampler, SimpleNamespace(num_speculative_tokens=1), torch.device("cpu"), AcceptancePolicy(top_k=1)
        )
        draft = torch.tensor([0, 1])
        pos, cumulative, mapping = torch.tensor([5, 6]), torch.tensor([0, 2]), torch.tensor([3])
        mapping_np, expanded, local = np.array([3]), torch.tensor([3, 3]), torch.tensor([0, 1])
        for (legacy, draft_logits), policy in product(
            ((False, None), (False, torch.randn(4, 1, 3)), (True, None)),
            (AcceptancePolicy(top_k=1), AcceptancePolicy("prob_ratio", threshold=0.5)),
        ):
            verifier.policy = policy
            utils.vllm_version_is.return_value = legacy
            result, sampled, counts = verifier._verify(
                logits, draft_logits, draft, pos, cumulative, mapping, mapping_np, expanded, local
            )
            self.assertIs(result, processed)
            self.assertEqual(sampled.tolist(), [[1, 0]])
            self.assertEqual(counts.tolist(), [2])
            kwargs = sampler.sample.call_args.kwargs
            for name, expected in dict(
                logits=logits,
                expanded_idx_mapping=expanded,
                idx_mapping_np=mapping_np,
                pos=pos,
                input_ids=draft,
                expanded_local_pos=local,
            ).items():
                self.assertIs(kwargs[name], expected)
            self.assertEqual("idx_mapping" in kwargs, not legacy)
            self.assertTrue(kwargs["return_logprobs"])

        verifier.trace_forward = True
        verifier.forward_events = (MagicMock(), MagicMock())
        verifier.forward_events[0].elapsed_time.return_value = 1.0
        with patch.object(module.logger, "debug") as debug:
            verifier._verify(logits, None, draft, pos, cumulative, mapping, mapping_np, expanded, local)
        verifier.forward_events[1].synchronize.assert_not_called()
        self.assertEqual(debug.call_args.args[1:3], ([1], [1]))


if __name__ == "__main__":
    unittest.main()
