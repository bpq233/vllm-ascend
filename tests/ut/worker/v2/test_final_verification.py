# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU tests: python -m unittest discover -s tests/ut/worker/v2 -p test_final_verification.py."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode"


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

    def test_batched_prefix_matches_sequential_reference(self):
        rng = torch.Generator().manual_seed(71)
        lengths = [1, 5, 3, 2, 4]
        boundaries = torch.tensor([0, *np.cumsum(lengths)])
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
                "vllm_ascend.worker.v2.spec_decode.acceptance": acceptance,
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
        for legacy, draft_logits in ((False, None), (False, torch.randn(4, 1, 3)), (True, None)):
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


if __name__ == "__main__":
    unittest.main()
