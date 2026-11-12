# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU tests: python examples/dflash_feedback/test_feedback.py -v."""

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[2] / "vllm_ascend/worker/v2/spec_decode/dflash/feedback_calibrator.py"
# Load dependency-light config modules directly: importing the plugin package
# would initialize its logger, which requires a real vLLM installation.
for name in ("config_utils", "dflash_config"):
    config_name = f"vllm_ascend.{name}"
    if config_name not in sys.modules:
        config_spec = importlib.util.spec_from_file_location(config_name, SOURCE.parents[4] / f"{name}.py")
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules[config_name] = config_module
        config_spec.loader.exec_module(config_module)
SPEC = importlib.util.spec_from_file_location("feedback_calibrator_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
FeedbackCalibrator, FeedbackConfig = MODULE.FeedbackCalibrator, MODULE.FeedbackConfig


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def calibrator(self, **kwargs):
        return FeedbackCalibrator(FeedbackConfig(output=str(Path(self.temp.name) / "feedback"), **kwargs))

    def batch(self, ids, tokens, start=11):
        width = len(tokens[0])
        inputs = [[2] + row for row in tokens]
        return SimpleNamespace(
            req_ids=ids,
            logits_indices=torch.arange(len(ids) * (width + 1)),
            positions=torch.tensor([list(range(start - 1, start + width))] * len(ids)).flatten(),
            input_ids=torch.tensor(inputs).flatten(),
            cu_num_logits_np=np.arange(len(ids) + 1) * (width + 1),
            idx_mapping_np=np.arange(len(ids)),
        )

    def verify(self, c, draft, target, ids=None, accepted=None, start=11):
        ids = ids or ["r"]
        tokens = c.propose(draft, ids, torch.arange(start, start + draft.shape[1]).repeat(len(ids), 1))
        batch = self.batch(ids, tokens.tolist(), start)
        n = draft.shape[1]
        targets = torch.cat((target, target[:, -1:]), 1).flatten(0, 1)
        accepted = accepted or [
            next((i for i in range(n) if tokens[r, i] != target[r, i].argmax()), n) for r in range(len(ids))
        ]
        out = SimpleNamespace(num_sampled=torch.tensor(accepted) + 1, num_rejected=n - torch.tensor(accepted))
        c.verify(targets, batch, out, torch.zeros(len(ids), 64, dtype=torch.int64))
        return [
            json.loads(line) for line in c.output.read_text().splitlines() if json.loads(line)["type"] == "verification"
        ]

    def test_empty_and_observe_preserve_argmax(self):
        logits = torch.randn(4, 7, 41)
        for mode in ("observe", "calibrate"):
            c = self.calibrator(mode=mode)
            result = c.propose(logits, list("abcd"), torch.arange(7).repeat(4, 1))
            self.assertTrue(torch.equal(result, logits.argmax(-1)))
            self.assertEqual(c.bias, {})

    def test_normalization_union_ema_rank_and_first_reject(self):
        c = self.calibrator(mode="observe", top_k=1)
        d = torch.tensor([[[4.0, 1.0, 0.0], [4.0, 2.0, 0.0], [0.0, 8.0, 0.0]]])
        t = torch.tensor([[[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [9.0, 0.0, 0.0]]])
        event = self.verify(c, d, t)[0]
        self.assertEqual(event["accepted"], 1)
        self.assertEqual([r["position"] for r in event["rows"]], [11, 12])
        self.assertEqual(event["rows"][1]["context"][-1], 0)
        self.assertEqual(event["rows"][1]["target_rank_in_draft"], 2)
        self.assertEqual(c.bias[0][1], 2)  # Deduplicated union, no third invalid row.
        self.assertEqual(c.bias[1][1], 1)
        delta0 = float(t[0, 0].log_softmax(-1)[0] - d[0, 0].log_softmax(-1)[0])
        delta1 = float(t[0, 1].log_softmax(-1)[0] - d[0, 1].log_softmax(-1)[0])
        expected = 0.9 * 0.1 * delta0 + 0.1 * max(-2, min(2, delta1))
        self.assertAlmostEqual(c.bias[0][0], expected, places=6)

    def test_history_only_threshold_and_change(self):
        c = self.calibrator(mode="calibrate", min_observations=2, strength=2)
        d = torch.tensor([[[1.0, 0.9]]])
        t = torch.tensor([[[0.0, 5.0]]])
        for _ in range(2):
            event = self.verify(c, d, t)[-1]
            self.assertEqual(event["changed"], 0)
        tokens = c.propose(d, ["r"], torch.tensor([[11]]))
        self.assertEqual(tokens.item(), 1)

    def test_multi_request_reorder_cleanup(self):
        c = self.calibrator(mode="observe")
        d = torch.tensor([[[0.0, 1.0]], [[2.0, 0.0]]])
        c.propose(d, ["a", "b"], torch.tensor([[11], [11]]))
        batch = self.batch(["b", "a"], [[0], [1]])
        out = SimpleNamespace(num_sampled=torch.tensor([2, 2]), num_rejected=torch.tensor([0, 0]))
        t = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        c.verify(t, batch, out, torch.zeros(2, 64, dtype=torch.long))
        self.assertEqual(c.pending, {})
        c.prune([])
        self.assertEqual(c.steps, {})

    def test_alignment_failure(self):
        c = self.calibrator(mode="observe")
        c.propose(torch.tensor([[[0.0, 1.0]]]), ["r"], torch.tensor([[12]]))
        with self.assertRaisesRegex(RuntimeError, "alignment"):
            c.verify(
                torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
                self.batch(["r"], [[1]]),
                SimpleNamespace(num_sampled=torch.tensor([2]), num_rejected=torch.tensor([0])),
                torch.zeros(1, 64),
            )

    def test_new_benchmark_reset(self):
        c = self.calibrator(mode="calibrate")
        c.bias[1] = (1.0, 9)
        fresh = self.calibrator(mode="calibrate")
        self.assertFalse(fresh.bias)
        self.assertFalse(fresh.pending)

    def test_config_validation(self):
        for kw in [dict(mode="bad"), dict(alpha=0), dict(top_k=0), dict(strength=float("nan"))]:
            with self.assertRaises(ValueError):
                FeedbackConfig(**kw)

    def test_additional_config_nested_schema(self):
        import dataclasses

        tree = ast.parse((SOURCE.parents[4] / "ascend_config.py").read_text(encoding="utf-8"))
        ascend = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendConfig")
        field = next(n for n in ascend.body if isinstance(n, ast.AnnAssign) and n.target.id == "dflash_calib_config")
        cls = ast.ClassDef(
            name="Schema", bases=[], keywords=[], body=[field], decorator_list=[ast.Name(id="config", ctx=ast.Load())]
        )
        namespace = dict(
            dataclasses=dataclasses,
            DFlashCalibrationConfig=FeedbackConfig,
            config=sys.modules["vllm_ascend.config_utils"].config,
        )
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "ascend_config.py", "exec"),
            namespace,
        )
        schema = namespace["Schema"]
        self.assertEqual(schema().dflash_calib_config.mode, "off")
        options = {
            "mode": "calibrate",
            "top_k": 8,
            "alpha": 0.2,
            "lambda": 0.7,
            "clip_value": 1.5,
            "min_observations": 3,
            "output": "custom",
        }
        parsed = schema(dflash_calib_config=options).dflash_calib_config
        self.assertEqual((parsed.mode, parsed.top_k, parsed.alpha, parsed.strength), ("calibrate", 8, 0.2, 0.7))
        self.assertEqual((parsed.clip_value, parsed.min_observations, parsed.output), (1.5, 3, "custom"))
        for invalid in (
            {"mode": "invalid"},
            {"top_k": 0},
            {"top_k": 1.5},
            {"lambda": -1},
            {"output": ""},
            {"min_observations": False},
            {"typo": 1},
        ):
            with self.subTest(options=invalid), self.assertRaises(ValueError):
                schema(dflash_calib_config=invalid)

    def test_legacy_environment_does_not_enable_feedback(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"DFLASH_CALIB_MODE": "calibrate", "DFLASH_CALIB_TOP_K": "0"}):
            options = FeedbackConfig()
            self.assertEqual((options.mode, options.top_k), ("off", 16))

    def test_off_delegates_without_computing_logits(self):
        # Compile the actual method without importing vLLM/NPU modules.
        source = SOURCE.with_name("speculator.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendDFlashSpeculator")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_draft")
        stub = ast.ClassDef(
            name="Subject", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
        )
        sentinel = object()

        class Base:
            def sample_draft(self, *args, **kwargs):
                return sentinel

        namespace = dict(Base=Base)
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), str(source), "exec"), namespace
        )
        subject = namespace["Subject"]()
        subject.feedback_calibrator = None
        self.assertIs(subject.sample_draft("arbitrary", option=True), sentinel)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_target_dispatch_uses_one_logits_call_then_verifies(self):
        c = self.calibrator(mode="observe")
        c.propose(torch.tensor([[[0.0, 1.0]]]), ["r"], torch.tensor([[11]]))
        batch = self.batch(["r"], [[1]])
        batch.has_structured_output_reqs = False
        batch.num_draft_tokens = 1
        batch.idx_mapping = torch.tensor([0])
        calls = []

        def compute(hidden):
            calls.append("logits")
            return torch.tensor([[0.0, 1.0], [0.0, 1.0]])

        def reject(logits, batch, draft_logits):
            self.assertFalse(c.bias)  # No current-round feedback before sampling.
            calls.append("strict")
            return SimpleNamespace(num_sampled=torch.tensor([2]), num_rejected=torch.tensor([0]))

        runner = SimpleNamespace(
            sampler=SimpleNamespace(sampling_states=SimpleNamespace(temperature=SimpleNamespace(gpu=torch.zeros(1)))),
            speculator=SimpleNamespace(draft_logits=None, feedback_calibrator=c),
            req_states=SimpleNamespace(req_id_to_index={"r": 0}, all_token_ids=SimpleNamespace(gpu=torch.zeros(1, 64))),
            model_state=SimpleNamespace(compute_sample_logits=compute),
            model=SimpleNamespace(compute_logits=compute),
            rejection_sampler=reject,
        )
        MODULE.sample_with_feedback(runner, torch.zeros(2, 8), batch, None)
        self.assertEqual(calls, ["logits", "strict"])
        self.assertEqual(c.target_forward_calls, 1)
        self.assertTrue(c.bias)

    def test_scheduler_truncated_drafts(self):
        c = self.calibrator(mode="observe")
        c.propose(torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]), ["r"], torch.tensor([[11, 12, 13]]))
        batch = self.batch(["r"], [[1]])
        c.verify(
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            batch,
            SimpleNamespace(num_sampled=torch.tensor([2]), num_rejected=torch.tensor([0])),
            torch.zeros(1, 64),
        )
        self.assertTrue(all(n == 1 for _, n in c.bias.values()))
        self.assertFalse(c.pending)

    def test_greedy_output_with_online_changed_candidates(self):
        # A deterministic CPU autoregressive oracle, not a claimed model run.
        c = self.calibrator(mode="calibrate", min_observations=1, strength=4)
        vocab, width = 5, 3
        output = [2]
        for _ in range(12):
            pos = len(output)
            d = torch.zeros(1, width, vocab)
            d[..., 0] = 1
            proposal = c.propose(d, ["r"], torch.arange(pos, pos + width)[None])[0].tolist()
            prefix = output.copy()
            target = []
            for token in proposal:
                row = torch.zeros(vocab)
                row[(prefix[-1] + 1) % vocab] = 5
                target.append(row)
                prefix.append(token)
            accept = next((i for i, row in enumerate(target) if row.argmax() != proposal[i]), width)
            bonus = (prefix[-1] + 1) % vocab if accept == width else int(target[accept].argmax())
            batch = self.batch(["r"], [proposal], start=pos)
            batch.input_ids[0] = output[-1]
            out = SimpleNamespace(num_sampled=torch.tensor([accept + 1]), num_rejected=torch.tensor([width - accept]))
            c.verify(torch.stack(target + [torch.zeros(vocab)]), batch, out, torch.tensor([output + [0] * 100]))
            output.extend(proposal[:accept] + [bonus])
        self.assertEqual(output, [(2 + i) % vocab for i in range(len(output))])


if __name__ == "__main__":
    unittest.main()
