# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run the production proposal method through real multi-round CPU decisions."""

import ast
import importlib.util
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import torch


def load_speculator():
    directory = Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify"
    name = "_sparse_propose_regression"
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    modules = {}
    for leaf in ("config", "decisions", "controller"):
        spec = importlib.util.spec_from_file_location(f"{name}.{leaf}", directory / f"{leaf}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[leaf] = module
    tree = ast.parse((directory / "speculator.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SparseDFlashSpeculator")
    propose = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "propose")

    class NativeDraft:
        def propose(self, batch, metadata, slots, hidden, aux, sampled, rejected, last_sampled, *args, **kwargs):
            # Native input prep indexes last_sampled storage by request-state
            # offset; require the CPU view to agree with that flat layout.
            self.native_anchors.append(last_sampled.clone())
            anchor = last_sampled[batch.idx_mapping[0]]
            return (anchor + torch.arange(1, self.num_speculative_steps + 1)).reshape(1, -1)

    def prepare_batch(template, buffers, ids, positions, index, req_id):
        return SimpleNamespace(idx_mapping=torch.tensor([index]), input_ids=ids, positions=positions)

    namespace = {
        "torch": torch,
        "asdict": asdict,
        "logger": MagicMock(),
        "NativeDraft": NativeDraft,
        "FeatureBundle": modules["controller"].FeatureBundle,
        "SparseForwardOutput": modules["controller"].SparseForwardOutput,
        "single_request_batch": prepare_batch,
    }
    actual = ast.ClassDef(
        name=cls.name,
        bases=[ast.Name(id="NativeDraft", ctx=ast.Load())],
        keywords=[],
        body=[propose],
        decorator_list=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[actual], type_ignores=[])),
            str(directory / "speculator.py"),
            "exec",
        ),
        namespace,
    )
    return namespace[cls.name], modules


class SparseRunner:
    def forward(self, template, ids, positions, index, req_id, **kwargs):
        logits = torch.full((ids.numel(), 64), -50.0)
        if ids.numel() > 1:
            logits[:-1].scatter_(1, ids[1:, None].long(), 0.0)
        features = positions.float().unsqueeze(1).repeat(1, 3)
        return logits, [features]


class TestSpeculatorAnchors(unittest.TestCase):
    def run_proposal(self, column, mode="fixed"):
        cls, modules = load_speculator()
        speculator = cls()
        config = modules["config"].SparseVerifyConfig(enabled=True, mode=mode, num_rounds=2, enable_metrics=False)
        speculator.sparse_config = config
        speculator.sparse_metrics = SimpleNamespace(enabled=False)
        speculator.controller = modules["controller"].MultiRoundDFlashController(config, 2, 4)
        speculator.sparse_model = SimpleNamespace(aux_layer_ids=(1,))
        speculator.sparse_runner = SparseRunner()
        speculator.total_candidates = 4
        speculator.max_model_len = 64
        speculator.num_speculative_steps = 2
        speculator.num_query_per_req = 3
        speculator.device = torch.device("cpu")
        speculator.target_vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=8),
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(eos_token_id=None)),
        )
        speculator.request_sampling = {}
        speculator.draft_context_buffers = object()
        speculator.block_tables = SimpleNamespace(gather_block_tables=MagicMock())
        speculator.native_anchors = []
        batch = SimpleNamespace(
            num_reqs=2,
            idx_mapping=torch.tensor([2, 0]),
            idx_mapping_np=[2, 0],
            req_ids=["second-state", "first-state"],
            query_start_loc_np=[0, 2, 4],
            input_ids=torch.tensor([7, 9, 17, 19]),
            positions=torch.tensor([0, 1, 0, 1]),
            is_prefilling_np=[True, True],
        )
        last_sampled = torch.tensor([20, 30, 10])
        if column:
            last_sampled = last_sampled[:, None]
        original = last_sampled.clone()
        result = speculator.propose(
            batch,
            {},
            {},
            torch.zeros(4, 3),
            [torch.zeros(4, 3)],
            torch.ones(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            last_sampled,
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
        )
        torch.testing.assert_close(last_sampled, original)
        self.assertEqual(result.tolist(), [[11, 12, 13, 14], [21, 22, 23, 24]])
        self.assertEqual([tuple(x.shape) for x in speculator.native_anchors], [(3,)] * 4)
        # Each inner round updates only its own local state-indexed anchor.
        self.assertEqual(
            [x.tolist() for x in speculator.native_anchors], [[20, 30, 10], [20, 30, 12], [20, 30, 10], [22, 30, 10]]
        )

    def test_native_column_anchors_two_requests_and_two_rounds(self):
        self.run_proposal(column=True)

    def test_legacy_vector_anchors(self):
        self.run_proposal(column=False)

    def test_confidence_mode_native_column_anchors(self):
        self.run_proposal(column=True, mode="confidence")


if __name__ == "__main__":
    unittest.main()
