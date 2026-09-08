import ast
import types
import unittest
from pathlib import Path

import torch


class FeatureTests(unittest.TestCase):
    def test_feature_capture_flag_restored_on_failure(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/via_sd/backend.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ViaSdPagedBackend")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_features")
        scope = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), scope)
        model = types.SimpleNamespace(capture_draft_features=False)

        def fail(*args, **kwargs):
            self.assertTrue(model.capture_draft_features)
            self.assertTrue(kwargs["return_features"])
            raise RuntimeError("forward failed")

        backend = types.SimpleNamespace(model=model, forward_batch=fail)
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            scope["forward_features"](backend, [[1]], [0], [0])
        self.assertFalse(model.capture_draft_features)

    def test_sparse_auxiliary_boundaries_and_normalized_output(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/via_sd/model.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ViaSdModel")
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        module = ast.Module(body=[forward], type_ignores=[])
        scope = dict(torch=torch, Any=object)
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        calls = []

        def layer_forward(layer, positions, hidden, residual):
            calls.append(layer)
            combined = hidden if residual is None else hidden + residual
            return torch.full_like(hidden, layer), combined

        model = types.SimpleNamespace(
            embed_tokens=lambda ids: ids.float().unsqueeze(-1),
            layer_ids=(0, 2),
            layers=(10, 20),
            total_layers=4,
            aux_hidden_state_layers=(0, 1, 2, 3, 4),
            capture_draft_features=True,
            adapter=types.SimpleNamespace(forward=layer_forward),
            norm=lambda h, r: (h + r) / 2,
        )
        result = scope["forward"](model, torch.tensor([1, 2]), torch.tensor([0, 1]))
        self.assertEqual(calls, [10, 20])
        expected = [[1, 2], [11, 12], [11, 12], [31, 32], [31, 32]]
        for value, row in zip(model.last_aux_hidden_states, expected):
            torch.testing.assert_close(value[:, 0], torch.tensor(row, dtype=torch.float32))
        torch.testing.assert_close(result[:, 0], torch.tensor([15.5, 16.0]))
        scope["forward"](model, torch.tensor([9]), torch.tensor([3]))
        self.assertEqual(len(model.last_aux_hidden_states), 5)
        self.assertEqual(model.last_aux_hidden_states[0].shape[0], 1)
        model.capture_draft_features = False
        scope["forward"](model, torch.tensor([9]), torch.tensor([3]))
        self.assertEqual(model.last_aux_hidden_states, [])


if __name__ == "__main__":
    unittest.main()
