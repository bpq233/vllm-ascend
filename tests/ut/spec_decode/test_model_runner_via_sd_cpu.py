"""Execute actual runner methods extracted by AST without the NPU imports."""

import ast
import copy
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
RUNNER = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")


def runner_class(names, namespace=None):
    names = names | {"_via_sd_config"}
    methods = [copy.deepcopy(node) for node in RUNNER.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for method in methods:
        method.decorator_list = []
    node = ast.ClassDef(
        name="Runner", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    parent = type("Parent", (), {"execute_model": Mock(), "sample_tokens": Mock(), "__init__": Mock(return_value=None)})
    scope = {"Parent": parent, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope["Runner"], parent


class RunnerContractTests(unittest.TestCase):
    def test_hierarchy_sample_dispatches_without_parent(self):
        cls, parent = runner_class({'sample_tokens', '_via_sd_mode'})
        runner = cls()
        runner.ascend_config = types.SimpleNamespace(via_sd_config=types.SimpleNamespace(enabled=True, mode='hierarchical'))
        runner._via_sd_runtime = types.SimpleNamespace(sample=Mock(return_value='hierarchy'))
        self.assertEqual(runner.sample_tokens(None), 'hierarchy')
        parent.sample_tokens.assert_not_called()

    def test_observe_verifies_without_building_plan(self):
        torch = MagicMock()
        cls, _ = runner_class({"_run_via_sd_validation_once"}, {"torch": torch})
        runner = cls()
        runner._via_sd_real_drafts = lambda value: value
        runner._via_sd_mode = lambda: "observe"
        runner._via_sd_config = lambda: object()
        runner._via_sd_timing_enabled = lambda: False
        runner._via_sd_logging_enabled = lambda: False
        runner.num_speculative_steps = 2
        runner.device = "cpu"
        runner._via_sd_qprime_validation_count = 0
        runner.block_tables = types.SimpleNamespace(input_block_tables=None)
        runner.via_sd_verifier = Mock()
        runner.req_states = MagicMock()
        token_view = runner.req_states.all_token_ids.gpu.__getitem__.return_value
        token_view.detach.return_value.cpu.return_value.tolist.return_value = [8, 9]
        mapping = MagicMock()
        mapping.__getitem__.return_value.detach.return_value.cpu.return_value.tolist.return_value = [0]
        mapping.__getitem__.return_value.to.return_value.detach.return_value.cpu.return_value.tolist.return_value = [0]
        batch = types.SimpleNamespace(num_reqs=1, req_ids=["a"], idx_mapping=mapping, num_computed_tokens_np=None)
        length_view = runner.req_states.total_len.gpu.__getitem__.return_value
        length_view.detach.return_value.cpu.return_value.tolist.return_value = [2]
        result = runner._run_via_sd_validation_once(batch, {"a": (1, 2)})
        self.assertIsNone(result)
        runner.via_sd_verifier.verify.assert_called_once()
        runner.via_sd_verifier.build_route_plan.assert_not_called()
        self.assertIs(runner.via_sd_last_logits, runner.via_sd_verifier.verify.return_value)

    def test_no_forward_timing_hooks(self):
        calls = [
            node.func.attr
            for node in ast.walk(RUNNER)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("register_forward_pre_hook", calls)
        self.assertNotIn("register_forward_hook", calls)
        self.assertIn("scope=parent_execute_model", SOURCE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
