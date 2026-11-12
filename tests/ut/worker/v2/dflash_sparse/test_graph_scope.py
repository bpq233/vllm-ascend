# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Test actual GraphParams scoping code without importing NPU libraries.

AST extraction executes the production dataclasses, context manager and access
functions. Only unrelated module imports and graph-execution classes are left
out; this is state-ownership evidence, not hardware graph-execution evidence.
"""

import ast
import importlib.util
import sys
import unittest
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

import torch


def load_graph_params_contract():
    source = Path(__file__).resolve().parents[5] / "vllm_ascend/compilation/acl_graph.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {
        "GraphParams",
        "GraphParamsOwner",
        "graph_params_scope",
        "set_graph_params",
        "get_graph_params",
        "update_graph_params_workspaces",
        "set_draft_graph_params",
        "get_draft_graph_params",
        "update_draft_graph_params_workspaces",
    }
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    # Include helper functions introduced by the implementation, if any.
    previous_names = set()
    while names != previous_names:
        previous_names = names.copy()
        for name in previous_names:
            for node in ast.walk(definitions[name]):
                if isinstance(node, ast.Name) and node.id in definitions:
                    names.add(node.id)
    referenced = set().union(
        *({node.id for node in ast.walk(definitions[name]) if isinstance(node, ast.Name)} for name in names)
    )
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in referenced for target in targets):
                selected.append(node)
    selected.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    extracted = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    spec = importlib.util.spec_from_loader("_sparse_graph_params_contract", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(
        dataclass=dataclass, field=field, contextmanager=contextmanager, ContextVar=ContextVar, torch=torch
    )
    sys.modules[spec.name] = module
    exec(compile(extracted, str(source), "exec"), module.__dict__)
    return module


class TestGraphParamsScope(unittest.TestCase):
    def setUp(self):
        self.contract = load_graph_params_contract()

    def test_target_sparse_and_draft_are_independent(self):
        graph = self.contract
        graph.set_graph_params([1, 8, 32])
        target = graph.get_graph_params()
        graph.set_draft_graph_params([8, 16])
        draft = graph.get_draft_graph_params()
        sparse_owner = graph.GraphParamsOwner()
        with graph.graph_params_scope(sparse_owner):
            self.assertIsNone(graph.get_graph_params())
            graph.set_graph_params([4, 8])
            sparse = graph.get_graph_params()
            self.assertIsNot(sparse, target)
            self.assertEqual(set(sparse.events), {4, 8})
            self.assertIs(graph.get_draft_graph_params(), draft)
            sparse.events[8].append("sparse event")
            graph.update_graph_params_workspaces(8, "sparse workspace")
        self.assertIs(graph.get_graph_params(), target)
        self.assertEqual(target.events[8], [])
        self.assertIsNone(target.workspaces[8])
        self.assertEqual(sparse.workspaces[8], "sparse workspace")
        self.assertEqual(draft.events[8], [])

    def test_nested_scopes_restore_each_owner(self):
        graph = self.contract
        graph.set_graph_params([1])
        target = graph.get_graph_params()
        outer, inner = graph.GraphParamsOwner(), graph.GraphParamsOwner()
        with graph.graph_params_scope(outer):
            graph.set_graph_params([2])
            outer_params = graph.get_graph_params()
            with graph.graph_params_scope(inner):
                graph.set_graph_params([3])
                self.assertEqual(set(graph.get_graph_params().events), {3})
            self.assertIs(graph.get_graph_params(), outer_params)
            with graph.graph_params_scope(outer):
                self.assertIs(graph.get_graph_params(), outer_params)
        self.assertIs(graph.get_graph_params(), target)

    def test_exception_resets_scope_without_overwriting_target(self):
        graph = self.contract
        graph.set_graph_params([1])
        target = graph.get_graph_params()
        with self.assertRaisesRegex(RuntimeError, "capture failed"), graph.graph_params_scope(graph.GraphParamsOwner()):
            graph.set_graph_params([2])
            raise RuntimeError("capture failed")
        self.assertIs(graph.get_graph_params(), target)

    def test_reinitializing_same_owner_is_rejected(self):
        graph = self.contract
        owner = graph.GraphParamsOwner()
        with graph.graph_params_scope(owner):
            graph.set_graph_params([8])
            with self.assertRaisesRegex(ValueError, "already"):
                graph.set_graph_params([16])
        with graph.graph_params_scope(owner):
            self.assertEqual(set(graph.get_graph_params().events), {8})
        graph.set_graph_params([1])
        with self.assertRaisesRegex(ValueError, "already"):
            graph.set_graph_params([2])

    def test_draft_workspace_is_unchanged_by_sparse_scope(self):
        graph = self.contract
        graph.set_graph_params([8])
        graph.set_draft_graph_params([8])
        graph.update_draft_graph_params_workspaces(8, "draft workspace")
        with graph.graph_params_scope(graph.GraphParamsOwner()):
            graph.set_graph_params([8])
            graph.update_graph_params_workspaces(8, "sparse workspace")
            self.assertEqual(graph.get_draft_graph_params().workspaces[8], "draft workspace")
        self.assertIsNone(graph.get_graph_params().workspaces[8])
        self.assertEqual(graph.get_draft_graph_params().workspaces[8], "draft workspace")


if __name__ == "__main__":
    unittest.main()
