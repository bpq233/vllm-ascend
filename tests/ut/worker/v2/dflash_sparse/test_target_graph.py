# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise the real Target graph factory and query padding on CPU."""

import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def load_contracts():
    path = Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "graph_manager_wrapper")
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    padding = next(
        n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "_pad_adaptive_query_start_loc_for_fia"
    )
    native = object()
    upstream = SimpleNamespace(ModelCudaGraphManager=native)
    namespace = {
        "contextmanager": contextmanager,
        "Any": Any,
        "np": np,
        "vllm_model_runner": upstream,
        "ModelAclGraphManager": lambda *a, **kw: kw,
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), factory, padding],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace, upstream, native


class TestTargetGraph(unittest.TestCase):
    def test_factory_selects_varlen_only_for_sparse_or_native_adaptive(self):
        namespace, upstream, native = load_contracts()
        for sparse in (False, True):
            for adaptive in (False, True):
                with self.subTest(sparse=sparse, adaptive=adaptive):
                    speculator = SimpleNamespace(sparse_config=object()) if sparse else SimpleNamespace()
                    with namespace["graph_manager_wrapper"](SimpleNamespace(speculator=speculator)):
                        result = upstream.ModelCudaGraphManager(None, None, None, 29, varlen_decode=adaptive)
                        self.assertEqual(result["varlen_decode"], sparse or adaptive)
                    self.assertIs(upstream.ModelCudaGraphManager, native)

    def test_factory_restores_on_error(self):
        namespace, upstream, native = load_contracts()
        with (
            self.assertRaisesRegex(RuntimeError, "failed"),
            namespace["graph_manager_wrapper"](SimpleNamespace(speculator=None)),
        ):
            raise RuntimeError("failed")
        self.assertIs(upstream.ModelCudaGraphManager, native)

    def test_padding_preserves_every_real_candidate_boundary(self):
        namespace, _, _ = load_contracts()
        pad = namespace["_pad_adaptive_query_start_loc_for_fia"]
        for lengths in ((3,), (1, 7, 29), (17, 24, 28, 32, 48, 64)):
            for extra in (0, 1, 17):
                with self.subTest(lengths=lengths, extra=extra):
                    starts = np.zeros(32, dtype=np.int32)
                    starts[1 : len(lengths) + 1] = np.cumsum(lengths)
                    expected = starts[: len(lengths) + 1].copy()
                    padded_requests = len(lengths) + 3
                    output, count = pad(None, sum(lengths) + extra, padded_requests, len(lengths), starts)
                    np.testing.assert_array_equal(output[: len(lengths) + 1], expected)
                    self.assertTrue(np.all(np.diff(output[: count + 1]) >= 0))
                    self.assertEqual(output[count], sum(lengths) + extra)


if __name__ == "__main__":
    unittest.main()
