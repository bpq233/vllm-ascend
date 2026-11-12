# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Variable proposal widths must not schedule the -1 padded suffix."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


class NativeHandler:
    def __init__(self, device=None):
        self._draft_tokens_by_req = {}

    def set_draft_tokens(self, batch, tokens):
        # Pinned upstream's fast path transports widths, not token IDs.
        for req_id in batch.req_ids:
            self._draft_tokens_by_req[req_id] = tokens.shape[1]

    def get_draft_tokens(self):
        return {req_id: [-1] * width for req_id, width in self._draft_tokens_by_req.items()}

    def discard(self, req_id):
        self._draft_tokens_by_req.pop(req_id, None)


def load_handler():
    source = (
        Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify/draft_tokens.py"
    )
    utils = ModuleType("vllm.v1.worker.gpu.spec_decode.utils")
    utils.DraftTokensHandler = NativeHandler
    spec = importlib.util.spec_from_file_location("_actual_sparse_draft_tokens", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {utils.__name__: utils}):
        spec.loader.exec_module(module)
    return module.SparseDraftTokensHandler


class TestSparseDraftWidths(unittest.TestCase):
    def test_actual_widths_exclude_padding_before_target_forward(self):
        lengths = [0, 1, 7, 17, 28, 60]
        batch = SimpleNamespace(req_ids=[f"r{i}" for i in range(len(lengths))], has_structured_output_reqs=False)
        drafts = torch.full((len(lengths), 60), -1, dtype=torch.int32)
        for row, length in enumerate(lengths):
            drafts[row, :length] = torch.arange(1, length + 1)
        old = NativeHandler()
        old.set_draft_tokens(batch, drafts)
        self.assertEqual([len(x) for x in old.get_draft_tokens().values()], [60] * len(lengths))
        handler = load_handler()()
        handler.set_draft_tokens(batch, drafts)
        for row, placeholders in enumerate(handler.get_draft_tokens().values()):
            self.assertEqual(len(placeholders), lengths[row])
            # Scheduler uses placeholder count to select real RequestState IDs.
            target_ids = drafts[row, : len(placeholders)]
            self.assertTrue((target_ids >= 0).all())
        self.assertTrue((drafts[0] == -1).all())

    def test_updates_and_removal_follow_request_lifetime(self):
        handler = load_handler()()
        batch = SimpleNamespace(req_ids=["r"], has_structured_output_reqs=False)
        handler.set_draft_tokens(batch, torch.tensor([[1, 2, -1]]))
        handler.set_draft_tokens(batch, torch.tensor([[-1, -1, -1]]))
        self.assertEqual(handler.get_draft_tokens(), {"r": []})
        handler.discard("r")
        self.assertEqual(handler.get_draft_tokens(), {})

    def test_empty_budget_and_malformed_prefixes(self):
        handler = load_handler()()
        batch = SimpleNamespace(req_ids=["r"], has_structured_output_reqs=False)
        handler.set_draft_tokens(batch, torch.empty(1, 0, dtype=torch.int32))
        self.assertEqual(handler.get_draft_tokens(), {"r": []})
        for ids in ([[1, -1, 2]], [[1, -2, -1]]):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "prefixes"):
                handler.set_draft_tokens(batch, torch.tensor(ids))


if __name__ == "__main__":
    unittest.main()
