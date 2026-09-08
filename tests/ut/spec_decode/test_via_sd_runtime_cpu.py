"""Exercise the real runtime with CPU tensors and boundary doubles."""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.distributed.tensor

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_via_runtime_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "vllm_ascend/worker/v2/spec_decode/via_sd")]
sys.modules[PACKAGE] = package
runtime_module = __import__(PACKAGE + ".runtime", fromlist=["ViaSdRuntime"])
ViaSdRuntime = runtime_module.ViaSdRuntime
NS = types.SimpleNamespace


class RuntimeTests(unittest.TestCase):
    def test_target_features_override_only_valid_positions(self):
        runtime, r, batch = self.make_runtime(torch.zeros((3, 3)))
        state = runtime.coordinator.states["r"]
        state.commit([0, 1, 2], qprime_computed_len=2, target_computed_len=2)
        hidden = torch.ones((3, 3))
        aux = [hidden + 10]
        runtime.target_features["r"] = [(0, hidden * 5, [hidden * 15])]
        selected, selected_aux = runtime._draft_features(batch, hidden, aux)
        torch.testing.assert_close(selected[:2], hidden[:2] * 5)
        torch.testing.assert_close(selected[2:], hidden[2:])
        torch.testing.assert_close(selected_aux[0][:2], hidden[:2] * 15)
        torch.testing.assert_close(selected_aux[0][2:], aux[0][2:])
        torch.testing.assert_close(hidden, torch.ones_like(hidden))
        state.commit([0, 9], qprime_computed_len=1, target_computed_len=1)
        selected, _ = runtime._draft_features(batch, hidden, aux)
        torch.testing.assert_close(selected[1:], hidden[1:])

    def test_low_feature_capture_reaches_original_drafter(self):
        runtime, r, batch = self.make_runtime(torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 3.0]]))
        target = NS(
            last_hidden_states=torch.tensor([[9.0, 8.0, 7.0]]),
            last_aux_hidden_states=[torch.tensor([[19.0, 18.0, 17.0]])],
        )
        def target_batch(token_ids, starts, table_indices):
            runtime._target_backend.last_forward_hidden = torch.tensor([[9.0, 8.0, 7.0]]).repeat(sum(map(len, token_ids)), 1)
            runtime._target_backend.last_forward_aux = [torch.tensor([[19.0, 18.0, 17.0]]).repeat(sum(map(len, token_ids)), 1)]
            return [torch.tensor([[0.0, 3.0, 0.0]]).repeat(len(tokens), 1) for tokens in token_ids]

        runtime._target_backend = NS(
            model=target,
            set_request_block_tables=Mock(),
            forward=Mock(return_value=torch.tensor([[0.0, 3.0, 0.0]])),
            forward_batch=Mock(side_effect=target_batch),
        )
        r.rejection_sampler = NS(_verify=Mock(return_value=(None, torch.tensor([[1, 0, -1]]), torch.tensor([2]))))
        result = runtime.sample(None)
        self.assertEqual(result.sampled_token_ids, [[1, 2]])
        args = r.speculator.propose.call_args.args
        torch.testing.assert_close(args[3][0], torch.tensor([9., 8., 7.]))
        torch.testing.assert_close(args[4][0][0], torch.tensor([19., 18., 17.]))
        self.assertEqual(runtime.target_features, {})
        self.assertEqual(runtime.stats["target_calls"], 2)

    def setUp(self):
        self.modules = {}
        for name in (
            "vllm",
            "vllm.v1",
            "vllm.v1.outputs",
            "vllm.logger",
            "vllm.config",
            "vllm.config.compilation",
            "vllm.v1.worker",
            "vllm.v1.worker.gpu",
            "vllm.v1.worker.gpu.cudagraph_utils",
            "vllm.v1.worker.gpu.attn_utils",
        ):
            self.modules[name] = types.ModuleType(name)
        self.modules["vllm.v1.outputs"].ModelRunnerOutput = lambda **kw: NS(**kw)
        self.modules["vllm.logger"].logger = Mock()
        self.modules["vllm.config.compilation"].CUDAGraphMode = NS(NONE=0)
        self.modules["vllm.v1.worker.gpu.cudagraph_utils"].BatchExecutionDescriptor = lambda **kw: NS(**kw)
        self.modules["vllm.v1.worker.gpu.attn_utils"].build_slot_mappings_by_layer = lambda *args: {}
        self.patch = patch.dict(sys.modules, self.modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def make_runtime(self, logits, drafts=(1, 2), index=2):
        count = len(drafts) + 1
        batch = NS(
            req_ids=["r"],
            num_reqs=1,
            num_reqs_after_padding=1,
            idx_mapping=torch.tensor([index], dtype=torch.int32),
            idx_mapping_np=np.array([index]),
            input_ids=torch.tensor([0, *drafts]),
            positions=torch.arange(count),
            query_start_loc_np=np.array([0, count]),
            query_start_loc=torch.tensor([0, count]),
            seq_lens=torch.tensor([count]),
            prefill_len_np=np.array([1]),
        )
        backend = NS(
            cache_enabled=True,
            set_request_block_tables=Mock(),
            forward_features=Mock(return_value=(logits, [logits + 10])),
            forward_batch=Mock(side_effect=lambda token_ids, starts, table_indices: [
                logits[: len(tokens)] for tokens in token_ids
            ]),
            _request_block_tables=[],
            _request_state_indices=[index],
        )
        sampler = Mock(side_effect=lambda logits, view: NS(sampled_token_ids=logits.argmax(-1).reshape(1, 1)))
        sampler.sampling_states = NS(temperature=NS(gpu=torch.zeros(4)), seeds=NS(gpu=torch.arange(4)))
        r = NS(
            _via_sd_config=lambda: NS(accept_ratio=0.7, escalate_ratio=0.5),
            _via_sd_timing_enabled=lambda: False,
            _via_sd_logging_enabled=lambda: False,
            device="cpu",
            max_num_tokens=32,
            max_model_len=128,
            num_speculative_steps=2,
            via_sd_verifier=NS(backend=backend, cache=NS(truncate=Mock())),
            via_sd_model=NS(compute_logits=lambda hidden: hidden),
            sampler=sampler,
            req_states=NS(
                all_token_ids=NS(gpu=torch.zeros((4, 128), dtype=torch.int64)),
                last_sampled_tokens=torch.zeros(4),
                next_prefill_tokens=torch.zeros((1, 4)),
                draft_tokens=torch.zeros((4, 2), dtype=torch.int64),
            ),
            speculator=NS(propose=Mock(return_value=torch.tensor([[2, 1]])), draft_logits=None),
            postprocess_sampled=Mock(),
            draft_tokens_handler=NS(set_draft_tokens=Mock()),
            kv_connector=NS(post_forward=Mock(return_value=None)),
            model_state=NS(prepare_attn=Mock(return_value={})),
            block_tables=NS(apply_staged_writes=Mock()),
            prepare_inputs=Mock(return_value=batch),
            prepare_attn=Mock(return_value=([], torch.zeros(1))),
            attn_groups=[],
            kv_cache_config=NS(),
        )
        for name in (
            "_discard_via_sd_requests",
            "update_pp_decode_requests",
            "finish_requests",
            "free_states",
            "add_requests",
            "update_requests",
        ):
            setattr(r, name, Mock())
        runtime = ViaSdRuntime(r)
        runtime.validate = Mock()
        r.execute_via_sd_hierarchical = lambda *args, **kwargs: runtime.coordinator.run(*args, **kwargs)
        scheduled = NS(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            total_num_scheduled_tokens=count,
            scheduled_new_reqs=[],
            num_scheduled_tokens={"r": count},
            scheduled_spec_decode_tokens={"r": list(drafts)} if drafts else {},
        )
        runtime.execute(scheduled)
        return runtime, r, batch

    def test_high_execute_sample_uses_qprime_features_and_no_target(self):
        logits = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]])
        runtime, r, batch = self.make_runtime(logits)
        runtime._catchup = Mock(side_effect=AssertionError("target called"))
        result = runtime.sample(None)
        self.assertEqual(result.sampled_token_ids, [[1, 2]])
        args = r.speculator.propose.call_args.args
        torch.testing.assert_close(args[3], logits)
        torch.testing.assert_close(args[4][0], logits + 10)
        self.assertEqual(args[5].tolist(), [2])
        self.assertEqual(args[6].tolist(), [1])
        self.assertEqual(runtime.coordinator.states["r"].target_computed_len, 0)
        self.assertEqual(runtime.coordinator.states["r"].qprime_computed_len, 2)
        self.assertEqual(r.req_states.draft_tokens[2].tolist(), [2, 1])

    def test_medium_rewrite_masks_suffix_for_existing_proposer(self):
        logits = torch.tensor([[0.0, -0.5, -4.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]])
        runtime, r, _ = self.make_runtime(logits)
        runtime._catchup = Mock(side_effect=AssertionError("target called"))
        self.assertEqual(runtime.sample(None).sampled_token_ids, [[0]])
        self.assertEqual(r.speculator.propose.call_args.args[6].tolist(), [2])
        self.assertEqual(runtime.coordinator.states["r"].qprime_computed_len, 1)
        r.via_sd_verifier.cache.truncate.assert_called_with("r", 1)

    def test_prefill_uses_qprime_without_target(self):
        runtime, r, _ = self.make_runtime(torch.tensor([[0.0, 2.0, 1.0]]), drafts=())
        runtime._catchup = Mock(side_effect=AssertionError("target called"))
        self.assertEqual(runtime.sample(None).sampled_token_ids, [[1]])
        self.assertEqual(runtime.coordinator.states["r"].target_computed_len, 0)

    def test_low_selects_original_probabilistic_position_and_state_row(self):
        runtime, r, _ = self.make_runtime(torch.zeros((3, 3)))
        original = torch.arange(24.0).reshape(4, 2, 3)
        r.speculator.draft_logits = original
        runtime.target_logits["r"] = torch.tensor([0.0, 2.0, 0.0])
        r.rejection_sampler = NS(_verify=Mock(return_value=(None, torch.tensor([[2, 1, -1]]), torch.tensor([2]))))
        event = NS(request_id="r", prefix_tokens=(0, 1), draft_token=2, position=1)
        decision = runtime._verify_target(event)
        args = r.rejection_sampler._verify.call_args.args
        torch.testing.assert_close(args[1][2, 0], original[2, 1])
        torch.testing.assert_close(r.speculator.draft_logits, original)
        self.assertEqual(args[5].tolist(), [2])
        self.assertEqual(args[3].tolist(), [1, 2])
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.computed_len, 2)

    def test_low_real_catchup_after_high_round(self):
        runtime, r, batch = self.make_runtime(torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]]))
        runtime.sample(None)
        self.assertEqual(runtime.coordinator.states["r"].target_computed_len, 0)
        writes = []

        def forward(tokens, start, row):
            writes.append((tuple(tokens), start, row))
            return torch.tensor([[0.0, 3.0, 0.0]]).repeat(len(tokens), 1)

        runtime._target_backend = NS(
            set_request_block_tables=Mock(),
            forward=forward,
            forward_batch=Mock(side_effect=lambda token_ids, starts, table_indices: [
                torch.tensor([[0.0, 3.0, 0.0]]).repeat(len(tokens), 1) for tokens in token_ids
            ]),
        )
        runtime.pending = (batch,)
        event = NS(request_id="r", prefix_tokens=(0, 1, 2), committed_len=3, target_computed_len=0)
        self.assertEqual(runtime._catchup(event), 3)
        self.assertEqual(writes, [((0, 1, 2), 0, 0)])
        self.assertEqual(runtime.coordinator.states["r"].target_computed_len, 3)
        self.assertEqual(runtime.stats["target_tokens"], 3)

    def test_0271_draft_count_branch_executes(self):
        tree = ast.parse((ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text(encoding="utf-8"))
        branches = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.UnaryOp)
            and isinstance(n.test.operand, ast.Name)
            and n.test.operand.id == "draft_tokens"
        ]
        branch = branches[0]
        scope = dict(
            np=np,
            torch=torch,
            draft_tokens={"a": [1, 2]},
            req_ids=["b", "a"],
            num_reqs=2,
            self=NS(device="cpu", model_state=NS(num_new_sampled_tokens_per_step=1), decode_query_len=3),
            idx_mapping=torch.tensor([3, 1]),
            async_copy_to_gpu=lambda x, **kw: torch.from_numpy(x),
            expand_idx_mapping=lambda *args: (None, None),
        )
        exec(compile(ast.Module(body=branch.orelse, type_ignores=[]), "<draft-count>", "exec"), scope)
        self.assertEqual(scope["num_draft_tokens_per_req"].tolist(), [0, 2])
        self.assertEqual(scope["total_num_logits"], 4)

    def test_mixed_batch_preserves_rows_and_runs_low_callbacks(self):
        runtime, r, batch = self.make_runtime(torch.zeros((3, 3)))
        batch.req_ids = ["high", "medium", "low"]
        batch.num_reqs = batch.num_reqs_after_padding = 3
        batch.idx_mapping = torch.tensor([3, 0, 2], dtype=torch.int32)
        batch.idx_mapping_np = np.array([3, 0, 2])
        batch.input_ids = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        batch.positions = torch.tensor([0, 1, 2] * 3)
        batch.query_start_loc_np = np.array([0, 3, 6, 9])
        batch.query_start_loc = torch.tensor([0, 3, 6, 9])
        batch.seq_lens = torch.tensor([3, 3, 3])
        batch.prefill_len_np = np.array([1, 1, 1])
        hidden = torch.tensor(
            [
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
                [3.0, 0.0, 0.0],
                [0.0, -0.5, -4.0],
                [0.0, 0.0, 3.0],
                [3.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0],
                [3.0, 0.0, 0.0],
            ]
        )
        scheduled = runtime.pending[-1]
        runtime.pending = (batch, hidden, [hidden + 10], {}, {}, [[0]] * 3, [[1, 2]] * 3, scheduled)
        r.speculator.propose.return_value = torch.tensor([[1, 1], [2, 2], [0, 0]])
        writes = []

        def forward(tokens, start, row):
            writes.append((tuple(tokens), start, row))
            return torch.tensor([[0.0, 3.0, 0.0]])

        runtime._target_backend = NS(
            set_request_block_tables=Mock(),
            forward=forward,
            forward_batch=Mock(side_effect=lambda token_ids, starts, table_indices: [
                torch.tensor([[0.0, 3.0, 0.0]]).repeat(len(tokens), 1) for tokens in token_ids
            ]),
        )
        r.rejection_sampler = NS(_verify=Mock(return_value=(None, torch.tensor([[1, 0, -1]]), torch.tensor([2]))))
        output = runtime.sample(None)
        self.assertEqual(output.sampled_token_ids, [[1, 2], [0], [1, 2]])
        self.assertEqual(writes, [])
        self.assertEqual(r.rejection_sampler._verify.call_args.args[5].tolist(), [2])
        self.assertEqual(r.speculator.propose.call_args.args[6].tolist(), [1, 2, 1])
        self.assertEqual(r.req_states.draft_tokens[[3, 0, 2]].tolist(), [[1, 1], [2, 2], [0, 0]])
        self.assertEqual(r._via_sd_route_plan.batch_rows, (0, 1, 2))
        self.assertEqual(len(r._via_sd_execution_result.results), 3)
        for row, offset in enumerate((0, 3, 6)):
            self.assertIsInstance(r._via_sd_route_plan.logits[row], torch.Tensor)
            self.assertEqual(r._via_sd_route_plan.logits[row].data_ptr(), hidden[offset].data_ptr())
        self.assertNotIn("build_route_plan", vars(runtime_module))


if __name__ == "__main__":
    unittest.main()
