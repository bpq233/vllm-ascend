# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Early capacity sizing and request validation, without NPU initialization."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_integration():
    directory = Path(__file__).resolve().parents[5] / "vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify"
    name = "_dflash_sparse_integration_tests"
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    modules = []
    for leaf in ("config", "integration"):
        spec = importlib.util.spec_from_file_location(f"{name}.{leaf}", directory / f"{leaf}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


config_module, integration = _load_integration()


def make_config(enabled=True, rounds=4, block_tokens=7):
    return SimpleNamespace(
        additional_config={"dflash_sparse_verify": {"enabled": enabled, "num_rounds": rounds}},
        speculative_config=SimpleNamespace(
            method="dflash",
            num_speculative_tokens=block_tokens,
            enable_adaptive_verification=False,
            num_speculative_tokens_per_batch_size=None,
            rejection_sample_method="standard",
            draft_sample_method="greedy",
            draft_model_config=SimpleNamespace(
                architectures=["DFlashDraftModel"],
                hf_config=SimpleNamespace(),
            ),
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3"),
            max_model_len=4096,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
        compilation_config=SimpleNamespace(cudagraph_mode="FULL", cudagraph_capture_sizes=[8, 16, 32, 64]),
        kv_transfer_config=None,
        lora_config=None,
        watermark_config=None,
        use_v2_model_runner=True,
    )


def sampling_params(**changes):
    values = dict(
        temperature=0.0,
        logprobs=None,
        prompt_logprobs=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        logit_bias=None,
        allowed_token_ids=None,
        bad_words=None,
        structured_outputs=None,
        min_tokens=0,
    )
    values.update(changes)
    return SimpleNamespace(**values)


class TestSparseIntegration(unittest.TestCase):
    def test_disabled_does_not_inspect_or_mutate_other_config(self):
        config = SimpleNamespace(additional_config={"dflash_sparse_verify": {"enabled": False}})
        integration.configure_sparse_verification(config)
        self.assertEqual(vars(config), {"additional_config": {"dflash_sparse_verify": {"enabled": False}}})
        integration.configure_sparse_verification(SimpleNamespace(additional_config=None))

    def test_capacity_expansion_is_idempotent_and_graph_config_is_shared(self):
        for rounds in (1, 2, 4, 8):
            with self.subTest(rounds=rounds):
                config = make_config(rounds=rounds)
                graph_config = config.compilation_config
                integration.configure_sparse_verification(config)
                integration.configure_sparse_verification(config)
                self.assertEqual(config.speculative_config.num_speculative_tokens, rounds * 7)
                self.assertEqual(config.speculative_config.draft_model_config.hf_config._ascend_sparse_block_tokens, 7)
                self.assertIs(config.compilation_config, graph_config)
                self.assertEqual(graph_config.cudagraph_capture_sizes, [8, 16, 32, 64])

    def test_marker_survives_worker_copy(self):
        config = make_config()
        integration.configure_sparse_verification(config)
        worker_config = copy.deepcopy(config)
        integration.configure_sparse_verification(worker_config)
        self.assertEqual(worker_config.speculative_config.num_speculative_tokens, 28)
        self.assertEqual(worker_config.speculative_config.draft_model_config.hf_config._ascend_sparse_block_tokens, 7)

    def test_block_copy_keeps_original_dflash_width(self):
        config = make_config()
        integration.configure_sparse_verification(config)
        draft = integration.draft_block_config(config)
        self.assertIsNot(draft, config)
        self.assertIsNot(draft.speculative_config, config.speculative_config)
        self.assertEqual(draft.speculative_config.num_speculative_tokens, 7)
        self.assertEqual(config.speculative_config.num_speculative_tokens, 28)
        self.assertIs(draft.compilation_config, config.compilation_config)
        self.assertIs(draft.speculative_config.draft_model_config, config.speculative_config.draft_model_config)

    def test_expansion_fails_before_modification_when_capacity_insufficient(self):
        for field, value in (("max_num_batched_tokens", 28), ("max_model_len", 29)):
            config = make_config()
            target = config.scheduler_config if field == "max_num_batched_tokens" else config.model_config
            setattr(target, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                integration.configure_sparse_verification(config)
            self.assertEqual(config.speculative_config.num_speculative_tokens, 7)
            self.assertFalse(
                hasattr(config.speculative_config.draft_model_config.hf_config, "_ascend_sparse_block_tokens")
            )

    def test_target_budget_larger_than_sixteen_does_not_change_dflash_block_limit(self):
        config = make_config(block_tokens=15, rounds=4)
        integration.configure_sparse_verification(config)
        self.assertEqual(config.speculative_config.num_speculative_tokens, 60)
        invalid = make_config(block_tokens=16)
        with self.assertRaisesRegex(ValueError, "at most 15"):
            integration.configure_sparse_verification(invalid)

    def test_model_and_method_mismatch(self):
        config = make_config()
        config.speculative_config.method = "eagle3"
        with self.assertRaisesRegex(ValueError, "method=dflash"):
            integration.configure_sparse_verification(config)
        config = make_config()
        config.speculative_config = None
        with self.assertRaisesRegex(ValueError, "method=dflash"):
            integration.configure_sparse_verification(config)
        for architecture in ("DFlash2DraftModel", "DSparkDraftModel", "LiLiCorrDraftModel"):
            config = make_config()
            config.speculative_config.draft_model_config.architectures = [architecture]
            with self.subTest(architecture=architecture), self.assertRaises(ValueError):
                integration.configure_sparse_verification(config)
        config = make_config()
        config.model_config.hf_text_config.model_type = "qwen3_moe"
        with self.assertRaisesRegex(ValueError, "dense Qwen3"):
            integration.configure_sparse_verification(config)

    def test_incompatible_parallel_and_cache_modes_are_explicit(self):
        for field in (
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        ):
            config = make_config()
            setattr(config.parallel_config, field, 2)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "PP=DP=PCP=DCP=1"):
                integration.configure_sparse_verification(config)
        config = make_config()
        config.parallel_config.tensor_parallel_size = 2
        integration.configure_sparse_verification(config)
        for group, field, value in (
            ("cache_config", "enable_prefix_caching", True),
            ("cache_config", "cache_dtype", "int8"),
            ("speculative_config", "enable_adaptive_verification", True),
            ("speculative_config", "num_speculative_tokens_per_batch_size", [(1, 4, 7)]),
        ):
            config = make_config()
            setattr(getattr(config, group), field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                integration.configure_sparse_verification(config)
        for field in ("kv_transfer_config", "lora_config"):
            config = make_config()
            setattr(config, field, object())
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "KV transfer or LoRA"):
                integration.configure_sparse_verification(config)

    def test_sampling_must_be_unmodified_greedy(self):
        config = config_module.SparseVerifyConfig(enabled=True)
        integration.validate_sparse_sampling(config, sampling_params())
        for changes in (
            {"temperature": 0.8},
            {"logprobs": 0},
            {"prompt_logprobs": 1},
            {"presence_penalty": 0.1},
            {"frequency_penalty": 0.1},
            {"repetition_penalty": 1.1},
            {"logit_bias": {1: -0.5}},
            {"allowed_token_ids": [1]},
            {"bad_words": ["word"]},
            {"structured_outputs": object()},
            {"min_tokens": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                integration.validate_sparse_sampling(config, sampling_params(**changes))

    def test_unsupported_runner_watermark_and_rejection_algorithms_fail_early(self):
        config = make_config()
        config.use_v2_model_runner = False
        with self.assertRaisesRegex(ValueError, "V2 model runner"):
            integration.configure_sparse_verification(config)
        config = make_config()
        config.watermark_config = object()
        with self.assertRaisesRegex(ValueError, "watermarking"):
            integration.configure_sparse_verification(config)
        for method in ("synthetic", "block", "fly"):
            config = make_config()
            config.speculative_config.rejection_sample_method = method
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "standard rejection"):
                integration.configure_sparse_verification(config)
        config = make_config()
        config.additional_config["finegrained_tp_config"] = {"lmhead_tensor_parallel_size": 2}
        with self.assertRaisesRegex(ValueError, "LM-head tensor parallelism"):
            integration.configure_sparse_verification(config)


if __name__ == "__main__":
    unittest.main()
