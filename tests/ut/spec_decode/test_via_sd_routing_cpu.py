"""CPU-only contract tests for the VIA-SD routing helpers.

The production package imports the vLLM/torch stack at package initialisation
time.  These tests deliberately load the four device-independent modules
under a private package name, so the routing and logical-cache contracts can
be checked on a workstation without an NPU (or a torch installation).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path


def _load_pure_via_modules():
    package_name = "_via_sd_cpu_contract_tests"
    source_dir = (
        Path(__file__).resolve().parents[3]
        / "vllm_ascend"
        / "worker"
        / "v2"
        / "spec_decode"
        / "via_sd"
    )
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source_dir)]
        sys.modules[package_name] = package

    loaded = {}
    # coordinator/core import routing relatively; load it first and retain the
    # private module identity for the rest of this test file.
    for module_name in ("routing", "coordinator", "core", "kv_cache"):
        qualified_name = f"{package_name}.{module_name}"
        module = sys.modules.get(qualified_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(
                qualified_name,
                source_dir / f"{module_name}.py",
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load VIA-SD module {module_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified_name] = module
            spec.loader.exec_module(module)
        loaded[module_name] = module
    return loaded


_MODULES = _load_pure_via_modules()
_routing = _MODULES["routing"]
_coordinator = _MODULES["coordinator"]
_kv_cache = _MODULES["kv_cache"]


def _logit_row(token_id: int, score: float, vocab_size: int = 4) -> list[float]:
    """Make logits whose selected token has ``score / max(score)``."""

    values = [-10.0] * vocab_size
    values[0] = 0.0
    if token_id == 0:
        return values
    values[token_id] = math.log(score)
    return values


class ViaSdRoutingCpuTests(unittest.TestCase):
    def test_threshold_boundaries_and_stable_relative_confidence(self):
        route = _routing.route_from_score
        self.assertIs(route(0.7), _routing.ViaSdRoute.HIGH)
        self.assertIs(route(0.5), _routing.ViaSdRoute.MEDIUM)
        self.assertIs(route(math.nextafter(0.7, 0.0)), _routing.ViaSdRoute.MEDIUM)
        self.assertIs(route(math.nextafter(0.5, 0.0)), _routing.ViaSdRoute.LOW)

        self.assertAlmostEqual(
            _routing.relative_confidence([1000.0, 1000.0 + math.log(0.6)], 1),
            0.6,
            places=12,
        )
        with self.assertRaises(ValueError):
            _routing.relative_confidence([math.nan, 0.0], 0)
        with self.assertRaises(ValueError):
            _routing.relative_confidence([-math.inf, -math.inf], 0)

    def test_route_plan_keeps_ragged_rows_and_original_batch_mapping(self):
        logits = [
            [_logit_row(0, 1.0), _logit_row(1, 0.6), _logit_row(2, 0.4)],
            [_logit_row(1, 0.8), _logit_row(1, 0.2)],
        ]
        drafts = [[0, 1, 2], [1, 1]]
        plan = _routing.build_route_plan(
            logits,
            drafts,
            request_ids=("first", "second"),
            batch_rows=(7, 3),
        )

        self.assertEqual(
            plan.route_names,
            (("high", "medium", "low"), ("high", "low")),
        )
        self.assertEqual(plan.fallback_rows, (7, 3))
        self.assertEqual(
            tuple((item.request_id, item.position) for item in plan.fallbacks),
            (("first", 2), ("second", 1)),
        )
        self.assertEqual(plan.draft_tokens, ((0, 1, 2), (1, 1)))

    def test_all_high_short_circuits_target(self):
        coordinator = _coordinator.ViaSdExecutionCoordinator()
        target_calls = []

        def target(_request):
            target_calls.append(True)
            return True

        result = coordinator.run(
            ["r0"],
            [[10, 11]],
            [[0, 0]],
            [[_logit_row(0, 1.0), _logit_row(0, 1.0)]],
            target_verify=target,
        )

        self.assertTrue(result.all_high)
        self.assertEqual(result.target_calls, 0)
        self.assertEqual(target_calls, [])
        self.assertEqual(result.results[0].scheduler_tokens, (0, 0))
        self.assertEqual(result.results[0].accepted_draft_tokens, (0, 0))

    def test_medium_rewrites_and_discards_the_later_suffix(self):
        coordinator = _coordinator.ViaSdExecutionCoordinator()
        target_calls = []

        def target(_request):
            target_calls.append(True)
            return True

        result = coordinator.run(
            ["r0"],
            [[10]],
            [[0, 1, 0]],
            [[
                _logit_row(0, 1.0),
                _logit_row(1, 0.6),
                _logit_row(0, 1.0),
            ]],
            target_verify=target,
            qprime_sampler=lambda *_args: 2,
        )
        request_result = result.results[0]

        self.assertEqual(request_result.source, "qprime_rewrite")
        self.assertEqual(request_result.qprime_rewrite, (1, 2))
        self.assertEqual(request_result.stop_position, 1)
        self.assertEqual(request_result.scheduler_tokens, (0, 2))
        self.assertEqual(request_result.target_fallback_positions, ())
        self.assertEqual(target_calls, [])

    def test_low_rows_are_compacted_and_target_length_is_actual(self):
        coordinator = _coordinator.ViaSdExecutionCoordinator()
        catchups = []
        batches = []

        def catchup(event):
            catchups.append(event)
            # The callback reports the length really materialised by target;
            # the coordinator must not infer this from accepted draft tokens.
            return len(event.prefix_tokens)

        def target_batch(events):
            batches.append(tuple(events))
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event.batch_row, 4)
            self.assertEqual(event.position, 0)
            self.assertEqual(event.prefix_tokens, (20, 21))
            return {
                (event.request_id, event.position): {
                    "accepted": True,
                    "token": event.draft_token,
                    "computed_len": event.committed_len + 1,
                }
            }

        result = coordinator.run(
            ["r0"],
            [[20, 21]],
            [[2, 0]],
            [[_logit_row(2, 0.4), _logit_row(0, 1.0)]],
            batch_rows=(4,),
            target_verify_batch=target_batch,
            target_catchup=catchup,
        )
        request_result = result.results[0]
        state = coordinator.state_for("r0")

        self.assertEqual(len(batches), 1)
        self.assertEqual(len(catchups), 1)
        self.assertEqual(result.fallback_rows, (4,))
        self.assertEqual(result.target_calls, 1)
        self.assertEqual(request_result.target_fallback_positions, (0,))
        self.assertEqual(request_result.target_accepted_positions, (0,))
        self.assertEqual(request_result.scheduler_tokens, (2, 0))
        self.assertEqual(request_result.target_computed_len, 3)
        self.assertEqual(state.target_computed_len, 3)

    def test_target_rewrite_stops_at_first_low_position(self):
        coordinator = _coordinator.ViaSdExecutionCoordinator()

        def target_batch(events):
            event = events[0]
            return {(event.request_id, event.position): (False, 9)}

        result = coordinator.run(
            ["r0"],
            [[20]],
            [[2, 0]],
            [[_logit_row(2, 0.4), _logit_row(0, 1.0)]],
            target_verify_batch=target_batch,
            target_catchup=lambda event: len(event.prefix_tokens),
        )
        request_result = result.results[0]

        self.assertEqual(request_result.target_rewrite, (2, 9))
        self.assertEqual(request_result.stop_position, 0)
        self.assertEqual(request_result.scheduler_tokens, (9,))
        self.assertEqual(request_result.target_accepted_positions, ())

    def test_target_callback_accepts_common_scalar_and_mapping_shapes(self):
        for callback_result in (True, {"accepted": True, "token": 2}):
            coordinator = _coordinator.ViaSdExecutionCoordinator()

            def target(_event):
                return callback_result

            result = coordinator.run(
                ["r0"],
                [[20]],
                [[2]],
                [[_logit_row(2, 0.4)]],
                target_verify=target,
                target_catchup=lambda event: len(event.prefix_tokens),
            )
            self.assertEqual(result.results[0].target_accepted_positions, (0,))

    def test_cache_reuse_is_capped_by_logical_qprime_length(self):
        manager = _kv_cache.ViaSdKVCacheManager(enabled=True)
        signature = ((4, 8, 9),)
        manager.commit(
            "r0",
            2,
            [1, 2, 3, 4],
            signature,
            committed_len=4,
            qprime_computed_len=2,
            target_computed_len=1,
        )

        self.assertEqual(
            manager.reusable_prefix("r0", 2, [1, 2, 3, 4], 4, ((4, 8),)),
            2,
        )
        state = manager.state("r0")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.committed_len, 4)
        self.assertEqual(state.qprime_computed_len, 2)
        self.assertEqual(state.target_computed_len, 1)

        manager.truncate("r0", 1)
        self.assertEqual(manager.lengths("r0"), {
            "committed_len": 1,
            "qprime_computed_len": 1,
            "target_computed_len": 1,
        })
        manager.discard(["r0"])
        self.assertIsNone(manager.state("r0"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
