# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise the real config factory before worker/model initialization.

The factory is loaded from its AST to avoid importing hardware dependencies on
a standalone CPU host. Its filtering, construction and cache logic are intact;
unrelated hardware/config derivations are stubbed. Validation uses the real
Pydantic config decorator, including the outer __post_init__ error wrapping.
"""

import ast
import logging
import sys
from copy import deepcopy
from dataclasses import field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from vllm_ascend.config_utils import config
from vllm_ascend.spec_decode.slim_verifier_config import SlimVerifierConfig


def _preconditions():
    return SimpleNamespace(_validate_preconditions=lambda vc: None)


@config
class FactoryAscendConfig:
    scheduler_config: Any
    sparse_kv_offload_config: Any
    kvpp_config: Any
    dump_config_path: Any
    runtime_config_path: Any
    runtime_config_hot_reload: bool
    runtime_config: Any
    enable_cpu_binding: bool = True
    finegrained_tp_config: Any = field(default_factory=_preconditions)
    xlite_graph_config: Any = field(default_factory=_preconditions)
    rl_config: Any = field(default_factory=lambda: SimpleNamespace(apply=lambda cfg: None))

    @staticmethod
    def _resolve_dump_config_path(additional_config):
        return None

    def derive_and_validate(self, vc):
        return self


@pytest.fixture
def config_factory(monkeypatch):
    path = Path(__file__).parents[3] / "vllm_ascend/ascend_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "init_ascend_config")
    runtime = ModuleType("vllm_ascend.observability.runtime_config")
    runtime.ADDITIONAL_CONFIG_STRIP_KEYS = frozenset()
    runtime.build_runtime_config_from_additional = lambda extra: SimpleNamespace(
        path=None, hot_reload=False, runtime_config=None
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    utils = ModuleType("vllm_ascend.utils")
    utils.clear_enable_sp = lambda: None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    namespace = {
        "VllmConfig": Any,
        "AscendConfig": FactoryAscendConfig,
        "SchedulerConfig": SimpleNamespace(from_additional_config=lambda extra: None),
        "SparseKVOffloadConfig": SimpleNamespace(from_additional_config=lambda vc, extra: None),
        "KVPPConfig": SimpleNamespace(from_vllm_config=lambda vc: None),
        "validate_additional_config_bool": lambda value, path: bool(value),
        "_is_ascend_config_initialized": lambda cfg: True,
        "_ASCEND_CONFIG": None,
        "_INIT_VLLM_CONFIG": None,
        "logger": logging.getLogger(__name__),
        # Exercise strict behavior without the optional Omni extension.
        "importlib": SimpleNamespace(util=SimpleNamespace(find_spec=lambda name: None)),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["init_ascend_config"]


def _outer_config_class(factory):
    # Upstream VllmConfig invokes platform/config initialization from
    # __post_init__. Pydantic reports nested constructor errors under this
    # OUTER class name, even if no unexpected keyword was passed to it.
    @config
    class VllmConfig:
        additional_config: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self):
            factory(self)

    return VllmConfig


@pytest.mark.parametrize("enabled", [True, False])
def test_slim_config_survives_platform_initialization(config_factory, enabled):
    extra = {
        "slim_verifier_eval": {
            "enabled": enabled,
            "keep_layers": [0, 4, 8, 12],
            "log_interval": 100,
            "top_k": 5,
            "profile_latency": False,
            "acceptance_strategy": "probability_ratio",
            "probability_ratio_threshold": 0.7,
        },
        "enable_cpu_binding": False,
    }
    original = deepcopy(extra)
    outer = _outer_config_class(config_factory)(additional_config=extra)
    ascend = config_factory(outer)
    assert outer.additional_config == original
    assert not ascend.enable_cpu_binding
    assert not hasattr(ascend, "slim_verifier_eval")
    slim = SlimVerifierConfig.from_additional_config(outer.additional_config, 16)
    assert (slim is not None) == enabled
    if enabled:
        assert slim.keep_layers == (0, 4, 8, 12)
        assert slim.acceptance_strategy == "probability_ratio"
        assert slim.probability_ratio_threshold == 0.7


def test_unrelated_unknown_config_is_still_rejected(config_factory):
    with pytest.raises(ValidationError) as caught:
        _outer_config_class(config_factory)(additional_config={"unrelated_typo": True})
    assert caught.value.title == "VllmConfig"
    assert caught.value.errors()[0]["loc"] == ("unrelated_typo",)
    assert caught.value.errors()[0]["type"] == "unexpected_keyword_argument"
