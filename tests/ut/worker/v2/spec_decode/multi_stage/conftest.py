# SPDX-License-Identifier: Apache-2.0
"""CPU tests load the pipeline without initializing the NPU plugin.

Only imports at the vLLM boundary are stubbed in adapter tests. Torch and all
pipeline, acceptance, packing, lifecycle and private-cache code are real.
Run with --confcutdir pointing to this directory on hosts without vLLM/NPU.
"""

import importlib
import importlib.util
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def core(monkeypatch):
    root = next(p for p in Path(__file__).resolve().parents if (p / "vllm_ascend").is_dir())
    path = root / "vllm_ascend/worker/v2/spec_decode/multi_stage"
    name = "_multi_stage_cpu"
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            monkeypatch.delitem(sys.modules, key)
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    vllm_logger = ModuleType("vllm.logger")
    vllm_logger.logger = logging.getLogger("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger)
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, package)
    spec.loader.exec_module(package)
    return SimpleNamespace(
        **{
            part: importlib.import_module(f"{name}.{part}")
            for part in (
                "acceptance",
                "pipeline",
                "state",
                "config",
                "interfaces",
                "metrics",
            )
        }
    )


@pytest.fixture
def adapters(core, monkeypatch):
    def module(name, **attrs):
        obj = ModuleType(name)
        obj.__path__ = []
        obj.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, obj)
        return obj

    class RequestData(SimpleNamespace):
        @property
        def prompt_len(self):
            return len(self.prompt_token_ids)

    class DraftTokenIds:
        def __init__(self, req_ids, draft_token_ids):
            self.req_ids, self.draft_token_ids = req_ids, draft_token_ids

    # Import-only stubs; adapter test doubles below model the execution boundary.
    module("vllm")
    module(
        "vllm.config",
        CompilationConfig=SimpleNamespace,
        ModelConfig=SimpleNamespace,
        SpeculativeConfig=SimpleNamespace,
        set_current_vllm_config=lambda _: nullcontext(),
    )
    module("vllm.config.compilation", CUDAGraphMode=SimpleNamespace(NONE=0), CompilationMode=SimpleNamespace(NONE=0))
    module("vllm.sampling_params", SamplingParams=SimpleNamespace)
    for name in (
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.spec_decode",
    ):
        module(name)
    module("vllm.v1.outputs", DraftTokenIds=DraftTokenIds)
    module("vllm.v1.core.kv_cache_utils", get_kv_cache_configs=None)
    module(
        "vllm.v1.core.sched.output",
        NewRequestData=RequestData,
        SchedulerOutput=SimpleNamespace(make_empty=lambda: SimpleNamespace()),
    )
    module("vllm.v1.kv_cache_interface", FullAttentionSpec=type("FullAttentionSpec", (), {}))
    module("vllm.v1.worker.gpu.spec_decode.rejection_sampler", RejectionSampler=type("RejectionSampler", (), {}))
    for part in ("sampler", "backend", "runtime"):
        setattr(core, part, importlib.import_module(f"_multi_stage_cpu.{part}"))
    return core
