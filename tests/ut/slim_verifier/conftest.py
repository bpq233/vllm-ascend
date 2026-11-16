# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Standalone pure-torch tests may bypass package-time vLLM registration.

With --confcutdir=tests/ut/slim_verifier the outer NPU fixtures are excluded.
All tested numerical code uses real torch tensors and operators.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

if importlib.util.find_spec("vllm") is None and "vllm_ascend" not in sys.modules:
    package = ModuleType("vllm_ascend")
    package.__path__ = [str(Path(__file__).parents[3] / "vllm_ascend")]
    sys.modules["vllm_ascend"] = package
