# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--multi-stage-models",
        nargs=4,
        default=None,
        help="Four local model paths: target, primary DFlash, intermediate, secondary DFlash",
    )


@pytest.fixture(scope="module")
def multi_stage_models(request):
    paths = request.config.getoption("--multi-stage-models")
    if paths is None:
        pytest.skip("Supply --multi-stage-models TARGET PRIMARY INTERMEDIATE SECONDARY on an NPU host")
    return paths
