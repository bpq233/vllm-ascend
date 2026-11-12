# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Opt-in real-NPU tests; no fake models or mocked graph counters.

Set DFLASH_SPARSE_TEST_TARGET and DFLASH_SPARSE_TEST_DRAFT to compatible local
Qwen3/DFlash checkpoints. Each configuration executes in an isolated process.
The FULL/FULL_DECODE_ONLY matrix requires observed replay of all three models,
and fails if the selected workload does not actually exercise long candidates.
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[5] / "examples/offline_inference/dflash_sparse_verify.py"
PROMPTS = (
    "Explain why the sky is blue in simple language.",
    "Write a Python function that returns the Fibonacci sequence.",
    "List the first ten prime numbers and explain what makes a number prime.",
)


@pytest.fixture(scope="module")
def checkpoints():
    target = os.environ.get("DFLASH_SPARSE_TEST_TARGET")
    draft = os.environ.get("DFLASH_SPARSE_TEST_DRAFT")
    if not target or not draft:
        pytest.skip(
            "Real-NPU opt-in: set DFLASH_SPARSE_TEST_TARGET and DFLASH_SPARSE_TEST_DRAFT local checkpoint paths"
        )
    for path in (target, draft):
        if not Path(path).is_dir():
            pytest.fail(f"Configured checkpoint directory does not exist: {path}")
    return target, draft


def invoke(checkpoints, output, *, variant="target", graph="FULL", extra=()):
    target, draft = checkpoints
    command = [
        sys.executable,
        str(EXAMPLE),
        "--model",
        target,
        "--draft-model",
        draft,
        "--variant",
        variant,
        "--graph-mode",
        graph,
        "--output",
        str(output),
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "512",
        "--batch-size",
        "3",
        "--max-tokens",
        "96",
        "--ignore-eos",
        "--warmup-runs",
        "1",
        "--repetitions",
        "1",
    ]
    for prompt in PROMPTS:
        command.extend(("--prompt", prompt))
    command.extend(extra)
    process = subprocess.Popen(
        command,
        env={**os.environ, "VLLM_USE_V2_MODEL_RUNNER": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=1800)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise
    assert process.returncode == 0, (
        f"NPU worker failed ({process.returncode}).\nstdout:\n{stdout[-6000:]}\nstderr:\n{stderr[-6000:]}"
    )
    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="module", params=("FULL", "FULL_DECODE_ONLY"))
def baseline(request, checkpoints, tmp_path_factory):
    graph = request.param
    output = tmp_path_factory.mktemp("dflash_sparse_baseline") / f"target_{graph}.json"
    return graph, invoke(checkpoints, output, graph=graph)


def assert_graph_evidence(result):
    assert result["worker_evidence"], "No real worker evidence was returned"
    for worker in result["worker_evidence"]:
        assert worker["device"].startswith("npu"), worker
        metrics = worker["metrics"]
        assert metrics["rounds"] and metrics["finals"], metrics
        counters = metrics["graph_calls"]
        for name in ("dflash", "sparse", "target"):
            assert counters.get(f"{name}.capture", 0) > 0, f"No observed {name} graph capture: {worker}"
            assert counters.get(f"{name}.replay", 0) > 0, f"No observed {name} graph replay: {worker}"
        assert all(0 <= record["accepted"] <= record["proposed"] for record in metrics["finals"])
        assert all(0 <= record["temporary_accepted"] <= record["proposed"] for record in metrics["rounds"])


@pytest.mark.parametrize("mode", ("fixed", "confidence"))
@pytest.mark.parametrize("final", ("strict", "topk", "confidence", "accept_all"))
def test_eight_strategies_graph_and_long_candidates(checkpoints, baseline, tmp_path, mode, final):
    graph, reference = baseline
    result = invoke(
        checkpoints,
        tmp_path / "result.json",
        variant=mode,
        graph=graph,
        extra=(
            "--intermediate-layers",
            "all",
            "--num-rounds",
            "4",
            "--draft-tokens",
            "7",
            "--final-method",
            final,
            "--require-graph-replays",
            "--require-min-final-proposal",
            "17",
        ),
    )
    assert_graph_evidence(result)
    assert result["observed_max_final_candidates"] > 16
    if final == "strict":
        assert result["runs"][0]["token_ids"] == reference["runs"][0]["token_ids"]
    else:
        assert result["strict_greedy_equivalence_expected"] is False
        assert all(len(tokens) == 96 for tokens in result["runs"][0]["token_ids"])
    for worker in result["worker_evidence"]:
        count = result["configuration"]["num_intermediate_layers"]
        assert worker["selected_layers"] == list(range(count)), "M=L must execute every target layer"


@pytest.mark.parametrize("rounds", (1, 2, 4))
def test_fixed_round_counts_single_request_and_all_layers(checkpoints, baseline, tmp_path, rounds):
    graph, _ = baseline
    reference = invoke(
        checkpoints,
        tmp_path / "target.json",
        graph=graph,
        extra=("--batch-size", "1"),
    )
    result = invoke(
        checkpoints,
        tmp_path / "sparse.json",
        variant="fixed",
        graph=graph,
        extra=(
            "--batch-size",
            "1",
            "--intermediate-layers",
            "all",
            "--num-rounds",
            str(rounds),
            "--draft-tokens",
            "7",
            "--final-method",
            "strict",
            "--require-graph-replays",
        ),
    )
    assert_graph_evidence(result)
    assert result["runs"][0]["token_ids"] == reference["runs"][0]["token_ids"]
    assert result["configuration"]["max_final_candidates"] == 7 * rounds


def test_eager_graph_and_original_dflash_strict_regression(checkpoints, baseline, tmp_path):
    graph, reference = baseline
    original = invoke(checkpoints, tmp_path / "original.json", variant="original", graph=graph)
    eager = invoke(
        checkpoints,
        tmp_path / "eager.json",
        variant="confidence",
        graph="NONE",
        extra=("--intermediate-layers", "all", "--num-rounds", "2", "--final-method", "strict"),
    )
    assert original["runs"][0]["token_ids"] == reference["runs"][0]["token_ids"]
    assert eager["runs"][0]["token_ids"] == reference["runs"][0]["token_ids"]
