# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import importlib.util
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/multi_stage"

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance", "acceptance.py")
    config = load("_intermediate_test_config", "config.py")
    pipeline = load("_intermediate_test_pipeline", "pipeline.py")
    return config, pipeline.IntermediatePipeline


def scores(predictions):
    logits = torch.full((len(predictions), 32), -10.0)
    logits[torch.arange(len(predictions)), predictions] = 10.0
    return logits


class Backend:
    def __init__(self, verification, proposals=()):
        self.verification = iter(verification)
        self.proposals = iter(proposals)
        self.calls = []
        self.retain_hidden = []

    def verify_batches(self, contexts, drafts, req_ids=None, retain_hidden=True):
        self.retain_hidden.append(retain_hidden)
        self.calls.append(("verify", contexts, drafts))
        rows = next(self.verification)
        yield torch.cat([scores(row) for row in rows]), [len(row) for row in rows]

    def propose(self, contexts, req_ids=None):
        self.calls.append(("propose", contexts))
        return next(self.proposals)


def test_three_rounds_preserve_primary_and_use_only_accepted_context(modules):
    config, Pipeline = modules
    backend = Backend(
        [[[1, 7, 8]], [[3, 4, 5]], [[9, 10]]],
        proposals=[[[3, 4]], [[9]]],
    )
    options = config.IntermediateConfig("verifier", "drafter", verification={"method": "topk", "top_k": 1})
    primary, prefixes = [[1, 2]], [[20]]
    result = Pipeline(backend, options, capacity=8).refine(prefixes, primary, [8])
    assert result == [[1, 7, 3, 4, 5, 9, 10]]
    assert backend.retain_hidden == [True, True, False]
    assert primary == [[1, 2]]
    assert prefixes == [[20]]
    assert backend.calls == [
        ("verify", [[20]], [[1, 2]]),
        ("propose", [[20, 1, 7]]),
        ("verify", [[20, 1, 7]], [[3, 4]]),
        ("propose", [[20, 1, 7, 3, 4, 5]]),
        ("verify", [[20, 1, 7, 3, 4, 5]], [[9]]),
    ]


def test_heterogeneous_eos_limit_and_inactive_requests(modules):
    config, Pipeline = modules
    backend = Backend([[[1, 2, 3], [4, 5, 6], [7]], [[8, 9]]], proposals=[[[8]]])
    options = config.IntermediateConfig("v", "d", verification={"method": "all"})
    result = Pipeline(backend, options, capacity=5, eos_token_id=[2]).refine(
        [[20], [21], [22], [23]], [[1, 2], [4, 5], [], [9]], [5, 2, 3, 0]
    )
    assert result == [[1, 2], [4, 5], [7, 8, 9], []]
    assert backend.calls == [
        ("verify", [[20], [21], [22]], [[1, 2], [4, 5], []]),
        ("propose", [[22, 7]]),
        ("verify", [[22, 7]], [[8]]),
    ]


def test_empty_active_batch_does_not_run_backend(modules):
    config, Pipeline = modules
    backend = Backend([])
    assert Pipeline(backend, config.IntermediateConfig("v", "d"), 4).refine([[1]], [[2]], [0]) == [[]]
    assert backend.calls == []


@pytest.mark.parametrize("method,top_k", [("all", 1), ("topk", 1), ("topk", 5)])
def test_fused_head_decision_matches_packed_reference(modules, method, top_k):
    config, Pipeline = modules
    acceptance = sys.modules["vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance"]
    policy = acceptance.AcceptancePolicy(method, top_k)
    runner = acceptance.IntermediateDecisionRunner(lambda hidden: hidden, policy, 5)
    reference = Pipeline(None, config.IntermediateConfig("v", "d", verification=vars(policy)), 16)
    generator = torch.Generator().manual_seed(91)
    for lengths in ([5], [2, 5], [1, 4, 3], [1, 1, 1, 1]):
        logits = torch.randn(sum(lengths), 17, generator=generator)
        logits[0].fill_(float("-inf"))
        # Include ties and masked rows in the semantic comparison.
        if len(logits) > 2:
            logits[2, :3] = 9
        tokens = torch.randint(17, (sum(lengths),), generator=generator)
        expected = reference._decide(logits, None, lengths, tokens=tokens)
        assert runner(logits, tokens, lengths).tolist() == expected


def test_all_policy_projects_only_bonus_rows(modules):
    acceptance = sys.modules["vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance"]
    projected = []
    runner = acceptance.IntermediateDecisionRunner(
        lambda hidden: projected.append(hidden.clone()) or hidden,
        acceptance.AcceptancePolicy("all"),
        16,
    )
    hidden = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    result = runner(hidden, None, [16, 16])
    assert result.tolist() == [[15, 7], [15, 7]]
    assert projected[0].shape[0] == 2
    torch.testing.assert_close(projected[0], hidden[[15, 31]])


@pytest.mark.parametrize("method", ["topk", "all"])
def test_decision_graph_reuses_buffers_for_changing_lengths(modules, monkeypatch, method):
    acceptance = sys.modules["vllm_ascend.worker.v2.spec_decode.multi_stage.acceptance"]
    policy = acceptance.AcceptancePolicy(method, 2)
    runner = acceptance.IntermediateDecisionRunner(lambda hidden: hidden, policy, 5, graph_enabled=True)
    eager = acceptance.IntermediateDecisionRunner(lambda hidden: hidden, policy, 5)
    state = NS(capturing=None, captures=0, streams=0, pools=0)

    class Graph:
        def replay(self):
            self.output.copy_(self.fn())

    def stream(**kwargs):
        state.streams += 1
        return NS(wait_stream=lambda other: None)

    def pool():
        state.pools += 1
        return object()

    @contextmanager
    def capture(graph, **kwargs):
        state.captures += 1
        state.capturing = graph
        try:
            yield
        finally:
            state.capturing = None

    original_run = runner._run

    def record(*args):
        output = original_run(*args)
        if state.capturing is not None:
            state.capturing.fn = lambda: original_run(*args)
            state.capturing.output = output
        return output

    runner._run = record
    tensor = torch.tensor
    monkeypatch.setattr(
        torch, "tensor", lambda *a, **kw: tensor(*a, **{k: v for k, v in kw.items() if k != "pin_memory"})
    )
    monkeypatch.setattr(
        torch,
        "npu",
        NS(
            Stream=stream,
            graph_pool_handle=pool,
            NPUGraph=Graph,
            graph=capture,
            current_stream=lambda: NS(wait_stream=lambda other: None),
            stream=lambda stream: nullcontext(),
        ),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.v2.utils", NS(communicator_switch=nullcontext))
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.parallel_state",
        NS(
            GraphCaptureContext=lambda stream: NS(stream=stream),
            get_tp_group=lambda: NS(graph_capture=lambda context: nullcontext()),
        ),
    )
    generator = torch.Generator().manual_seed(4)
    addresses = {}
    for lengths in ([5, 5, 5], [1], [1, 4, 2, 3], [4, 1, 1], [3]):
        hidden = torch.randn(sum(lengths), 11, generator=generator)
        tokens = torch.randint(11, (sum(lengths),), generator=generator)
        torch.testing.assert_close(runner(hidden, tokens, lengths), eager(hidden, tokens, lengths))
        for bucket, entry in runner.entries.items():
            pointers = tuple(t.data_ptr() for t in entry[:4])
            assert addresses.setdefault(bucket, pointers) == pointers
    assert state.captures == 2  # request buckets 1 and 4, independent of lengths
    assert state.streams == state.pools == 1
    assert runner.replays == 5


def test_resident_requests_run_first_and_results_restore_original_order(modules):
    config, Pipeline = modules
    backend = NS(max_num_reqs=1, cache=NS(slots={"b": 0}))
    pipe = Pipeline(backend, config.IntermediateConfig("v", "d"), 4)
    calls = []

    def refine(prefixes, drafts, limits, req_ids):
        calls.extend(req_ids)
        return drafts

    pipe._refine = refine
    assert pipe.refine([[1], [2], [3]], [[4], [5], [6]], [4, 4, 4], ["a", "b", "c"]) == [[4], [5], [6]]
    assert calls == ["b", "a", "c"]


def vllm_config():
    return NS(
        use_v2_model_runner=True,
        speculative_config=NS(num_speculative_tokens=8, use_dflash=lambda: True),
        parallel_config=NS(),
        scheduler_config=NS(async_scheduling=False),
        lora_config=None,
        model_config=NS(is_multimodal_model=False),
    )


def settings(**intermediate):
    return {
        "intermediate": {"verifier": {"model": "v"}, "drafter": {"model": "d"}, **intermediate},
        "final_verification": {"method": "all"},
    }


def test_config_disabled_and_zero_rounds(modules):
    config, _ = modules
    config.validate_multi_stage(None, None)
    config.validate_multi_stage(None, {})
    cfg = vllm_config()
    cfg.speculative_config.use_dflash = lambda: False
    config.validate_multi_stage(cfg, settings(num_rounds=0))
    invalid = settings(num_rounds=0)
    invalid["primary_num_speculative_tokens"] = 4
    with pytest.raises(ValueError, match="zero intermediate rounds"):
        config.validate_multi_stage(cfg, invalid)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_rounds", -1),
        ("num_rounds", True),
        ("num_speculative_tokens", 0),
        ("max_num_seqs", 0),
        ("max_model_len", 0),
    ],
)
def test_invalid_intermediate_numbers(modules, field, value):
    config, _ = modules
    with pytest.raises(ValueError, match=field):
        config.IntermediateConfig.from_dict(settings(**{field: value})["intermediate"])


@pytest.mark.parametrize(
    "invalid",
    [
        {"unexpected": True},
        {"primary_num_speculative_tokens": 4},
        {**settings(), "primary_num_speculative_tokens": 9},
        {"intermediate": settings()["intermediate"]},
    ],
)
def test_invalid_multi_stage_options(modules, invalid):
    config, _ = modules
    with pytest.raises(ValueError):
        config.validate_multi_stage(vllm_config(), invalid)


@pytest.mark.parametrize("attribute,value", [("use_v2_model_runner", False), ("lora_config", object())])
def test_unsupported_runner_modes(modules, attribute, value):
    config, _ = modules
    cfg = vllm_config()
    setattr(cfg, attribute, value)
    with pytest.raises(ValueError):
        config.validate_multi_stage(cfg, settings())


def test_long_budget_validation(modules):
    config, _ = modules
    cfg = vllm_config()
    cfg.speculative_config.num_speculative_tokens = 32
    cfg.scheduler_config.max_num_batched_tokens = 64
    cfg.cache_config = NS(cache_dtype="auto")
    options = {**settings(), "primary_num_speculative_tokens": 4}
    config.validate_multi_stage(cfg, options)
    # Omitting primary width must not inherit the long final budget.
    config.validate_multi_stage(cfg, settings())
    assert config.primary_draft_width(settings(), 32) == 4
    assert config.primary_draft_width(settings(), 8) == 8
    assert config.primary_draft_width(options, 32) == 4
    with pytest.raises(ValueError, match="primary_num_speculative_tokens=32"):
        config.validate_multi_stage(cfg, {**settings(), "primary_num_speculative_tokens": 32})
    with pytest.raises(ValueError, match="intermediate.num_speculative_tokens=32"):
        config.validate_multi_stage(cfg, settings(num_speculative_tokens=32))
    for changes in ({"primary_num_speculative_tokens": 16}, {"intermediate": settings(num_rounds=0)["intermediate"]}):
        with pytest.raises(ValueError):
            config.validate_multi_stage(cfg, {**options, **changes})
    cfg.scheduler_config.max_num_batched_tokens = 32
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        config.validate_multi_stage(cfg, options)


def test_packed_decisions_match_individual_requests(modules):
    config, Pipeline = modules
    pipeline = Pipeline(None, config.IntermediateConfig("v", "d", verification={"method": "topk", "top_k": 1}), 32)
    drafts = [[1, 2, 3], [], [5, 6]]
    packed = torch.cat([scores([1, 9, 3, 4]), scores([7]), scores([5, 6, 8])])
    assert pipeline._decide(packed, drafts, [4, 1, 3]) == [[1, 9], [0, 7], [2, 8]]


def test_sparse_capture_sizes_cover_verifier_and_secondary(modules):
    config, _ = modules
    sizes = config.intermediate_capture_sizes(4096, 4, 15)
    assert sizes == [1, 16, 32, 48, 64, 256, 1024, 4096]
    # Reported NPU failure used max_model_len=40960 and 44 legacy gears.
    assert config.intermediate_capture_sizes(40960, 4, 15) == [1, 16, 32, 48, 64, 256, 1024, 4096, 16384, 40960]
    assert config.intermediate_capture_sizes(40960, 4, 15, [64, 256]) == [64, 256, 40960]
    assert config.intermediate_capture_sizes(4096, 4, 15, [256, 64, 256]) == [64, 256, 4096]
    assert config.intermediate_capture_sizes(128, 4, 4) == [1, 5, 10, 15, 16, 20, 64, 128]
    for requests in range(1, 5):
        assert any(requests * 16 <= size <= 4 * 16 and size % 16 == 0 for size in sizes)
    with pytest.raises(ValueError, match="token buffer"):
        config.intermediate_capture_sizes(4096, 4, 15, [8192])


@pytest.mark.parametrize("sizes", [[], [0], [-1], [True], [2.5], "64"])
def test_invalid_intermediate_capture_sizes(modules, sizes):
    config, _ = modules
    with pytest.raises(ValueError, match="cudagraph_capture_sizes"):
        config.IntermediateConfig.from_dict(settings(cudagraph_capture_sizes=sizes)["intermediate"])


@pytest.mark.parametrize("top_k", [1, 2, 7, 40])
def test_decisions_preserve_ties_and_nonfinite_policy(modules, top_k):
    config, Pipeline = modules
    pipe = Pipeline(None, config.IntermediateConfig("v", "d", verification={"top_k": top_k}), 32)
    drafts = [[1, 2, 3], [], [0, 6]]
    lengths = [4, 1, 3]
    for seed in range(8):
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randint(-2, 3, (8, 8), generator=generator).float()
        logits[0, 1] = float("inf")
        logits[2, 3] = float("-inf")
        logits[5, 0] = float("nan")
        expected = []
        offset = 0
        for draft, size in zip(drafts, lengths):
            flags = pipe.policy.accept(logits[offset : offset + size], torch.tensor([*draft, 0]))
            stop = next((i for i in range(len(draft)) if not flags[i]), len(draft))
            expected.append([stop, logits[offset + stop].argmax().item()])
            offset += size
        assert pipe._decide(logits, drafts, lengths) == expected
        # Bonus values need not be zero: they are masked before acceptance.
        tokens = torch.tensor([1, 2, 3, 31, 30, 0, 6, 29])
        assert pipe._decide(logits, None, lengths, tokens=tokens) == expected


def test_shape_metadata_reused_and_bounded(modules):
    config, Pipeline = modules
    pipe = Pipeline(None, config.IntermediateConfig("v", "d"), 32)
    logits = scores([1, 2, 3])
    first = pipe._decision_shape(logits, [3])
    second = pipe._decision_shape(logits, [3])
    assert all(a is b for a, b in zip(first, second))
    for length in range(1, 24):
        pipe._decision_shape(torch.zeros(length, 32), [length])
    assert len(pipe._decision_shapes) == 16


def test_proposal_context_is_reused_without_mutating_retained_history(modules):
    _, Pipeline = modules
    backend = Backend([[[1, 7, 8]], [[3, 4, 5]], [[9, 10]]], proposals=[[[3, 4]], [[9]]])
    pipe = Pipeline(backend, NS(num_rounds=3, verification={"method": "topk", "top_k": 1}), 8)
    prefixes = [[20] * 8192]
    original = prefixes[0].copy()
    assert pipe.refine(prefixes, [[1, 2]], [8]) == [[1, 7, 3, 4, 5, 9, 10]]
    assert prefixes == [original]
    assert backend.calls[1][1][0] is backend.calls[2][1][0]
    assert backend.calls[3][1][0] is backend.calls[4][1][0]
    assert backend.calls[1][1][0] == original + [1, 7]
