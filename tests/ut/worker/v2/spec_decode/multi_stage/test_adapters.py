# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("method,top_k", [("all", 5), ("topk", 1), ("topk", 5)])
@pytest.mark.parametrize("seed", [3, 8, 17])
def test_packed_target_policy_matches_request_oracle(adapters, method, top_k, seed):
    torch.manual_seed(seed)
    lengths = [0, 1, 3, 5]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length + 1)
    logits = torch.randn(offsets[-1], 12)
    tokens = torch.randint(0, 12, (offsets[-1],))
    target_samples = torch.randint(0, 12, (offsets[-1],))
    policy = adapters.config.make_policy(adapters.config.VerificationConfig(method=method, top_k=top_k))
    output, counts, accepted = adapters.sampler.pack_accepted_prefixes(
        policy, logits, tokens, target_samples, torch.tensor(offsets), 6
    )
    for row, (start, end) in enumerate(zip(offsets, offsets[1:])):
        draft = tokens[start + 1 : end]
        mask = policy.accept(logits[start : end - 1], draft)
        expected_accepted = adapters.acceptance.accepted_prefix_length(mask)
        expected = torch.cat((draft[:expected_accepted], target_samples[start + expected_accepted :][:1]))
        assert accepted[row] == expected_accepted
        assert output[row, : counts[row]].tolist() == expected.tolist()
        assert (output[row, counts[row] :] == -1).all()


def make_runtime(adapters):
    runtime = adapters.runtime.MultiStageRuntime.__new__(adapters.runtime.MultiStageRuntime)
    runtime.runner = SimpleNamespace(req_states=SimpleNamespace(index_to_req_id={7: "a", 2: "b", 4: "c", 1: "d"}))
    runtime.stop_ids = {"a": frozenset({9}), "b": frozenset(), "c": frozenset({9}), "d": frozenset()}
    runtime.context_lengths = {name: 3 for name in "abcd"}
    runtime.max_lengths = {"a": 15, "b": 5, "c": 15, "d": 3}
    return runtime


def test_runtime_logs_enabled_configuration_at_startup(adapters, caplog):
    config = SimpleNamespace(
        metrics_enabled=False,
        summary_logging=True,
        primary_num_speculative_tokens=8,
        secondary_num_speculative_tokens=4,
        num_intermediate_rounds=3,
        intermediate_verification=SimpleNamespace(method="topk"),
        final_verification=SimpleNamespace(method="all"),
    )
    runner = SimpleNamespace(num_speculative_steps=15)
    with caplog.at_level("INFO"):
        adapters.runtime.MultiStageRuntime(runner, config)
    assert "multi_stage_enabled primary_tokens=8 secondary_tokens=4" in caplog.text
    assert "intermediate_rounds=3" in caplog.text
    assert "intermediate_verification=topk" in caplog.text
    assert "final_verification=all" in caplog.text
    assert "target_capacity=15" in caplog.text


def test_runtime_summary_logging_false_suppresses_startup_log(adapters, caplog):
    config = SimpleNamespace(
        metrics_enabled=False,
        summary_logging=False,
        primary_num_speculative_tokens=8,
        secondary_num_speculative_tokens=4,
        num_intermediate_rounds=3,
        intermediate_verification=SimpleNamespace(method="topk"),
        final_verification=SimpleNamespace(method="all"),
    )
    with caplog.at_level("INFO"):
        adapters.runtime.MultiStageRuntime(SimpleNamespace(num_speculative_steps=15), config)
    assert not caplog.records


def test_target_policy_sampler_reuses_normal_target_samples(adapters, monkeypatch):
    sampler = adapters.sampler.PolicyRejectionSampler.__new__(adapters.sampler.PolicyRejectionSampler)
    logits = torch.tensor([[0.0, 9.0, 1.0], [9.0, 1.0, 0.0], [0.0, 1.0, 9.0]])
    sampler.sampler = SimpleNamespace(sample=lambda *args, **kwargs: (torch.tensor([2, 0, 2]), logits))
    sampler.policy = adapters.acceptance.TopKPolicy(1)
    sampler.num_speculative_steps = 2
    sampler.config = SimpleNamespace(debug_logging=False, summary_logging=False, metrics_enabled=False)
    sampler.runtime = SimpleNamespace(record_final=lambda *args: None)
    monkeypatch.setattr(adapters.sampler.torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    processed, sampled, count = sampler._verify(
        logits, None, torch.tensor([0, 1, 2]), None, torch.tensor([0, 3]), None, [0], None, None
    )
    assert processed is logits
    assert sampled.tolist() == [[1, 0, -1]]
    assert count.tolist() == [2]


def test_accept_all_keeps_masked_candidates_and_target_bonus(adapters, monkeypatch):
    sampler = adapters.sampler.PolicyRejectionSampler.__new__(adapters.sampler.PolicyRejectionSampler)
    processed = torch.tensor([[0.0, 0.0, 0.0], [0.0, -torch.inf, 0.0], [0.0, 0.0, -torch.inf]])
    sampler.sampler = SimpleNamespace(sample=lambda *args, **kwargs: (torch.tensor([0, 0, 2]), processed))
    sampler.policy = adapters.acceptance.AcceptAllPolicy()
    sampler.num_speculative_steps = 2
    sampler.config = SimpleNamespace(debug_logging=False, summary_logging=False, metrics_enabled=False)
    recorded = []
    sampler.runtime = SimpleNamespace(record_final=lambda *args: recorded.append(args))
    monkeypatch.setattr(adapters.sampler.torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)

    _, sampled, count = sampler._verify(
        processed,
        None,
        torch.tensor([0, 1, 2]),
        None,
        torch.tensor([0, 3]),
        None,
        [0],
        None,
        None,
    )

    assert sampled.tolist() == [[1, 2, 2]]
    assert count.tolist() == [3]
    assert not recorded


def test_target_verification_summary_reports_policy_and_acceptance(adapters, caplog):
    runtime = make_runtime(adapters)
    runtime.config = SimpleNamespace(
        debug_logging=False,
        summary_logging=True,
        final_verification=adapters.config.VerificationConfig(method="topk", top_k=5),
    )
    runtime.metrics = adapters.metrics.PipelineMetrics(True)
    with caplog.at_level("INFO"):
        runtime.record_final(
            torch.zeros(3, 8),
            torch.tensor([1, 2, 3]),
            torch.tensor([0, 3]),
            torch.tensor([[2, 7, -1]]),
            torch.tensor([2]),
            [7],
            torch.tensor([1]),
            1.25,
        )
    assert "multi_stage_target_verification method=topk top_k=5" in caplog.text
    assert "candidates_by_request={'a': 2}" in caplog.text
    assert "accepted_by_request={'a': 1}" in caplog.text
    assert "acceptance_rate=50.0%" in caplog.text
    assert "verification_ms=1.250" in caplog.text


def test_target_accept_all_summary_reports_one_hundred_percent(adapters, caplog):
    runtime = make_runtime(adapters)
    runtime.config = SimpleNamespace(
        debug_logging=False,
        summary_logging=True,
        final_verification=adapters.config.VerificationConfig(method="all"),
    )
    runtime.metrics = adapters.metrics.PipelineMetrics(False)
    with caplog.at_level("INFO"):
        runtime.record_final(
            torch.full((3, 8), -torch.inf),
            torch.tensor([1, 2, 3]),
            torch.tensor([0, 3]),
            torch.tensor([[2, 3, 7]]),
            torch.tensor([3]),
            [7],
            torch.tensor([2]),
            1.25,
        )
    assert "method=all top_k=None" in caplog.text
    assert "total_candidates=2 total_accepted=2 acceptance_rate=100.0%" in caplog.text


def test_ragged_handler_reports_real_lengths_and_takes_snapshot(adapters):
    runtime = SimpleNamespace(drafts={"a": [2, 3, 4], "b": [], "c": [5], "d": [6, 7]})
    handler = adapters.runtime.RaggedDraftTokensHandler(runtime)
    handler.set_draft_tokens(SimpleNamespace(req_ids=list("abcd")), torch.zeros(4, 16))
    runtime.drafts["a"].clear()
    output = handler.get_draft_tokens()
    assert output.req_ids == list("abcd")
    assert output.draft_token_ids == [[2, 3, 4], [], [5], [6, 7]]


def test_runtime_logs_ready_candidate_count_and_stage_times(adapters, caplog):
    runtime = adapters.runtime.MultiStageRuntime.__new__(adapters.runtime.MultiStageRuntime)
    runtime.config = SimpleNamespace(summary_logging=True)
    runtime.runner = SimpleNamespace(
        num_speculative_steps=16,
        max_model_len=64,
        device=torch.device("cpu"),
        req_states=SimpleNamespace(
            total_len=SimpleNamespace(gpu=torch.tensor([3, 2])),
            all_token_ids=SimpleNamespace(gpu=torch.tensor([[1, 2, 3], [4, 5, 0]])),
        ),
    )
    runtime.stop_ids = {"a": frozenset(), "b": frozenset()}
    runtime.context_lengths = {}
    runtime.max_lengths = {"a": 64, "b": 64}
    runtime.last_primary_ms = 1.25

    class Pipeline:
        last_run = {"intermediate_verifier_ms": 2.5, "secondary_drafter_ms": 0.75}

        @staticmethod
        def run(states, primary):
            return {"a": [7, 8, 9], "b": [6]}

    runtime.pipeline = Pipeline()
    input_batch = SimpleNamespace(req_ids=["a", "b"], idx_mapping=torch.tensor([0, 1]), num_reqs=2)
    with caplog.at_level("INFO"):
        output = runtime.expand(input_batch, torch.tensor([[7, 8], [6, 0]]), torch.tensor([1, 1]))
    assert output.shape == (2, 16)
    assert "candidates_by_request={'a': 3, 'b': 1}" in caplog.text
    assert "total_candidates=4" in caplog.text
    assert "primary_model_ms=1.250" in caplog.text
    assert "multi_stage_candidates_ready" in caplog.text


def test_runtime_logs_exact_candidates_scheduled_to_target(adapters, monkeypatch, caplog):
    runtime = adapters.runtime.MultiStageRuntime.__new__(adapters.runtime.MultiStageRuntime)
    runtime.config = SimpleNamespace(summary_logging=True, metrics_enabled=False)
    runtime.metrics = adapters.metrics.PipelineMetrics(True)
    runtime.current_target_candidate_counts = {"a": 15, "b": 3}
    monkeypatch.setattr(adapters.runtime.torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    with caplog.at_level("INFO"):
        start = runtime.before_forward()
        runtime.after_forward(start, 20)
    assert "target_candidates_by_request={'a': 15, 'b': 3}" in caplog.text
    assert "total_target_candidates=18" in caplog.text
    assert "scheduled_tokens=20" in caplog.text
    assert "target_model_forward_ms=" in caplog.text


def test_finished_and_preempted_requests_release_private_cache(adapters):
    runtime = make_runtime(adapters)
    released = []
    runtime.backend = SimpleNamespace(release=released.append)
    runtime.drafts = {"a": [1], "b": [2]}
    runtime.observe(SimpleNamespace(finished_req_ids={"a"}, preempted_req_ids={"b"}, scheduled_new_reqs=[]))
    assert set(released) == {"a", "b"}
    assert "a" not in runtime.context_lengths
    assert "b" in runtime.context_lengths  # retained until resume metadata arrives
    assert not runtime.drafts


class FakeRunner:
    """Only device execution is fake; _forward, offsets and rollback are real."""

    def __init__(self, chunk_size=100):
        self.max_model_len = 100
        self.max_num_tokens = chunk_size
        self.execute_calls = 0
        self.spans = []
        self.zeroed_blocks = []
        self.kv_block_zeroer = SimpleNamespace(zero_block_ids=self.zeroed_blocks.extend)
        self.model = SimpleNamespace(compute_logits=lambda hidden: hidden.repeat(1, 12))

    def execute_model(self, scheduled):
        self.execute_calls += 1
        positions = []
        offsets = [0]
        req_ids = []
        for req in scheduled.scheduled_new_reqs:
            start = req.num_computed_tokens
            end = start + scheduled.num_scheduled_tokens[req.req_id]
            self.spans.append((start, end))
            positions.append(torch.arange(start, end))
            offsets.append(offsets[-1] + end - start)
            req_ids.append(req.req_id)
        packed_positions = torch.cat(positions)
        self.execute_model_state = SimpleNamespace(
            hidden_states=packed_positions.float()[:, None],
            aux_hidden_states=None,
            input_batch=SimpleNamespace(
                req_ids=req_ids,
                positions=packed_positions,
                query_start_loc_np=offsets,
                num_tokens=len(packed_positions),
            ),
        )


def make_backend(adapters, chunk_size=100):
    backend = adapters.backend.IntermediateBackend.__new__(adapters.backend.IntermediateBackend)
    backend.config = SimpleNamespace(
        secondary_num_speculative_tokens=2,
        num_intermediate_rounds=1,
        debug_logging=False,
        intermediate_verification=SimpleNamespace(top_k=5),
        metrics_enabled=False,
        summary_logging=False,
    )
    backend.timing_enabled = False
    backend.device = torch.device("cpu")
    backend.runner = FakeRunner(chunk_size)
    backend.block_sizes = [4, 4]
    backend.free_blocks = list(range(1, 100))
    backend.caches = {}
    backend.next_drafts = {}
    backend.vllm_config = SimpleNamespace()
    backend.policy = adapters.acceptance.AcceptAllPolicy()
    backend._store_draft_context = lambda execution, cache: None
    backend._store_draft_contexts = lambda execution, caches: None
    return backend


def test_private_cache_rollback_and_incremental_resynchronization(adapters):
    backend = make_backend(adapters)
    _, logits, cache = backend._forward("r", (1, 2, 3, 4, 5, 6), 4)
    assert logits[:, 0].tolist() == [3, 4]  # row before each candidate
    backend.rollback("r", 3)
    assert cache.tokens == (1, 2, 3)
    _, logits, cache = backend._forward("r", (1, 2, 3, 8, 9, 10, 11), 5)
    assert backend.runner.spans == [(0, 6), (3, 7)]
    assert logits[:, 0].tolist() == [4, 5]
    assert cache.tokens == (1, 2, 3, 8, 9, 10, 11)
    backend.rollback("r", 3)
    after = len(backend.free_blocks)
    backend.rollback("r", 3)
    assert len(backend.free_blocks) == after


def test_chunked_private_prefill_keeps_all_verification_logits(adapters):
    backend = make_backend(adapters, chunk_size=2)
    _, logits, _ = backend._forward("r", tuple(range(8)), 5)
    assert backend.runner.spans == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert logits[:, 0].tolist() == [4, 5, 6]


def test_intermediate_verification_packs_active_requests_into_one_forward(adapters):
    backend = make_backend(adapters)
    states = [
        adapters.state.SpeculativeState("a", (1, 2), 8),
        adapters.state.SpeculativeState("b", (5, 6), 8),
    ]
    states[0].current_draft = [3, 4]
    states[1].current_draft = [7, 8]

    results = backend.verify(states)

    assert backend.runner.execute_calls == 1
    assert backend.runner.spans == [(0, 4), (0, 4)]
    assert results["a"].accepted_tokens == [3, 4]
    assert results["b"].accepted_tokens == [7, 8]


def test_private_block_groups_and_requests_never_alias(adapters):
    backend = make_backend(adapters)
    first = backend._reserve("a", 9)
    second = backend._reserve("b", 5)
    allocated = [block for cache in (first, second) for group in cache.blocks for block in group]
    assert len(allocated) == len(set(allocated))
    assert 0 not in allocated
    assert set(backend.runner.zeroed_blocks) == set(allocated)
    backend.free_blocks.clear()
    with pytest.raises(RuntimeError, match="budget exhausted"):
        backend._reserve("a", 99)


@pytest.mark.parametrize("accepted_end", [5, 8])
def test_secondary_anchor_has_no_one_token_shift(adapters, accepted_end):
    backend = make_backend(adapters)
    captured = {}

    def propose(*args):
        captured["num_rejected"] = args[6].item()
        return torch.tensor([[7, 8]])

    backend.runner.speculator = SimpleNamespace(propose=propose)
    backend.runner.req_states = SimpleNamespace(
        req_id_to_index={"r": 0}, last_sampled_tokens=torch.zeros(1, 1, dtype=torch.int64), next_prefill_tokens=None
    )
    backend.runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(temperature=SimpleNamespace(gpu=None), seeds=SimpleNamespace(gpu=None))
    )
    positions = torch.arange(2, 9)
    execution = SimpleNamespace(
        input_batch=SimpleNamespace(positions=positions, num_tokens=len(positions)),
        attn_metadata={},
        slot_mappings_by_layer={},
        hidden_states=None,
        aux_hidden_states=None,
    )
    backend._draft("r", execution, accepted_end, 6)
    last_valid = positions[len(positions) - captured["num_rejected"] - 1].item()
    assert last_valid + 1 == accepted_end - 1
    assert backend.runner.req_states.last_sampled_tokens[0, 0] == 6
    assert backend.propose([SimpleNamespace(request_id="r")]) == {"r": [7, 8]}


def test_secondary_context_slots_use_private_group_and_position(adapters):
    backend = make_backend(adapters)
    captured = {}

    def store(hidden, positions, slots):
        captured.update(hidden=hidden, positions=positions, slots=slots)

    backend.runner.speculator = SimpleNamespace(
        draft_kv_cache_group_ids=[0, 1],
        _layer_group_idx=[1, 0],
        model=SimpleNamespace(combine_hidden_states=lambda x: x, precompute_and_store_context_kv=store),
    )
    execution = SimpleNamespace(
        hidden_states=torch.zeros(4, 2),
        aux_hidden_states=[torch.ones(4, 1), torch.full((4, 1), 2.0)],
        input_batch=SimpleNamespace(
            req_ids=["r"],
            positions=torch.tensor([0, 3, 4, 5]),
            query_start_loc_np=[0, 4],
            num_tokens=4,
        ),
    )
    # Restore the real method (the generic fake-runner helper stubs this boundary).
    adapters.backend.IntermediateBackend._store_draft_contexts(
        backend,
        execution,
        {"r": adapters.backend.PrivateCache(blocks=[[10, 11], [20, 21]])},
    )
    assert captured["slots"][0].tolist() == [80, 83, 84, 85]
    assert captured["slots"][1].tolist() == [40, 43, 44, 45]
    assert captured["hidden"].tolist() == [[1.0, 2.0]] * 4


def test_backend_constructor_does_not_share_target_registry_or_quantization(adapters, monkeypatch):
    class Config(SimpleNamespace):
        @staticmethod
        def _get_quantization_config(model, load):
            return (model.model, load)

    target_registry = {"target.layer": object()}
    target = Config(
        model_config=SimpleNamespace(
            dtype=torch.float32, max_model_len=64, trust_remote_code=False, tokenizer="target", tokenizer_revision=None
        ),
        compilation_config=SimpleNamespace(static_forward_context=target_registry, custom_ops=["all"]),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(num_gpu_blocks_override=99, enable_prefix_caching=False),
        additional_config={"multi_stage_spec_config": {"enabled": True}},
        attention_config=SimpleNamespace(),
        load_config=object(),
        quant_config="target_quant",
    )

    def model_config(**kwargs):
        return SimpleNamespace(**kwargs, tokenizer=kwargs["model"], tokenizer_revision=None, is_moe=False)

    monkeypatch.setattr(adapters.backend, "ModelConfig", model_config)
    monkeypatch.setattr(
        adapters.backend, "CompilationConfig", lambda **kwargs: SimpleNamespace(**kwargs, static_forward_context={})
    )
    spec = adapters.backend.FullAttentionSpec()
    spec.sliding_window = spec.attention_chunk_size = None
    spec.block_size = 4
    kv = SimpleNamespace(num_blocks=32, kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
    monkeypatch.setattr(adapters.backend, "get_kv_cache_configs", lambda *args: [kv])

    class Runner:
        def __init__(self, config, device):
            self.config = config

        def load_model(self):
            self.config.compilation_config.static_forward_context["private.layer"] = object()

        def get_kv_cache_spec(self):
            return {"private.layer": spec}

        def initialize_kv_cache(self, config):
            self.kv_config = config

        def _init_kv_zero_meta(self):
            self.kv_block_zeroer = SimpleNamespace(zero_block_ids=lambda blocks: None)

    runners = ModuleType("vllm_ascend.worker.v2.model_runner")
    runners.NPUModelRunner = Runner
    monkeypatch.setitem(sys.modules, runners.__name__, runners)
    tokenizers = ModuleType("vllm.tokenizers")
    tokenizers.get_tokenizer = lambda *args, **kwargs: SimpleNamespace(get_vocab=lambda: {"a": 1, "b": 2})
    monkeypatch.setitem(sys.modules, tokenizers.__name__, tokenizers)
    config = adapters.config.MultiStageConfig(
        enabled=True, intermediate_model="intermediate", secondary_model="secondary"
    )
    backend = adapters.backend.IntermediateBackend(
        target, torch.device("cpu"), config, adapters.acceptance.TopKPolicy(1)
    )
    assert set(target_registry) == {"target.layer"}
    assert set(backend.vllm_config.compilation_config.static_forward_context) == {"private.layer"}
    assert backend.vllm_config.quant_config[0] == "intermediate"
    assert target.quant_config == "target_quant"
    assert backend.vllm_config.parallel_config is not target.parallel_config
    assert backend.vllm_config.parallel_config.tensor_parallel_size == 2
    assert backend.vllm_config.speculative_config.draft_tensor_parallel_size == 2
    assert target.cache_config.num_gpu_blocks_override == 99
    assert backend.vllm_config.cache_config.num_gpu_blocks_override is None
    assert "multi_stage_spec_config" in target.additional_config
    assert "multi_stage_spec_config" not in backend.vllm_config.additional_config
