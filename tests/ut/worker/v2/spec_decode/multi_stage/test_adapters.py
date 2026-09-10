# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("method", ["all", "topk1", "topk5"])
@pytest.mark.parametrize("seed", [3, 8, 17])
def test_packed_final_sampling_matches_request_oracle(adapters, method, seed):
    torch.manual_seed(seed)
    lengths = [0, 1, 3, 5]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length + 1)
    logits = torch.randn(offsets[-1], 12)
    tokens = torch.randint(0, 12, (offsets[-1],))
    target_samples = torch.randint(0, 12, (offsets[-1],))
    policy = (
        adapters.acceptance.AcceptAllPolicy() if method == "all" else adapters.acceptance.TopKPolicy(int(method[-1]))
    )
    output, counts = adapters.sampler.pack_accepted_prefixes(
        policy, logits, tokens, target_samples, torch.tensor(offsets), 6
    )
    for row, (start, end) in enumerate(zip(offsets, offsets[1:])):
        expected = adapters.acceptance.finalize_candidates(
            policy, logits[start:end], tokens[start + 1 : end], target_samples[start:end]
        )
        assert output[row, : counts[row]].tolist() == expected.tolist()
        assert (output[row, counts[row] :] == -1).all()
        assert counts[row] <= lengths[row] + 1


def make_runtime(adapters):
    runtime = adapters.runtime.MultiStageRuntime.__new__(adapters.runtime.MultiStageRuntime)
    runtime.runner = SimpleNamespace(req_states=SimpleNamespace(index_to_req_id={7: "a", 2: "b", 4: "c", 1: "d"}))
    runtime.stop_ids = {"a": frozenset({9}), "b": frozenset(), "c": frozenset({9}), "d": frozenset()}
    runtime.context_lengths = {name: 3 for name in "abcd"}
    runtime.max_lengths = {"a": 15, "b": 5, "c": 15, "d": 3}
    return runtime


def test_final_eos_bonus_and_max_tokens_before_worker_history(adapters):
    runtime = make_runtime(adapters)
    tokens = torch.tensor([[2, 9, 5, 6], [2, 3, 4, 5], [9, 3, 4, 5], [2, 3, 4, 5]])
    result, counts = runtime.limit_final_output(tokens, torch.tensor([4, 4, 4, 4]), [7, 2, 4, 1])
    assert counts.tolist() == [2, 2, 1, 0]
    assert result.tolist() == [[2, 9, -1, -1], [2, 3, -1, -1], [9, -1, -1, -1], [-1] * 4]


def test_ragged_handler_reports_real_lengths_and_takes_snapshot(adapters):
    runtime = SimpleNamespace(drafts={"a": [2, 3, 4], "b": [], "c": [5], "d": [6, 7]})
    handler = adapters.runtime.RaggedDraftTokensHandler(runtime)
    handler.set_draft_tokens(SimpleNamespace(req_ids=list("abcd")), torch.zeros(4, 16))
    runtime.drafts["a"].clear()
    output = handler.get_draft_tokens()
    assert output.req_ids == list("abcd")
    assert output.draft_token_ids == [[2, 3, 4], [], [5], [6, 7]]


def test_policy_sampler_reuses_target_samples_and_passes_acceptance_count(adapters):
    sampler = adapters.sampler.PolicyRejectionSampler.__new__(adapters.sampler.PolicyRejectionSampler)
    logits = torch.tensor([[0.0, 9.0, 1.0], [9.0, 1.0, 0.0], [0.0, 1.0, 9.0]])
    sampler.sampler = SimpleNamespace(sample=lambda *args, **kwargs: (torch.tensor([2, 0, 2]), logits))
    sampler.policy = adapters.acceptance.TopKPolicy(1)
    sampler.num_speculative_steps = 2
    sampler.config = SimpleNamespace(debug_logging=False, metrics_enabled=False)
    sampler.runtime = SimpleNamespace(limit_final_output=lambda tokens, counts, _: (tokens, counts))
    processed, sampled, count = sampler._verify(
        logits, None, torch.tensor([0, 1, 2]), None, torch.tensor([0, 3]), None, [0], None, None
    )
    assert processed is logits
    assert sampled.tolist() == [[1, 0, -1]]
    assert count.tolist() == [2]


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
        self.spans = []
        self.zeroed_blocks = []
        self.kv_block_zeroer = SimpleNamespace(zero_block_ids=self.zeroed_blocks.extend)
        self.model = SimpleNamespace(compute_logits=lambda hidden: hidden.repeat(1, 12))

    def execute_model(self, scheduled):
        req = scheduled.scheduled_new_reqs[0]
        start = req.num_computed_tokens
        end = start + scheduled.total_num_scheduled_tokens
        self.spans.append((start, end))
        positions = torch.arange(start, end)
        self.execute_model_state = SimpleNamespace(
            hidden_states=positions.float()[:, None],
            aux_hidden_states=None,
            input_batch=SimpleNamespace(positions=positions, num_tokens=end - start),
        )


def make_backend(adapters, chunk_size=100):
    backend = adapters.backend.IntermediateBackend.__new__(adapters.backend.IntermediateBackend)
    backend.config = SimpleNamespace(secondary_num_speculative_tokens=2, metrics_enabled=False)
    backend.device = torch.device("cpu")
    backend.runner = FakeRunner(chunk_size)
    backend.block_sizes = [4, 4]
    backend.free_blocks = list(range(1, 100))
    backend.caches = {}
    backend.next_drafts = {}
    backend._store_draft_context = lambda execution, cache: None
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
        input_batch=SimpleNamespace(positions=torch.tensor([0, 3, 4, 5]), num_tokens=4),
    )
    # Restore the real method (the generic fake-runner helper stubs this boundary).
    adapters.backend.IntermediateBackend._store_draft_context(
        backend, execution, adapters.backend.PrivateCache(blocks=[[10, 11], [20, 21]])
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
        parallel_config=SimpleNamespace(),
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
    assert target.cache_config.num_gpu_blocks_override == 99
    assert backend.vllm_config.cache_config.num_gpu_blocks_override is None
    assert "multi_stage_spec_config" in target.additional_config
    assert "multi_stage_spec_config" not in backend.vllm_config.additional_config
