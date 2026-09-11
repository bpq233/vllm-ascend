# Experimental MRV2 multi-stage speculative decoding

## Existing execution path

This design is based on the Ascend checkout and its pinned upstream vLLM commit
`ba07e4a48fc951300d97eb506217dd530583dea3`.

1. Scheduler puts request-specific drafts in `scheduled_spec_decode_tokens`.
2. Ascend `NPUModelRunner.execute_model` delegates execution to upstream MRV2.
   `prepare_inputs` combines the anchor and draft tokens from
   `req_states.draft_tokens`, using per-request cumulative logits offsets.
3. One packed target forward computes all scheduled candidate positions.
4. `sample_tokens` invokes `sample`: logits row i verifies input token i+1;
   the last logits row supplies a bonus token.
5. `postprocess_sampled` writes only final sampled output to request history,
   advances computed positions and subtracts rejected tokens.
6. DFlash proposes using target hidden states and the sampled anchor. The
   returned tensor is saved in `req_states.draft_tokens`. `take_draft_token_ids`
   returns lengths to the scheduler, which stores `request.spec_token_ids`.
7. Scheduler updates accepted output and KV accounting, and schedules the next
   verification. Scheduler may truncate a draft to its token budget.

The original draft handler assumes a uniform tensor width, even though input
preparation and scheduler support ragged sequences. Multi-stage decoding replaces
that handler with one which returns actual per-request token lists.

## Design and state ownership

`MultiStageDFlashSpeculator` reuses the primary DFlash implementation, then calls
`SpeculativePipeline`. The pipeline passes per-request speculative states through
an `IntermediateVerifier` and a `SecondaryDrafter`. An independent MRV2 runner
implements these roles, loading the configured intermediate model and its DFlash
model. `AcceptancePolicy` implements either full acceptance or approximate top-k
prefix acceptance. Only the existing final target sampler can commit output.

Intermediate accepted tokens remain private candidates. Neither the pipeline nor
the intermediate runner receives a scheduler request object to modify. The
intermediate runner has independent request buffers, attention layer registry,
block tables, and KV allocation. Target and primary DFlash retain their existing
separate layer KV tensors. Equal block numbers across runners do not imply shared
storage.

There are exactly `num_intermediate_rounds` intermediate verification passes at
most, including verification of the primary draft. Secondary drafting happens
only when another intermediate verification remains. The last unverified draft
is never appended. Zero acceptance, EOS, candidate budget and finished state stop
only the affected request. On exit, private speculative tails are logically
truncated; on the next outer iteration the private runner synchronizes with the
actual committed prefix, computing only the missing suffix.

The final target subclasses the upstream MRV2 rejection sampler and changes only
its candidate-acceptance rule. Parent MRV2 chunking, logprobs, sampled/rejected
counts, replacement and bonus sampling, EOS clamping, and request-history updates
remain in place. Final acceptance can use approximate top-k membership or accept
all valid, unmasked candidates. Replacement and bonus tokens still come from the
target's normal sampler with its request sampling parameters.

## Files

| File | Change | Purpose |
| --- | --- | --- |
| `worker/v2/spec_decode/multi_stage/config.py` | New | Opt-in validation and model composition |
| `multi_stage/state.py`, `interfaces.py` | New | Private request state and role contracts |
| `multi_stage/acceptance.py`, `pipeline.py` | New | Acceptance policies and bounded per-request loop |
| `multi_stage/backend.py` | New | Intermediate MRV2 runner, secondary DFlash and private KV |
| `multi_stage/speculator.py`, `runtime.py` | New | Primary reuse, ragged draft handoff and lifecycle |
| `multi_stage/sampler.py` | New | Configurable target top-k/full acceptance |
| `multi_stage/metrics.py` | New | Stage timing and token counters |
| `worker/v2/spec_decode/__init__.py` | Modify | Select the opt-in speculator |
| `worker/v2/model_runner.py` | Modify | Attach runtime, load hooks and request lifecycle |
| `ascend_config.py` | Modify | Accept the new additional-config namespace |
| `tests/ut/worker/v2/spec_decode/multi_stage/` | New | CPU policy, lifecycle, ragged and integration contracts |

## Validation boundary

Development environment: Windows with a separate CPU PyTorch test environment.
NPU end-to-end execution requires model checkpoints and an Ascend environment.
CPU tests do not establish NPU kernel correctness or a performance speedup.

## Configuration and running

Keep upstream `speculative_config.model` as the primary DFlash checkpoint.
Upstream `num_speculative_tokens` is the **total candidate capacity**, used by
the scheduler, buffers and target KV lookahead. The additional configuration
separately sets the primary and secondary draft widths. For 8 primary tokens and
three intermediate verification rounds with 4 secondary tokens per round, the
largest candidate is `8 + (3 - 1) * 4 = 16` tokens. Multi-stage initialization
raises a smaller configured capacity only up to the effective platform limit
before the scheduler and runner allocate their buffers. Although
the pinned native sampler supports 128 candidates, Ascend FIA's TND decode path
supports at most 16 query tokens, including the target bonus position. Therefore
the effective candidate limit is 15. If the configured rounds could produce
more, expansion stops as soon as that request has accepted 15 candidates and
the remaining intermediate rounds are skipped. An explicitly larger upstream
capacity is reduced to 15 during configuration, before attention builders run.
Each individual primary or secondary draft width must also be no greater than
15; the accumulated candidate list is truncated independently at the same limit.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/models/target",
    tensor_parallel_size=2,
    distributed_executor_backend="mp",
    enforce_eager=True,
    async_scheduling=False,
    enable_prefix_caching=False,
    max_model_len=4096,
    max_num_seqs=4,
    speculative_config={
        "method": "dflash",
        "model": "/models/primary-dflash",
        "num_speculative_tokens": 8,
    },
    additional_config={
        "multi_stage_spec_config": {
            "enabled": True,
            "intermediate_model": "/models/intermediate",
            "secondary_model": "/models/secondary-dflash",
            "primary_num_speculative_tokens": 8,
            "secondary_num_speculative_tokens": 4,
            "num_intermediate_rounds": 3,
            "kv_cache_memory_bytes": 1073741824,
            "intermediate_verification": {"method": "topk", "top_k": 5},
            "final_verification": {"method": "topk", "top_k": 5},
            "debug_logging": False,
            "metrics_enabled": False,
        }
    },
)
outputs = llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=64))
```

Select MRV2 using the existing upstream `VLLM_USE_V2_MODEL_RUNNER=1` setting.
At the default INFO level, enabling multi-stage decoding always emits summary
logs through vLLM's worker logger. It first reports `multi_stage_enabled` and
`multi_stage_target_sampler_installed`, then reports each intermediate round's
proposed and accepted counts, candidate counts ready for scheduling, the exact
candidate counts scheduled into each target forward, the final target acceptance
rate, and primary/intermediate/secondary/target-forward/target-verification time.
Accurate NPU timing adds device synchronizations. The legacy `summary_logging`
option remains accepted for configuration compatibility but no longer suppresses
these logs. Set
`VLLM_LOGGING_LEVEL=DEBUG` and `debug_logging=true`
to see token traces; these traces include prompt content. `metrics_enabled=true`
keeps cumulative in-process counters independently of summary logs.
For both `intermediate_verification` and `final_verification`, `method="topk"`
accepts the contiguous candidate prefix whose tokens belong to the corresponding
model's top k, while `method="all"` accepts every candidate with a valid token ID,
including candidates removed by target-side probability or grammar masks. `top_k`
must remain a positive integer and is ignored by `all`. Final top-k is evaluated
after the target's normal logits processing.
These configurable final policies are approximate and do not preserve the
target distribution in the same way as standard rejection sampling. Secondary
DFlash drafts greedily.

The upstream `Per-position acceptance rate` uses the total number of speculative
iterations as every position's denominator. With ragged multi-stage drafts it is
therefore a reach/survival rate: later values fall when fewer requests propose
that many candidates, even if every proposed candidate is accepted. Upstream
`Avg Draft acceptance rate` and the multi-stage `acceptance_rate` instead divide
total accepted candidates by total proposed candidates. With final `method="all"`,
the latter rates should be 100%.

Omit `multi_stage_spec_config` or set `enabled=false` to retain ordinary MRV2
decoding and the original DFlash implementation.

Initial supported scope is synchronous eager dense-model text generation with tensor parallelism,
with full-attention intermediate KV, identical target/intermediate token-ID
mappings, and DFlash checkpoints trained for their respective target models.
LoRA, structured outputs, min_tokens, trace replay, prefix caching, KV transfer,
sampling masks and custom logprob-token lists are rejected explicitly. Model
weights for both extra models and private KV are loaded during target model
loading, before target memory profiling. Increase the private KV budget if it
cannot retain the batch's prefixes. Allocation failure is explicit rather than
silently reusing another request's blocks.

The target, primary DFlash, intermediate, and secondary DFlash models all use
the same `tensor_parallel_size` and the existing vLLM TP process group. Setting
`speculative_config.draft_tensor_parallel_size` to a different value is rejected
because the per-rank multi-stage loop must execute the same collective sequence.
`kv_cache_memory_bytes` is a per-rank budget: each rank allocates that many bytes
for its intermediate-model KV shard. Pipeline, data, decode-context, and
prefill-context parallel sizes must remain 1. The experimental
`additional_config.enable_reduce_sample` path keeps vocabulary-sharded logits;
leave it disabled because the multi-stage top-k policies require gathered logits.

Intermediate requests currently run serially; each verification forwards its
whole candidate span, and initial long prefill is chunked. Target requests still
use the existing packed batch forward. Secondary drafting executes while the
intermediate request's buffers are bound, then the pipeline consumes its cached
proposal. For a candidate spanning several prefill chunks, secondary expansion
is omitted if its anchor falls outside the final chunk; already accepted
candidates still proceed to final verification. Near model-length capacity,
secondary expansion stops if the full DFlash query block cannot fit.

CPU regression tests (no NPU or vLLM install needed):

```bash
python -m pytest -q \
  --confcutdir=tests/ut/worker/v2/spec_decode/multi_stage \
  tests/ut/worker/v2/spec_decode/multi_stage
```

Real four-model NPU tests (12 parameter combinations, plus request reuse):

```bash
python -m pytest -sv tests/e2e/nightly/one_card/multi_stage \
  --multi-stage-models /models/target /models/primary-dflash \
                       /models/intermediate /models/secondary-dflash
```

Two-rank TP smoke test:

```bash
python -m pytest -sv \
  tests/e2e/pull_request/two_card/spec_decode/test_multi_stage.py \
  --multi-stage-models /models/target /models/primary-dflash \
                       /models/intermediate /models/secondary-dflash
```

The NPU tests compare intermediate/final top-k=1 greedy output to ordinary target
decoding, and exercise intermediate/final top-k=5/full acceptance with batch 1/4
and rounds 1/3. They are supplied
but have not been executed in the Windows development environment. The CPU
tests use real Torch policies and cache/lifecycle code with explicitly simulated
device forwards; they cannot validate NPU attention, weight loading or latency.

KG reference: `model-infer-kvcache`,
`model-infer-kvcache_res_references_standalone_kv_reference_md` (direct skill
loads, no search score). Retrieval anchor:
`ascendinfo_case_ascendpae_02vllm系列_vllmascendmodelrunner架构解析_vllmascendmodelrunner架构解析_draft_tokens`,
score `0.9094939`, source
`ascend-doc/ascend_info/case/Ascend_PAE/02_vLLM系列/【vLLM-Ascend】 ModelRunner架构解析/【vLLM-Ascend】 ModelRunner架构解析.md`.
That overview predates MRV2; execution contracts above were checked against the
pinned upstream source rather than inferred from that overview.
