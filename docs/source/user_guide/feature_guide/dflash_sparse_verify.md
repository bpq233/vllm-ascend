# Experimental sparse-layer multi-round DFlash

This opt-in experiment reuses the first N consecutive layers of a dense Qwen3
Target for a causal prefix verifier. There are no skipped layers inside that
prefix. The legacy `dflash_sparse_verify` configuration name is retained.
It alternates original parallel DFlash proposals and prefix verification,
then submits the accumulated candidates to the normal scheduler for one logical
complete Target verification. Intermediate acceptance never commits output tokens.

Hardware validation is pending. CPU reference and interface tests do not establish
CANN kernel compatibility, ACL capture/replay correctness, or a performance gain.

## Configuration

```python
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

llm = LLM(
    model="/home/data/Qwen3-8B",
    enable_prefix_caching=False,
    max_model_len=4096,
    max_num_seqs=16,
    max_num_batched_tokens=4096,
    speculative_config={
        "method": "dflash",
        "model": "/home/data/Qwen3-8B-DFlash-b16",
        "num_speculative_tokens": 7,
    },
    compilation_config=CompilationConfig(cudagraph_mode="FULL"),
    additional_config={
        "dflash_sparse_verify": {
            "enabled": True,
            "mode": "confidence",
            "num_intermediate_layers": 12,
            "num_rounds": 4,
            "threshold_high": 0.9,
            "threshold_low": 0.5,
            "feature_source": "sparse",
            "enable_metrics": True,
            "final_verification": {
                "method": "strict",
                "topk": 2,
                "target_threshold": 0.1,
            },
        }
    },
)
outputs = llm.generate(["Explain causal attention."], SamplingParams(temperature=0, max_tokens=128))
print(outputs[0].outputs[0].text)
```

| Option | Meaning |
| --- | --- |
| `enabled` | Defaults to false; ordinary DFlash uses its existing class. |
| `mode` | `fixed` uses sparse argmax; `confidence` uses two thresholds. |
| `num_intermediate_layers` | N: execute original layers 0 through N-1. Must accommodate the checkpoint's feature count. |
| `num_rounds` | Fixed round limit or confidence-mode maximum; zero-accept rounds consume the limit. |
| `threshold_high`, `threshold_low` | Confidence requires `0 <= low < high <= 1`; fixed mode ignores their values. |
| `feature_source` | `sparse` uses newly computed approximate features. `target` hands off to Target when true features do not cover the new prefix. |
| `enable_metrics` | Enables synchronized NPU event timings, bounded detail records and cumulative counters. |
| `final_verification.method` | `strict`, `topk`, `confidence` or `accept_all`. |
| `final_verification.topk` | Positive k, used only by `topk`; defaults to 2. |
| `final_verification.target_threshold` | `(0, 1]`, used only by final `confidence`; defaults to 0.1. |

Only `strict` is intended to reproduce unmodified greedy Target decoding.
`topk`, final `confidence` and `accept_all` are approximate generation policies.
Confidence at threshold 1 accepts tied maxima; strict follows argmax's first-index
tie rule. Nonfinite Target distributions raise an error. Invalid sparse distributions
stop expansion and hand the pending proposal to Target.

The public speculative token count remains the DFlash block width K. Platform
validation reserves K times `num_rounds` candidate slots before scheduler, cache,
sampler and graph initialization; the copied DFlash configuration retains K.
The complete Target receives the entire retained prefix, not only the last block.
Context, generation, EOS and scheduler budgets may shorten a request's proposal.
Near the context end, a full native DFlash mask block is not launched if it would
exceed the context. This safely hands off the accumulated prefix to Target.

## Weights, features and caches

The sparse view shares embeddings, selected decoder weights, RMSNorm and LM Head.
It clones only mutable Attention bookkeeping and registers independent cache
owners before upstream KV memory profiling. The scheduler accounts for those
additional cache tensors. Target, sparse verifier and DFlash have separate KV data.
Gathered block-table buffers and their cached device pointer arrays are also private.

The loaded Target's auxiliary capture boundaries determine the checkpoint's
feature count. A capture boundary c denotes the output after original layer c-1.
The verifier always executes layers `[0, ..., N-1]`. If all original feature
boundaries lie within `[1, N]`, keep them. Otherwise select the same number of
distinct feature boundaries uniformly from the first through the Nth layer.
A checkpoint with a single feature uses the Nth layer. N smaller than the
feature count raises an error; no features are silently dropped or repeated.

For example, with L=32, N=12 and original capture boundaries `[1, 16, 31]`, execute
layers `[0, 1, ..., 11]` and capture `[1, 6, 12]` (zero-based layers `[0, 5, 11]`).
Both Target and prefix collectors are updated before draft loading or graph
capture. Feature count, width and increasing channel order remain unchanged.
The DFlash FC weights are not retrained: remapped channels change its checkpoint
feature semantics, which is explicitly logged. N=L preserves original valid
feature boundaries and executes every original layer.

Each round labels features with source, layer order, tensor shape, token identity
and absolute position interval. The initial round consumes true Target states.
Further sparse rounds consume states strictly before the new anchor; rejected
states are excluded. For identical tokens and positions, these states should
match the same early layers of full Target, subject to numerical tolerance.
The prefix's final logits still apply the original final RMSNorm and LM Head
at an early exit, and need not match full Target or improve acceptance rates.
Sparse KV suffix invalidation uses actual sequence endpoints: subsequent calls
overwrite from the anchor and cannot attend the stale rejected suffix.

In confidence mode, only consecutive high-threshold candidates are temporarily
accepted. The first middle-threshold candidate causes a retry; the first low one
stops expansion and hands off the accumulated prefix plus the current pending
block. Four final policies share correction/bonus and upstream commit logic.
Accept-all executes all Target Transformer rows but computes only the last
required LM-head row per request.

## Graphs and queries longer than 16 tokens

No private graph configuration is introduced. Target retains its native
`ModelAclGraphManager`; DFlash retains `DFlashAclGraphManager`. The sparse forward
uses another native `ModelAclGraphManager` and captures its selected layers,
normalization, LM Head and auxiliary outputs. Scoped graph parameter ownership
isolates its event/handle/workspace lists from Target and DFlash.

The bounded Python round controller runs outside capture. Replays read refreshed
token, position, block-table, sequence-length and feature buffers. Capture sizes
are upstream **total token counts**, not per-request counts. Explicit sizes are
preserved. Missing shapes use the native fallback and emit a warning identifying
the actual sparse execution mode. DFlash follows its upstream compatibility rule
for PIECEWISE; sparse PIECEWISE needs the native breakable wrapper, otherwise the
native manager reports its unsupported configuration. Neither enables FULL silently.

For causal FIA, a long logical query becomes virtual query rows of at most 16
tokens, sharing its request's block-table row. Chunk j exposes KV only through
`historical_prefix + chunk_end[j]`. All query KV is written before the paged
attention call. No candidate is dropped or reordered. This is one model forward
with bounded Attention query rows, followed by one decision over the whole
candidate sequence. Empty padding query rows are omitted; real token rows remain.
FlashAttention backends keep their native long-query path.

The reference tests compare every causal Attention row for lengths
`1, 7, 15, 16, 17, 24, 28, 32, 48, 64`, including mixed request lengths,
historical prefixes and invalidated suffixes. **CANN FIA semantics and graph
parameter updates for these virtual rows still require real NPU validation.**

## Offline, Serve and benchmark commands

```bash
python examples/offline_inference/dflash_sparse_verify.py \
  --model /home/data/Qwen3-8B --draft-model /home/data/Qwen3-8B-DFlash-b16 \
  --variant confidence --final-method strict --num-rounds 4 --draft-tokens 7 \
  --graph-mode FULL --require-graph-replays --output sparse-full.json

vllm serve /home/data/Qwen3-8B \
  --no-enable-prefix-caching --max-model-len 4096 --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"dflash","model":"/home/data/Qwen3-8B-DFlash-b16","num_speculative_tokens":7}' \
  --compilation-config '{"cudagraph_mode":"FULL"}' \
  --additional-config '{"dflash_sparse_verify":{"enabled":true,"mode":"fixed","num_intermediate_layers":12,"num_rounds":4,"feature_source":"sparse","final_verification":{"method":"strict"}}}'

python benchmarks/dflash_sparse_verify.py \
  --model /home/data/Qwen3-8B --draft-model /home/data/Qwen3-8B-DFlash-b16 \
  --graph-modes NONE FULL FULL_DECODE_ONLY --layers 12 all --rounds 1 2 4 \
  --draft-lengths 7 --batch-sizes 1 4 --output sparse-benchmark.json
```

Serve requests must specify `temperature: 0`. The scripts use actual worker RPC
to collect selected layers and observed capture/replay counters. Flags
`--require-graph-replays` and `--require-min-final-proposal 17` fail when a workload
does not produce the requested evidence. They do not manufacture long proposals.
The benchmark separates initialization from steady-state measurements and writes
raw case JSON/logs as well as output agreement and throughput/TPOT comparisons.
Approximate-policy output agreement is a quality measurement, not losslessness.

Metrics retain at most 2048 detail records per kind and separate cumulative counts.
With `enable_metrics=true`, `DFlash sparse rejection` and `DFlash Target rejection`
report the first rejected candidate's `candidate_token_id`, `candidate_logit`,
`max_token_id`, `max_logit`, `logit_gap` (maximum minus candidate), and relative
`confidence`. Sparse records also include the absolute `candidate_position`.
The rejection index is zero-based within the current draft block. These values
come from the verifier's prediction row for that candidate, before softmax;
they are not the DFlash proposal logits. All-accepted rounds have no rejection record.
Counts refer to sampled proposals before the engine applies final stop-string or
generation-length truncation. Event timings include the corresponding worker
submission's transfers and kernels; controller `host_records` are separately
labelled submission durations. Accept-all does not compute a counterfactual strict
acceptance length, since doing so would require the skipped LM-head rows.

## Validation and current limits

```bash
python -m pytest -q --confcutdir=tests/ut/worker/v2/dflash_sparse \
  tests/ut/worker/v2/dflash_sparse

export DFLASH_SPARSE_TEST_TARGET=/home/data/Qwen3-8B
export DFLASH_SPARSE_TEST_DRAFT=/home/data/Qwen3-8B-DFlash-b16
pytest -q tests/e2e/pull_request/one_card/model_runner_v2/test_dflash_sparse_verify.py
```

The first implementation supports dense Qwen3, original DFlash, plain greedy
sampling, ordinary tensor parallelism and multiple independent requests.
PP, DP, PCP, DCP, LoRA, KV transfer, prefix caching, quantized KV, specialized
LM-head TP, KVPP/offload, watermarking, probability sampling, penalties,
constraints, logprobs and min-tokens EOS masking are rejected explicitly.
Requests are processed sequentially inside each proposal batch; host decisions
and opt-in event synchronization impose overhead. Throughput improvements are
unproven. Prefix priming is chunked using upstream capture capacity to bound
temporary logits memory. Sparse KV and graph output buffers increase memory use.

No compatible checkpoint, `vllm`, `torch_npu` or NPU was available in the local
Windows environment. Real capture/replay logs, BF16/FP16 graph/eager comparisons,
strict end-to-end token equality, Serve execution and performance measurements
remain unrun. The opt-in hardware tests deliberately skip without explicit model
paths. Do not treat CPU tests or configured FULL mode as proof of NPU execution.
