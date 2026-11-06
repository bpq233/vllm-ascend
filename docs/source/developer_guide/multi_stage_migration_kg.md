# Hierarchical speculative decoding: KG evidence

Checked on 2026-11-06. Health, search/source/cypher golden queries, and all five onboarding guides succeeded. Relevant category indexes and executable skills were loaded before implementation.

## Sources and scope

| Node ID | Source file | Search score | Purpose |
| --- | --- | --- | --- |
| `python-refactoring` | `agent-skills/official/Common/python-refactoring/SKILL.md` | N/A: category-index lookup | Incremental migration, upstream reuse, regression coverage before cleanup |
| `pytest-writer` | `agent-skills/official/MindSpeed/mindspeed-llm-auto-ut-skills/skills/pytest-writer/SKILL.md` | N/A: category-index lookup | Isolated fixtures, boundary and error cases, parametrization |
| `model-infer-kvcache` | `cannbot-skills/model/model-infer-kvcache/SKILL.md` | N/A: category-index lookup | Consistent KV writes and attention reads |
| `model-infer-graph-mode` | `cannbot-skills/model/model-infer-graph-mode/SKILL.md` | N/A: category-index lookup | Stable graph buffers and addresses; avoiding device synchronization |
| `vllmascend_vllmascend_ops_triton_docs_v2_specdecode_preparedflashinputs_dflash_speculative_decoding` | `inference-serving/vllm-ascend/vllm_ascend/ops/triton/docs/v2/spec_decode/prepare_dflash_inputs.md` | 0.91843 | DFlash request mapping, padding, context/query contracts |

Loaded companion resources: `references/refactoring-config.json`, `references/code-smells.md`, `references/refactoring-in-practice.md`, `references/fixtures.md`, and `references/best-practices.md`.

The KV-cache and graph-mode skills primarily describe standalone `cann-recipes-infer` deployments. Only their general mapping and buffer rules apply here. The current checkout and its pinned vLLM source define the actual plugin interfaces and graph-dispatch behavior.

## Checks applied

- Preserve the distinction between batch rows and persistent request-state indexes in `idx_mapping`.
- Chunked prefill with `num_sampled == 0` must not carry rejected speculative tokens. Keep individual DFlash queries short; route accumulated hierarchical candidates through the target verification path.
- Initialize graph-visible padding and keep query boundaries consistent with query/output capacity. Remove padding only at a proven request boundary; never discard real query tokens.
- Use consistent slot-mapping and block-table addressing. Physical KV blocks and attention kernel pages can have different sizes; DCP ownership follows physical blocks.
- Check integer indexing and direct data movement bit-for-bit. Distinguish individual-kernel checks from wrapper integration and graph replay checks.
- Reuse upstream implementations; avoid duplicate progress-only D2H transfers, per-token `.item()`, and output-buffer self-copies.

## Evidence limits

**[Unverified by KG]** The user's custom old `multi_stage` algorithm was not found in KG. These sources do not prove that algorithm's correctness. Local old/new source comparisons, mathematical references, and regression tests provide the migration evidence.

The retrieved DFlash document describes a vLLM 0.31.0 ABI. Its constraints are review prompts, not a replacement for the current checkout's interfaces. Its scalar-position optimized path also does not establish multimodal support.

CPU/mock regressions validate only the covered host logic and interface contracts. Actual NPU execution, ACL capture/replay, and performance gains require hardware evidence. No NPU acceptance or measured speedup is claimed from these tests.
