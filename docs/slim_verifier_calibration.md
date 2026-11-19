# Slim Verifier：分位置诊断、子层消融与组合搜索

本工具用于定位减层误差并选择候选层集合。默认采用与线上影子评估相同的语义：全层模型生成历史 KV，Slim 只计算当前块。搜索结果写入 JSON，不会自动修改服务配置或采样行为。

## 线上配置

原来的 `keep_layers` / `skip_layers` 继续表示整个 Decoder Layer。新增两个独立选项：

```python
additional_config = {
    "slim_verifier_eval": {
        "enabled": True,
        "kv_source": "target",
        "skip_attention_layers": [18],
        "skip_mlp_layers": [20, 24],
        "acceptance_strategy": "probability_ratio",
        "probability_ratio_threshold": 0.5,
        "log_interval": 100,
    }
}
```

这个例子跳过第 18 层的 Attention 分支，以及第 20、24 层的 MLP 分支。仅提供子层选项时，默认保留全部 Decoder 层容器。层号仍为 Target 的原始 0-based 编号，具体集合需通过实际模型校准；上例不是推荐层集合。

也可与整层选择组合使用，但子层索引必须属于 `keep_layers`，不能对已经整层删除的位置重复指定。Attention 与 MLP 同时跳过，等价于删除整个 Decoder Layer。跳过分支时相应的前置 Norm 也不执行；`hidden` / `residual` 继续遵守原模型的残差约定。跳过 Attention 的层不收集历史 KV。

## 新增指标

以下指标均为原始 Logits 的影子诊断。**Target 的接受标准仍是 Top-1 匹配，不是随机采样、Penalty 或 Grammar 处理后的实际接受结果。** 日志中的 `target_acceptance_basis` 明确标记为 `raw_top1_diagnostic`。

| 指标 | 定义与用途 |
| --- | --- |
| `per_position` | 按草稿块中的 1-based 位置分组，包含该位置样本数、两侧接受率、Top-1 一致率和 KL。变长块只统计实际存在的位置 |
| `mean_kl_target_to_slim` | 所有预测位置的 KL(Target ∥ Slim) 均值；对整体 Logit 常数平移不敏感 |
| `target_prefix_tokens` | Target 在第一次拒绝前连续接受的 Token 总数 |
| `slim_rejection_rate_in_target_prefix` | Target 连续可接受前缀内，被 Slim 拒绝的位置数 / 该前缀 Token 数 |
| `joint_avg_accepted_length` | 两侧都接受的连续前缀长度之和 / Block 数 |
| `target_prefix_preservation_rate` | 两侧共同连续接受的 Token 总数 / Target 连续接受 Token 总数 |
| `mean_kl_in_target_prefix` | 仅在 Target 连续可接受前缀内计算的平均 KL |
| `target_rejection_boundaries` | 存在 Target 第一次拒绝位置的 Block 数 |
| `slim_acceptance_at_target_rejection_rate` | 在 Target 第一次拒绝的位置，Slim 接受的比例 |

分母为零时，相应条件指标返回 `null`。Window 清空后不会继承上一窗口的长草稿位置计数。Cumulative 按实际 Token 数加权。概率比策略下，Slim 接受率与其 Top-1 一致率分别统计。

诊断时先比较 `per_position[0]` 与后续位置：首位置也差，优先检查层集合和计算路径；首位置接近、后续变差，再检查块内 Slim KV 引入的偏差。不要只通过降低阈值提高接受率，还需同时观察 Target 拒绝边界处的 Slim 接受比例。

## 离线审计

准备两个互不重合的 JSON 文本列表：`calibration.json` 用于搜索，`validation.json` 仅用于最终候选评价。它们的格式均为：

```json
[
  "一段代表实际任务的完整上下文与续写文本……",
  "另一段代表实际任务的上下文与续写文本……"
]
```

工具使用文本最后 `draft_length + 1` 个 Token 作为当前块，前面作为历史；短文本使用实际剩余长度。这里的续写是固定文本，不是运行真实 Draft Model 得到的草稿，相关接受率只能用于校准诊断。为每个真实任务选择有代表性的文本；内置的两条演示文本仅用于连通性检查。工具不自动应用 Chat Template，按传入文本直接分词。

将当前层配置保存到 `current_mask.json`，例如：

```json
{"skip_layers": [18]}
```

也支持 `{"slim_verifier_eval": {...}}` 形式。该文件提供层选择；KV 来源、接受策略和阈值以命令行选项为准。

在安装匹配版本 vLLM / vLLM-Ascend、CANN 和 torch_npu 的 NPU 环境运行：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 python -m vllm_ascend.spec_decode.slim_verifier_audit \
  --model /path/to/Qwen3-8B \
  --prompts-json calibration.json \
  --validation-prompts-json validation.json \
  --mask-json current_mask.json \
  --draft-length 15 \
  --compare-kv-refresh \
  --output slim_audit.json
```

默认依次评价全层基线、每个整层删除、每个 Attention 分支删除、每个 MLP 分支删除，再评价指定集合。每个候选对照独立的“显式残差相加 + FP32 分组 Attention”参考计算。全层基线还对照完整历史前向的 Logits；基线不通过则停止。可用 `--skip-single-ablations` 仅检查指定集合。

`--kv-source replay` 保留旧版完整历史重算作为对照；现在默认是 `target`，不会再把旧模式的层排名误当成新版排名。

## 层组合搜索

搜索默认保护首尾层，以 Token 加权的 KL(Target ∥ Slim) 为目标，每次重新评价整个候选组合。`beam-width=1` 是逐步贪心搜索，更大的值会保留多个候选继续扩展。它是预算受限的启发式搜索，不是 DIMR 的贝叶斯实现，也不保证全局最优。

例如搜索删除 6 个整层的组合：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 python -m vllm_ascend.spec_decode.slim_verifier_audit \
  --model /path/to/Qwen3-8B \
  --prompts-json calibration.json \
  --validation-prompts-json validation.json \
  --search-scope layer --search-skip-count 6 \
  --beam-width 2 --max-evaluations 512 \
  --skip-single-ablations --compare-kv-refresh \
  --output slim_search.json
```

`search-scope` 可以是 `layer`、`attention`、`mlp` 或 `sublayer`；最后一种允许在 Attention/MLP 两类分支中联合选择。`search-skip-count` 的单位随之变化：6 个整层对应 12 个分支，但相同分支数量不代表相同 FLOPs 或延迟。若要放开首尾层，显式添加 `--allow-edge-skips`。

验证集不参与搜索决策。所有 TP Rank 在每个候选后同步评价结果，保证后续 Forward 和 LM Head 集合通信顺序一致。

报告中的 `search.completed=false` 表示预算内没有达到要求的删除数量，此时不会生成 `search_best` 配置。达到数量也不表示搜索完所有组合；检查 `budget_exhausted` 和 `optimality`。完整候选通过数值参考检查后，才会提供 `candidate_additional_config`。它仍只是候选：需查看独立验证集上的分布、条件接受率及真实 NPU 性能后再选用。

## 将候选配置接入现有脚本

先查看 `selected_masks.search_best.validation.metrics`，重点比较 KL、`slim_rejection_rate_in_target_prefix` 和 `target_prefix_preservation_rate`；前两项越小、后一项越大，说明分布和有效前缀更接近 Target。同时观察 `slim_acceptance_at_target_rejection_rate`，避免只靠放宽阈值提高接受率。固定文本上的诊断结果仍需用真实草稿复测。

确认要试用该候选后，在现有 `test.py` 的 `llm = LLM(**llm_kwargs)` 之前读取报告：

```python
import json

with open("slim_search.json", encoding="utf-8") as handle:
    report = json.load(handle)
if not report.get("search", {}).get("completed", False):
    raise RuntimeError("Search did not reach the requested skip count")
selected = report["selected_masks"]["search_best"]
candidate = selected["candidate_additional_config"]
if candidate is None:
    raise RuntimeError("Candidate failed numerical reference checks")

additional_config = dict(llm_kwargs.get("additional_config") or {})
additional_config["slim_verifier_eval"] = {
    **candidate["slim_verifier_eval"],
    "log_interval": 100,
    "top_k": 5,
    "profile_latency": False,
}
llm_kwargs["additional_config"] = additional_config
llm = LLM(**llm_kwargs)
```

保留原有草稿模型、`speculative_config` 和其他推理参数，启动时使用 `VLLM_USE_V2_MODEL_RUNNER=1 python test.py`。上面的代码替换整个 `slim_verifier_eval` 配置，避免旧 `skip_layers` 与新 `keep_layers` 同时存在；其他 `additional_config` 项会保留。加载后检查启动日志中的 `kv_source`、保留层和子层跳过列表，再查看 `[SlimVerifier] Window/Cumulative` 的新指标。

如果暂时只诊断当前实现，原有层选择可直接继续使用，新增位置和前缀指标自动输出。仅需明确设置 `kv_source="target"`；使用概率比策略时设置 `acceptance_strategy="probability_ratio"` 和对应阈值。

报告文件不会覆盖已有文件；重复运行请更换 `--output` 文件名，或省略该参数自动生成带时间戳的文件名。

## 逐位置刷新 Target KV 对照

`--compare-kv-refresh` 对指定集合和搜索最终候选执行额外诊断：每个预测位置都重新使用该位置之前的全层 Target KV，再用 Slim 计算单个当前输入 Token。它使用已经计算好的全层历史，只用于隔离误差来源，不会启用到线上，也不是可直接替代论文流程的加速方案。

比较同一数据集上的普通 `target` 结果与 `target_kv_refreshed_each_position`：

- 若刷新后明显接近 Target，说明普通块内由 Slim 产生的历史 K/V 是主要差异来源之一。
- 若首位置及刷新后仍明显偏离，优先调整层集合、跳层粒度或检查数值实现。
- 在相同前缀下，两种计算的第一个位置应数值接近；测试覆盖了这个不变量。

有验证集时，对照默认在验证集执行，报告用 `refresh_comparison_dataset` 标明，不能拿它与校准集的平均数交叉比较。

## 报告与边界

- `full_layer_control`：全层基线及参考一致性检查。
- `validation_full_layer_control`：独立验证集的全层基线，有验证集时同样必须通过。
- `single_ablation_effects_least_kl_first`：三类单项消融，按 KL 从小到大排列。
- `selected_masks.configured` / `selected_masks.search_best`：指定集合和搜索候选，含校准、验证及可选刷新对照。
- `reference_mismatch_count`：数值参考不一致的候选评价次数。
- `held_out_evaluated`：是否执行了独立验证集；不表示质量已达标。

为避免无界保存历史激活，默认每段截断到 256 Token，校准集与验证集的总 Token 预算为 4096。可分别通过 `--max-model-len`、`--max-total-tokens` 显式调整。长上下文会增加全层预计算、显式参考 Attention 和保存 KV 的内存开销；搜索评价次数只约束搜索本身，不包含单项消融和最终验证。

离线全层 KV 由原始权重、原始 Q/K Norm、RoPE 和 SDPA 前向产生，语义与 Target 历史 KV 对齐，但不经过线上 Paged Attention 后端。实际 NPU 缓存读取、TP、Eager/Graph 以及输出不变性仍由 `tests/manual/test_slim_verifier_npu.py` 验证。当前 Windows CPU 环境无法运行真实 Qwen3-8B 校准或 NPU 测试；不能将随机小模型数值测试当作接受率改善的实测证据。

CPU 回归覆盖子层跳过与 Hugging Face Qwen3 的 FP32/BF16 对照、有效前缀和位置统计、组合相互作用、预算耗尽、验证集隔离、逐位置 KV 刷新以及非主 TP Rank 的搜索调用顺序。模拟 TP 测试只验证调用协议，不能替代真实分布式设备测试。JSON 报告使用 `schema_version: 2`，旧版的 `vs_full_replay` 排名已替换为明确标注 KV 模式的三类消融结果。
