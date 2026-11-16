# VIA-SD 第一阶段：Slim Verifier 影子校验

本功能从当前 Target 的原始权重构建减层校验路径，对已经执行的同一批 Draft Token 进行额外的影子诊断。Slim 支持 Top-1 匹配或概率比阈值策略，Target 诊断仍使用 Top-1 匹配。结果只进入统计模块，不回传给采样器、调度器或 Draft Model。不实现分级接受、实际采样接受修正、重新生成、Target 回退或 DIMR 搜索。

## 支持范围

- Model Runner V2，标准稠密文本 `Qwen2ForCausalLM` / `Qwen3ForCausalLM`。
- 任意实际层数和任意合法保留层集合；不限 Target/Draft 的模型规模及组合。
- 普通 TP 的词表并行，所有 TP Rank 按同一顺序参加模型和 LM Head 集合通信，仅 TP Rank 0 汇总和输出。
- 原 Target 可以使用 Eager 或已有 NPU Graph 配置；影子路径始终在采样后于 Graph 外执行，不改变 Target 的图配置。
- 使用现有公共 Verification Batch，不检查草稿方法或草稿模型类型。

当前会在启用时明确拒绝：MRV1、其他 Target 架构、混合/滑窗/非因果 Attention、PP/DP/PCP/DCP、Sequence Parallel、专用跨 DP 的 LM-head TP、LoRA、量化权重/量化 KV、Prompt Embeddings 和 Adaptive Verification。普通词表 TP 与这里的专用 LM-head TP 是两种不同配置。

上述支持范围是代码实现范围；**实际 NPU、TP 及 Graph 数值行为尚待真机验证**。本地 KG 健康检查超时，[未核实] 本实现未获得 KG 的交叉核验；设计依据为当前仓库与配套本地上游源码，并完成了 CPU 数值测试。模型结构也可对照 [vLLM 官方 Qwen3 源码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)，具体接口以本工作区版本为准。

## 配置与启动

默认不启用。关闭时不创建 Verifier、独立 KV 或统计状态，也不额外执行 Forward。

`slim_verifier_eval` 保留在 `VllmConfig.additional_config` 中，由 Runner 读取；`init_ascend_config()` 会在构造严格校验的 `AscendConfig` 前过滤该键。初版漏掉这一过滤，即使下方嵌套写法正确，也会在启动时出现 `VllmConfig / slim_verifier_eval / Unexpected keyword argument`。Pydantic 将内部构造错误包装为外层 `VllmConfig` 错误，因此该消息不能证明调用方把键传到了顶层。遇到此错误需同步更新 `vllm_ascend/ascend_config.py` 的过滤逻辑，无需修改上游 `VllmConfig` 接口。

```python
additional_config = {
    "slim_verifier_eval": {
        "enabled": True,
        "keep_layers": [0, 4, 8, 12],
        "log_interval": 100,
        "top_k": 5,
        "profile_latency": False,
        "acceptance_strategy": "probability_ratio",
        "probability_ratio_threshold": 0.5,
    }
}
```

`keep_layers` 使用 Target 的 0-based 编号，校验范围来自加载模型的 `len(target.model.layers)`，随后按升序执行。重复、越界、负数、布尔编号以及未知配置项均报错。也可用 `skip_layers` 指定补集；两者只能提供一个。保留全部层可写 `skip_layers: []`；`keep_layers: []` 是仅 Embedding → Final Norm → LM Head 的消融实验，三者仍全部执行。

`log_interval` 是非空 Verification Block 数量，不是请求数、Token 数或 Scheduler Step 数。`top_k` 自动取不超过词表大小的值。`profile_latency` 默认关闭；开启时使用 NPU Event，且在每个 Block 完成后同步结束 Event，适用于实验测量。

`acceptance_strategy` 默认是 `"top1"`，保持原有行为。设为 `"probability_ratio"` 时，对每个草稿 Token x，Slim 根据下式生成影子接受掩码：

```text
q(max) = 同一预测位置整个全局词表中的最大概率
accept(x) = q(x) / q(max) > probability_ratio_threshold
```

阈值可配置为 `[0, 1]` 内的有限数，默认 `0.5`（仅作为实验初值），只在概率比策略下生效。按严格 `>` 比较，等于阈值时拒绝。`q` 是原始模型 Logits 在温度 1 下的 Softmax 分布，不应用实际采样的 Temperature、Top-K/Top-P、Penalty 或 Grammar。

Softmax 归一化项会约掉，概率比等于 `exp(logit[x] - max(logits))`。实现直接比较 Logit 差与 `log(threshold)`，避免极小概率先算 Softmax 再相除造成下溢。阈值为 0 时接受正概率候选，精确零概率候选仍拒绝；阈值为 1 时全部拒绝，包括最高概率候选。比值本身可在 `last_results` 中检查，但 FP32 下极小比值仍可能显示为 0，接受判断以稳定的 log 空间比较为准。

Top-1 策略选用确定性 Argmax；概率比策略下，与最高 Logit 并列的候选比值都是 1。即使保留全部层使两模型的 Logits 一致，选择不同判定策略后，两套影子接受率仍可能不同。两种策略都只统计当前相同上下文下的候选，连续接受长度仍在第一个拒绝处停止。

启动示例（替换模型路径，按实际 Target 层数选择索引）：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/target \
  --speculative-config '{"method":"draft_model","model":"/path/to/draft","num_speculative_tokens":3}' \
  --additional-config '{"slim_verifier_eval":{"enabled":true,"keep_layers":[0,4,8,12],"log_interval":100,"top_k":5,"acceptance_strategy":"probability_ratio","probability_ratio_threshold":0.5}}'
```

原有 `speculative_config` 的配置方式保持不变。例如已有 n-gram、EAGLE 或其他提案方式时，继续使用其原有配置，只添加 `slim_verifier_eval`。日志启动时打印实际总层数、保留层、跳过层和保留比例。

## 执行路径与权重共享

公共接入点为 `vllm_ascend/worker/v2/model_runner.py::NPUModelRunner.sample()`：

1. 原 `super().sample()` 完整执行 Target 采样、接受/拒绝与 Bonus 处理，保留其原始返回对象。
2. 启用时执行 `SlimVerifierEval.evaluate()`；读取这次已执行 Batch 的 Token、位置和 Target Hidden States。
3. 返回步骤 1 的同一对象。后续原有前缀提交、KV 更新、输出与 Draft 生成不变。

`SlimVerifier` 直接持有 Target 的 Embedding、Final Norm 和 CausalLM。所选 Decoder 和 Self-Attention 仅浅拷贝 Python 模块容器，以独立替换其 Attention 叶节点。QKV/O Projection、Q/K Norm、RoPE、Residual/RMSNorm 和 MLP 都沿用原有模块与参数对象；LM Head 通过原 `target.compute_logits()` 调用。不构造新模型、不复制参数存储、不改原层列表、不替换全局 Forward。

影子 Attention 使用 PyTorch 的 `scaled_dot_product_attention` 因果计算，经当前设备分发，不含 CUDA 专用调用。其 K/V 来自减层路径当前重算的隐藏状态，既不读取也不写入 Target Attention 后端或 Paged KV。

每个有效 Block 单独处理一个无 Padding 的请求：

- 从 `req_states.all_token_ids.gpu` 取已提交前缀，再拼接当前实际执行的完整 Query。
- 前缀长度使用实际 `input_batch.positions`；不使用拒绝后仍可能偏大的 CPU computed-token 计数。
- 取 `cu_num_logits_np` 中该请求的 `n` 个 Draft 预测行，排除最后的 Bonus 行；第 `i` 行对应的 Draft 是 `input_ids[logits_indices[i] + 1]`。
- Slim 对整个历史执行一次减层 Forward，仅将这些相同的绝对预测位置送入 LM Head。

因此第 i 个 Draft 的预测条件是实际前缀和之前所有 Draft，而不是只有已经被 Target 接受的 Draft。各请求的上下文独立，Padding 不参与指标。EOS 按调度器提供的有效候选范围处理；不会用 EOS 或 Token 0 作为 Padding 哨兵额外删除候选。后续 Step 的前缀只取已提交历史，拒绝后的旧 Draft、抢占后重新分配的请求槽位及 Prefix Cache 命中不会复用 Slim 状态。

本阶段采用任务允许的“完整历史重算”方式：没有持久化 Slim Paged KV，只有独立、临时的 K/V，Block 结束后释放。未来若增加增量缓存，需要自行维护拒绝回滚和前缀同步，不能复用 Target KV。

Target 诊断 Logits 从这次正常 Target Forward 的 Hidden States 额外调用共享 LM Head 得到，避免使用已被 Grammar、Temperature 或 Penalty 原地处理的采样 Logits。普通 TP 下通过模型自己的全词表接口取得结果，并检查词表宽度，不能对局部分片 Argmax。影子调用保存并恢复 CPU/NPU RNG 状态。

## 指标定义

默认策略的日志标识为 **Top-1 Diagnostic**。概率比策略标识为 **Shadow Diagnostic**，明确打印 `Target: top1`、`Slim: probability_ratio` 和严格阈值。即使推理使用随机采样，以下指标也是影子诊断结果，不是原有概率拒绝采样的实际接受率。

设 B 为窗口/累计非空 Block 数，N 为其有效 Draft 总数。Target 接受掩码为 `argmax(target_logits_i) == draft_i`；Slim 接受掩码由所选策略产生，另外始终计算 Slim 的 Top-1 匹配掩码。各模型的 p 为其接受掩码遇到第一个拒绝前的连续长度：

| 日志字段 | 计算方式 |
| --- | --- |
| `blocks` / `draft_tokens` | B / N |
| `*_avg_accepted_length` | Σp / B |
| `*_prefix_acceptance_rate` | Σp / N |
| `*_token_match_rate` | 全部位置 Top-1 匹配数量 / N，包含首次不匹配之后的位置；Slim 此项始终是 Top-1 指标 |
| `slim_token_acceptance_rate` | 全部位置被 Slim 所选策略接受的数量 / N；与 Top-1 匹配率分开 |
| `*_full_block_acceptance_rate` | 按各自策略接受全部 Draft 的 Block 数 / B |
| `*_avg_draft_logprob` | Σ(logit[draft] − logsumexp(logits)) / N |
| `*_avg_top1_confidence` | Σexp(max(logits) − logsumexp(logits)) / N |
| `*_avg_draft_rank` | Σ(1 + 严格高于 Draft Logit 的候选数量) / N；并列取相同名次 |
| `prediction_agreement_rate` | 两模型 Top-1 ID 相同的位置数 / N |
| `decision_agreement_rate` | Target Top-1 接受掩码与 Slim 所选策略接受掩码相同的位置数 / N |
| `slim_only_rate` | Slim 策略接受、Target Top-1 拒绝的位置数 / N |
| `target_only_rate` | Target Top-1 接受、Slim 策略拒绝的位置数 / N |
| `mean_logprob_gap` | Target 平均 Draft LogProb − Slim 平均 Draft LogProb |
| `top_k_overlap` | Σ(两模型 Top-K 集合交集大小 / K) / N |
| `slim_replay_ms_per_block` | NPU Event 测得的减层历史 Forward + Slim LM Head 毫秒数 / 被计时 Block 数 |
| `slim_acceptance_strategy` | 当前 Slim 判定策略名称 |
| `probability_ratio_threshold` | 概率比策略使用的阈值，仅该策略输出 |

`*` 分别为 `target` 和 `slim`。分母为 0 的空统计只返回 Block/Token 数，不输出伪造的平均值。窗口到达 `log_interval` 时同时输出 Window 和 Cumulative，随后只清空窗口。计数和概率在设备侧累积，到日志边界才回传紧凑汇总。主 Rank 的 `last_results` 保存最近 Batch 的 Request ID、预测位置、两套预测和接受掩码，另有 `slim_top1_match_mask`、策略名称以及概率比策略下的 `slim_probability_ratios`，仅供实验检查。

没有输出 Target Verify Time、JS Divergence 或 Logit Margin，因为本阶段未测量这些量。`SlimVerifierEval.flush()` 可输出尾部不足一个窗口的数据；下面的 NPU 测试通过 Worker RPC 在实验结束时调用它。

## 测试

CPU 测试使用真实 PyTorch 张量和独立的 Attention 矩阵参考，不需要安装 vLLM/CANN：

```bash
python -m pytest --confcutdir=tests/ut/slim_verifier tests/ut/slim_verifier -q
```

覆盖：全层 Logits 一致性、真实跳层、零层消融、参数存储共享、Target KV 保护、配置校验、连续前缀和 Token 匹配区别、概率与 Rank、窗口/累计、跨请求变长/槽位映射、Padding/Bonus 排除、实际位置对齐、RNG/输入保护、非主 TP Rank 参加全部计算、全词表宽度检查，以及原采样结果对象保持不变。概率比测试还覆盖与 Softmax 比值的一致性、严格阈值边界、0/1 阈值、并列最大值、下溢、大幅平移 Logits、策略接受率与 Top-1 匹配率分离及日志标识。

真实 NPU 测试入口：

```bash
python -m pytest tests/manual/test_slim_verifier_npu.py -v -s
```

该测试使用 Qwen3-0.6B Target 和同检查点 Draft，分别测试 TP=1/2、Eager/Graph、两种 Slim 判定策略、关闭/保留全部层/只保留第 0 层；检查真实全词表 Logits 的 BF16 数值误差、真实执行层、非空周期指标和 NPU Event 时间，并比较最终生成 Token ID 序列。还包括变长 Prompt、Chunked Prefill、Prefix Cache 和第二轮请求槽位复用。需要当前仓库匹配的 vLLM、CANN、torch_npu、检查点与相应数量的 NPU。

当前 Windows 环境使用 Python 3.11.9、PyTorch 2.10.0+cpu 运行上述测试，**50 项通过**。初始化回归执行实际 `init_ascend_config()` 函数体，使用轻量依赖替身和真实 Pydantic 严格校验，覆盖配置开启/关闭以及其他未知项仍被拒绝；新策略和阈值也沿此入口验证。新增代码通过 Ruff 检查、格式检查、Python 编译检查，文档通过 PyMarkdown，发布说明通过 YAML 解析；差异通过 `git diff --check`。当前机器没有 NPU/torch_npu，未运行完整仓库的 NPU/vLLM 测试或全量 `format.sh ci`。上述真实 NPU 测试尚未执行，不能将 CPU 的简化 Decoder 合约测试当作真实模型/NPU 验证。

## 开销与限制

这是准确性实验路径，不能用于宣称 VIA-SD 加速收益。每个 Block 额外重算完整历史：保留 K 层、历史长 S 时，稠密 Attention 计算随 K·S² 增长，并执行整个历史的投影和 MLP。GQA 临时扩展 K/V 以兼容设备 SDPA；峰值内存取决于 S 和 SDPA 实现，长上下文可能显著占用显存。

还包括每个 Block 一次 Target 诊断 LM Head、一次 Slim LM Head，以及概率、全词表 Rank、Top-K 指标计算。每个 Batch 有一次实际 Query 起始位置回传；日志边界有紧凑统计回传；开启计时时每个 Block 另有 Event 同步。没有额外模型权重和持久 Slim KV，但这些临时激活未计入原 Target KV 容量规划，实验时应预留设备内存并降低 `gpu_memory_utilization`。未获得 NPU 实测数据，不报告速度或显存数值。

不支持运行期间替换 Parameter 对象的热换权重、Sleep/Wake 或并发外部调用同一个 Runner 的实验；初版不处理这些生命周期下模块视图的重新绑定。上述功能没有被加入实际推理决策。

## 文件

- `vllm_ascend/spec_decode/slim_verifier_config.py`：开关、层选择和参数校验。
- `vllm_ascend/spec_decode/slim_verifier.py`：共享权重、独立临时 K/V 的减层重算。
- `vllm_ascend/spec_decode/slim_verifier_metrics.py`：两套结果、指标与窗口日志。
- `vllm_ascend/spec_decode/slim_verifier_eval.py`：MRV2 Batch/前缀/位置对齐和影子调用。
- `vllm_ascend/worker/v2/model_runner.py`：加载时创建与采样后调用。
- `vllm_ascend/worker/v1/model_runner.py`：启用时明确报 MRV2 要求。
- `tests/ut/slim_verifier/`：独立 CPU 回归。
- `tests/manual/test_slim_verifier_npu.py`：真实模型/NPU 回归入口。
