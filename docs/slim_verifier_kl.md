# Draft / Slim / Target 三方 KL 调试

启用 Slim Verifier 后，线上默认 `metrics_mode="kl"`，仅输出三组 KL 的窗口值和累计值。原来的接受长度、Top-K、Rank 和分位置接受率日志可通过 `metrics_mode="acceptance"` 恢复。原有层选择、Target 历史 KV 和实际投机采样逻辑不变。

## 使用

在现有 `LLM(**llm_kwargs)` 之前设置：

```python
config = llm_kwargs["additional_config"]["slim_verifier_eval"]
config["enabled"] = True
config["metrics_mode"] = "kl"
config["log_interval"] = 1  # 每个有效草稿块输出；调试后可改回 100
```

保留现有 `keep_layers` / `skip_layers` / Attention、MLP 跳过配置和 `speculative_config`。运行：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 python test.py
```

输出字段：

| 字段 | 方向 |
| --- | --- |
| `kl_draft_to_slim` | KL(Draft ∥ Slim) |
| `kl_draft_to_target` | KL(Draft ∥ Target) |
| `kl_slim_to_target` | KL(Slim ∥ Target) |

例如日志形态如下，数值仅为格式示例：

```text
[SlimVerifier] KL Window: {'blocks': 1, 'draft_tokens': 15, 'kl_draft_to_slim': 0.2, 'kl_draft_to_target': 0.3, 'kl_slim_to_target': 0.1, 'distribution_basis': 'raw_logits_temperature_1', 'draft_distribution_source': 'captured_proposal_logits'}
[SlimVerifier] KL Cumulative: {...}
```

## 计算口径

三者都先将原始 Logits 转成 FP32，并在完整词表上计算温度 1 的 Log Softmax。每个位置计算 `KL(P || Q) = sum_v P(v) * (log P(v) - log Q(v))`，再按有效草稿 Token 数平均，单位为自然对数的 nat。不会除以词表大小；窗口和累计都按 Token 加权，不是先对不同长度的 Block 求平均。

只统计实际校验的草稿位置，排除 Bonus 和 Padding；首次拒绝之后、仍在该次校验块内的草稿位置也参与。KL 有方向性，第三项是 Slim ∥ Target，与旧接受率诊断中的 `mean_kl_target_to_slim` 方向相反。这里的“loss”仅用于观察，不会进行反向传播或训练。

`acceptance_strategy`、`probability_ratio_threshold`、`top_k`、`profile_latency` 在 KL 模式下不参与指标计算。实际采样的 Temperature、Top-P、Penalty、Grammar 等不会应用到这三组诊断分布中，也不会被调试配置修改。

## 草稿分布来源和对齐

草稿生成经过 `sample_draft → compute_draft_logits` 时，观察器在采样处理前将完整原始 Logits 写入独立缓存。下一轮校验按请求状态槽、草稿位置和实际生成 Token 对齐读取。不会将原采样器的 `draft_logits=None` 改成概率采样缓存，因此贪心和随机生成路径保持原样；也不会用 One-hot、已选 Token 的单个概率或 Top-K 近似完整分布。

重新生成草稿前只清理本次请求槽的记录，避免请求重排、拒绝回滚或槽复用导致串样本。日志发布前检查 Token 和位置匹配；缺失或过期记录会报错，不会输出伪造 KL。检查发生在日志窗口边界，初次调试建议 `log_interval=1`。

Draft 一侧是实际生成提案时的分布。对自回归草稿，它对应逐步生成的上下文；对 DFlash 等并行草稿，它是该方法自身的提案分布，不能解释为把已生成草稿逐 Token 回放给 Draft 后得到的条件分布。Target / Slim 则评价当前已执行的同一组草稿位置。

## 支持范围与开销

沿用 Slim Verifier 的 MRV2、Qwen2/Qwen3 Target、普通 TP 等限制。神经草稿器必须提供上述完整 Logits 采样接口；该接口覆盖当前上游自回归草稿、EAGLE/MTP 和 DFlash 的公共采样路径。只提供 Token 的 n-gram 没有可计算的草稿模型分布；绕过公共接口的自定义草稿路径也不能直接使用此模式。`use_local_argmax_reduction` 必须为 `False`，因为该优化不生成完整词表 Logits。完整词表宽度和 Token ID 空间必须与 Target 对齐。

只在统计用 TP Rank 0 分配私有 FP32 Logits 缓存，大小约为 `(max_num_seqs + 1) × num_speculative_tokens × vocab_size × 4` 字节；额外一行用于隔离图 Padding。启动日志打印该缓存的 MiB。该记录在模型加载后、图捕获前安装，缓存写入随草稿图回放执行。其他 Rank 保持原有模型集合通信顺序。

此外需要原始 Logits 拷贝和三组全词表 KL 计算。调试时可降低 `max_num_seqs`，并在短实验结束后调用现有 Worker 的 `slim_verifier_eval.flush()` 输出最后不足窗口的部分。关闭 `enabled` 时不安装观察器、不分配该缓存。

离线选层工具只加载 Target，无法获得真实 Draft 分布；它仍使用原有 Target/Slim 审计指标。本次三方 KL 调试需要运行现有的在线或离线 `LLM.generate` 投机生成脚本。

CPU 回归覆盖 KL 方向和 Token 加权、BF16/FP32、零概率和无穷 KL、草稿原始 Logits 捕获、贪心/随机输出及 RNG 不变性、请求重排/变长块/Padding/槽复用，以及 TP 非统计 Rank 的调用顺序。真实 NPU 的 Eager/Graph、TP=1/2 回归入口为：

```bash
pytest tests/manual/test_slim_verifier_npu.py -k three_way_kl -v -s
```

当前 Windows 环境没有 NPU/torch_npu，未执行上述真机回归。可先用全层 `skip_layers=[]` 检查 `kl_slim_to_target` 接近零，以及 `kl_draft_to_slim` 与 `kl_draft_to_target` 接近，再切回减层配置。
