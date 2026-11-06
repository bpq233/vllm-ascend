# Target-Aligned Proposal Sampling

这是 MRV2 层次化投机解码的显式可选模式。默认关闭，不改变已有 Top-k、Prob-ratio、全接受或普通投机解码路径。

当前实现真实 greedy DFlash 提议、严格轮数的整块中间校验、逐位置混合采样及最终标准拒绝采样。正确性目标是保持经过请求采样参数处理后的 Target 概率分布，允许 FP32 数值误差；不是与普通 Target 使用同一 seed 时逐 token 轨迹完全一致，也不保证提升吞吐量。

## 阶段 1：实际调用链与概率边界

现有 `MultiStageDFlashSpeculator` 将 Primary DFlash 接到独立的 intermediate verifier/DFlash pair；不启用 Primary 时，`IntermediateOnlySpeculator` 直接使用独立 pair。候选经原 MRV2 draft 缓冲区和 scheduler 交给下一次 Target forward，提交、EOS、长度和 KV 回退继续使用原流程。

已有近似路径的 `IntermediateDecisionRunner` 只返回接受长度和 argmax 修正，不保留真实条件提议分布。它不能直接用于无损混合采样。新模式单独使用 `ProposalPipeline` 和 `ProposalVerificationSampler`，不会把近似策略改名为标准拒绝采样。

实际 greedy DFlash 的采样器走 argmax，因此其真实条件提议分布是确定性输出 `d` 的 delta：

```text
qD(v | h, proposal_state) = 1[v = d]
qMix(v | h, proposal_state) = (1 - lambda) * pM(v | h) + lambda * qD(v | h, proposal_state)
```

`proposal_state` 包括当前有效并行块及其来源。DFlash 原始 logits 的 softmax 不是 greedy 采样过程的分布，不能拿来代替 `qD`。本版要求 `draft_sample_method="greedy"`，没有接通 stochastic DFlash 的真实并行联合/条件概率；配置为 probabilistic 时会明确拒绝。

`IntermediateBackend.proposal_block_logits(prefixes, drafts, req_ids)` 每轮通过一次整块 teacher-forced 校验取得各草稿位置的设备 logits，并在图输出复用前保存独立存储。只有首次拒绝之前、包含修正位置的行会被消费；这些行对应实际生成前缀。拒绝之后的旧尾部 logits 一律丢弃，下一轮按修正后的前缀复用共同 KV 并重算不同后缀。`intermediate_probabilities` 将 temperature、Top-k、Top-p 和 min-p 应用于 FP32 logits，再归一化为 `pM`。

最终 Target 通过原 sampler 的请求参数处理入口得到当前 prefix 下的 `pT`。`ProposalSamplingContext` 保存每个位置实际使用的 `pM`、`qMix`、delta token、lambda、来源、context 和位置，最终验证读取这一 ledger，不根据旧 draft logits 重建概率。

## 阶段 2：固定混合权重

中间阶段从实际 `qD` 得到 token `x`，按 `min(1, qMix(x)/qD(x))` 接受；拒绝时从归一化的 `max(qMix-qD, 0)` 采样修正。lambda 小于 1 时，该残差与标准 `pM/qD` 拒绝采样的残差一致。lambda 为 1 时，delta DFlash 提议全部保留。

最终阶段逐位置按 `min(1, pT(x)/qMix(x))` 接受；首次拒绝从归一化的 `max(pT-qMix, 0)` 采样修正并终止该次验证。全部接受时追加一个从 `pT` 采样的 bonus token。所有比例、残差、归一化及类别抽样在 FP32 中执行，不插入 epsilon、均匀回退或概率地板来掩盖无效状态。

lambda 为 0 是“标准 Intermediate 提议分布 + Target 标准拒绝采样”基线；lambda 为 0.25、0.5、0.75 是固定混合实验；lambda 为 1 是实际 greedy DFlash delta 提议基线。它们使用相同正温度请求参数时可以比较，不能与 temperature=0 的严格贪婪速度直接混为同一组。

## 阶段 3：多轮、来源及 KV

`intermediate.num_rounds=N` 严格限制中间校验轮数。一轮处理一个 DFlash 草稿块：整块计算 `pM`，按顺序执行混合拒绝采样，首次拒绝时保留残差修正 token、丢弃剩余尾部并结束该轮；全部接受时也结束该轮。只有还有下一轮时才重新提案，不在同一轮内补齐候选。Primary 首块计入第 1 轮；独立三模型模式的第 1 轮使用 Intermediate DFlash。

一个位置只生成一次实际 `qMix` 样本，不在后续 intermediate 轮次中重新筛选、重验或改写成另一种提议分布。没有未记录分布的中间 bonus：块耗尽后重新取得有效提议，每个追加位置仍记录实际混合分布。最终 Target bonus 单独服从 `pT`。

候选容量为 `Primary 首块宽度 + (N-1) * Intermediate 草稿宽度`，不预留中间 bonus。`intermediate.max_generated_tokens` 是累计候选硬上限：配置为 100 时最多提交 100 个候选，包含中间修正 token，不包含最终 Target bonus；不要求填满 100，也不会为填满它增加轮数。达到上限时在当前块内截断剩余预算并结束，不允许整轮结束后超出 100。

因此，两个草稿宽度均为 15、N=3、`max_generated_tokens=100` 时，正常执行 3 轮，实际候选为 3 至 45 个，取决于每轮首次拒绝位置；最终 Target 全部接受后另加一个 bonus。N=10、同样宽度时最多 100 个候选；全部中间接受时第 7 轮生成剩余 10 个后结束。EOS、请求/上下文长度上限或 drafter 无可用输出时也允许提前结束。

可变候选长度不等于有偏：每个已抽取的修正 token 都保留，是否到达下一位置只取决于此前生成结果；当前位置的混合权重及分布在该位置抽样前确定，不用未来 token 选择当前输出。最终 Target 对每个实际到达的位置使用记录的条件 `qMix`，没有后续候选时从 `pT` 生成 bonus。小词表测试跨多个 Target cycle 精确枚举前 3 个输出的联合分布，验证这种停止规则。

EOS 或 DFlash 无法提案的上下文边界可以结束候选前缀。无可用 DFlash token 时，不伪造 `qD` 或静默切换到近似验收，而将后续生成留给 Target。候选 context、anchor、位置或 token 与 ledger 不一致时直接报错。请求结束或 ID 重用时清理状态；抢占时清理失效候选，保留随机流进度供恢复使用。不同请求使用独立 RNG 状态，Target 和 intermediate 也使用不同随机流。

## 阶段 4：只使用历史 Target 反馈

`ProposalFeedback` 默认维护 lambda 集合 `0.0, 0.1, ..., 1.0`，按 intermediate/draft 熵、margin、分布差异、来源、轮次和位置分桶。只在 Target 完成验证后，对 `1-TV(pT,qMix(lambda))` 更新 EMA；样本不足时使用默认 `mixture_lambda`。

一次 proposal cycle 开始时冻结反馈快照，整次生成只读该快照。当前尚未完成的 Target 结果不会反过来决定当前 token 的权重，更新也不修改已记录的提议概率。

`fixed` 固定权重，`adaptive` 读取历史反馈。`feedback_enabled=false` 可关闭固定权重实验的学习开销；adaptive 要求该字段为 true。`ema_decay` 默认 0.9，`min_observations` 默认 8，均可显式配置。

Oracle 仅用于离线诊断：`proposal_feedback.offline_oracle_lambda(overlaps, candidates)` 在已经取得 Target 结果后选择重叠最大的权重，估计可改进空间。在线配置 `mode="oracle"` 会报错。CPU benchmark 记录事后 Oracle；模型 benchmark 在日志含 `lambda_candidates` 与 `lambda_candidate_overlaps` 标量网格时，也事后计算 Oracle 重叠、改进空间和权重直方图，否则保留 null。两者都不使用 Oracle 决定当前在线 proposal。

## 启用方式

以下示例启用四模型链路、固定 lambda=0.5 和同步调度；省略 `final_verification`、`intermediate.verification` 时自动选择新标准采样路径，也可以将二者显式设为 `{"method": "standard"}`。不能同时配置 Top-k 或 Prob-ratio 策略。

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/models/Qwen3-8B",
    tensor_parallel_size=1,
    enforce_eager=True,
    async_scheduling=False,
    max_model_len=4096,
    max_num_batched_tokens=4096,
    max_num_seqs=2,
    speculative_config={
        "model": "/models/Qwen3-8B-DFlash",
        "method": "dflash",
        "num_speculative_tokens": 4,
        "draft_sample_method": "greedy",
        "rejection_sample_method": "standard",
    },
    additional_config={
        "multi_stage_speculative": {
            "use_primary_drafter": True,
            "primary_num_speculative_tokens": 4,
            "intermediate": {
                "verifier": {"model": "/models/Qwen3-4B"},
                "drafter": {"model": "/models/Qwen3-4B-DFlash"},
                "num_rounds": 6,
                "num_speculative_tokens": 4,
                "max_generated_tokens": 100,
                "max_num_seqs": 2,
                "max_model_len": 4096,
            },
            "proposal_sampling": {
                "enabled": True,
                "mode": "fixed",
                "mixture_lambda": 0.5,
                "feedback_enabled": True,
                "debug": False,
            },
        }
    },
)
outputs = llm.generate(
    ["Explain speculative decoding."],
    SamplingParams(temperature=0.8, top_p=1.0, top_k=-1, seed=17, max_tokens=128),
)
```

将 `mixture_lambda` 改为 0、0.25、0.75 或 1 即可运行固定权重对照；改为 `mode="adaptive"` 则使用历史反馈，`mixture_lambda` 仍是冷桶默认值。独立三模型模式将 `use_primary_drafter` 设为 false，并移除 `primary_num_speculative_tokens`。

新模式要求请求 `temperature > 0`；temperature=0 会明确报错。严格贪婪对照应关闭新模式，在原路径使用最终 Top-1 验证。不要将非 Top-1 接受称为与 Target 贪婪完全一致。

## 测试与实验

从仓库根目录运行 CPU 测试；需要 CPU PyTorch，但不需要完整 vLLM 或 NPU 运行环境：

```bash
python -m pytest --noconftest -q \
  tests/ut/worker/v2/test_proposal_math.py \
  tests/ut/worker/v2/test_proposal_backend.py \
  tests/ut/worker/v2/test_proposal_config.py \
  tests/ut/worker/v2/test_proposal_feedback.py \
  tests/ut/worker/v2/test_proposal_runtime.py \
  tests/ut/worker/v2/test_proposal_round_law.py
```

运行可复现的三 token 条件模型实验，包含小词表原始 Target、近似基线、固定 lambda 和 adaptive。它只报告 CPU toy workload 的耗时、接受长度、输出频率和已声明范围的 tensor 内存，不估计真实 NPU 加速比：

```bash
python benchmarks/proposal_sampling_cpu.py --tokens 2000 --capacity 4 --seed 17 --output proposal_cpu.json
```

模型实验需要已安装匹配的 vLLM/CANN/torch_npu 和对应模型权重。在 NPU 机器执行：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 python examples/offline_inference/target_aligned_proposal_benchmark.py \
  --model /models/Qwen3-8B \
  --primary-drafter /models/Qwen3-8B-DFlash \
  --intermediate-model /models/Qwen3-4B \
  --intermediate-drafter /models/Qwen3-4B-DFlash \
  --use-primary-drafter --tp 1 \
  --temperature 0.8 --top-p 1 --top-k -1 \
  --rounds 3 --draft-tokens 15 --max-generated-tokens 100 --max-num-seqs 2 --max-tokens 128 \
  --repetitions 3 --output-dir proposal_model_results
```

脚本为每组配置启动独立子进程，避免模型、KV 或图资源跨组残留。默认 eager；只有显式 `--graphs` 才使用 FULL model graphs。`--prompts` 接受 JSON 字符串列表或每行一个 prompt 的文本文件，`--log-dir` 可单独指定日志目录。

默认实验包含原始正温度 Target、严格贪婪、Top-k、Prob-ratio、标准 intermediate 提议、固定 0.25/0.5/0.75 和 adaptive。`--cases target_random,standard,lambda_05,adaptive` 可缩小范围。每组生成次数、prompt、max tokens、随机 seed、正温度 Top-k/Top-p 参数一致；严格贪婪单独分组，近似策略也标明不保证 Target 分布。

结果保存为每组 JSON、原始日志、测量窗口内的 `proposal_sampling` JSONL 和 `summary.json`。吞吐量的 wall-time 包含 prefill；只有 request metrics 提供 first/last-token 时间时才另报 decode interval。EAL 和接受率按 `target_cycle_id` 去重统计；按请求统计的验证 cycle 数不冒充实际 batched Target forward 次数。无法确认的 NPU 峰值内存、纯采样开销或 forward 次数保留 null。

中间层 EAL 按 `proposal_generated` 事件中的实际 DFlash 块数统计。该事件在生成结束时记录完整候选的中间接受数、各阶段 host wall time 及驻留概率 tensor 字节数，避免只统计 Target 接受前缀造成截断偏差。`max_logged_retained_probability_bytes` 只表示已记录的 ledger 存储，不是 NPU 峰值显存。

### 本地验证记录（2026-11-06）

Windows / Python 3.11 / CPU PyTorch 2.10 环境的测试范围包括原多级路径、FIA 输入契约、小词表精确枚举、每组 60,000 次采样频率检查、严格轮数与硬候选上限、上下文修正、Primary/Secondary 来源、多请求重排、首次注册、ID 复用、抢占恢复和日志统计。Ruff、Python 编译及仓库相关静态检查通过。完整 vLLM/NPU 启动未在该环境执行。

当前相关回归为 709 项测试及 231 个 subtest 通过；其中严格轮数的可变长度正确性由跨 Target cycle 的联合分布枚举另行覆盖。

CPU toy benchmark 使用 `--tokens 2000 --capacity 4 --seed 17`，结果如下。toy drafter 使用已知的确定性 token；它验证采样和统计机制，不代表 Qwen/DFlash 模型精度或硬件性能。

| 正温度精确模式 | Target EAL | Target 调用次数 | CPU toy token/s |
| --- | ---: | ---: | ---: |
| 原始 Target 采样 | 不适用 | 2000 | 7669 |
| lambda=0 | 2.189 | 628 | 468 |
| lambda=0.25 | 2.733 | 536 | 523 |
| lambda=0.5 | 2.598 | 557 | 502 |
| lambda=0.75 | 1.724 | 735 | 396 |
| lambda=1 | 1.129 | 941 | 328 |
| adaptive | 3.167 | 480 | 565 |

本次 toy 实验的 adaptive 提高了 EAL，但显式 Python/CPU 采样成本使其慢于原始 toy Target；该结果不能证明真实模型吞吐收益。严格贪婪与近似模式的结果单独分组保存在 benchmark JSON 中。数学分布结论以枚举和频率测试为依据，不能仅由本表的接受长度推出。

### 15-token 块、严格 3 轮、100 候选硬上限

使用 `benchmarks/proposal_sampling_hierarchical_cpu.py` 直接调用生产 `ProposalPipeline`、请求 RNG、概率 ledger 和历史反馈，模型 forward 由同一三 token 条件模型替代。toy drafter 沿自己的预测前缀构造确定性块；拒绝后由真实流水线丢弃尾部并重提案。此测试没有执行真实 DFlash 网络、Qwen 权重、KV 或 NPU 图。

```bash
python benchmarks/proposal_sampling_hierarchical_cpu.py \
  --primary-tokens 15 --draft-tokens 15 --rounds 3 --max-generated-tokens 100 \
  --tokens 2000 --seed 17 --output hierarchical_strict_cap100_p15_d15_r3_seed17.json
```

配置派生容量为 `min(100, 15 + (3-1)*15) = 45`。临近输出长度上限时缩短候选并预留 Target 修正/bonus 位置。所有模式开启反馈统计，`min_observations=4`、temperature=1，debug 关闭。严格轮数会在每轮首次修正后转入下一轮，实际候选长度由中间接受结果决定；此前补齐 48 个候选的实验属于已替换的旧实现，不能作为当前性能结果。

| 模式 | Target EAL | Target 接受率 | Target 调用次数 | CPU toy token/s |
| --- | ---: | ---: | ---: | ---: |
| lambda=0 | 2.300 | 42.56% | 606 | 334.3 |
| lambda=0.25 | 3.444 | 48.65% | 450 | 351.1 |
| lambda=0.5 | 3.274 | 31.35% | 468 | 274.2 |
| lambda=0.75 | 2.231 | 11.59% | 619 | 123.9 |
| lambda=1 | 1.375 | 3.08% | 842 | 42.4 |
| adaptive | 4.362 | 58.89% | 373 | 432.4 |

严格轮数实验中，adaptive 平均每次生成约 7.4 个候选，接受约 4.36 个；这比补齐固定预算减少了大量无效候选生成。中间校验按块计算，CPU 逐位置采样仍有成本，不能由此推导真实模型吞吐提升。新脚本使用生产请求 RNG 和实际块失效规则，旧脚本是概率运算实验，两张表不构成只改变候选长度的严格对照。

严格轮数回归覆盖 N=1/2/3 的全部接受和连续拒绝，检查来源、概率与最终 bonus；候选硬上限测试覆盖 100 上限不生效、达到 100 时截断、上限小于一块和请求剩余长度先耗尽。另有 14 项精确枚举测试验证跨多个 Target cycle 的三 token 联合分布。

## 日志、性能边界与限制

`proposal_sampling.debug=true` 产生独立 `proposal_sampling` JSON 日志，包括 request、轮次、候选位置、来源、lambda、原始 draft 与实际 proposal token、两级接受情况、标量概率、TV 距离和时间字段。`event="target_verified"` 与 `target_cycle_id` 标识完成的 Target 验证，可可靠区分反馈与提案阶段。时间是带同步/等待的 host wall time，不能当成纯 NPU kernel 时间；缺失字段为 null，不用假数字填充。

词表概率及残差运算保留在设备，回传 CPU 的是 token、接受决定、特征和 TV/overlap 标量包。CPU 为每个请求维护独立随机流并将小型 uniform 包传到设备；轮次控制仍在 CPU。本版不是全设备控制循环，也不把整个采样流程捕获成一张 NPU 图。模型 forward 可继续使用既有 graph manager，但逐 token 概率、验收和控制在模型图外，需要真机验证图生命周期和算子支持。

每个逻辑轮次只请求一次整块中间校验，不再为每个 token 单独调用 `proposal_logits`。冷上下文填充、KV 重建或图预热的额外计算不属于新增投机轮次；逐位置采样与特征回传仍有同步开销。ledger 保留每个有效位置的 `pM` 与 `qMix`，FP32 存储量约为 `2 * 请求数 * 候选数 * 词表大小 * 4` 字节，不含整块 logits 临时值、KV 和其他状态。不要只凭接受长度上升宣称吞吐提升。

固定权重实验同时关闭 `feedback_enabled` 和 `debug` 时，跳过不再使用的熵、margin、TV 特征计算及对应标量同步；新模式也不构造或预热旧的近似验收 decision graph。开启反馈或日志时恢复这些统计。

当前显式拒绝 context-dependent penalty、bad words、logit bias、allowed-token 限制、structured output、自定义 logits processor、min_tokens、额外 thinking-budget 参数等尚未对齐的请求处理。支持范围不包括 stochastic DFlash、DFlash2/LiLiCorr、PP/DP/CP 大于 1、LoRA、多模态、LM-head TP、KV 传输/PD、动态投机长度、非标准最终拒绝采样、block/entropy 验收、watermark 或 trace replay。非法组合不会静默退化为近似路径。

## 分阶段实现位置

| 阶段 | 文件与核心入口 | 与 MRV2 的关系 |
| --- | --- | --- |
| 概率与上下文分析 | `backend.py::proposal_block_logits` | 复用原 context KV 与整块微批 forward，仅消费首次拒绝前含修正位置的预测行 |
| 固定权重与标准验收 | `proposal_math.py`、`proposal_sampling.py::intermediate_probabilities` | 概率空间混合、真实类别抽样、残差和 provenance，不改模型权重 |
| 多轮集成 | `proposal_pipeline.py::ProposalPipeline`、`adapter.py`、`proposal_verification.py::ProposalVerificationSampler` | opt-in 接到原 draft 发布和 Target rejection sampler 接口 |
| 历史反馈 | `proposal_feedback.py::ProposalFeedback`、`proposal_sampling.py::target_feedback` | Target 完成后更新 EMA，生成时冻结快照，Oracle 离线使用 |
| 实验与验证 | `tests/ut/worker/v2/test_proposal_*.py`、CPU 概率/流水线 benchmark 与模型 benchmark | 枚举/频率与生命周期验证和可运行实验；NPU 性能须另行测量 |

本地 CPU 测试不能证明真实 Qwen 权重、NPU 图、跨流内存和吞吐已通过。应在匹配的模型与设备上比较同温度的 Target 输出分布、EAL、实际提交数、延迟、吞吐、图资源及峰值内存，保留软件版本和完整参数。
