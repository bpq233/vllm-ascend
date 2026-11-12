# DFlash 校验反馈 MVP

本实验不训练、不改模型权重、不增加 Target Forward。它只在完成严格校验后学习全局 Token Bias，下一轮并行 Draft 的 argmax 使用历史 Bias。默认关闭。

## 本地调用链与位置约定

当前分支包含 V1、V2 两套路径，平台配置默认走 V2；运行脚本显式设置 `VLLM_USE_V2_MODEL_RUNNER=1`。
修改前检查了本地插件及 `../../.sparse-reference/vllm` 中的 vLLM 源码参考。
本机未安装 vLLM，因此该参考与部署机器安装版本的一致性仍待确认。
KG 健康检查 TLS 握手超时，未取得远端资料；生态兼容性结论为 **[未核实]**，此处调用链来自本地源码。

1. `AscendDFlashSpeculator → LmheadTPDraftSamplingMixin → DFlashSpeculator → DraftModelSpeculator`。
2. `propose → prepare_dflash_inputs → _generate_draft → sample_draft`。
3. `_generate_draft` 抽取 `sample_indices` 指向的 mask hidden states；其传入 sampling 的位置是 `sample_pos - 1`。
4. 原始 logits 来自 `model.compute_logits`，Greedy 路径默认不缓存完整 draft logits，直接 `argmax(-1)`。
5. 新 `sample_draft` 在实验模式缓存原始 logits，在 argmax 前加历史 Bias。`off` 完全委托原父类。
6. Target `NPUModelRunner.sample` 中的实验分支只计算一次 sample logits，仍调用原 sampler/rejection sampler，之后才更新反馈。
7. Target 的 row j 预测 `positions[logits_indices[j]] + 1`，与输入的 row j+1 候选比较。最后一行预测 Bonus，不作 Draft 反馈。
8. DFlash 的 query row 0 是上一轮确认的 Anchor/Bonus，mask 1..N 才是候选。缓存按 Request ID 和绝对位置匹配，同时检查实际输入候选 ID。
9. 原 ACL Graph 捕获 `_generate_draft`，包含 Draft Forward、LM head 和采样。实验强制 Target/Draft Eager，避免捕获 CPU 状态和陈旧 Bias。

KV Cache、Anchor 准备、共享 embedding/LM head、Attention 和校验器均沿用原代码。
实验模式仅支持普通 DFlash、单 rank、同步调度、相同词表、Greedy、standard 校验；不支持 DFlash2、分层变体、自适应校验、local argmax 或 watermark。
关闭时上述功能继续使用原有实现。非默认 logits processor/penalty 不在实验支持范围；raw argmax 与严格校验不一致时会报错。

## 配置

所有校准参数通过 `additional_config["dflash_calib_config"]` 配置，由 `AscendConfig` 的嵌套 schema 校验。
省略该字典时默认 `off`；不再读取任何 `DFLASH_CALIB_*` 环境变量。
可以直接集成到自己的运行脚本，不依赖本目录的 benchmark 脚本或 worker extension。

```python
# 加入你现有的 additional_config 字典，其他 Ascend 配置项可以同时保留。
additional_config["dflash_calib_config"] = {
    "mode": "calibrate",  # off / observe / calibrate
    "top_k": 16,
    "alpha": 0.1,
    "lambda": 0.3,
    "clip_value": 2.0,
    "min_observations": 5,
    "output": "/path/to/results/dflash_feedback",
}
# 构造 LLM / EngineArgs 时传入 additional_config=additional_config。
```

运行时仍需遵守 MVP 的 Eager、同步调度和 Greedy 限制：Target 设置 `enforce_eager=True`，
`speculative_config` 设置 `enforce_eager=True`，`async_scheduling=False`，采样 `temperature=0`。
`additional_config` 是 vLLM 的实际参数名；这些运行时开关继续使用其原有配置位置。

| 字段 | 默认值 | 范围/意义 |
| --- | --- | --- |
| `mode` | `off` | `off` / `observe` / `calibrate` |
| `top_k` | 16 | 正整数，运行时截到词表大小 |
| `alpha` | 0.1 | `(0,1]` |
| `lambda` | 0.3 | 非负有限值 |
| `clip_value` | 2.0 | 非负有限值 |
| `min_observations` | 5 | 正整数，Token 观测门槛 |
| `output` | `dflash_feedback` | 文件前缀，追加 PID 和 `.jsonl` |

Top-K 并集去重后更新 EMA；log-prob 使用完整词表 logsumexp 归一化。
未观察/样本不足的 Token 不修改。Bias 表在一个 worker 的一次 benchmark 生命周期中共享，Request 的 pending 缓存分别管理；每次独立进程运行初始化新表。
只反馈连续接受位置和首个拒绝位置；完全接受时只反馈 N 个候选，不反馈 Bonus。
Target Token 在 Draft 中的排名使用完整设备端 logits 计算，概率偏差记录为 Top-K 并集上的部分 L1，不能解释为全分布 KL。

## 运行

以下脚本仅为可选实验工具；自己的脚本只需要传入上述配置即可启用反馈和自动 JSONL 记录。
端到端耗时和 Target Forward 计数等 benchmark 指标由自己的脚本统计，或使用这里的可选工具。
在已有可用的 Linux Ascend/vLLM 环境及本插件源码目录运行。模型路径由命令行传入。
先安装实验工具依赖：`pip install datasets matplotlib`。这些依赖不加入生产包。

```bash
python examples/dflash_feedback/prepare_prompts.py --output prompts.jsonl --per-dataset 120

# 功能检查：预热 20 条 + 正式 40 条，交错后的正式样本为每数据集 20 条。
python examples/dflash_feedback/compare.py \
  --model /path/to/Qwen3-8B --draft /path/to/Qwen3-8B-DFlash-b16 \
  --prompts prompts.jsonl --output results/smoke --limit 40 --warmup 20

# 正式评估 200 条，依次测试 batch=1、4；可指定修改前副本进行基线测试。
python examples/dflash_feedback/compare.py \
  --model /path/to/Qwen3-8B --draft /path/to/Qwen3-8B-DFlash-b16 \
  --prompts prompts.jsonl --output results/full --limit 200 --warmup 20 \
  --baseline-root /path/to/untouched-vllm-ascend --batch-sizes 1 4

# 单独运行一个模式。
python examples/dflash_feedback/run.py \
  --model /path/to/Qwen3-8B --draft /path/to/Qwen3-8B-DFlash-b16 \
  --prompts prompts.jsonl --output results/one --mode calibrate --limit 40 --warmup 20

python examples/dflash_feedback/analyze.py results/full
python examples/dflash_feedback/test_feedback.py -v
```

不传 `--baseline-root` 时，脚本测试 Target Only/off/observe/calibrate，**不声称完成修改前 Baseline 比较**。
对比脚本会在输出或 observe 接受轨迹不一致时失败。Baseline 目录必须是与当前模型和 vLLM 版本兼容的修改前源码；备份目录不能未经检查就当成等价基线。
参数固定为 seed=42、7 个候选、temperature=0、max_model_len=1024、max_num_batched_tokens=4096、max_tokens=256。
Prompt 过长导致超过上下文限制时应报错并统一修改输入文件，不能只为某个模式过滤。
自备 JSONL 每行需包含唯一 `id`、`prompt`，可选 `dataset`。输入文件 hash 和模型配置保存在结果中。

预热和评估样本不重叠。预热性能和日志不计入正式结果；保留预热学习的全局 Bias，评估阶段继续在线学习。
每个模式、batch size 使用新进程。相同正式输入顺序和分组保证可比性。

## 输出与指标解释

- `result.json`：配置、真实生成 token 数、端到端耗时/吞吐、平均每 token 延迟、worker Target 采样调用计数和接受轨迹。
- `outputs.jsonl`：逐样本实际输出 token IDs，用于逐 token 正确性比较。
- `feedback.<pid>.jsonl`：Request/step/绝对位置/上下文、原始 Top-K logits、候选、Target argmax、严格接受结果、排名、部分概率偏差、稀疏 log-prob、修改/反馈耗时和表内存。
- `summary.csv`、`rejection_histogram.json`：各模式汇总；无观测的指标留空。
- `paired_online.csv`、`paired_counts.json`：相同样本和绝对 Anchor 的实际在线接受长度增/减/不变。候选数不同的配对不计入增减汇总；未配对 Anchor 不作推断。
- `acceptance.png`、`throughput.png`、`first_rejection.png`、`bias_acceptance.png`：仅从已完成运行生成。

Mean Accepted Length 定义为接受的 Draft Token 数，不含恢复/Bonus Token；接受比例为总接受数/总候选数。
Top-1 一致率只使用真实有效前缀；Bias 改变比例使用全部实际提交校验的候选。
单轮因果改善不能从原路径 logits 离线得出；配对图属于跨真实运行的描述性比较，输出一致且相同 Anchor 时才配对。

同步开销允许但应如实计入：实验代码使用少量 `.cpu()` 和上下文标量同步，整词表不会传至 CPU。
`adjust_ms` 包括复制、Bias 构建、argmax 和结果同步；`feedback_ms` 包括统计、归一化、稀疏传输及 EMA。
`target_forward` 事件另记 JSONL 写入耗时和未消费 pending logits 字节数。计时是同步墙钟估计，可能包含前序设备工作等待，不是独立 kernel 时间。
临时设备缓存约 `batch * 7 * vocab * dtype_bytes`，另有临时 float32/target snapshot；CPU 表内存由 Python 对象大小求和。
JSONL I/O、日志同步以及 benchmark-only 接受轨迹计数都在端到端耗时内，必须据此解释吞吐，不应外推生产性能。
Target Forward 计数以本同步 MRv2 路径每次 `sample` 对应的 Target 执行为单位，含 prefill，不是所有内部层/算子的调用数。

## 本机验证结果与限制

当前环境为 Windows、PyTorch `2.15.0+cpu`，无 vLLM/torch_npu/NPU，也未发现已部署模型路径。
已运行 13 项 CPU 单元测试并全部通过，包含嵌套配置校验及旧环境变量不生效的检查；测试中的确定性 oracle 只验证协议，不能代替真实模型结果。
**尚无真实 NPU 接受率或吞吐结果，研究假设未得到证实。**

| 正确性要求 | 本机状态 |
| --- | --- |
| off 与修改前 Baseline 一致 | 待 NPU，脚本支持 pristine 副本比较 |
| observe 候选/最终结果不变 | CPU argmax 等价通过；模型/接受轨迹待 NPU |
| calibrate 与 Target Only 一致 | CPU 在线 Greedy oracle 通过；模型待 NPU |
| Draft/Target/Anchor 对齐 | CPU 位置与 ID 检查通过；模型待 NPU |
| 首拒之后不更新 | CPU 通过 |
| 多请求不串扰 | CPU 4 请求与重排通过；真实 batch=4 待 NPU |
| 空历史等价 | CPU 通过 |
| off 无反馈计算/清理 | 代码委托原路径；CPU 路径测试；实际性能待 NPU |
| 新 benchmark 表初始化 | CPU 通过；NPU 独立进程流程待验证 |

当前主要瓶颈是实验日志/CPU 同步、全局表构造索引、全词表设备归一化和一轮设备缓存。
先完成上述 smoke/正式测试；若有真实收益，再考虑设备侧稀疏更新、日志采样和 Context-Aware 消融。
本版不实现 Context-Aware Bias，也不调整动态候选长度。

本机还完成 Ruff 检查/格式检查、`compileall` 和 `git diff --check`。
`bash format.sh ci` 已尝试，但原有 CRLF 脚本在 WSL Bash 中解析失败，未完成仓库全套 pre-commit 检查。
技能更新自检脚本也因 CRLF 解析失败，未修改安装内容；KG 健康检查失败后未继续访问服务。
