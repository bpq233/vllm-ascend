# 任务：在原始 vLLM-Ascend 中实现基于稀疏层校验的多轮 DFlash 投机解码

## 一、任务背景和目标

请直接修改当前工作目录中的原始 vLLM-Ascend 代码，实现一种新的投机解码机制：

**DFlash → Target Sparse-Layer Verifier → 多轮迭代 → Full Target Verification**

注意：

- 当前任务基于原版 vLLM-Ascend，不使用我之前实现的任何层次化投机解码代码。
- 保留原始 DFlash 的模型结构、权重和并行草稿生成方式。
- 不引入额外的独立中间语言模型。
- 中间校验模型由 Target 原有 Transformer 层中选取的部分层构成，所有参数复用 Target 原始权重，不进行训练。
- 实现两种模式：固定轮数模式和置信度自适应模式。
- 两种模式共用中间校验模型、DFlash、KV Cache 管理及最终 Target 校验逻辑。
- 原有 DFlash 模式必须保持兼容，新增功能默认关闭。

请先分析当前代码架构和现有接口，然后直接完成代码修改、测试和示例，不要仅输出设计方案。

## 二、中间校验模型的构建

### 1. 从 Target 中抽取指定数量的 Transformer 层

假设 Target 有 L 层 Transformer，用户指定中间模型层数为 M。

从 Target 的原始层中均匀选取 M 层，形成轻量级中间校验模型：

- 层数 M 可配置。
- 使用原始 Target 的 Transformer 层权重。
- 不创建新的可训练参数，不进行微调。
- 中间模型保留 Target 的 Token Embedding、最终 RMSNorm 和 LM Head。
- 只执行选中的 Transformer 层，其余层跳过。
- 保留选中层的原始顺序及模型层编号。
- 模型初始化时打印最终选择的层编号。

### 2. 必须包含 DFlash 所依赖的 Target 隐藏状态层

DFlash 会从 Target 的多个中间层抽取 Hidden States 作为输入特征。

请先检查当前 DFlash checkpoint 配置、DFlash Speculator 以及 Target Hidden States 提取代码，获取实际需要的层编号。

注意识别 `target_layer_ids`、`eagle_aux_hidden_state_layer_ids` 等配置的来源及其层编号语义，检查是否存在 0-based、1-based、pre-layer 和 post-layer 的转换。

不要硬编码 DFlash 的隐藏状态层编号。

令其对应的 Transformer 层集合为 S，要求：

- 最终中间模型选层集合必须包含 S。
- 当 M 小于必须保留的层数时，直接报错。
- 剩余层从 Target 中均匀选择，尽可能缩小相邻层之间的间隔。
- 不允许重复选层。
- 所有选中层编号必须合法。
- M 等于 L 时，中间模型应与完整 Target 在模型计算层面保持一致，可作为正确性测试基准。

建议设计独立的选层函数，例如 `select_sparse_layers(total_layers, num_layers, required_layers)`，并编写单元测试。

### 3. 中间校验模型的前向计算

中间模型接收 DFlash 提出的候选 Token，在当前已确认前缀下执行因果前向计算。

只执行选中的 Transformer 层，最终通过原始 RMSNorm 和 LM Head 得到词表上的 Logits。

定义：

`q_i(v) = softmax(logits_i)[v]`

其中 `q_i` 是中间稀疏层模型在当前上下文下预测第 i 个候选 Token 时得到的分布。

重点处理：

- 正确的 Token Position 和 RoPE。
- 各个候选位置的 Logits 对齐，不能出现 one-token shift 错误。
- Attention Mask 必须满足因果约束。
- 中间模型使用独立、正确管理的 KV Cache。
- 只提交有效前缀，丢弃被拒绝的后缀。
- 中间模型可以与 Target 共享权重，但不得混用语义不一致的 KV Cache。

中间模型输出至少包括：

1. 每个候选 Token 对应的 Logits 或所需概率。
2. 每个候选 Token 的校验结果。
3. DFlash 所需层的 Hidden States。
4. 本轮可继续扩展的有效前缀长度。

### 4. DFlash Hidden States 的更新

这是本任务必须重点解决的问题。

原始 DFlash 使用完整 Target 产生的中间层 Hidden States。现在希望在多轮迭代中，利用稀疏层模型继续为 DFlash 提供特征，避免每轮调用完整 Target。

需要实现以下机制：

- 初始轮使用已有的、与当前上下文正确对齐的 Target Hidden States。
- 中间模型校验通过后，抽取其对应层的 Hidden States，尝试作为下一轮 DFlash 的条件特征。
- 保证特征层的数量、顺序、张量维度与 DFlash checkpoint 的要求一致。
- 必须检查 Token Position 和特征有效性，不能复用已拒绝 Token 对应的特征。
- 明确区分完整 Target 特征和稀疏模型近似特征。

由于稀疏层模型跳过了部分 Transformer 层，其 Hidden States 与完整 Target 对应层的真实 Hidden States 并不等价。

请增加特征来源控制：

- `sparse`：使用中间稀疏层模型重新计算的特征。
- `target`：使用完整 Target 真实特征，仅在其确实覆盖当前上下文时有效。

默认实验模式为 `sparse`。如果无法获得当前前缀所需的有效特征，不得把旧位置的 Target 特征冒充新特征，而应该报出明确错误或采用有正确性保证的回退路径。

同时记录特征来源、形状与位置，方便分析特征分布偏移对 DFlash 接受率的影响。

## 三、方案一：固定 N 轮迭代

配置：

- `mode = "fixed"`
- `num_rounds = N`

流程：

1. 使用 DFlash 并行生成 K 个候选 Token。
2. 中间稀疏层模型对这 K 个 Token 进行一次并行前缀校验。
3. 默认采用贪婪前缀校验：候选 Token 必须等于中间模型对应位置的 argmax Token 才能通过。
4. 从前向后找到第一个不匹配的 Token。
5. 保留该位置之前连续通过校验的前缀，丢弃当前及后续候选。
6. 使用新的有效前缀和正确更新的 Hidden States，再次调用 DFlash 扩展。
7. 重复上述步骤，最多进行 N 轮。
8. N 轮完成后，将累计候选 Token 交给完整 Target 进行一次最终校验。
9. 完整 Target 根据原始目标模型的校验规则确定真正接受的连续前缀，以及必要的修正或 Bonus Token。

中间模型接受的 Token 只能视为“临时接受”，不能提前输出给用户。

无论中间执行多少轮，最终必须经过完整 Target 校验。

对于没有接受任何 Token 的轮次，也要继续消耗轮数，不能无限重试。

当达到 Target 单次校验的最大 Token Budget 时，应严格遵守该 Budget。请检查当前 Ascend 内核限制，根据剩余容量缩短每轮草稿长度；若配置无法满足固定 N 轮和单次 Target 校验两个要求，应在初始化时报错，不允许悄悄拆成多次 Target 校验。

## 四、方案二：基于置信度的自适应迭代

配置：

- `mode = "confidence"`
- `max_rounds = N`
- `threshold_high = τ1`
- `threshold_low = τ2`

要求：

`0 <= τ2 < τ1 <= 1`

### 1. 置信度计算

中间模型对第 i 个候选 Token `x_i` 产生 Logits。

计算：

`confidence_i = q_i(x_i) / max_v q_i(v)`

推荐直接通过 Logits 计算：

`confidence_i = exp(logit_i[x_i] - max_v(logit_i[v]))`

这样可以避免对整个词表执行 Softmax 后再计算比值，提升数值稳定性和效率。

注意：这里 q 指的是中间稀疏层校验模型的输出分布，而不是 DFlash 的草稿分布。

### 2. 三种决策

**情况 A：confidence >= τ1**

表示中间模型对该候选 Token 的支持程度较高。

处理方式：

- 临时接受该 Token。
- 继续检查下一个候选 Token。
- 如果当前草稿块全部通过，更新有效前缀。
- 在轮数和 Token Budget 允许时，进入下一轮 DFlash 扩展。

这里的接受只是中间模型层面的接受，最终仍需 Target 校验。

**情况 B：τ2 <= confidence < τ1**

表示当前 Token 置信度不足，但还没有必要立即调用完整 Target。

处理方式：

- 保留当前 Token 之前所有连续通过高阈值校验的前缀。
- 拒绝当前 Token，丢弃其后的全部候选。
- 不直接调用 Target。
- 更新中间模型的有效前缀和 KV Cache。
- 进入下一轮 DFlash 重新生成候选。

如果尚未达到最大轮数，则继续迭代；如果已达到最大轮数，则进入完整 Target 校验。

**情况 C：confidence < τ2**

表示当前候选 Token 与中间模型预测存在较大差异。

处理方式：

- 立即停止后续 DFlash 扩展。
- 标记当前轮为低置信度提前终止。
- 将之前累计的有效前缀和当前待校验候选交给完整 Target。
- 立即执行一次完整 Target 校验。
- 完整 Target 决定实际接受长度和修正 Token。

### 3. 顺序处理原则

必须从左到右处理候选 Token。

如果某个位置触发情况 B 或情况 C，则当前块后续 Token 不允许越过该位置单独接受。

不能只根据每个 Token 的独立阈值判断结果拼接离散接受位置。

例如：

置信度为 `[0.98, 0.95, 0.72, 0.93, 0.99]`

假设 `τ1=0.9，τ2=0.5`，则：

- 前两个 Token 临时接受。
- 第三个 Token 触发情况 B。
- 第三个及后续 Token 丢弃。
- 保留前两个 Token，进入下一轮 DFlash。

假设置信度为 `[0.98, 0.95, 0.32, 0.93]`：

- 前两个 Token 临时接受。
- 第三个 Token 触发情况 C。
- 立即进入 Target 校验。

### 4. 终止条件

以下情况必须结束中间迭代，执行最终 Target 校验：

- 触发低阈值。
- 达到最大轮数。
- 达到 Target 单次可校验 Token Budget。
- 剩余模型上下文长度不足。
- EOS 或请求终止等边界条件需要结束当前投机块。

必须避免零接受长度导致的死循环，保证每个外层解码步骤最终能够通过 Target 推进生成。

## 五、最终 Target 校验

无论使用哪种模式，都必须保留完整 Target 的最终校验。

请尽量复用 vLLM-Ascend 原始 DFlash 的 Target Verification 和 Rejection Sampler 机制。

关键要求：

1. 最终 Target 必须校验整个中间阶段累计保留的候选序列，而不能只校验最后一轮。
2. Target 必须从上一次真实确认的 Token 前缀出发。
3. 中间阶段的临时接受结果不能修改已经正式提交的 Target KV Cache。
4. Target 对候选 Token 的验证必须满足连续前缀接受语义。
5. 当 Target 拒绝某个 Token 时，需要正确处理修正 Token 和后续上下文。
6. 当 Target 接受全部候选时，需要正确处理 Bonus Token。
7. 必须正确同步序列长度、Token Position、Block Table 和 KV Cache 状态。
8. 必须支持批量请求，每个请求有独立的轮数、候选长度和提前终止状态。
9. 不能因为某一个请求触发 Target 校验而破坏其他请求的状态。

首版限定 `temperature=0` 的贪婪推理。

原因是中间模型置信度筛选、拒绝后重新生成等操作改变了最终候选的有效提议分布，不能在没有证明的情况下把原始 DFlash 概率直接交给标准概率拒绝采样器并宣称无损。

在贪婪模式下，最终结果必须与原始完整 Target 的贪婪输出一致。

如果用户使用非零 temperature，应明确提示当前实验模式不支持，不允许静默运行可能有偏的采样流程。

## 六、配置接口

尽量复用现有 vLLM-Ascend 的 `additional_config` 配置机制；如果当前版本不支持，则选择与原始配置验证流程兼容的实现。

建议配置结构：

`speculative_config`：

- `method: dflash`
- `model: /home/data/Qwen3-8B-DFlash-b16`
- `num_speculative_tokens: 7`

`additional_config.dflash_sparse_verify`：

- `enabled: true`
- `mode: fixed` 或 `confidence`
- `num_intermediate_layers: 12`
- `num_rounds: 4`
- `threshold_high: 0.90`
- `threshold_low: 0.50`
- `feature_source: sparse`
- `enable_metrics: true`

要求：

- 支持 Python `LLM()` 离线推理。
- 尽可能支持 vLLM Serve。
- 新增配置必须经过类型和取值校验。
- 禁用该功能时，保持原版 DFlash 行为。
- 固定模式忽略置信度阈值。
- 置信度模式使用 `num_rounds` 作为最大迭代轮数。
- 不允许存在配置参数定义了但实际上未生效的问题。

## 七、代码架构和实现要求

请首先检查本地版本中以下模块或对应实现：

- `vllm_ascend/worker/v2/spec_decode/dflash/speculator.py`
- DFlash 所继承的原始 Speculator。
- Target Model Runner。
- Qwen3 模型的 Transformer Layer Forward。
- Target Hidden States 提取。
- Speculative Config 和 additional_config。
- Rejection Sampler。
- KV Cache Manager。
- Scheduler 与输出 Token 提交逻辑。

不要假设最新上游文件一定与本地代码一致，应以本地实际代码为准。

建议拆分为以下职责清晰的组件，具体文件路径由实际代码组织决定：

- `SparseLayerSelector`：负责中间模型选层。
- `SparseTargetVerifier`：负责稀疏层执行、Logits、Hidden States 和 KV 管理。
- `ConfidenceDecision`：负责双阈值分类与连续前缀判断。
- `MultiRoundDFlashController`：负责多轮草稿、临时前缀和最终 Target 调用。
- `SparseVerificationMetrics`：负责统计与调试。

尽量复用现有权重和模型模块，避免复制完整 Target 或重复分配没有必要的参数。

不要为了方便实现而把整套原有 DFlash 推倒重写。

首版允许新增功能使用 eager 模式，但不得因此破坏普通 DFlash 的 ACL Graph 运行路径。

如果多轮动态形状与 ACL Graph 不兼容，应明确隔离该路径，确保可以在昇腾 NPU 上正常运行。

## 八、统计与日志

增加可配置的统计功能，至少记录：

- 每个请求实际执行的中间迭代轮数。
- 每轮 DFlash 提议 Token 数。
- 每轮中间模型临时接受 Token 数。
- 每轮首次拒绝位置。
- 每轮置信度分布。
- 高阈值接受次数。
- 中间阈值重试次数。
- 低阈值提前进入 Target 次数。
- 每次最终 Target 提议长度、真实接受长度及拒绝位置。
- DFlash 耗时。
- 中间模型前向耗时。
- 完整 Target 校验耗时。
- 单轮总耗时。
- 总吞吐量、TPOT 和平均 Target 接受长度。

提供固定模式和置信度模式的性能对比统计，并区分中间模型临时接受长度与 Target 最终真实接受长度。

## 九、测试要求

请添加可运行的测试和 Benchmark。

### 1. 选层正确性测试

测试：

- 保留全部 DFlash 所需层。
- 选层数量恰好为 M。
- 层编号不重复、单调递增。
- 小于必选层数时正确报错。
- M=L 时覆盖完整模型。
- 不同 M 对应的选层合理均匀。

### 2. 中间模型正确性测试

- M=L 时，中间模型 Logits 应与完整 Target 对齐。
- 检查 RoPE、Position 和 Logits Shift。
- 验证 KV Cache 增量计算和完整上下文重新计算的一致性。
- 验证中间模型 Hidden States 的层顺序和形状。

### 3. 固定模式测试

测试 N=1、2、4 等不同轮数，验证：

- 正确执行指定轮数或明确处理容量边界。
- 候选接受与拒绝逻辑正确。
- 多轮候选能正确拼接。
- 最终 Target 校验完整候选序列。
- 不发生无限循环。

### 4. 置信度模式测试

构造人工 Logits，覆盖：

- 所有 Token 高于高阈值。
- 第一个 Token 落入中间区间。
- 中间位置落入中间区间。
- 第一个 Token 低于低阈值。
- 中间位置低于低阈值。
- 恰好等于阈值。
- NaN、Inf 及非法阈值。
- 达到最大轮数。

确保所有阈值分支和停止逻辑符合要求。

### 5. 端到端正确性测试

使用相同 Target 和测试 Prompt，对比：

1. 不启用投机的完整 Target。
2. 原版 DFlash。
3. 固定轮数稀疏校验 DFlash。
4. 置信度自适应稀疏校验 DFlash。

在 `temperature=0` 下，最终输出应与完整 Target 的贪婪结果一致。

测试单请求和多请求，并覆盖 EOS、长上下文、部分拒绝、全部接受和 KV Cache 回滚情况。

### 6. 性能测试

在相同环境下测试：

- 不同稀疏层数 M。
- 不同固定轮数 N。
- 不同阈值 τ1、τ2。
- 不同 Batch Size。
- 不同草稿长度 K。

输出统计数据用于研究层数、校验准确性、Target 接受长度和吞吐量的权衡。

## 十、实现顺序

请按以下顺序开发并在每一阶段验证：

**阶段 1：** 阅读现有 DFlash 和 Target 代码，梳理入口、隐藏状态提取及 KV Cache 生命周期。

**阶段 2：** 实现 SparseLayerSelector 和 SparseTargetVerifier，验证单次稀疏前向的正确性。

**阶段 3：** 实现 DFlash 使用稀疏 Hidden States 继续扩展，验证新一轮草稿真正依赖更新后的特征。

**阶段 4：** 实现固定 N 轮模式，验证连续前缀管理及最终 Target 校验。

**阶段 5：** 在相同框架上实现双阈值置信度模式，完善提前终止和回退。

**阶段 6：** 集成配置、日志、测试和 Benchmark。

不要跳过阶段 2 和阶段 3 的正确性验证直接修改最终采样器。

## 十一、最终交付

完成后请提供：

1. 实际修改和新增的代码文件。
2. 每个核心文件的修改说明。
3. 中间模型选层方法及实际选层结果。
4. DFlash 隐藏状态更新机制说明。
5. 两种模式的完整调用流程。
6. 可直接运行的 Python 推理脚本。
7. vLLM Serve 启动示例。
8. 单元测试和端到端测试结果。
9. 性能测试脚本和指标输出。
10. 当前版本下未解决的限制与风险。

必须实际修改代码并运行当前环境能够执行的测试。如果环境缺少模型、NPU 或依赖，应明确列出未执行的测试，不得伪造通过结果。

**优先保证算法语义正确、Target 最终验证正确、KV Cache 状态正确，再进行性能优化。**

# 补充需求：最终 Target 支持四种可配置校验策略

在前面实现的 **DFlash → 稀疏层中间模型 → 多轮迭代 → 完整 Target** 架构基础上，新增最终 Target 校验策略配置。

本次修改只针对最终完整 Target 的校验模块，不改变 DFlash 草稿生成、稀疏层中间校验及多轮迭代的基本流程。

最终支持四种模式：

1. `strict`：严格校验。
2. `topk`：Top-k 近似校验，默认 Top-2。
3. `confidence`：基于 Target 概率比值的置信度校验。
4. `accept_all`：全部候选直接放行，Target 只生成下一个 Token。

## 一、四种最终 Target 校验模式

### 1. 严格校验（strict）

完整 Target 对累计候选 Token 进行一次前向计算，得到每个候选位置对应的 Logits。

对于候选 Token x_i，计算：

`target_token_i = argmax_v p_T(v | prefix_i)`

校验条件：

`x_i == target_token_i`

如果相等，则接受当前 Token，继续验证后续候选。

如果不相等，则立即拒绝当前及其后续候选，并使用 Target 对应位置的 argmax Token 作为修正 Token。

如果全部候选通过，且已计算下一位置的 Target Logits，则生成一个 Bonus Token。

在 `temperature=0` 的条件下，该模式必须与完整 Target 的贪婪解码结果一致。

### 2. Top-2 近似校验（topk）

完整 Target 计算每个候选位置的 Logits，并选取概率最高的两个 Token：

`top2_i = topk(target_logits_i, k=2)`

校验条件：

`x_i in top2_i`

如果候选 Token 属于 Target 当前预测的 Top-2，则接受该 Token，即使它不是 Target 的 argmax。

否则，拒绝当前及后续候选，并使用 Target 的 argmax Token 修正。

默认 `k=2`，但代码中允许配置 k，方便后续扩展 Top-1、Top-3 等实验。

重要要求：

- 必须按照候选序列从左到右连续校验。
- 允许接受 Target 的非最高概率 Token。
- 后续 Target Logits 必须基于实际候选 Token 构成的上下文计算。
- 该方法属于近似校验，不保证与原始 Target 贪婪输出完全一致。
- 不能将 Top-2 校验描述为无损投机解码。

### 3. 置信度校验（confidence）

完整 Target 对每个候选 Token 计算概率相对比值：

`ratio_i = p_T(x_i | prefix_i) / max_v p_T(v | prefix_i)`

为了提高计算效率，直接采用 Logits：

`ratio_i = exp(logit_i[x_i] - max_v(logit_i[v]))`

设置最终 Target 置信度阈值 `target_threshold`。

校验条件：

`ratio_i >= target_threshold`

满足条件则接受，否则拒绝当前及后续 Token，并使用 Target 的 argmax Token 修正。

要求：

- 阈值可配置，范围为 (0, 1]。
- 阈值为 1 时，在无最大值并列等边界差异时，退化为严格 argmax 校验。
- 较低阈值可以接受更多非 argmax Token。
- 必须按照候选序列从左到右执行连续前缀校验。
- 该方法属于近似校验，不保证 Target 原始分布不变。
- 对数值异常、阈值边界和并列最大 Logits 做明确处理。

注意：最终 Target 的 `target_threshold` 与中间模型自适应模式的 `threshold_high`、`threshold_low` 完全独立，不允许共用。

### 4. 全部放行（accept_all）

该模式不对 DFlash 多轮累计候选 Token 进行任何接受或拒绝判断。

所有候选 Token 直接接受，完整 Target 只负责根据完整候选序列生成下一个 Token（Bonus Token）。

执行流程：

1. DFlash 生成候选 Token。
2. 中间稀疏层模型按照固定 N 轮或置信度自适应模式完成多轮迭代。
3. 得到累计保留的候选 Token 序列。
4. 将全部候选 Token 输入完整 Target。
5. 跳过严格匹配、Top-k 判断及置信度判断。
6. 全部候选 Token 直接接受。
7. Target 执行必要的前向计算，更新 KV Cache，并预测候选序列之后的下一个 Token。
8. 输出全部候选 Token 和 Target 生成的下一个 Token。

例如：

DFlash 多轮累计候选：

`[A, B, C, D, E, F]`

完整 Target 不判断 A～F 是否正确，直接全部放行。

Target 根据上下文：

`[已确认前缀, A, B, C, D, E, F]`

预测下一个 Token G。

最终输出：

`[A, B, C, D, E, F, G]`

其中 G 必须来自完整 Target 对候选序列之后位置的预测，首版使用贪婪 argmax。

#### Accept All 的关键实现要求

**（1）跳过校验，但不跳过 Target 前向**

虽然不进行候选校验，但 Target 仍必须执行必要的前向计算，以保证 KV Cache 正确并生成下一个 Token。

不能直接使用中间模型 Logits 生成 Bonus Token，也不能使用过期的 Target KV Cache。

**（2）无需计算中间候选位置的完整词表 Logits**

Accept All 模式不需要逐 Token 检查候选概率。

如果现有 Target Model Runner 支持只计算最后一个有效位置的 Logits，应直接复用该能力，避免对全部候选位置执行不必要的 LM Head 计算。

但是，Target Transformer 仍必须正确处理所有尚未计算的候选 Token。

**（3）全部接受**

定义：

`accepted_length = proposed_length`

`rejected_length = 0`

`bonus_length = 1`

其中 Bonus Token 只有在生成长度、EOS 和上下文预算允许时才生成。

这里的 accepted_length 表示直接放行的长度，不代表经过 Target 校验的正确长度。

**（4）KV Cache 正确性**

- 所有放行的 Token 必须正确纳入 Target 上下文。
- Target KV Cache 必须与实际输出的 Token 序列保持一致。
- 不能直接复用不兼容的稀疏层模型 KV Cache。
- 正确处理 Position、RoPE、Attention Mask 和 Block Table。
- 下一轮 DFlash 必须能够基于更新后的上下文继续生成。
- 如果存在最后一个已输出 Token 尚未写入 KV Cache 的延迟计算机制，应遵循原始 vLLM 的缓存生命周期，避免重复计算或遗漏。

**（5）生成质量说明**

Accept All 属于实验性近似生成策略。

该模式不保证输出与 Target 贪婪解码一致，也不能视为无损投机解码。

其主要用途是研究在省略最终 Token 校验后，中间稀疏层模型筛选候选的质量，以及最终 Target 校验带来的性能开销。

## 二、统一校验器

建议实现统一的 `FinalTargetVerifier`，支持四种模式：

- `strict`
- `topk`
- `confidence`
- `accept_all`

### 输入

1. 完整 Target 的 Logits（仅在对应模式需要时计算）。
2. 所有中间迭代累计保留的候选 Token。
3. 最终校验方法及参数。
4. 各请求的有效候选长度和位置信息。

### 输出

1. Target 最终接受长度。
2. 首次拒绝位置。
3. 修正 Token 或 Bonus Token。
4. 每个位置的接受结果。
5. Target 校验耗时及相关指标。

### 统一实现要求

四种模式应共享完整 Target 前向和统一的 Token 提交及 KV Cache 管理机制，仅在候选 Token 接受条件和所需 Logits 计算范围上存在差异。

- `strict`：使用 argmax 判定。
- `topk`：使用 Top-k Membership 判定。
- `confidence`：使用概率比值判定。
- `accept_all`：不执行接受条件判断，直接全部放行。

前三种模式共用连续前缀接受、首次拒绝、后缀丢弃和修正 Token 逻辑。

Accept All 模式不发生拒绝，因此直接提交所有候选 Token，并生成 Bonus Token。

不要分别实现四套 Target 前向逻辑。

特别注意：即使 Top-2 或置信度校验接受了非 argmax Token，后续位置的 Target Logits 也必须对应包含这些实际候选 Token 的上下文，不能使用不一致的上下文继续验证。

如果当前原始 vLLM-Ascend 的 Rejection Sampler 不支持自定义 Top-k、置信度或 Accept All 判定，需要新增独立的验证策略组件，而不是破坏原始严格校验实现。

## 三、配置方式

新增统一配置：

```python
"additional_config": {
    "dflash_sparse_verify": {
        "enabled": True,

        # 中间校验模型
        "num_intermediate_layers": 12,

        # 中间迭代方式
        "mode": "confidence",
        "num_rounds": 4,
        "threshold_high": 0.90,
        "threshold_low": 0.50,

        # DFlash 特征来源
        "feature_source": "sparse",

        # 最终完整 Target 校验
        "final_verification": {
            "method": "topk",
            "topk": 2,
            "target_threshold": 0.10
        },

        "enable_metrics": True
    }
}
```

`final_verification.method` 支持：

| Method | 说明 | 主要参数 |
|---|---|---|
| strict | 严格 argmax 校验 | 无 |
| topk | Top-k 近似校验 | topk，默认 2 |
| confidence | Target 置信度校验 | target_threshold，默认 0.1 |
| accept_all | 全部放行，仅预测下一个 Token | 无 |

配置规则：

- `method="strict"`：忽略 topk 和 target_threshold。
- `method="topk"`：使用 topk 参数。
- `method="confidence"`：使用 target_threshold。
- `method="accept_all"`：忽略 topk 和 target_threshold，全部候选直接接受。

应检查当前配置解析方式，确保实际配置结构符合本地 vLLM-Ascend 接口。

新增功能默认关闭，不能影响原始 DFlash 运行。

## 四、两种中间迭代模式与四种 Target 校验模式组合

总共支持八组实验：

| 中间迭代模式 | 最终 Target 校验 |
|---|---|
| Fixed N | Strict |
| Fixed N | Top-2 |
| Fixed N | Confidence |
| Fixed N | Accept All |
| Adaptive N | Strict |
| Adaptive N | Top-2 |
| Adaptive N | Confidence |
| Adaptive N | Accept All |

其中：

- Fixed N：中间稀疏层模型按指定轮数执行校验与 DFlash 重新扩展。
- Adaptive N：中间模型根据双阈值判断继续接受、重新生成或提前进入 Target。

不允许因为最终 Target 校验策略不同而改变中间阶段的轮数、候选选择及接受规则。

八组实验应能够使用统一脚本，通过修改配置直接切换。

## 五、统计与测试

### 1. 统计指标

针对每次最终 Target 调用记录：

- 中间模型实际迭代轮数。
- 累计候选 Token 长度。
- Target 最终接受 Token 长度。
- Target 首次拒绝位置。
- Target 严格 argmax 匹配长度。
- 近似校验额外接受的 Token 数。
- Accept All 直接放行 Token 数。
- Target 前向与校验耗时。
- 修正 Token 和 Bonus Token 数量。
- 总解码吞吐量与平均 TPOT。

对于 Top-k 和 Confidence 模式，统计：

`approx_extra_accepted = approximate_accepted_length - strict_accepted_length`

其中严格接受长度应在同一次 Target 前向得到的相同候选序列上计算，避免比较不同候选序列。

对于 Accept All 模式：

- `accepted_length = proposed_length`
- `rejected_length = 0`

Accept All 不需要默认计算每个候选位置的严格匹配长度，否则会破坏仅计算下一 Token Logits 的性能优化目标。

如需分析 Accept All 与严格校验的差距，可以提供独立的调试或离线评估模式，不纳入默认性能测试路径。

### 2. 严格校验测试

1. 全部候选通过。
2. 第一个 Token 被拒绝。
3. 中间位置被拒绝。
4. 全部接受后的 Bonus Token 处理。
5. Target 拒绝后的 KV Cache 回滚与修正。
6. 最终输出与完整 Target 贪婪解码一致。

### 3. Top-k 校验测试

1. Top-1 与 Strict 的一致性。
2. Strict 拒绝但 Top-2 接受。
3. 候选不属于 Top-k 时正确拒绝。
4. Top-k 参数边界检查。
5. 接受非 argmax Token 后，后续上下文计算正确。

### 4. Confidence 校验测试

1. 候选置信度高于阈值。
2. 候选置信度低于阈值。
3. 候选置信度恰好等于阈值。
4. 阈值为 1 的边界行为。
5. 非法阈值与数值异常处理。
6. 非 argmax Token 被接受后，后续上下文正确。

### 5. Accept All 校验测试

1. 全部候选 Token 均被接受。
2. 不执行逐 Token 接受或拒绝判定。
3. 不调用不必要的逐位置 Top-k 或置信度计算。
4. Bonus Token 必须由完整 Target 生成。
5. 在相同前缀和候选条件下，Bonus Token 与完整 Target 下一位置贪婪预测一致。
6. 多轮候选 Token 拼接正确。
7. Target KV Cache 与实际输出序列一致。
8. 正确处理 EOS、最大输出长度和上下文边界。
9. Batch 内不同请求具有不同候选长度时正确运行。
10. 下一轮 DFlash 能够正确接续生成。
11. 不因所有候选直接放行而出现 Position 偏移、重复 Token 或上下文缺失。

### 6. 端到端测试

端到端测试必须覆盖全部八种组合。

同时对比：

1. 原始完整 Target。
2. 原版 DFlash。
3. Fixed N + 四种 Target 校验。
4. Adaptive N + 四种 Target 校验。

严格模式在 `temperature=0` 下必须与完整 Target 贪婪解码一致。

Top-k、Confidence 和 Accept All 模式允许生成结果不同，但必须保证 Token 序列、位置、Attention Mask、KV Cache 和上下文一致性正确。

其中：

- Strict 属于确定性贪婪无损校验。
- Top-k 属于近似校验。
- Confidence 属于近似校验。
- Accept All 属于无条件放行策略。

后三种策略均不能宣称保持完整 Target 原始生成分布不变。

首版限定 `temperature=0`，非零温度应明确报错或提示未支持。

## 六、实现要求

请在已经实现的 `SparseLayerVerifier` 和 `MultiRoundDFlashController` 基础上新增最终校验策略，不需要重写中间校验逻辑。

优先复用原始 vLLM-Ascend 中 DFlash 的 Target Forward、Rejection Sampler、Output Processor 和 KV Cache 管理机制。

实现时特别注意：

1. 四种策略使用统一配置入口。
2. 严格校验保持原版 Target 贪婪输出的一致性。
3. Top-k、Confidence、Accept All 的非严格接受不能错误地触发原版严格拒绝采样器。
4. Accept All 必须跳过候选校验，但保留完整 Target 必需的前向计算。
5. Accept All 尽量只计算 Bonus Token 所需的最后位置 Logits。
6. 四种模式下都必须正确处理 Target KV Cache 的提交与下一轮推理。
7. 不允许因候选 Token 被近似接受而错误复用旧的 Target Hidden States。
8. 确保 DFlash 下一轮使用的 Target 特征与对应 Token 前缀保持一致。
9. 不允许修改已正确运行的原始 DFlash 路径。
10. 新增配置和日志默认不影响原版 DFlash 性能。

先检查本地原版 vLLM-Ascend 的具体接口，再实际修改代码、补齐测试和运行示例。

最终请交付：

- 修改及新增的文件列表。
- 四种校验策略的关键实现说明。
- 完整配置示例。
- 可运行的推理和 Benchmark 脚本。
- 八种模式的测试覆盖情况。
- 实际执行的测试结果。
- 当前实现的已知限制。

请直接完成代码实现，不要只输出方案或伪代码。若当前环境无法运行 NPU 测试，应明确说明原因，不得虚构测试结果。


# 七、FULL Graph 图模式支持与长候选序列兼容

在已经实现的 DFlash → 稀疏层中间模型 → 多轮迭代 → 完整 Target 架构中，进一步实现完整的 ACL Graph 支持。

核心目标：

1. DFlash Forward 支持 FULL Graph。
2. 中间稀疏层模型 Forward 支持 FULL Graph。
3. 完整 Target Forward 支持 FULL Graph。
4. 三个模型统一使用上游 vLLM-Ascend 的图模式配置和管理机制。
5. 支持中间模型多轮扩展后，累计候选 Token 长度超过 16 的情况。
6. 不改变原有固定 N 轮、自适应 N 轮和四种最终 Target 校验策略的算法语义。

## 1. 图模式配置完全沿用上游

不得单独为本功能添加图模式开关或图参数。

统一使用上游：

- `compilation_config`
- `cudagraph_mode`
- `cudagraph_capture_sizes`
- `max_cudagraph_capture_size`
- 上游 ACL Graph capture/replay 机制
- 上游图缓存、Warmup 和内存管理机制

不要新增：

- `enable_sparse_graph`
- `enable_dflash_graph`
- `enable_target_graph`
- `graph_mode`
- `graph_capture_sizes`

等重复配置。

首先检查当前 vLLM-Ascend 版本中图模式的实现，识别 `FULL`、`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE`、`PIECEWISE` 和 `NONE` 的真实运行行为。

具体要求：

- 当上游配置为 FULL 时，三个模型的 Forward 都应使用 FULL Graph。
- 当配置为 FULL_DECODE_ONLY 时，三个模型的 Decode Forward 应使用完整图，Prefill 遵循上游行为。
- 当配置为 FULL_AND_PIECEWISE 时，遵循上游图调度策略。
- 当配置为 PIECEWISE 或 NONE 时，也必须遵循上游，不得强行开启 FULL Graph。
- 原始 DFlash、Target 与新增中间模型必须继承同一套有效图模式配置。
- 不得因为新增中间模型而强制将 Target 或 DFlash 切换为 eager。

如果当前 Attention Backend 或硬件不支持某个图模式，必须按照上游兼容策略处理，明确报告实际生效模式。

特别注意：不能仅仅因为配置中写了 FULL，就认为三个模型已经成功入图，必须实际确认 capture 和 replay。

## 2. DFlash Forward 入图

首先检查原始 DFlash 的图捕获实现。

尽可能复用原始 DFlash Speculator 中已有的完整图机制，不要重新实现一套图管理系统。

要求：

- DFlash 模型的完整 Forward 可通过 ACL Graph 捕获和重放。
- 保持原有 DFlash 的并行候选生成逻辑。
- 支持不同 Batch Size。
- 支持中间模型提供的最新 Hidden States。
- 图重放时能够正确更新输入 Token、Position、Attention Metadata、Hidden States 和其他动态输入。
- 不允许图重放时继续读取上一轮残留的 Hidden States。
- DFlash 原始权重和模型结构保持不变。

原有 DFlash 的单轮候选长度仍然遵守 checkpoint 和 Attention Backend 的限制，不要求通过增加 DFlash 单次输出长度来突破 16 Token 限制。

本次需要支持的是：**多个 DFlash 迭代轮次累计产生超过 16 个候选 Token。**

## 3. 中间稀疏层模型 Forward 入图

此前实现的 SparseLayerVerifier 需要支持完整图捕获与重放。

中间模型由 Target 中选取的指定数量 Transformer 层组成，因此需要注意其计算图与完整 Target 不同。

要求：

1. 模型初始化时完成稀疏层选择，选中层集合在一次运行期间固定。
2. 选中层复用 Target 原始权重，不重复创建不必要的权重副本。
3. 捕获中间模型完整 Forward，包括选中 Transformer 层、必要的最终归一化、LM Head 以及 Hidden States 输出。
4. 固定层集合后重复重放同一套图，不允许每轮重新构建计算图。
5. 支持不同 Batch Size 和候选长度。
6. 支持中间模型 KV Cache 的增量计算。
7. 图重放后能够取得用于候选校验的 Logits 和供下一轮 DFlash 使用的 Hidden States。
8. 输出的 Hidden States 必须与实际 Token Position 对齐。

如果稀疏层模型使用独立的 KV Cache，图捕获与重放时应正确绑定其缓存地址和 Block Table。

需要特别检查中间模型的 Layer Index、Position、RoPE、Attention Mask 和 KV Cache Metadata 是否与稀疏执行逻辑兼容。

不能因为部分 Transformer 层被跳过，就直接复用完整 Target 对应层的 KV Cache。

## 4. Target Forward 入图

完整 Target 的最终校验必须支持 FULL Graph。

四种最终校验策略：

- strict
- topk
- confidence
- accept_all

均使用同一套 Target Forward 图执行路径。

要求：

- Strict、Top-k 和 Confidence 使用 Target Forward 输出的对应位置 Logits 进行校验。
- Accept All 不需要中间候选位置的校验 Logits，尽可能只计算 Bonus Token 所需的最终 Logits。
- 四种校验模式不能分别创建重复的完整 Target 模型。
- Target Forward 不得因为多轮中间校验产生长候选序列而直接报错。
- Target 的 KV Cache 必须正确处理临时候选和最终接受前缀。

优先复用上游 Target Model Runner 中已有的图捕获、图选择和图重放机制。

如果原有 Target Graph 只覆盖常规投机解码长度，需要扩展其对长候选输入的兼容能力。

## 5. 重点：Target 支持超过 16 个候选 Token

这是本次实现的关键要求。

原始 vLLM-Ascend 的部分昇腾 Attention Backend 使用 `npu_fused_infer_attention_score`，在 Decode 路径下单请求 Query Token 数可能受 16 的限制。

但在本方案中，DFlash 经过 N 轮中间校验后，最终候选长度可能超过 16。

例如：

- DFlash 单轮生成 7 个 Token。
- 中间模型执行 4 轮迭代。
- 最终累计候选长度可能达到 28。
- Target 需要对这 28 个 Token 执行最终校验。

因此必须支持候选长度 17、24、28、32 等情况。

### 5.1 不允许直接截断候选

不得通过以下方式规避问题：

- 强制将累计候选长度截断为 15 或 16。
- 超过 16 时直接丢弃剩余 Token。
- 将剩余候选视为已通过验证。
- 使用错误的 Position 或 Attention Mask 将长序列伪装成 16 Token。
- 静默切换为语义不一致的校验流程。

中间模型交给 Target 的全部有效候选必须得到正确处理。

### 5.2 实现长候选序列兼容路径

请先确认当前硬件、CANN 版本及 Attention Backend 的实际限制，再选择合理实现。

优先考虑两种方案。

**方案 A：使用支持长 Query 的 Attention 路径**

如果当前 Ascend Attention Backend 的 Prefill 或其他 Attention Kernel 支持超过 16 Token 的 Query，并且可以正确处理带有 KV Cache 的长候选序列，则优先复用该路径。

要求：

- 保留因果 Attention 语义。
- 正确使用已有 Target KV Cache。
- 正确处理历史前缀与新候选的 Attention Mask。
- 每个候选位置只能看到历史前缀和当前位置之前的候选。
- 支持图捕获与重放。
- 验证不同长度下的 Logits 与正确的 eager 参考实现一致。

不能仅通过切换某个 Attention Backend 标志，就认为长 Query 已经得到支持。

**方案 B：将 Target Forward 按 Query 长度分块**

如果单次 Decode Query 只能处理最多 16 Token，则实现内部 Chunked Forward。

例如累计 28 个候选：

- Chunk 1：前 16 个 Token。
- Chunk 2：后 12 个 Token。

每个 Chunk 执行合法的 Target Forward，使用正确的因果上下文和 KV Cache。

Chunk 2 必须能够看到 Chunk 1 对应的 Token。

要求：

1. 每个 Chunk 的 Query 长度不超过实际 Kernel 限制。
2. 各 Chunk 保持原始 Token 顺序。
3. Chunk 之间正确传递 KV Cache 状态。
4. 各 Chunk 的 Attention Mask 和 Position 正确。
5. 合并各 Chunk 对应的 Logits，供最终统一校验。
6. 不能在每个 Chunk 结束后单独执行最终接受/拒绝。
7. 所有 Chunk 完成后，对完整候选序列统一执行一次最终校验。
8. 如果某个 Token 最终被拒绝，应正确回滚或使其后续 KV Cache 失效。
9. 必须支持 Batch 内不同请求具有不同候选长度。
10. 在 FULL Graph 模式下，Chunk Forward 也应使用图捕获与重放。

注意：这里允许内部多个 Chunk Forward，但它们属于同一次逻辑 Target Verification，不能改变原有的一次最终校验语义。

不能把分块 Forward 误实现成多次独立的 Target 校验和 Token 提交。

### 5.3 超过 16 Token 时保持 FULL Graph

长候选兼容不能简单通过超过 16 后全部切回 eager 来完成。

优先实现：

- 正常长度：Target FULL Graph。
- 长候选：使用可入图的长 Query Attention 路径，或者多个 Chunk Graph Replay。
- 最终候选接受/拒绝逻辑统一执行。

如果当前运行环境确实存在无法在图模式下处理的算子或形状，应明确记录实际限制和回退原因，不能在日志中错误报告为 FULL Graph。

实现目标是让长候选路径同样能够使用图执行，而不是仅保证不报错。

## 6. 图捕获形状与动态长度支持

三种模型的 Forward 输入形状不完全一致：

- DFlash：每轮固定或有限变化的草稿块。
- 中间模型：每轮不同数量的候选 Token。
- Target：多轮累计后的可变长候选序列。

需要基于上游图管理机制处理这些形状。

要求：

1. 优先复用上游 `cudagraph_capture_sizes`。
2. 上游没有显式配置时，遵循其默认捕获尺寸生成规则，并检查是否覆盖新增工作负载。
3. 上游明确指定捕获尺寸时，不得静默覆盖。
4. 支持候选长度 Padding、有效长度 Mask 和静态 Buffer 复用。
5. Padding Token 不能改变真实 Token 的 Attention 结果。
6. 避免每次候选长度变化都重新 capture。
7. 支持图缓存复用，避免动态轮数引起反复捕获。
8. 保证多个请求共享 Graph 时，Token、Position、KV Block Mapping 和有效长度信息相互隔离。
9. 避免因图数量过多导致 NPU 显存显著增长。
10. 检查上游图捕获尺寸表示的是总 Token 数，而不是单请求 Token 数，避免 Batch Size 扩大后选择错误的 Graph。

对于超过 16 Token 的 Target 校验，应优先通过合法的长 Query Graph 或可复用的 Chunk Graph 处理，而不是为每个任意长度都捕获一张新图。

如果显式的上游 capture sizes 无法覆盖所需形状，应给出清晰的错误或按照上游规定回退，并记录实际执行路径；不得悄悄把未入图的执行报告为 FULL。

## 7. 多轮动态控制与图重放

本方案中，固定模式和自适应模式可能执行不同数量的中间轮次。

实现时不要求将整个动态多轮 Python 循环强制合并成一个静态 ACL Graph。

建议采用：

`DFlash Graph Replay → Sparse Verifier Graph Replay → Decision → 下一轮或 Target Graph Replay`

其中：

- DFlash Forward 使用完整图。
- 中间稀疏层 Forward 使用完整图。
- Target Forward 使用完整图。
- 循环次数与高低阈值分支仍由控制器调度。
- 每轮尽可能复用已捕获的 Graph。

避免动态控制逻辑破坏三个模型自身的 FULL Graph 支持。

不得因为启用了自适应迭代，就默认禁用 DFlash 或中间模型的图执行。

## 8. Graph Capture 正确性要求

请重点检查以下问题：

- Capture 阶段是否存在不支持的动态算子。
- Graph Replay 是否读取了正确的输入 Buffer。
- Hidden States 是否在不同迭代间正确更新。
- KV Cache 的物理地址、Block Table 和逻辑位置是否一致。
- Padding Query 是否影响有效 Token 的 Attention 结果。
- 多 Batch 下各请求的有效长度是否正确。
- Query 长度超过 16 时是否进入了错误的 Decode Attention Kernel。
- 捕获过程中是否存在不允许的 CPU/GPU 同步。
- 多轮 Graph Replay 是否产生额外内存泄漏。
- 图捕获和重放是否出现未初始化 Buffer 或旧缓存复用。
- 不同 Chunk 的 Query 是否真正看到各自应有的 KV 前缀。

特别注意昇腾 FIA Attention 的因果 KV 长度处理：不能仅依据输出形状正确就认为 Attention 计算正确。

应通过与参考实现逐位置比较，验证真实 Attention 输出和 Target Logits。

## 9. 测试要求

### 9.1 三模型图捕获测试

分别验证：

1. DFlash Forward 能成功 Capture 和 Replay。
2. 中间稀疏层 Forward 能成功 Capture 和 Replay。
3. Target Forward 能成功 Capture 和 Replay。
4. 三个 Forward 能在同一多轮解码流程中正常调用。
5. 实际记录到 Graph Replay，而不是全部执行 eager。

### 9.2 长候选序列测试

至少测试：

`1、7、15、16、17、24、28、32、48、64`

验证：

- 所有候选 Token 均得到正确处理。
- 不因长度超过 16 而报错。
- 不发生 Token 丢失或重复。
- 正确处理 Chunk 边界。
- 每个位置的 Logits 与参考实现一致。
- 最终接受长度正确。
- 修正 Token 和 Bonus Token 正确。
- KV Cache 状态正确。
- FULL Graph 路径实际生效。

如果某些长度超出用户设置的生成预算或上下文容量，应调整测试环境，而不是静默跳过。

### 9.3 Graph 与 Eager 一致性

对比相同输入下：

- DFlash Graph / Eager 输出。
- Sparse Verifier Graph / Eager Logits。
- Sparse Hidden States Graph / Eager。
- Target Graph / Eager Logits。
- Target 长 Query / Chunk Forward 参考结果。

检查数值误差，并在 temperature=0 时验证严格校验的最终生成结果。

需要考虑 BF16/FP16 的合理数值容差，同时检查接近 argmax 边界的 Token，不能只验证平均误差。

### 9.4 八种模式回归测试

在 FULL Graph 下测试：

- Fixed N + Strict
- Fixed N + Top-2
- Fixed N + Confidence
- Fixed N + Accept All
- Adaptive N + Strict
- Adaptive N + Top-2
- Adaptive N + Confidence
- Adaptive N + Accept All

每一种模式至少覆盖短候选和超过 16 Token 的长候选。

### 9.5 性能测试

分别统计：

- DFlash Graph Capture 时间。
- Sparse Verifier Graph Capture 时间。
- Target Graph Capture 时间。
- 各模型 Graph Replay 时间。
- Target 长候选分块数量。
- 长候选 Target Forward 耗时。
- 完整 Target Verification 耗时。
- 图缓存命中率。
- 图捕获带来的额外显存开销。
- Eager 与 FULL Graph 的吞吐量、TPOT 对比。

测试时应区分 Graph Capture 的初始化开销与稳定运行阶段的 Replay 开销。

## 10. 配置和运行示例

不增加任何图相关的自定义配置。

直接沿用上游接口：

```python
from vllm import LLM
from vllm.config import CompilationConfig

llm = LLM(
    model="/home/data/Qwen3-8B",
    tensor_parallel_size=1,
    max_model_len=4096,
    max_num_seqs=16,

    speculative_config={
        "method": "dflash",
        "model": "/home/data/Qwen3-8B-DFlash-b16",
        "num_speculative_tokens": 7,
    },

    compilation_config=CompilationConfig(
        cudagraph_mode="FULL",
    ),

    additional_config={
        "dflash_sparse_verify": {
            "enabled": True,
            "num_intermediate_layers": 12,
            "mode": "confidence",
            "num_rounds": 4,
            "threshold_high": 0.9,
            "threshold_low": 0.5,
            "feature_source": "sparse",
            "final_verification": {
                "method": "topk",
                "topk": 2,
                "target_threshold": 0.1,
            },
            "enable_metrics": True,
        }
    },
)
```

其中 `cudagraph_mode="FULL"` 是上游原生配置，并非新增配置。

需要按照本地版本的 `CompilationConfig` 接口适配。

也需要测试上游的 `FULL_DECODE_ONLY` 模式。

当用户未显式配置图模式时，遵循上游默认行为，不得自行修改默认值。

## 11. 最终交付要求

请实际修改代码，使 DFlash、中间稀疏层模型以及完整 Target 的 Forward 均能使用上游图模式执行。

特别重点：

1. 三个模型的 Forward 都真正完成图捕获和重放。
2. 图参数全部继承上游配置。
3. 多轮中间模型累计候选超过 16 Token 时，Target 仍可正确处理。
4. 长候选路径同样尽可能保持 FULL Graph。
5. 分块 Target Forward 不改变一次最终 Target 校验的逻辑。
6. 四种 Target 校验策略全部兼容长候选。
7. 固定和自适应两种中间迭代模式均兼容 FULL Graph。
8. 不破坏原版 DFlash 的运行与性能。

请首先检查本地代码中已有的 ACL Graph、Attention Backend 和 Speculative Decoding 实现，优先复用上游机制。

必须提供：

- 具体修改文件。
- 三个模型各自的图捕获入口。
- 长候选超过 16 Token 的处理方案。
- 图形状选择与缓存复用机制。
- 实际 Graph Capture/Replay 日志。
- Graph 与 Eager 的正确性对比。
- 长候选和八种策略的测试结果。
- FULL Graph 与 Eager 的性能对比。

不要只修改配置或声称已开启 FULL Graph，必须证明三个模型的 Forward 都实际使用了图执行。

请直接完成代码实现，而不是仅输出设计方案。