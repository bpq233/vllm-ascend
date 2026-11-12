# Inter.md 实现与验证记录

## 2026-11-12：修复变长草稿与调度器宽度传输的适配

第二份真机日志在最终校验中报“NaN/+Inf/全 -Inf/非法候选 ID”的合并异常，
单独该堆栈不能区分数值问题与非法 ID。核对锁定上游后确认了一个必须修复的问题：
普通 `DraftTokensHandler` 在无结构化输出时仅传输张量宽度。稀疏多轮候选用 -1
补齐到 K×N，沿用该处理器会把补齐槽位也调度给 Target。

新增 `sparse_verify/draft_tokens.py`，仅为实验路径安装 `SparseDraftTokensHandler`，
按每请求真实连续候选前缀上报宽度，并复用原生请求生命周期和占位计数传输。
实际 Token 仍存于 RequestState。回归覆盖有效长度 0/1/7/17/28/60、零预算、
不同请求及非法非连续前缀。最终校验仍严格拒绝无效数值，同时细分异常诊断并附请求信息。
本地完整测试目录结果：**119 passed，316 subtests passed**；Ruff 检查通过。
这些改动未证明服务器上的异常全部消除；同步后仍需真机重跑确认。

需同步的运行时代码：`worker/v2/model_runner.py`，以及
`worker/v2/spec_decode/dflash/sparse_verify/` 下的 `draft_tokens.py`（新增）、
`decisions.py`、`sampling.py`。新增测试 `test_draft_tokens.py` 并扩充 `test_sampling.py`。

## 2026-11-12：修复首次真机反馈的 Anchor 形状错误

用户提供的 Qwen3-32B、TP=2、FULL_DECODE_ONLY 日志显示，首次草稿扩展在
`int(anchors[row])` 处触发 TypeError。当前上游 RequestState 将 `last_sampled_tokens`
存成 `[max_num_reqs, 1]`，原代码误按一维处理，`.tolist()` 返回嵌套列表。

`sparse_verify/speculator.py` 现在将二维单列转换为一维视图，同时兼容已有一维输入；
其他形状明确报错。内部各轮使用本地副本更新 Anchor，不修改 Target 请求状态。
新增 `test_speculator.py` 执行生产 propose 方法和真实多轮控制器，覆盖双请求、两轮、
固定/置信度模式。已确认旧版本复现同一 TypeError，修复后完整目录 **115 passed，
314 subtests passed**，Ruff 与 diff 检查通过。服务器修复后的 NPU 重跑尚未执行。

日志中的 `Sparse Target forward used NONE ... prefill=True` 属于 FULL_DECODE_ONLY
对 Prefill 的回退提示，不是本次退出原因；不能据此认定 Decode 图重放已成功。

日期：2026-11-12。工作目录：`vllm-ascend`。

已实际修改原始代码，新增默认关闭的稀疏层多轮 DFlash 实验路径。
固定轮数、双阈值自适应、四种最终策略、独立缓存、原生图管理器接入及长 Query
元数据路径均有实现。**当前仅完成 CPU 参考和接口验证，尚未完成 NPU 验收。**
没有使用旁边的 backup 或旧层次化投机实现。

## 修改文件与职责

| 文件 | 实际修改 |
| --- | --- |
| `vllm_ascend/platform.py` | 在调度器与图尺寸初始化前校验配置，预留 K×N 候选容量。 |
| `vllm_ascend/ascend_config.py` | 将新配置交给专用校验器，避免通用配置拒绝该键。 |
| `vllm_ascend/worker/v2/spec_decode/__init__.py` | 仅 enabled 时选择新的 Speculator。 |
| `vllm_ascend/worker/v2/model_runner.py` | 接入统一最终采样，拒绝不支持的请求参数，记录真实 Target 事件时间及捕获计数。 |
| `vllm_ascend/attention/attention_v1.py` | 实验路径支持有因果 KV 端点的虚拟 Query 行，不截断长候选。 |
| `vllm_ascend/compilation/acl_graph.py` | 新增作用域图资源所有者，避免第二个模型图管理器与 Target 单例冲突。 |
| `vllm_ascend/worker/v2/aclgraph_utils.py` | 实际完成 Target 重放后记录计数。 |
| `vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py` | 实际完成 DFlash 重放后记录计数。 |
| `mkdocs.yml` | 注册新功能指南。 |

新增核心目录：`vllm_ascend/worker/v2/spec_decode/dflash/sparse_verify/`。

| 文件 | 职责 |
| --- | --- |
| `config.py` | 类型、范围、模式及独立最终阈值校验。 |
| `selection.py` | 必选层转换及确定性的均匀选层。 |
| `model.py` | 共享权重、独立 Attention/KV 所有者、稀疏层执行、Norm/LM Head 和辅助状态输出。 |
| `controller.py` | 每请求临时候选、轮次、阈值、EOS、预算和特征来源控制。 |
| `decisions.py` | 稀疏双阈值决策、四种最终策略、连续前缀和修正/Bonus。 |
| `integration.py` | 提前容量扩展、原始 DFlash K 配置副本和请求约束。 |
| `speculator.py` | 复用原始 DFlash propose，接入多轮控制、缓存预填充和事件计时。 |
| `runtime.py` | 私有输入与 Block Table、原生 Metadata、稀疏模型捕获/重放及明确回退。 |
| `long_query.py` | 长 Query 分成最多 16 Token 的虚拟 Attention 行，保留全部输出。 |
| `sampling.py` | 对完整累计候选统一决策，输出原生 SamplerOutput；accept-all 只计算 Bonus LM Head。 |
| `metrics.py` | 有界明细、累计计数、临时/最终接受区分和图执行证据。 |
| `__init__.py` | 核心组件导出。 |

新增交付文件：

- [完整配置、Python 与 Serve 使用指南](docs/source/user_guide/feature_guide/dflash_sparse_verify.md)。
- [可运行离线推理脚本](examples/offline_inference/dflash_sparse_verify.py)。
- [Benchmark 矩阵脚本](benchmarks/dflash_sparse_verify.py)。
- [NPU 端到端测试](tests/e2e/pull_request/one_card/model_runner_v2/test_dflash_sparse_verify.py)。
- `tests/ut/worker/v2/dflash_sparse/test_core.py`：配置、选层、数值与四种策略。
- `test_model.py`：真实 CPU 因果 Attention、RoPE、M=L、增量 KV、覆盖拒绝后缀及共享权重。
- `test_controller.py`：固定/自适应及八种组合、长候选、零接受、位置和来源。
- `test_long_query.py`：全部指定长度逐位置 Attention 参考比较。
- `test_integration.py`：提前容量扩展、幂等初始化和非法配置。
- `test_sampling.py`：真实最终采样适配器、对齐、Bonus-only LM Head、空 Prefill。
- `test_runtime.py`：运行时接口、私有设备指针、动态输入与图 Padding。
- `test_graph_scope.py`：生产图资源作用域函数的隔离和异常恢复。
- `test_attention_metadata.py`：执行真实 FIA Metadata Builder 方法，验证长 Query 与 Padding 接入。
- `test_target_graph.py`：执行真实 Target 图工厂和可变 Query Padding 方法，验证实验路径启用 varlen。
- `release_notes/fragments/0-dflash-sparse-verify.yaml`：本地发布说明占位；提交 PR 时需要真实编号。

## 选层、特征和执行流程

选层从已加载 Target 的 `aux_hidden_state_layers` 取得实际捕获边界。
锁定上游将 DFlash `target_layer_ids` 加一转换为该边界；边界 c>0 对应必选原始层 c−1，
零边界表示 Embedding。选层保留必选集合，以最小平方间距选择其余层；M=L 执行全部原始层。
实际 checkpoint 不在本机，不能提供其真实选层。已执行的 L=32/M=12/必选 `[0,15,30]`
测试得到 `[0,2,4,6,9,12,15,18,21,24,27,30]`，不是硬编码配置。

初始 DFlash 使用刚完成 Target 前向的真实有效特征。后续 sparse 模式使用对应选中层
重新计算的近似特征，携带层序、形状、Token 身份和绝对位置；只传入新 Anchor 之前
连续有效的特征。target 模式缺少新前缀的真实特征时交回 Target，绝不冒用旧位置。

固定模式每轮生成 K 个候选，稀疏 argmax 保留连续前缀；零接受也消耗轮数。
自适应模式高阈值接受、中阈值重试、低阈值保留当前待验块并终止扩展。
两者最后都把累计候选交给调度器的下一次 Target 前向和统一最终决策。
strict 与 Target 贪婪输出的一致性是验收目标；其他三种策略明确属于近似生成。

缓存由上游内存规划分配，稀疏 Attention 是独立所有者，所有参数共享原始存储。
私有 Block Table 同时重建设备指针数组；仅换 Python 列表会错误写入 Target 缓冲区。
后缀失效由后续实际序列端点和覆盖写实现，原始 Target 缓存不被中间轮次提交。
最终拒绝计数为 `候选数−最终接受数`，复用上游提交与延迟 KV 生命周期。

## 图入口与长候选

| 模型 | 捕获/重放入口 |
| --- | --- |
| DFlash | 原 `AscendDFlashSpeculator.capture` → `DFlashAclGraphManager.capture/run_fullgraph`。 |
| 稀疏模型 | `SparseForwardRunner.capture/_forward` → 原生 `ModelAclGraphManager.capture/run_fullgraph`。 |
| Target | 原 `NPUModelRunner.capture_model/execute_model` → 原生 Target 图管理器。 |

全部沿用上游 compilation_config，不新增图开关或捕获尺寸。稀疏模型捕获包含
选中层、归一化、LM Head 与辅助状态；图资源作用域独立。动态 Python 控制留在图外。
显式捕获尺寸不被覆盖；缺少形状或上下文末端无法安全 Padding 时明确记录原生回退。

长 Query 保留所有 Token；每个虚拟行共享逻辑请求的 Block Table，KV 可见端点严格
限制到历史前缀加本块末尾，从而保留完整因果语义。它不产生多个接受/输出提交步骤。
CPU Float64 测试逐位置比较长度 `1/7/15/16/17/24/28/32/48/64`，含异长 Batch、
多个历史前缀长度和拒绝后缀。CANN 真机上的虚拟行与图参数更新仍待验收。

## 实际验证记录

本机：Windows、Python 3.11、PyTorch `2.15.0+cpu`；未安装 `vllm` 或 `torch_npu`，
没有可用 checkpoint/NPU。

| 检查 | 结果 |
| --- | --- |
| 新增 CPU 测试目录 | 112 passed，314 subtests passed。 |
| NPU 端到端用例收集 | 24 skipped：未设置真实模型路径。不是运行通过。 |
| 33 个修改/新增 Python 文件编译 | 通过。 |
| Ruff check 与 format check | 通过。 |
| 仓库 forbidden-imports、boolean-context-manager、long-functions 检查 | 通过。 |
| Python package `__init__.py` 检查 | 通过。 |
| 两个 CLI 的 `--help` | 通过。 |
| `bash format.sh ci` | 未通过环境初始化：GitHub ruff-pre-commit fetch 连接被重置。 |
| 真机 Capture/Replay、BF16/FP16、端到端严格一致性、Serve | 未执行。 |
| 吞吐、TPOT、图显存开销、真实性能对比 | 未执行；没有虚构结果。 |

测试 XML 保存在相邻工作区 `../proposal_sampling_results/dflash_sparse_cpu.xml` 与
`../proposal_sampling_results/dflash_sparse_npu.xml`。重新执行命令：

```bash
python -m pytest -q --confcutdir=tests/ut/worker/v2/dflash_sparse tests/ut/worker/v2/dflash_sparse
python -m pytest -q --confcutdir=tests/e2e/pull_request/one_card/model_runner_v2 \
  tests/e2e/pull_request/one_card/model_runner_v2/test_dflash_sparse_verify.py
```

## 尚未解决的限制与验收风险

- **[未核实] 昇腾真机兼容性**：KG 健康检查超时，按技能要求未继续请求；采用本地代码与
  `.github/vllm-main-verified.commit` 锁定的上游 `24420b7a341af37a9bc3186e6fc6062512e0aa58`
  源码核对接口。未获得真实 CANN/硬件版本，因此不能确认内核或图执行效果。
- 实验路径仅限 dense Qwen3、原始 DFlash、temperature=0、普通 TP 与多请求。
  PP/DP/PCP/DCP、LoRA、KV 传输、Prefix Cache、量化 KV、专用 LM-head TP、KVPP/Offload、
  watermark、logprobs、penalty/constraint、min_tokens 均明确拒绝。
- Batch 内请求逐个执行中间循环；控制标量回读、启用统计时的事件同步有性能成本。
  没有证明性能优于原始 DFlash。额外 KV、Logits 与图输出缓冲区有显存成本。
- PIECEWISE 的稀疏模型需要上游 breakable wrapper，否则原生管理器明确报错；
  DFlash 延续上游 PIECEWISE 不支持的兼容行为。不能声称所有图组合已真机通过。
- native DFlash 每次仍执行完整 K 个 Mask，接近上下文末端时安全停止扩展；未通过非法位置
  或单纯切片返回值掩盖超范围 Kernel。严格的部分 Mask 执行优化尚未实现。
- Accept-all 为避免额外 LM Head，不提供反事实严格接受长度。图捕获耗时目前只在初始化
  总耗时中体现；独立三个模型捕获耗时、图缓存命中率及图增量显存仍需补充真机剖析。
- 必须在真实 NPU 上完成三模型 Capture/Replay、所有长度、八策略、Graph/Eager 逐位置
  比较和严格输出一致性，才能按 Inter.md 的硬件要求验收。当前代码不能视为已通过该验收。
