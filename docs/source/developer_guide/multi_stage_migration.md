# 层次化投机迁移记录

本次将 `vllm-ascend-backup` 中已经接通的层次化 DFlash 投机迁移到当前 MRV2，并保留三模型默认路径和可选四模型路径。配置和使用方式见[多级投机解码](../user_guide/feature_guide/multi_stage_speculative_decoding.md)。

## 对齐基线

| 项目 | 版本 |
| --- | --- |
| 旧版 Ascend 工作树基线 | `e013970baa18ae62a8f04f6dac2fb44a3c7cffbe` |
| 新版 Ascend 工作树基线 | `a7a1cc386dad45f53477bc11f3f221b1d70f6f82` |
| 新版 `.github/vllm-main-verified.commit` | `24420b7a341af37a9bc3186e6fc6062512e0aa58` |
| 新版 release 兼容标签 | `v0.32.0` |

本机没有安装完整 vLLM、torch_npu、CANN、模型权重或 NPU。CPU 验证使用 Python 3.11、PyTorch `2.10.0+cpu`、NumPy 和 pytest；通过加载实际业务模块或提取实际方法、替换外部模型与设备边界执行测试。因此，CPU 结果不能证明完整 vLLM 启动、NPU 图捕获或性能达标。

## 接口适配

- 提前推导最终候选容量，再交给平台默认值、scheduler 和图尺寸初始化。
- 将请求重排从旧 `prepare_inputs` 移到 `gather_batch_req_state`：先保留短 query，再处理长验证 query；`BatchReqState` 所有逐请求 NumPy 数组一起重排，维持请求、slot、prefill 和 query 数据对应。
- 复用新版 `_prepare_inputs_impl`、输入 metadata 和进度缓存生命周期，移除 `0.27.1` 专用分支。
- 中间 KV 使用当前分配描述的 `layers`、`layer_stride`、`block_stride`，复用 `AscendBlockTables` 和新版初始化接口。
- DFlash 提案转发物理 KV block size，区分其与 attention kernel page size；将 DP dispatch 返回的状态转换为模型生成入口需要的 token-count 数据。
- 最终校验转发 `seq_lens_upper_bound_np`，支持独立的 `verify_draft_sampled`，保留上游 trace replay 钩子。
- 草稿发布器实现新版 `begin_step`、`discard`、`set_draft_tokens` 与 `get_draft_tokens`，保留未调度请求的候选，并按当前 deferred grammar 请求范围返回结果。
- `FULL_DECODE_ONLY` 的长验证使用实际 mixed FULL manager，保持真实累计 query 边界，仅在图 padding 存在时新增 dummy request。

## 本次修复与代码清理

- 独立三模型路径关闭 Target 的 auxiliary hidden 输出，避免错误地将中间 DFlash 绑定到 Target。
- 中间 DFlash 图重放跳过已有 context KV 的再次投影与写入，仅刷新设备 metadata。
- 无效 token ID（包含 `-1` placeholder）不参与 logits 索引或接受；首个无效候选终止该请求连续接受前缀。
- 候选数量遵守新版 runner 传入的预算；Primary 的单轮预算与最终候选容量独立，零预算仍同步正式进度。
- 已结束或新加入的请求 ID 会清理旧 host history、中间 KV 和 hidden；相同 ID、相同长度的新 prompt 不会命中旧请求缓存。
- 针对未接入的动态长度、非标准拒绝采样、block/entropy 验证、watermark、LM-head TP、KV 传输/PD 以及 DFlash2/LiLiCorr 组合，在初始化阶段明确报错。
- 稳定 metadata 与 KV 页表复用缓存，避免重复小传输和 gather；进度与现有同步点共用，保留新版普通投机和 PP 分支。
- 删除旧版本兼容分支和未被启用的生产死路径。保留异步 H2D、图固定输入写入及 CPU 控制循环所需复制，避免破坏数据所有权。

## 未完成的旧 exact 草稿

旧工作树有未跟踪的 `multi_stage/exact_sampling.py`、`test_exact_acceptance.py`、`test_exact_pipeline.py`、`test_exact_sampling.py`、`test_multi_stage_exact_config.py` 和 `examples/offline_inference/multi_stage_exact_benchmark.py`。它们引用的部分 exact 校验器与流水线接口尚不存在，未形成可运行的旧功能。本次不将这些未接通代码加入新版生产路径或常规测试；源文件未修改，完整保留在 `vllm-ascend-backup`。

`topk`、`all` 和 `prob_ratio` 均为近似策略，不保证采样分布与 Target 等价。未来接通 exact 模式时，需保留逐级提案分布，验证残差拒绝采样、独立 RNG、不同 batch/轮数和采样参数下的分布一致性。

## 验证命令与范围

2026-11-06 初次迁移结果：18 个测试文件共 **452 项测试、231 项子测试通过**。全部 38 个变更 Python 文件的 Ruff、格式和编译检查通过，仓库源码规则与三份新增 Markdown 文档检查通过。旧工作树保持原样。

在具备 CPU PyTorch 的环境中，从仓库根目录执行：

```bash
python -m pytest --noconftest -q \
  tests/ut/worker/v2/test_dummy_fia_metadata.py \
  tests/ut/worker/v2/test_intermediate*.py \
  tests/ut/worker/v2/test_multi_stage*.py \
  tests/ut/worker/v2/test_final_verification.py \
  tests/ut/worker/v2/test_long_verification.py \
  tests/ut/worker/v2/test_verification*.py \
  tests/ut/worker/v2/test_target_metadata_cache.py \
  tests/ut/attention/test_fia_input_contract.py
```

测试覆盖接受策略、ragged batch、EOS、长度/预算边界、请求重排与复用、候选设备发布、KV 淘汰、图 metadata、stream 依赖、FIA 输入契约与新版 API 转发。整数索引和移动结果要求精确相等。

源码检查使用 `ruff check`、`ruff format --check`、Python 编译检查、仓库禁止导入/布尔上下文管理器/长函数检查和 Markdown 检查。`bash format.sh ci` 在本机 WSL 因工作树 CRLF 换行无法启动，未运行完整 pre-commit 检查；没有将该失败计作通过。PR 编号尚未分配，发布说明先保存在 `release_notes/drafts/multi-stage-speculative.yaml`，提交 PR 时按模板填写编号并移入 `fragments/`。

以下集成验证需要匹配的模型权重与 NPU 环境，本机尚未执行：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 python -m pytest -q \
  tests/e2e/nightly/single_node/spec_decode/test_multi_stage_dflash.py
```

该文件覆盖 eager、FULL、长 query 的 FULL_DECODE_ONLY、独立/Primary 两种路径、最终 Top-1 与 greedy Target 对照、FIA FP32 因果参考和请求复用。`all`/`prob_ratio` 检查结构与长度，不宣称与 greedy Target token 等价。

原用户指南中的 CPU profiler 计数和性能观察来自旧实现历史记录，本次未重新测量。新版吞吐、TTFT、TPOT、NPU 内存与图资源需使用相同 prompts、采样和并发对照普通 Target/DFlash，记录设备、软件版本、TP 和上下文长度后评估。

KG 检索来源与执行约束见[迁移知识核对附录](multi_stage_migration_kg.md)。

## 启动错误：中间 KV layout 尚未解析

用户在 NPU 上反馈 `backend._init_scratch → init_kv_cache → get_resolved_kv_cache_layout` 抛出 `KV cache layout has not been resolved yet`。中间模型在 worker 加载模型时已经分配私有 KV，而主模型的 layout 解析由 EngineCore 在 worker 模型加载完成后执行。之前的 CPU 分配替身未调用该 accessor，漏检了这一前置条件。

修复在中间 attention backend 初始化后，汇总 verifier 与 secondary drafter 支持的布局，调用上游 `resolve_kv_cache_layout`，传入全部中间 KV 规格并写入私有 `KVCacheConfig.kv_cache_layout`，再分配 KV 和图资源。配置保持独立，不提前解析或覆盖 Target 的布局；上游继续处理显式布局和不兼容组合。[上游解析规则](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/v1/attention/backends/utils.py#L277)，[EngineCore 初始化顺序](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/v1/engine/core.py#L240)。

回归测试的分配替身实际调用 `get_resolved_kv_cache_layout()`，覆盖未解析布局、已有布局、verifier/drafter 规格传递、Target 配置隔离，以及不兼容布局在内存分配前报错。本次修复后完整 CPU 回归结果为 **455 项测试、231 项子测试通过**，变更 Python 文件的 Ruff、格式和编译检查通过，文档检查通过。该修复仍需同步到 NPU 机器并重启验证。
