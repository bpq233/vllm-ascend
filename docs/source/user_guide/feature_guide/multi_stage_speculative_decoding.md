# MRV2 多级投机解码

原 DFlash 先生成 token，中间 verifier 校验这些 token，再由中间 DFlash 继续生成；最终候选通过原 MRV2 draft 缓冲区交给 Target。原 DFlash 实现和 Target 的提交、拒绝回退流程保持不变。

实现统一放在 `vllm_ascend/worker/v2/spec_decode/multi_stage/`：`adapter.py` 接入原 DFlash，`config.py` 管理配置，`pipeline.py` 控制迭代，`backend.py` 包含中间模型、KV/hidden 缓存及图状态，`acceptance.py` 与 `final_verification.py` 管理接受策略和最终验证。包入口保持轻量，配置校验时不会提前加载模型模块。原来的独立 cache/graph 辅助文件已合并，不保留重复实现；部署更新需要同步新目录和引用路径修改。

## 热路径审计与传输边界

调用链为：CPU scheduler 生成请求和候选长度 → Target 图执行及设备端采样/提交 → Primary DFlash 图生成候选 → `adapter._read_step` 回传有界历史增量 → `pipeline.refine` 驱动中间轮次 → `backend.verify_batches` 增量 KV 验证 → `_decide` 设备端验收并回传决策 → Secondary DFlash 图生成下一轮候选 → adapter 发布 CPU 候选并写入设备 draft buffer → 下一步 Target 验证。中间 KV 按请求隔离、核对前缀并截断分歧后缀；它不提交正式 Target 状态。

| 优先级 | 位置及频率 | 成本与处理 |
|---|---|---|
| P0 | `adapter._read_step`，每个外层 decode step | 长度、Primary 和历史尾部一次 D2H；冷请求原来逐请求读完整历史，现在额外合并一次 D2H。热请求仍保留一次同步。 |
| P0 | `pipeline._decide` / `backend.propose`，每个中间 microbatch | 决策和 Secondary token 仍需 D2H，供 CPU 活跃请求控制及 KV 前缀记录使用；尚未实现全设备控制循环。 |
| P0 | `FinalVerificationSampler._verify`，仅 DEBUG | 原来 event 显式同步加两次统计 D2H；现在统计合并一次阻塞 D2H，该传输也等待此前 forward 完成。普通路径没有这项回传。 |
| P1 | `pipeline._decide`，每轮 top-k 验收 | 删除从 Python draft 再构造 pinned tensor 并 H2D；直接使用当前 verifier 输入在设备上移位得到的 token。bonus 行被 mask，异长请求不会串入验收。 |
| P1 | `backend._forward`，每个中间 forward | pinned CPU buffer 中批量生成位置和 request row，不再逐 token 构造多层 Python 整数列表；整个输入仍一次异步 H2D。CPU `.numpy()` 是 pinned 内存视图。 |
| P2 | backend / adapter / final sampler，每步 | 固定 request offsets、padding、历史 offsets 和验收 steps 复用；正常输出采用 empty 后完整覆盖，dummy 输出仍清零；最终 top-k 使用返回值过滤非有限 logits，省去一次 vocabulary gather。 |
| P3 | 图、KV 和模型初始化 | 固定 buffer 分配保留，不计作 decode 同步；不增加图捕获档位和 stream。 |

复查范围包含 multi_stage 和 DFlash 中的 `.item/.tolist/.cpu/.numpy`、标量转换及 tensor 工厂调用。NumPy query 长度和 CPU metadata 上的转换不是 D2H；上游 DFlash 的 `seq_lens_cpu_upper_bound.max().item()` 同样操作 CPU tensor。DFlash replay 的 DP token-count tensor 创建仍保留，修改涉及原始 DFlash 通信上下文，应在 NPU 上验证其跨 stream 生命周期后再调整。

设备 token 只在 `verify_batches` 当前 yield 期间有效，恢复生成器后即清除引用，防止下一次图重放覆盖后误用。异步 H2D 的 pinned 源仍按调用独立分配，避免未经 event 保护的主机缓冲区复用造成数据竞争。CPU 控制循环、EOS/长度裁剪、前缀比较和 scheduler 发布仍存在；当前实现不能宣称全流水线已图捕获，也不能以 CPU 测试推断 NPU 加速比。

## 配置

启用 MRV2（`VLLM_USE_V2_MODEL_RUNNER=1`），使用同步调度（`--no-async-scheduling`）。例如：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /models/Qwen-8B \
  --no-async-scheduling \
  --max-model-len 4096 \
  --speculative-config '{"method":"dflash","model":"/models/DFlash-8B","num_speculative_tokens":20}' \
  --additional-config '{
    "multi_stage_speculative": {
      "primary_num_speculative_tokens": 4,
      "intermediate": {
        "verifier": {"model": "/models/Qwen-4B"},
        "drafter": {"model": "/models/DFlash-4B"},
        "num_rounds": 4,
        "num_speculative_tokens": 4,
        "max_num_seqs": 4,
        "max_model_len": 4096,
        "verification": {"method": "topk", "top_k": 5}
      },
      "final_verification": {"method": "topk", "top_k": 5}
    }
  }'
```

将示例路径替换为兼容的实际模型。中间 verifier 与 Target 必须具有相同词表和 token ID，中间 DFlash 必须与中间 verifier 匹配。

- `primary_num_speculative_tokens`：原 DFlash 的单轮生成长度。省略时，最终容量 ≤15 则沿用最终容量，最终容量 >15 则默认 4。显式设置超过 15 不会被自动截断，会提示具体字段和值。
- 原 `speculative_config.num_speculative_tokens`：最终候选容量，供原 scheduler、Target KV 和缓冲区预分配使用。中间候选最多占用这一容量，实际长度单独返回 scheduler。
- `intermediate.num_rounds`：中间校验次数。3 表示“校验原 draft → 中间 draft → 校验 → 中间 draft → 校验”，不会把最后一批未经中间校验的 draft 追加进候选。
- `intermediate.num_speculative_tokens`：每次中间 DFlash 的生成长度。
- `intermediate.max_num_seqs`：中间模型每个小批次的请求数，默认 4；同时受 token 预算限制。
- `intermediate.max_model_len`：独立 KV 的上下文容量，默认继承 Target；超过该容量的请求直接回到 Target 解码。

每轮保留连续接受前缀，并追加中间 verifier 的 greedy 修正或 bonus token。完全接受时，最大候选长度为 `primary_length + (num_rounds - 1) * secondary_length + num_rounds`；示例为 20。容量更小时会截断，遇到 EOS 或请求长度上限也会停止扩展。

## 长候选 Target 验证

原 DFlash 及中间 DFlash 的每次 proposal 长度仍不超过 15；只有最终候选容量允许超过 15。`max_num_batched_tokens` 至少为最终容量加 1，确保 scheduler 能容纳完整验证块。

最终候选的 query 包含一个已提交的末尾 context token 和 candidate，所以 15 个 candidate 对应 16 个 query token。短 query 保留原 attention 路径；超过 16 的 query 显式使用 `ChunkedPrefill`（paged cached-prefill/extend）。混合 batch 将短 query 排在长 query 前，保证 backend 的 decode/prefill 切分正确。metadata 的 Decode 分类门槛仍为 16，scheduler、采样缓冲区和 logits 数量仍使用完整候选容量。

cached-prefill 复用 `attention_v1` 的 paged FIA：Q 长度为累计 query 长度，KV 长度为各请求的有效总长度，`block_table` 指向已有前缀，右下因果遮罩使用 `sparse_mode=3`。同一 FIA 接口由实际 query 形状执行多 token prefill，未提高 Decode kernel 的上限，也不重新计算正式前缀。该路径限制为 full-attention Target 和未量化 Target KV；MLA、混合状态模型不在首版范围内。参数语义参见 [TorchNPU FIA 文档](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md)。

长容量配置在启用图时，Target 使用 PIECEWISE 分段图：动态 paged attention 留在图外，其余模型计算按捕获的 token 桶重放。编译配置在模型加载前设置 attention 分割边界；Primary DFlash 继续使用独立的短 query 图管理器。显式 enforce_eager/NONE 仍关闭图，容量不超过 15 的原配置保持原图行为。NPU 集成测试检查实际捕获、重放调用以及重复生成时没有新增捕获。

logits 仍通过原 `combine_sampled_and_draft_tokens` / `logits_indices` 选取：context 行预测 candidate 第一个 token，最后一个 candidate 行预测 bonus。拒绝后复用原 `num_rejected` 和 `postprocess_sampled` 更新有效 computed length；多算的尾部 KV 留在预分配 slot 中，但下一轮的长度和位置不会读取它，并在新 token forward 时覆盖。修正/bonus token 在下一轮 forward 才获得自己的 KV，不能把 sampled token 数直接当成已计算 KV 长度。

## 接受策略

中间和最终校验共用独立 `AcceptancePolicy`：

- `{"method": "topk", "top_k": 5}`：遇到第一个不在 Top-k 的 token 时截断。
- `{"method": "all"}`：全部接受；最终 Target 仍生成 bonus。

最终 Target 的修正/bonus 复用原 sampler，包括请求采样参数；Top-k 使用经过采样约束处理的 logits。中间修正/bonus 和中间 DFlash 使用 greedy。Top-k 和全接受均属于近似策略，不保证严格投机解码的分布等价性。

仅设置 `final_verification` 可以单独替换最终接受策略。完全移除 `multi_stage_speculative` 即恢复原流程。`num_rounds=0` 不加载中间模型，此时 primary 长度必须与原 speculative 长度一致。

## 状态与 KV

中间模型使用独立模型配置、attention metadata、KV 张量和 RoPE 缓冲区，不访问 Target 或原 DFlash 的 KV。按请求 ID 保留独立物理页和有效前缀；forward 仅计算变化的后缀及必要的最后一行预测 hidden state，并同步填充 Secondary DFlash 的 context KV。拒绝后比较 token 前缀，从分歧位置覆盖旧尾部；两套 KV 都写入成功后才提交有效长度。请求结束释放槽位，容量不足按 LRU 淘汰，同一批活跃请求受到保护。中间接受 token 不写入正式请求历史。

原 Target 的 `postprocess_sampled` 提交结果后，适配层才读取请求历史并处理原 draft。适配层只替换返回的 draft tensor 和交给 scheduler 的候选列表；Target 最终拒绝时仍使用原计数和 KV 回退流程。

支持文本、full-attention、未量化的中间 verifier，复用 TP；不支持中间流水线的 PP/DP/CP、LoRA、异步调度或 adaptive verification。启用图时，Target 和中间 verifier 使用分段图，Primary/Secondary DFlash 使用独立完整 query 图。中间 verifier 在模型加载后预捕获小 token 桶和覆盖完整 context 的几何桶；稳定输入缓冲区填入实际 query，padding 槽置为 -1，实际 KV 长度保持不变。中间图参数、更新流及 RoPE 与主模型隔离；缺少匹配图会报错，不静默回退 eager。显式 enforce_eager/NONE 用于关闭图作对照。图捕获与中间 KV 在加载期计入内存预算，中间激活也参与 profile。按缓存容量分组完成多轮，避免轮间反复淘汰；microbatch token 预算按实际新增 query 计算。尚未提供真机吞吐收益数据。

这里的四模型图执行不代表整个 Python 调度循环是单张完整图：两个 verifier 的动态 attention 仍在分段图边界外执行，候选列表、接受决策回传和调度控制也在图外。中间图首次捕获会增加加载时间和常驻内存。

遇到 `EE1023 / Alloc Stream resource failed / Too many streams are created` 时，需要降低总捕获桶数量，而不是增加候选 token 限额。中间层默认采用稀疏桶（例如 token buffer 为 4096、4 请求、DFlash 宽度 15 时为 `[1,16,64,256,1024,4096]`），不再捕获完整的 `1..32` 小桶。可在 `intermediate` 中设置 `"cudagraph_capture_sizes": [64,256]`；实现会补齐最大中间 token buffer 和 Secondary 最大批次桶，确保全部输入长度仍有图覆盖。更少桶会增加 padding 计算量，需要真机测量权衡。主模型的 `compilation_config.cudagraph_capture_sizes` 单独控制 Target/Primary，不能替代此中间层选项。捕获资源耗尽后应退出并重新启动该任务，不能在已报异步错误的进程内继续捕获。不要同时开启 `ASCEND_LAUNCH_BLOCKING=1` 与 ACL 图。

资源诊断依据：KG `runtime_docs_zh_faq_ee1023资源不足问题_too_many_streams_are_captured_to_the_acl_graph`，`source_file=cann-runtime/runtime/docs/zh/FAQ/EE1023资源不足问题.md`，score `0.943825`。该文档要求检查 stream 创建/销毁、设备上其他进程和共享资源占用；减少捕获桶是针对本实现的修正，不保证覆盖所有 EE1023 原因。

数据传输：正式历史在 CPU 按请求 ID 缓存，稳定 decode 每轮只将长度、Primary 草稿及最多 final_capacity+1 个尾部 token 合并为一次 D2H；首次请求、长度缩短或大步 prefill 才读取完整历史。中间输入 IDs、位置、页槽、长度和 query 边界合并为一次 pinned H2D，设备上生成派生 metadata；最终候选也使用 pinned H2D。接受决策保持每个 microbatch 一次小结果 D2H，仍需同步以驱动 CPU 迭代控制，尚不是全设备端流水线。

稳态热路径优化：CPU 正式历史和 KV token 记录只追加新增 token；KV 前缀按 1024-token 块比较，仅在分歧块逐 token 定位。中间验证全部 hidden 行都用于预测时直接计算 logits，省去索引 H2D 和 hidden gather；Secondary 使用常驻 anchor、计数、温度和 seed 缓冲区。中间接受判断缓存最多 16 种小形状 metadata，直接利用 top-k 返回值检查有效性，避免再次 gather 词表 logits；最终全接受路径跳过接受 mask 和首拒绝位置归约。先执行仍驻留缓存的请求，再将结果还原为原批次顺序，降低跨 Target 步的缓存淘汰。

稀疏图桶存在性能代价：只有 `[64,256,40960]` 时，4-token query 会补到 64，257-token query 原本会补到 40960。现在当 padding 超过实际 query 的 8 倍、且存在待计算的冷前缀和较小图桶时，先用已有小图分块补齐前缀，再批量验证预测行，不增加捕获桶。该保护以更多小图调用换取更少 padding，并非所有场景的实测最优值。debug 日志的 `executed_tokens`（包含 padding）与 `padding_tokens` 可与 `forward_tokens` 对比。

Verifier 仅保存预测短后缀的、已投影的 context hidden，复制到独立存储以防下一次图重放覆盖；不保存整段 prompt 激活。Secondary 所需的末尾行仍在缓存、且对应 KV 前缀完全一致时，直接复用该行，跳过额外 Verifier forward 和重复辅助层投影；缓存缺失、前缀变化或槽位回收时沿用正常模型计算。后端的 `reused_hidden_tokens` 记录跳过的 predictor 行数，NPU 集成测试检查其增长。以上优化保持现有接受策略；真实 NPU 精度、吞吐和图资源占用仍须验证。

每个中间小批次保持 packed logits，只进行一次 Top-k 和一次决策 D2H；slot mapping 按整个小批次向量化。模型依赖链上的 verifier → drafter 必须顺序执行，各阶段内部按 batch 计算。首版不跨轮复用中间 KV，因此实际收益需要结合上下文长度和接受率测量。

## 调试与性能

初始化记录三套 draft/intermediate 模型信息及轮数、策略。设置现有 `VLLM_LOGGING_LEVEL=DEBUG` 后记录：

- `multi_stage_intermediate`：round、proposed、accepted、accepted_by_request、intermediate_model_ms、secondary_model_ms。时间为包含组装、模型、必要 D2H 和接受策略的主机墙钟时间。
- `multi_stage_final`：candidate_lengths、accepted_lengths、verification_path、target_ms。accepted 为策略接受数（不含修正/bonus，后续 EOS/长度截断仍由原流程处理）；target_ms 使用 NPU event，包含 forward 区间，不含中间流水线。分块采样时按采样块输出长度，共用同一 forward 时间。

最终详细日志会同步 NPU；正式性能测试关闭 DEBUG，使用相同 prompts、输出长度和采样设置，对比普通 DFlash、15 容量及 20 容量配置的总吞吐、TTFT、TPOT，并记录 NPU 型号、CANN/torch_npu/vLLM 版本、TP 和上下文长度。不能用 CPU 耗时推断 NPU 加速比。

## 验证

CPU 测试不依赖安装完整 vLLM/NPU 环境：

```bash
python -m pytest --confcutdir=tests/ut/worker/v2 \
  tests/ut/worker/v2/test_final_verification.py \
  tests/ut/worker/v2/test_intermediate.py \
  tests/ut/worker/v2/test_intermediate_backend.py \
  tests/ut/worker/v2/test_intermediate_graph.py \
  tests/ut/worker/v2/test_long_verification.py
```

NPU 集成测试使用仓库已有的 Qwen3-8B/DFlash 配对作为主、中间两套独立实例，覆盖短/长候选、Top-1 对照、全接受、不同请求长度和请求结束后复用。长候选用中间全接受生成 20-token candidate，再以最终 Top-1 对照普通 greedy 输出。图用例断言四模型都有实际重放、中间 KV 命中、请求复用后没有新增图捕获。实际 8B/4B 配对仍需使用对应权重执行验证：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 pytest -q \
  tests/e2e/nightly/single_node/spec_decode/test_multi_stage_dflash.py
```

同文件另有无需模型权重的实际 FIA 测试：混合 1/16/17/33 query 长度、非连续物理 block 和不同前缀长度，对比 FP32 因果 attention 参考结果。它用于验证 cached-prefill 的位置和遮罩语义，仍需要 NPU、CANN 和 torch_npu。

接口核对参考：KG `vllmascend_docs_source_userguide_featureguide_speculativedecoding_vllm_ascend`，`source_file=inference-serving/vllm-ascend/docs/source/user_guide/feature_guide/speculative_decoding.md`，score `0.923520`；独立 KV 工作流参考 KG `model-infer-kvcache`（直接加载，无 score）。中间辅助隐藏层沿用 [上游 DFlash/EAGLE3 层选择接口](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py)。
