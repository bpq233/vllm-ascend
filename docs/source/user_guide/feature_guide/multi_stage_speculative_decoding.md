# MRV2 多级投机解码

原 DFlash 先生成 token，中间 verifier 校验这些 token，再由中间 DFlash 继续生成；最终候选通过原 MRV2 draft 缓冲区交给 Target。原 DFlash 实现和 Target 的提交、拒绝回退流程保持不变。

## 配置

启用 MRV2（`VLLM_USE_V2_MODEL_RUNNER=1`），使用同步调度（`--no-async-scheduling`）。例如：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /models/Qwen-8B \
  --no-async-scheduling \
  --max-model-len 4096 \
  --speculative-config '{"method":"dflash","model":"/models/DFlash-8B","num_speculative_tokens":15}' \
  --additional-config '{
    "multi_stage_speculative": {
      "primary_num_speculative_tokens": 4,
      "intermediate": {
        "verifier": {"model": "/models/Qwen-4B"},
        "drafter": {"model": "/models/DFlash-4B"},
        "num_rounds": 3,
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

- `primary_num_speculative_tokens`：原 DFlash 的生成长度。保留原配置的长度即可；省略时等于最终候选容量。
- 原 `speculative_config.num_speculative_tokens`：最终候选容量，供原 scheduler、Target KV 和缓冲区预分配使用。中间候选最多占用这一容量，实际长度单独返回 scheduler。
- `intermediate.num_rounds`：中间校验次数。3 表示“校验原 draft → 中间 draft → 校验 → 中间 draft → 校验”，不会把最后一批未经中间校验的 draft 追加进候选。
- `intermediate.num_speculative_tokens`：每次中间 DFlash 的生成长度。
- `intermediate.max_num_seqs`：中间模型每个小批次的请求数，默认 4；同时受 token 预算限制。
- `intermediate.max_model_len`：独立 KV 的上下文容量，默认继承 Target；超过该容量的请求直接回到 Target 解码。

每轮保留连续接受前缀，并追加中间 verifier 的 greedy 修正或 bonus token。完全接受时，最大候选长度为 `primary_length + (num_rounds - 1) * secondary_length + num_rounds`；示例为 15。容量更小时会截断，遇到 EOS 或请求长度上限也会停止扩展。

## 接受策略

中间和最终校验共用独立 `AcceptancePolicy`：

- `{"method": "topk", "top_k": 5}`：遇到第一个不在 Top-k 的 token 时截断。
- `{"method": "all"}`：全部接受；最终 Target 仍生成 bonus。

最终 Target 的修正/bonus 复用原 sampler，包括请求采样参数；Top-k 使用经过采样约束处理的 logits。中间修正/bonus 和中间 DFlash 使用 greedy。Top-k 和全接受均属于近似策略，不保证严格投机解码的分布等价性。

仅设置 `final_verification` 可以单独替换最终接受策略。完全移除 `multi_stage_speculative` 即恢复原流程。`num_rounds=0` 不加载中间模型，此时 primary 长度必须与原 speculative 长度一致。

## 状态与 KV

中间模型使用独立模型配置、attention metadata、KV 张量和 RoPE 缓冲区，不访问 Target 或原 DFlash 的 KV。每次中间 forward 从位置 0 批量重算传入前缀，旧尾部不在当前 attention 长度内；拒绝后和下一次 Target 提交后均从新的有效前缀重建。因此中间接受 token 不会写入正式请求历史，不需要额外维护跨轮 rollback 状态。

原 Target 的 `postprocess_sampled` 提交结果后，适配层才读取请求历史并处理原 draft。适配层只替换返回的 draft tensor 和交给 scheduler 的候选列表；Target 最终拒绝时仍使用原计数和 KV 回退流程。

首版支持文本、full-attention、未量化的中间 verifier，复用 TP；不支持中间流水线的 PP/DP/CP、LoRA、异步调度或 adaptive verification。中间模型使用 eager 模式，Target 和原 DFlash 保留原图模式。scratch KV 在加载期分配并计入内存预算，中间激活也参与 profile。重算完整前缀有明显开销，本实现优先保证状态正确，尚未提供真机吞吐收益数据。

## 验证

CPU 测试不依赖安装完整 vLLM/NPU 环境：

```bash
python -m pytest --confcutdir=tests/ut/worker/v2 \
  tests/ut/worker/v2/test_final_verification.py \
  tests/ut/worker/v2/test_intermediate.py \
  tests/ut/worker/v2/test_intermediate_backend.py
```

NPU 集成测试使用仓库已有的 Qwen3-8B/DFlash 配对作为主、中间两套独立实例，覆盖 Top-1 对照、全接受、不同请求长度和请求结束后复用。实际 8B/4B 配对仍需使用对应权重执行验证：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 pytest -q \
  tests/e2e/nightly/single_node/spec_decode/test_multi_stage_dflash.py
```

接口核对参考：KG `vllmascend_docs_source_userguide_featureguide_speculativedecoding_vllm_ascend`，`source_file=inference-serving/vllm-ascend/docs/source/user_guide/feature_guide/speculative_decoding.md`，score `0.923520`；独立 KV 工作流参考 KG `model-infer-kvcache`（直接加载，无 score）。中间辅助隐藏层沿用 [上游 DFlash/EAGLE3 层选择接口](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py)。
