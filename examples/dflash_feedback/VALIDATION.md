# 本机验证记录

日期：2026-11-12（Asia/Shanghai）。工作目录：`vllm-ascend/`。

## 已完成

```text
python examples/dflash_feedback/test_feedback.py -v
Ran 13 tests in 0.145s — OK

python -m ruff check vllm_ascend/dflash_config.py vllm_ascend/ascend_config.py \
  vllm_ascend/worker/v2/model_runner.py \
  vllm_ascend/worker/v2/spec_decode/dflash examples/dflash_feedback
All checks passed!

python -m compileall -q examples/dflash_feedback \
  vllm_ascend/worker/v2/spec_decode/dflash \
  vllm_ascend/worker/v2/model_runner.py vllm_ascend/ascend_config.py vllm_ascend/dflash_config.py
成功，无输出

git diff --check
成功，无输出
```

`run.py --help`、`compare.py --help` 可执行。
分析器对不存在实验结果的目录返回明确错误，不生成 CSV 或图表。
测试运行时间仅属于 CPU 单元测试，不是模型性能结果。

13 项测试覆盖：空历史及 observe argmax 等价、归一化/并集去重/EMA/完整排名、首拒截断、历史门槛及延迟生效、多请求重排和清理、位置错配、初始化、配置校验、在线 Greedy oracle、off 委托父类、一次 Target logits 计算与校验先于更新、调度器截短、嵌套 additional_config schema、旧环境变量不生效。
其中部分检查合并于同一个测试方法。

## 未完成

- 未安装 vLLM，未能导入实际运行时核实继承和签名；分析使用本地 vLLM 源码参考。
- CPU PyTorch `2.15.0+cpu`，无 `torch.npu`，未进行实际 NPU 编译或推理。
- 模型级 Baseline/off/observe/Target Only 一致性及 batch=1/4 待 NPU。
- GSM8K/HumanEval 的 20/200 条实验均未运行，没有接受率、Target Forward 次数或吞吐收益结论。
- `bash format.sh ci` 因原有脚本 CRLF 换行导致 Bash 解析失败；尚未完成整个仓库的检查套件。
- ascend-kg 更新脚本同样因 CRLF 无法执行；KG 健康检查 TLS 握手超时，依赖 KG 的兼容性结论为 [未核实]。

## 修改范围

| 文件 | 作用 |
| --- | --- |
| `vllm_ascend/ascend_config.py` | 登记 additional_config.dflash_calib_config 嵌套配置 |
| `vllm_ascend/dflash_config.py` | 模式及超参数的共享 schema，校验范围/类型及未知字段 |
| `vllm_ascend/worker/v2/spec_decode/dflash/speculator.py` | Eager 配置守卫及 Draft argmax 前插入 |
| `vllm_ascend/worker/v2/spec_decode/dflash/feedback_calibrator.py` | 稀疏 EMA、有效前缀对齐、反馈日志及 Target 采样分发 |
| `vllm_ascend/worker/v2/model_runner.py` | 实验采样入口及请求结束清理 |
| `examples/dflash_feedback/run.py` | 单模式 benchmark |
| `examples/dflash_feedback/compare.py` | 独立进程对比及输出/候选一致性检查 |
| `examples/dflash_feedback/worker_metrics.py` | 仅 benchmark 使用的 Target 采样计数及接受轨迹 |
| `examples/dflash_feedback/prepare_prompts.py` | 固定顺序公共数据集输入 |
| `examples/dflash_feedback/analyze.py` | CSV、JSONL 分析及四类图 |
| `examples/dflash_feedback/test_feedback.py` | 独立 CPU 单元测试 |
| `examples/dflash_feedback/README.md` | 调用链、参数、命令与限制 |
| `examples/dflash_feedback/VALIDATION.md` | 本验证记录 |
| `release_notes/fragments/0.yaml` | 新特性的临时 release note，创建 PR 后改成实际编号 |
