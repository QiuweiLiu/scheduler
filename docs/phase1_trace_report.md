# Phase 1 — Trace Adapter v0.1 真实采集验收

## 结论

Phase 1 的最小交付已完成：在用户指定的新远端上，使用未修改的 VideoSeek 固定 commit 和外层 Trace Adapter，连续完成 3 条真实 VideoSeek → Qwen API 运行。3 个 `trace.jsonl` 均通过 validator，manifest 与实际事件数一致，必需字段完整，未将 API 密钥写入产物。

这 3 条运行证明了 Trace Adapter 可以记录真实 action、远端 API 调用、状态、时间和 GPU 快照；它们仍然不能作为 144p 视频的准确率结论，也不能证明本地多 GPU 推理收益。

## 固定输入与环境

| 项目 | 已验证值 |
|---|---|
| 远端连接 | 用户指定的 `connect.westc.seetacloud.com:12469` |
| 远端容器 | `autodl-container-41d846924e-d62d0601` |
| 远端工作根目录 | `/root/autodl-tmp/scheduler` |
| VideoSeek checkout | `/root/autodl-tmp/scheduler/third_party/videoseek` |
| VideoSeek commit | `443f8710bd1f9023f4a52ac95cb83610ad29c5ac` |
| 工作树 | clean |
| 输入视频 | `/root/autodl-tmp/scheduler/data/phase0/lvbench/wgBlACG927Y.mp4` |
| 输入 SHA-256 | `adf60db34dad3d81b0a091045e356b1722c6d32d0139a5c2db2e4fee0f7e76e7` |
| 输入属性 | 256×144，25 fps，2331.423 秒，39,024,195 字节 |
| Python | `/root/miniconda3/envs/finetooling/bin/python`，3.10.20 |
| 模型 | `openai/qwen3-vl-plus` |
| API base | `https://dashscope.aliyuncs.com/compatible-mode/v1`（仅通过环境变量） |
| GPU | NVIDIA GeForce RTX 4080 SUPER，driver 595.58.03，32760 MiB |

API key 只在远端当前进程环境中使用，未写入 YAML、命令行参数、manifest、trace、日志或 Git。

## 三条真实 trace

| task_id | run_id | seed | 状态 | 耗时 (s) | 事件数 | 事件类型 | 错误 / 重试 | 预测 | validator |
|---|---|---:|---|---:|---:|---|---:|---|---|
| `phase1_run_01` | `wgBlACG927Y_1785461520` | 42 | success | 425.067 | 37 | api_call 27 / action 9 / run 1 | 0 / 0 | D | VALID |
| `phase1_run_02` | `wgBlACG927Y_1785462040` | 43 | success | 374.539 | 41 | api_call 30 / action 10 / run 1 | 0 / 0 | D | VALID |
| `phase1_run_03` | `wgBlACG927Y_1785462510` | 44 | success | 210.619 | 21 | api_call 15 / action 5 / run 1 | 0 / 0 | D | VALID |

远端证据目录：

```text
/root/autodl-tmp/scheduler/results/raw/phase1/wgBlACG927Y_1785461520/
/root/autodl-tmp/scheduler/results/raw/phase1/wgBlACG927Y_1785462040/
/root/autodl-tmp/scheduler/results/raw/phase1/wgBlACG927Y_1785462510/
```

每个目录包含 `trace.jsonl`、`run_manifest.json`、`run_status.json` 以及 VideoSeek 原生轨迹/预测产物（如 wrapper 配置所启用）。三条 trace 的 manifest `trace_event_count` 与实际 JSONL 行数分别为 37、41、21，完全一致。

## Schema 与资源字段验收

- validator 对 3 个 `trace.jsonl` 均返回 `VALID`。
- Schema v0.1 的 21 个顶层必需字段在全部事件中均存在；逐事件缺失计数为 0。
- `resource.runtime_ms` 与 `resource.api_wait_ms` 在全部事件中均存在；缺失计数为 0。
- trace 中保留 `local_runtime_ms`、API 等待、进程 RSS 和 GPU 型号/驱动/显存/利用率快照；API 等待不被标记为 GPU runtime。
- 三条真实运行没有自然产生 API 或工具错误，也没有重试事件。Phase 0 质量门槛中要求的失败/重试样例因此仍未触发；此处记录为缺失原因，不用人为构造的失败伪装成真实样例。

## 可复查命令

```bash
/root/miniconda3/envs/finetooling/bin/python -B \
  /root/autodl-tmp/scheduler/tracing/validators/validate_trace.py \
  /root/autodl-tmp/scheduler/results/raw/phase1/<run_id>/trace.jsonl
```

本地实现与测试入口：

- `tracing/collectors/videoseek_wrapper.py`：不改第三方源码的运行时 wrapper。
- `tracing/schema/trace_v0_1.json`：Trace Adapter v0.1 schema。
- `tracing/validators/validate_trace.py`：依赖无关的结构/时序/资源/错误校验器。
- `tests/test_trace_validator.py`：覆盖正常、GPU、错误、重试、API 返回 `None` 和工具失败 sentinel 的单元测试。

## 边界与下一步

这轮只完成 Trace Adapter 的真实采集与验收。视频为 Phase 0 的 144p 最小样本，且 VideoSeek 本地路径主要是 CPU 解码/图像打包加远端 API；因此不从本报告推导视觉准确率、GPU 推理性能或多 GPU 调度收益。若要进入 Phase 2，应先补充可复现的失败/重试采样策略，并按项目门槛收集足够的路径与资源差异证据。
