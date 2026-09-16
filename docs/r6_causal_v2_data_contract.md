# R6 causal-v2 数据契约与正式派生结果

日期：2026-08-16（Asia/Shanghai）  
运行环境：远端 `/root/autodl-tmp/scheduler`，Python `/root/miniconda3/envs/finetooling/bin/python`  
实现脚本：`scripts/build_r6_causal_v2.py`

## 目标和边界

R6 把旧的 768 条 C1 轨迹变成可以接入后续调度器的、可审计的派生数据。原始
`trace.jsonl`、旧的 v1 processed 文件和 768 条 legacy collection 都没有覆盖。
R6 使用 C1 视频级 48/8/8 划分；C2 只登记来自 600 视频 provenance 库、且排除了
远端已有 trace id/hash 的 32 个候选和 8 个替补。C2 尚未有新轨迹，所以本轮不生成
C2 workload，也不把候选说成测试结果。

## 决策时刻契约

策略在当前节点真正执行前，只能从 `causal_prefixes_v2.jsonl` 的 `input` 读取：

- 已完成 prefix 的动作摘要和结构化 state；
- 当前 ready 节点的 role/action/model/lane 身份；
- 任务结构和当前 split；
- 由后续 provider 返回的 H=1/3/5 场景和 train-only 资源预测。

`input` 不含当前节点的 runtime/load/memory/status/retry，也不含未来节点、答案、
remaining steps/runtime、video id 特征或整条 template。标签单独放在 `label`，只用于
训练/评估，不交给在线策略。完整 template、`execution_truth_v2.jsonl` 和测量资源
字段标记为 `execution_engine_only`。

未来 artifact 的 prototype 只从 C1 `train` 的 566 个 template 拟合，输出 H1/H3/H5，
每行概率和为 1；未知质量用显式 `unknown` 回退，不把未知当作零概率。

资源 artifact 只从 C1 train 的 566 个 template、6,916 个 truth node 拟合，输出 9,366
个节点预测。`resource_profile_v2.json` 记录分层 quantile，
`scheduler_resource_contract_v2.json` 记录 `runtime_p50/p90`、`load_p50`、
`peak_memory_p95` 和 uncertainty 的接口。资源真值不写回 scheduler-visible workload。

## 正式远端结果

唯一 canonical R6 目录：

```text
/root/autodl-tmp/scheduler/results/processed/r6_causal_v2_20260816_fix1/
```

主要文件：

```text
causal_prefixes_v2.jsonl
causal_future_h1_v2.jsonl
causal_future_h3_v2.jsonl
causal_future_h5_v2.jsonl
resource_predictions_scheduler_v2.jsonl
resource_profile_v2.json
scheduler_resource_contract_v2.json
execution_truth_v2.jsonl
job_templates_causal_v2_truth.jsonl
job_templates_causal_v2_scheduler.jsonl
workload_train_v2.jsonl
workload_validation_v2.jsonl
workload_test_v2.jsonl
r6_manifest.json
r6_gate_report.json
```

split registry：

```text
/root/autodl-tmp/scheduler/data/manifests/video_split_registry_r6_v1_causal_v2.jsonl
```

## 数量与 gate

| 项目 | 数量/结果 |
|---|---:|
| C1 train/validation/test-retrospective 视频 | 48 / 8 / 8 |
| C2 pending/backup 视频 | 32 / 8 |
| prefix rows | 4,683 |
| future rows（H1/H3/H5） | 9,366 / 9,366 / 9,366 |
| resource prediction rows | 9,366 |
| train/validation/test workload episodes | 20,000 / 1,000 / 6,750 |
| workload jobs | 743,680 / 34,160 / 248,400 |
| prefix duplicate/leakage/causal-edge errors | 0 / 0 / 0 |
| future probability normalization errors | 0 |
| scheduler-visible forbidden workload fields | 0 |
| R6 gate | **通过** |

episode 的 job 只保留 `template_id`、`arrival_ms`、`service_class` 和固定相对 deadline；
整条任务的 predicted runtime、GPU runtime、source video/trace refs 被移到
`workload_truth_metadata` 或完全不暴露给 policy。Alibaba PAI 到达间隔仍作为 workload
生成的 arrival source，路径和 SHA256 写入 `r6_manifest.json`。

## 后续恢复点

R6 完成后进入 R7：先把 `SchedulerStateView`、`ExecutionTruthProvider` 和
`FutureProvider` 接到唯一事件引擎，再在同一份 R6 validation workload 上复现
Round-Robin/Myopic/Oracle，并实现 Optimizer-H、RL-H。C2 只有在 predictor/resource
冻结后才采集新视频轨迹、生成 C2 templates 和 workload。
