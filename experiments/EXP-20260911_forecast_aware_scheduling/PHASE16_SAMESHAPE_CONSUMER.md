# Phase 16 — Truth-same-consumer arms（oracle 对照的 P0 混淆修正）

**Date:** 2026-09-17
**Status:** `implemented_not_run`（已实现 + 单元验证；**未跑任何调度实验，本文不含新数字**）
**Supersedes:** 无（不修改既有数字，只撤回一条被混淆的结论）

## 1. P0 发现：oracle 对照是 confounded

`EXPERIMENT_GATE.json` 里 `baseline_matrix_status.headline` 曾声称 q95 同时胜过两个 oracle 臂
（`-3.5k vs truth-resource oracle`、`-4.2k vs known-DAG oracle`）。**该结论目前不成立**，因为被比较的三个臂
并不是"同一个消费函数，只把预测值换成真值"——它们的排序 key 形状不同（代码证据）：

| 臂 | 排序 key | priority 位置 |
|---|---|---|
| `predopt_h5_q95`（当前冠军） | `(priority, current + future, future, ready_time, job_index, node_id, gpu.index)` | 第 1 |
| `trueopt_h5` | `(limited_future_truth_cost(...), priority, ready_time, job_index, node_id, gpu.index)` | 第 2（future 压过 priority） |
| `predopt_h5` | `(current + future, future, priority, ready_time, job_index, node_id, gpu.index)` | 第 3 |

证据：`src/tracing/analysis/workload_v02_simulator.py` 的 `_q95/_lam` 分支、`trueopt_h*` 分支、
`_predicted_candidate_key()`；`aligned_h5_policy_key()` 的文档明确要求 priority 在 cost 之前
（"after priority, regardless of whether the values are predicted or true"），因此 `trueopt_h5` 与
`predopt_h5` 都违反了该共享硬约束。

后果：`r95 = 180.4s` 与 `trueopt_h5 ≈ 183.8s` 的差值**不能**用于任何"预测优于真值"或"真值反而更差"的表述；
它同时混合了 (a) 信息源、(b) 排序 key 形状、(c) future 项定义（真拓扑 + 真 load + residency + DAG layer vs
预测链 + runtime-only + 预测长度）。

## 2. 本次实现（additive, opt-in）

新增三个**同 key 形状**的臂（`src/tracing/analysis/workload_v02_simulator.py`）：

| 臂 | future 项 | 说明 |
|---|---|---|
| `sameshape_h5_p50` | `Σ_h p50_h`（冻结 artifact，逐步骤） | 基础未来工作量 |
| `sameshape_h5_p95` | `Σ_h p95_h`（冻结 artifact，逐步骤） | **按构造等价于 `predopt_h5_q95`** |
| `sameshape_h5_truth` | 真实后继展开 `limited_future_truth_cost(job, node, gpu, stats, 5)` | 真值 future |

三者 current 项、key 顺序、tie-break **完全一致**：`(priority, current_table + future, future, ready_time,
job_index, node_id, gpu.index)`；仅 future 信息不同。既有臂（`predopt_h5`、`trueopt_h5`、`predopt_h5_q95`、
`aligned_*`）**未做任何修改**，保持已发布数字可复现。

### 已知的不可消除差异（必须写进论文）

冻结 artifact 的 future step 只有 identity（`model_id` / `execution_lane` / `resource` 分位数），**没有 node_id**，
因此"预测链上每一步的真值 runtime"在数据上不可观测。`sameshape_h5_truth` 因此只能把 future 项替换为
**真实后继展开**：key 形状与 current 项已对齐，但 future 的**链内容**（真 DAG successor walk vs 预测 synthetic chain）
仍有本质差异。这是该对照不可再压缩的剩余混淆，必须在文中显式声明。

### 单元验证（可在无 GPU 环境运行）

`tests/test_sameshape_consumer.py`：
- 三个臂都注册在 `POLICIES` 中；
- **`sameshape_h5_p95` 与 `predopt_h5_q95` 在同一 episode 上的 summary 逐字段相等**（把新代码钉在已验证冠军上）；
- `p50` 与 `p95` future 值单调（500 vs 2000，合成 artifact）；
- truth 臂可端到端跑通（completed 3 / failed 0）；缺 artifact 时报错。

命令：`PYTHONPATH=src python -m unittest tests.test_sameshape_consumer`
（本机实测通过；同时发现一个**既有**失败：`tests.test_workload_v02_simulator.AdmissionTests.test_round_robin_joint_action_keeps_oldest_ready_node`，
在未改动的 public repo 副本上同样失败，与本改动无关。）

## 3. 预注册判据（本次**未执行**，需门禁批准后跑）

dev700 上跑三臂 + `predopt_h5_q95` + `E0/E2`，配对 bootstrap：

- `Truth ≪ r95 ≪ p50` → 正常：真值最好，p95 部分补偿预测误差（叙事 = 校准）；
- `Truth ≈ r95 ≪ p50` → 支持"p95 ≈ 聚合纠偏"，需追加 scale-matched / tail-shuffle 证据；
- `r95 < Truth`（CI 上界 < 0）→ **真正有研究价值**：不确定性溢价改变了排序且有利于系统目标；
- 若三臂排序在 dev 与 confirm 间反转，则记为不稳定，不得包装成机制。

## 4. 拒绝/暂停的表述

- 禁止："prediction outperforms oracle" / "oracle 资源反而更差"；
- 暂停引用：`baseline_matrix_status.headline` 中两个 oracle 差值（改为"key 形状未对齐，见 Phase 16"）；
- 本文件不改变任何已完成实验的数值；`trueopt_h5` / `predopt_h5` 的历史结果仍可引用，但只能作为
  "legacy key shape" 下的结果，不得作为 oracle 上界。
