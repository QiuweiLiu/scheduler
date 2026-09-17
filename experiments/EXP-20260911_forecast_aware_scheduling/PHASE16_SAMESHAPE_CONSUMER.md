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
| `predopt_h5`（live greedy 路径） | `(priority, current + future, future, ready_time, job_index, node_id, gpu.index)` | 第 1（与冠军同形） |
| `trueopt_h5` | `(limited_future_truth_cost(...), priority, ready_time, job_index, node_id, gpu.index)` | 第 2（future 压过 priority，且**不含当前节点**） |

证据：`src/tracing/analysis/workload_v02_simulator.py` 的 `_q95/_lam` 分支、generic `predopt_h*` 分支、
`trueopt_h*` 分支；`aligned_h5_policy_key()` 的文档明确要求 priority 在 cost 之前
（"after priority, regardless of whether the values are predicted or true"），因此 **`trueopt_h5` 违反了该共享硬约束**。

> **勘误（2026-09-17，GPT 审阅后本地复核）**：本文件与提交 `ae61c0a` 的 message 曾把混淆写成"三臂 priority 分别第 1/第 2/第 3 位"，
> 其中"`predopt_h5` priority 第 3"**是错的**：live greedy 的 generic `predopt_h*` 分支同样是 priority 第一。
> priority 第 3 的形状（`_predicted_candidate_key()`）只出现在 **MPC/rollout 路径**
> （`_predicted_rollout_score()` / `_pred_mpc_choose()`），不是 `choose_action` 的 greedy 派发路径。
> 结论不变（旧的 "q95 胜 oracle" 仍然撤回），但**混淆只有一处**：legacy `trueopt_h5` vs 预测族。

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

冻结 artifact 的 future step 只有 identity（`model_id` / `execution_lane` / `resource` 分位数），**没有 node_id**。
`sameshape_h5_truth` 因此把 future 项替换为**真实后继展开**：key 形状与 current 项已对齐，但 future 的
**链内容**（真 DAG successor walk vs 预测 synthetic chain）仍有本质差异，且 truth 项还额外带入了
GPU-dependent residency/load 与 "H = DAG layer" 语义（预测臂的 H 是 synthetic slot）。

**该差异是否可再压缩尚未定论**（初版曾写成"不可再压缩"，已按审阅修正）：v3.1 是 linear slot 预测，
artifact 带 anchor `node_id` 与每槽 `step_offset`，若 evaluated continuation 内 outdegree ≤ 1，
则可能用 **ordinal successor 对齐**构造 `TruthRuntime | PredictedShape`（保持预测长度 L̂，只把第 h 个预测
runtime 换成第 h 个真实 successor runtime）。这需要先做 path-invariant 断言验证，属**待验证推断**。

### 审阅后修订（2026-09-17，GPT 审阅 `docs/research/2026-09-17_fas_phase16_sameshape_review_gpt.md`）

- **判定**：本次实现可作为 "same-key system sensitivity"（Phase16-A），但**还不是**严格的 runtime-oracle 机制实验。
- **本轮仍未修的 P0（下一轮）**：
  1. `_q95_step_cost` 在 GPU step 且 `load_occurrence_probability ≥ 0.5` 时会加 conditional load p95 ⇒
     `predopt_h5_q95` 不是"纯 runtime p95 求和"（`sameshape_h5_p95 ≡ predopt_h5_q95` 仍成立，但语义描述要改）；
     真正 runtime-only 的既有实现是 `predopt_h5_r95`（`_runtime_only_step_cost()`）。
     建议新增严格正交四臂：`Pred50_R → Pred95_R → TruthRuntime|PredShape → TruthRuntime|TrueShape`。
  2. **label contract 审计**：需确认 J 预测器的 `runtime_ms_quantiles` 目标语义是 total runtime 还是 compute-only
     （`compute_ms = runtime_ms - load_ms`）——若标签已含 load，则 `runtime + load` 存在重复计 load 的风险。
  3. 单测需补：多候选、非零 conditional load、≥3 步 successor 链、双 GPU 不同 residency、**直接断言每臂的 key**
     （而不只是最终 summary）。
- **判据修订**：`Truth ≈ r95` 不能用"差异不显著"定义，需预注册 practical equivalence margin δ 做 equivalence CI；
  必须报告 signed bias / MAE / underestimation rate、候选集 top-1 agreement 与 pairwise disagreement / Kendall τ、
  **common-state ranking audit**（避免 trajectory divergence 污染 episode 级比较）、score margin、
  分位违例率与 tie 率（多少决策由 `gpu.index` 决胜）。

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
