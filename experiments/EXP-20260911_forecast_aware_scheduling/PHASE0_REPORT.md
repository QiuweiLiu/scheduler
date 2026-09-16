# Forecast-aware scheduling — Phase 0 报告（接口验证，2026-09-11）

状态：**Phase 0 通过**（接口/覆盖/一致性/成本全部确认）；并得到 E2 预览的正面信号。

## 0. 目的与结论

- 目的：验证 J3 预测器 artifacts 能真正进入调度器矩阵（node_id 对接、命名/manifest 兼容、策略无崩溃），并测量本地成本。
- 结论：
  1. node_id 命名空间完全一致（template node_id ≡ trace event_id ≡ artifact node_id），**无需映射**。
  2. 覆盖：**8,935/8,935 template 节点全部命中**（我们的 artifacts 是超集，9,575 keys）。
  3. 调度器 loader 需要 `b05_*` 命名 + `b05_artifact_manifest.json`；已生成兼容目录 `outputs/sstar_predictor_artifacts_sched/prediction_artifacts/`（原目录保留）。
  4. 100 episodes × 3 策略：**0 failed jobs、状态 passed、无容量违规**（capacity_violation_count=0）。
  5. 一致性：myopic 与 oracle 在两个 provider 下**逐位相同**（接口不改变非未来策略），predopt_h5 有差异（未来信息不同）。
  6. 成本：100 eps × 3 策略 ≈ **31–33 秒**（本地 CPU）；1,000 eps × 6 策略 ≈ 11 分钟/seed → 3 seeds ≈ 33 分钟。

## 1. 100-episode 对照（同一 workload、同一策略、只换 future provider）

| policy | J3 预测（我们） | B05（旧 provider） | 差异 |
|---|---|---|---|
| myopic（无未来） | 160,958 ms | 160,958 ms | 0（一致性 ✓） |
| **predopt_h5** | **152,782 ms** | 158,169 ms | **−5,387 ms（−3.4%）** |
| oracle（上界） | 141,576 ms | 141,576 ms | 0（一致性 ✓） |

predopt_h5 其它指标（J3 vs B05）：queue 62,118 vs 67,465（−7.9%）；deadline miss **3.25% vs 4.06%**（−0.8pp）；evictions 20.04 vs 17.75（+2.3）。

oracle gap：J3 的 predopt_h5 距 oracle 还差 152,782/141,576 = **+7.9%**（上界空间）。

## 2. 关键实现事实

- 当前策略的 future 成本只读 `model_id`/`execution_lane` + 静态资源表（`train_stats` 的 P50）；
  **我们 artifact 里的 `resource` 块（runtime/load 分位数）被忽略**（simulator 行 126/698-699/760-761）。
  ⇒ 本次对照的差异**完全来自未来结构预测（E2 通路）**；E1/E3 需要资源适配层。
- 预测长度分布（全部 9,575 anchors）：{0:1272, 1:656, 2:582, 3:581, 4:621, 5:5863}；template 节点覆盖零缺失。
- 报告元数据：`future_artifacts_manifest_sha256`、`information_boundary`、`oracle_gaps` 均由 runner 记录。

## 3. Phase 1 需要的实现（资源适配层）

- **E2（已完成预览）**：J3 拓扑 + 静态资源表 = 现有策略 + 我们的 artifacts。
- **E3（待实现）**：opt-in 读取 `step["resource"]`：
  - runtime 用 `runtime_ms_quantiles.p50`（或按目标函数取分位数）；
  - load 用 `load_occurrence_probability × load_duration_ms_quantiles.p50`；
  - 以**新策略名**（如 `predopt_h5_jres`）落地，不改既有策略行为（保持可比性）。
- **E1（待定设计）**：resource-only 臂（无未来拓扑）；建议定义为"当前节点资源预测 + 无未来链"的 myopic 变体，具体口径需在 Phase 1 设计评审中冻结。
- **E4**：oracle 已在矩阵中（真值），无需实现。

## 4. 下一步

Phase 1 pilot（100 episodes，E0/E2/E3/E4 四臂；E3 需先实现资源适配层并 smoke）→ 看 effect size 决定 Phase 2（1,000 eps × 6 臂 × 3 seeds）。
