# Forecast-aware — Phase 9 基线矩阵（1,000 episodes，2026-09-15）

目的：回应审稿质疑 R3"是否只打赢弱 baseline"。补三个经典/近期基线（FCFS、SJF-LTR、state-aware），
与已有臂在同一 workload、同一 artifacts、同一运行目录下比较（成对 bootstrap B=2000，seed 20260914）。

## 新增基线（本机实现）

- `fcfs`：先来先服务（ready 顺序 + GPU 顺序，无未来信息）。
- `sjf_pred`：SJF/LTR-style——预测剩余工作量 = 当前表成本 + 预测未来长度 × 每步表成本（只用预测长度，无拓扑/资源分位数）。
- `state_aware`：SOLA/SlackFit-style——只用当前状态（当前节点成本、deadline slack、GPU 驻留），不含任何未来信息。

## 结果（mean completion ms；miss 为违约率）

| 策略 | completion | miss | q95 − 该臂（配对） |
|---|---|---|---|
| **predopt_h5_q95（冠军）** | **180,362** | **7.25%** | — |
| trueopt_h5（真值资源 oracle） | 183,845 | 7.30% | **−3,484 [−4,054, −2,974]** |
| oracle_topology_h5_tab（known-DAG oracle） | 184,514 | 6.90% | **−4,152 [−4,741, −3,618]**（miss +0.35pp） |
| predopt_h5（E2：未来结构+表点成本） | 198,052 | 8.64% | −17,690 [−19,327, −16,211] |
| fcfs | 213,282 | 11.73% | −32,920 [−35,588, −30,370] |
| sjf_pred（LTR-style） | 215,108 | 10.98% | −34,746 [−37,792, −31,897] |
| myopic（E0，当前节点贪心） | 215,116 | 10.98% | −34,755 [−37,809, −31,896] |
| state_aware（SOLA/SlackFit-style） | 215,658 | 8.67% | −35,296 [−38,352, −32,395] |

**E2 vs 无未来基线**：−15,230（FCFS）、−17,065（E0）、−17,056（SJF）、−17,606（state-aware），全部 CI 不含 0。

## 关键发现

1. **q95 显著优于全部经典/近期基线**：相对 FCFS/SJF/E0/state-aware 快 33–35 秒（−15%~−16%），miss 也全面更低。
2. **q95 甚至优于两个 oracle 臂**（真值拓扑 + 点估计）：
   - vs known-DAG oracle：−4,152ms [−4,741, −3,618]（但 miss +0.35pp 略差）；
   - vs 真值资源 oracle：−3,484ms [−4,054, −2,974]（miss 差异 CI 跨 0）。
   - 解释（C2 的直接证据）：**消费机制的形状（相关尾部惩罚）比信息精度更重要**——用不完美的预测 + 风险形状，胜过完美拓扑 + 点估计。
3. **未来信息本身有价值**：E2 相对四个无未来基线一致 −15~−18 秒。

## 边界与注意事项

- `predopt_h5_surv` 在当前（dist）artifacts 下与 `predopt_h5` 数值完全相同 → **退化臂**（artifacts 无 `survival_probability` 字段）。
  不得作为独立基线报告；Phase 4 的 survival 结果使用的是 C1/C2 artifacts，两者不可混用。若要保留该基线需重新打包带 survival 字段的 artifacts。
- oracle 臂的目标函数为 completion（当前节点表 p50 + 真值拓扑/资源），与 q95 的风险形状不同；"q95 优于 oracle"应表述为
  **在同一调度目标下的完成时间优势**，并同时报告 miss（known-DAG oracle 的 miss 略优）。
- 单 workload family、单预测器（J3:seed11）、仿真数据；仍需真实 replay 与第二 workload 才能外推。
- 运行：`outputs/phase9_baselines_1000/`（9 臂 × 1,000 集，88 分钟）；新增策略已在 `workload_v02_simulator.py`（POLICIES）；脚本 `.scratch/phase9_stats.py`。
