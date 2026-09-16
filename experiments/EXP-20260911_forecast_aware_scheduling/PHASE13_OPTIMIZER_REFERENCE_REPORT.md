# Forecast-aware — Phase 13：优化型参照（CP-SAT / MPC）对照（1,000 episodes，2026-09-16）

目的：回答"我们的收益是不是贪心框架的局限"——把 **CP-SAT 滚动优化**与 **MPC rollout** 拉进同一接口做对照。

## 结果（同 artifacts、同 episodes，配对 B=2000）

| 臂 | 框架 | 消费 | completion | miss |
|---|---|---|---|---|
| **r95** | 贪心 | runtime p95 求和 | **180,407** | 7.27% |
| cp_rho_h5 | CP-SAT（0.25s/决策 + fallback） | E2 式（表 p50 系数） | 222,535 | 11.78% |
| cp_rho_h3 | CP-SAT | E2 式 | 223,033 | 11.82% |
| pred_mpc_h5 | MPC rollout | E2 式 | 223,805 | 12.01% |

配对：**r95 − cp_rho_h5 = −42,128 [−45,740, −38,834]**；r95 − pred_mpc_h5 = −43,398 [−47,392, −39,748]。

## 为什么这么差（诊断）

1. **消费项不同**：两者内部都用 `predicted_future_cost`（E2 式：表 p50），不是 r95；因此这不是纯粹的"框架对照"。
2. **CP-SAT V1 目标函数被流程时间主导**：目标里 `flow_weight × end_time` 的量级是 `≈1000×ms`，而 `future_cost` 是 `≈ms`；
   实际效果接近"窗口内最短作业优先"，未来项几乎没有作用。
3. **只执行窗口内第一步**（rolling-horizon V1 边界），且窗口只含当前 ready 节点。
4. **历史一致性**：R8 时代同一实现（`cp_rho_matrix_report.json`）在旧口径下 cp_rho_h5 = 223,464ms、
   `predopt_v2_h1` = 221,469ms —— 即当时它也没能超过贪心基线，本次结果复现了该行为（222.5k）。
5. MPC 的 rollout 同样是 deepcopy + E2 式成本，结果与 CP-SAT 同档（223.8k）。

## 结论（诚实表述）

- **在本接口与当前配置下，优化型参照（CP-SAT/MPC）不优于简单贪心 + 状态对齐尾部消费**；
  但它们同时受"消费项未参数化 + CP-SAT 目标量级失衡 + 窗口化 V1"三重限制，
  因此这**不能**作为"贪心优于优化"的证据，只能作为 exploratory/negative 结果。
- 若要得到干净的"框架 vs 消费"2×2，需要：① 把它们的 future 系数换成 r95 式；② 重新平衡 CP-SAT 目标权重；
  ③ 之后重跑 cp_rho_h5_r95（约 1.6 小时/臂）。
- 论文中建议的表述：主结论限定为"**在线贪心列表调度框架下**，消费函数决定收益"，并把优化型对照列为探索性结果与未来工作。

## 产物

- `outputs/phase13_cprho5/`、`phase13_cprho3/`、`phase13_predmpc5/`；脚本 `.scratch/phase13_stats.py`。
- 依赖：`ortools==9.9.3963`（本轮为本地 `scheduler` 环境安装）。
