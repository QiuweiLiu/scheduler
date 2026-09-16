# 重训与消费方式：网页版规划（2026-09-11）

问题：① 重训预测器；② 找最佳消费方式（需文献调研）。结论：**先消费机制、后重训；H=10 是重训的正确方向，资源质量最后。**

## 总判断

当前证据把问题重新定位为：
- 未来 **horizon** 是主要信息瓶颈（H5→H10 ≈10s，H20 饱和）；
- **资源预测精度不是主要瓶颈**（oracle resource ≈ table）；
- **内容属性精度不是主要瓶颈**（真值 content 反而更差）；
- **消费方式是最大未知变量**（risk-aware p90 已拿到 ~10.5s）。

→ **不建议立即重训 predictor**（容易优化错误目标）。优先级：**① 消费机制（主线）→ ② H=10 重训 → ③ 资源质量**。

## 1. Predictor 重训规格（H=10，不做资源增强）

- 标签：保持 v3.1 契约；`future length L ∈ [0,10]`；termination 保留 terminated/censored；每 slot 预测 node existence / runtime quantile / load occurrence+duration；**不扩大 node attribute**（属性精度非价值来源）。
- **不需要新数据**：现有 P_dev 重新生成 `future_window=10` 即可；但需新实验 ID、新 checkpoint、新 validation gate；**不能与 H5 直接混用**。
- 对照：P0 = H5 J3 frozen；P1 = H10 J3-style。预测指标（length MAE / termination acc / runtime pinball）+ 系统指标（同调度器、同消费策略）；真正目标 = H10 是否把真值 oracle 收益转化为 deployable。

## 2. 消费机制（文献方向与候选）

领域：**uncertainty-aware predictive scheduling / stochastic MPC**。关键词：chance-constrained scheduling、stochastic MPC、risk-sensitive scheduling、CVaR optimization、quantile regression scheduling、robust scheduling under uncertainty；代表方向：MPC 滚动优化、Chance-Constrained Optimization、CVaR、Distributional RL、Quantile Regression。

| 优先级 | 候选 | 形式 | 最小实验 | 成本 |
|---|---|---|---|---|
| **①** | **Quantile / CVaR 风险调度** | `Cost=Σ_h (p50_h + λ(p90_h−p50_h))` 或 `CVaR_α(T)` | λ ∈ {0, 0.25, 0.5, 1} 四个策略 | <1 小时 |
| ② | Chance-constrained scheduling | `P(JCT>D)<ε`；选 `P(T_future<deadline)` 最高者 | deadline 三档 1.2×/1.5×/2× | 低 |
| ③ | Stochastic MPC / rollout（长期推荐） | 每步模拟候选动作、执行第一步、重新规划（receding horizon） | 小规模 pilot | 中等 |
| ④ | Survival / hazard | `E[C]=Σ P(T≥h)c_h`；升级为预测 hazard `P(T=h|T≥h)` 而非长度分类 | — | 低 |
| ⑤ | Cache-aware stochastic simulation | `Cost(node_i | history)`（需 residency/reuse） | 暂缓（需数据） | 高 |

## 3. 避免 validation 过拟合

1,000 episodes 已被多次探索（多个阶段 × 多臂）。建议：**70% development / 30% frozen confirm subset**；或**预注册** λ / CVaR α / horizon，只跑一次。

## 4. 论文主线（网页版更新）

> "Agent scheduling benefits from predicting future workflow evolution and explicitly reasoning over uncertainty, rather than merely estimating per-node execution cost."

证据链：future prediction > no future；H10 > H5；resource oracle ≈ table；risk-aware consumption > naive expected cost。

## 5. 推荐执行顺序

- **Phase 1（马上，1–3 天，无需训练）**：消费机制——p90/CVaR、chance constraint、stochastic rollout。
- **Phase 2（≤1 周）**：H=10 predictor 重新训练（新 ID/门禁/checkpoint）。
- **Phase 3（后续，需采数据）**：cache-aware + residency。
