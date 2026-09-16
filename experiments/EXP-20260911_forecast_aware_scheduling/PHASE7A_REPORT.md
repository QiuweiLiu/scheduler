# Forecast-aware scheduling — Phase 7A：dev/confirm 切分 + 场景采样 CVaR 消费（2026-09-14）

状态：**Step 0（分割）已完成；Phase 7-A（场景采样 + CVaR）完成——优于 E2 但未超过 q95；q95 在冻结 confirm300 上确认。**

## Step 0：validation 70/30 分割（防过拟合）

- `data/manifests/validation_split_dev700_confirm300.json`（seed 20260914；sorted ids → shuffle → 前 700 dev / 后 300 confirm）；
  另导出 `validation_split_dev700_ids.txt` / `validation_split_confirm300_ids.txt`。
- 运行器新增 `--episode-ids-file`（R7 matrix），报告状态改为按策略覆盖检查。
- 协议：所有消费器调参只在 dev700；冠军只在 confirm300 跑一次。q95 为预注册冠军。

## Phase 7-A：场景采样 + CVaR（新 artifacts 带分布）

- 新打包（`--emit-distributions`）：`outputs/sstar_predictor_artifacts_dist[_sched]/`（每步 model_probabilities + 每行 length_probabilities；J3:seed11 同 checkpoint）。
- 新消费器：采样 32 条未来链 → `Score = current + E[F] + κ·CVaR₀.₉(F)`。
  - `predopt_h5_scen_k{0,50,100}`：各步独立采样（分位数有界插值，尾部截断 9×(p95−p90)）。
  - `predopt_h5_comon_k{50,100,200}`：共单调耦合（同一分位水平贯穿各步）。
- 采样器修复记录：初版 lognormal(p50,p90) 拟合在 p50 极小的行上爆炸（均值被拉到千万级）；
  改为有界分段线性逆 CDF（0→0, 0.5→p50, 0.9→p90, 0.95→p95, 尾部≤p95+9(p95−p90)）。

### dev700 排行榜（mean_completion_ms，n=700，0 失败）

| 策略 | 完成时间 | vs E2（配对） |
|---|---|---|
| **predopt_h5_q95** | **179,125** | **−17,229 [−19,037, −15,476]** |
| scen_k50（独立） | 183,116 | −13,238 [−14,703, −11,837] |
| scen_k100（独立） | 183,555 | −12,800 [−14,125, −11,485] |
| comon_k50 | 185,880 | −10,474 [−11,678, −9,301] |
| comon_k100 | 187,374 | −8,981 [−10,062, −7,947] |
| comon_k200 | 188,592 | −7,762 [−8,763, −6,804] |
| E2（predopt_h5） | 196,354 | — |
| scen_k0（纯均值） | 210,116 | +13,761 [12,053, 15,505] |

关键对比：**q95 − scen_k50 = −3,991 [−4,492, −3,473]（q95 显著更好）**；comon_k100 − scen_k50 = +4,258（共单调更差）。

### confirm300（冻结，首次使用，预注册冠军）

- q95 − E2：**−18,765 [−21,948, −15,932]**；queue −19,436；miss −1.36pp [−1.74, −1.01]。
- 绝对：q95 = 183,248 vs E2 = 202,012（miss 8.04% vs 9.41%）。

## 结论与解释

1. **q95（逐步骤 p95 求和）仍是当前最强消费方式，并在未见过的 confirm300 上复现**（−18.8s，CI 不含 0）。
2. 场景采样 + CVaR **有效但弱于 q95**：独立采样 −13.2s、共单调 −7.8~−10.5s。
   解释：5 步求和使场景成本向均值集中（步骤间独立 → 方差收缩），尾部度量被稀释；
   而 q95 隐含"各步同时取 p95"（完全相关最坏情形），在本工作负载下恰好更有效。
   加大 κ（2.0）或改共单调耦合都无法弥补，说明不是"保守程度不够"而是"惩罚形状"不同。
3. **κ=0 明显差于 E2**（+13.8s）→ 消费器的价值来自尾部/风险项，而非分布均值。
4. 场景采样没有转化为增量收益，但给出了干净的负结果与机制解释（对论文的"消费方式"一章有价值）。

## 边界与产物

- 训练/预测器不变（J3:seed11，v3.1 契约）；新增仅消费器与打包分布字段。
- 产物：`outputs/phase7a_dev700/`、`outputs/phase7a_confirm300/`、`outputs/sstar_predictor_artifacts_dist{,_sched}/`；
  统计脚本 `.scratch/phase7a_stats.py`、`phase7a_stats2.py`、`phase7a_confirm_stats.py`。
- 新增策略：`predopt_h5_scen_k{0,50,100}`、`predopt_h5_comon_k{50,100,200}`（均在 POLICIES）。
- 未跑：7-B 自适应风险、7-C 分量拆分、7-D queue-aware rollout、H10-lite。
