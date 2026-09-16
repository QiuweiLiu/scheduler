# Forecast-aware — Phase 11：P0 修复 + runtime-only 正交族（1,000 episodes，2026-09-15）

目的：按 GPT 代码审查的 P0 清单修复实现问题，并重建"消费方式"的**干净对照**，判定 C2（尾部机制）是否成立。

## P0 修复内容

1. **缓存污染修复**：`_SCEN_COST_CACHE` / `_JRES_COST_CACHE` / `_STEP_COST_CACHE` 的键加入 artifact 与 train-stats 上下文
   （`set_artifact_context` / `set_train_stats_context`，由 runner 在加载后设置）；`_JRES_COST_CACHE` 不再用 `id(cost_fn)`。
2. **表查询层级化**：`_step_estimate_cost` / `_table_step_components` 改为 model_lane → lane → global 三级查找
   （不再把 exact / model_lane / lane 混在一起取中位数），并记录 `lookup_path_counts` 供报告披露（本轮 5 次解析：4 次 model_lane、1 次 lane；无 global/missing）。
3. **runtime-only 正交族**（future load=0；相同 current_cost、链长、fallback）：
   `predopt_h5_r50`、`predopt_h5_r90`、`predopt_h5_r95`；
   `predopt_h5_r50k` = k·Σp50，k=**6.2293** 在 dev700 节点上冻结（`phase11_r50_scale.json`；mean Σp95 / mean Σp50）；
   plus 打乱对照：`tail_shuffle` artifacts（保留 (p95−p50) 边缘分布、跨 step 重新配对）+ `predopt_h5_q95`。
4. **Stage0 分位数审计**（`scripts/audit_future_artifacts_quantiles.py`）：runtime 83/66,680 非单调（0.12%）、
   **load duration 5,364/66,680 非单调（8.0%）**、load occurrence 全部合法（66,680/66,680）。审计产物 `stage0_quantile_audit.json`。

## 结果（1,000 集，配对 B=2000）

| 臂 | completion | miss |
|---|---|---|
| q95（旧冠军，含 load 项） | 180,362 | 7.25% |
| **Pred95-R（runtime-only p95）** | **180,407** | 7.27% |
| ShuffledTail + q95（打乱尾部配对） | 187,424 | 7.66% |
| E2（层级化查表后的静态表基线） | 187,615 | 8.27% |
| Pred90-R | 187,778 | 7.68% |
| **ScaledPred50-R（k=6.23 缩放版 p50）** | **193,371** | 8.61% |
| Pred50-R | 214,359 | 10.95% |

**关键配对**：

| 对比 | Δ completion | 结论 |
|---|---|---|
| **Pred95-R − ScaledPred50-R** | **−12,963 [−14,250, −11,818]** | **不是"数值更大"**：放大 p50 到同等量级仍差 13.0s |
| **Pred95-R − ShuffledTail** | **−7,016 [−7,767, −6,292]** | **尾部必须与具体预测对齐**（打乱配对即损失 7.0s） |
| Pred95-R − E2 | −7,208 [−7,953, −6,464] | 干净对照下的静态表差距（非旧报告的 17.7s） |
| Pred95-R − q95 | +46 [−61, +150] | load 项无关（CI 跨 0） |
| Pred90-R − Pred50-R | −26,581 | 单调：p50 → p90 大幅改善 |
| Pred95-R − Pred90-R | −7,371 | 单调续：p90 → p95 仍显著 |

## 结论

1. **C2 成立（在 runtime 维度）**：收益来自**尾部的形状与对齐**，不是"把未来权重放大"；
   缩放版 p50（同量级）与打乱版尾部（同边缘分布）都显著更差。
2. **冠军应改称 runtime-tail 求和**（load 项无贡献，`q95 ≡ r95`），并可去掉 load 分支以简化实现。
3. **E2 的旧数字被查表 bug 抬高**：198,052 → 187,615（−10,437）；诚实差距为 **7.2s**（仍显著）。
4. **"预测资源优于静态表"被证伪**：Pred50-R 214,359 比 E2 187,615 更差（预测 p50 尺度偏小 → 决策近似 myopic）。
5. load duration 分位数有 8% 非单调 → 已在 Stage0 披露；由于 load 项不影响结果，不阻塞主线。

## 边界

- 仍是单 workload、单预测器（J3:seed11）、纯仿真；cell 分层与 confirm 纪律未变。
- **CVaR/场景族结论仍待 P1 修复**（链长截断、人造逆 CDF、N=32、真采样）——在此之前不得引用"q95 优于 CVaR"。
- 产品：`outputs/phase11_*`（7 臂）、`outputs/phase11_artifacts_tail_shuffle/`、`phase11_r50_scale.json`、`stage0_quantile_audit.json`；
  脚本 `.scratch/phase11_stats.py`；新增策略已在 `workload_v02_simulator.py`。
