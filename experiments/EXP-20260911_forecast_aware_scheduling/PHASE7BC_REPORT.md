# Forecast-aware scheduling — Phase 7B/7C：自适应风险 + 分量拆分（2026-09-14，dev700）

状态：7B（自适应风险）与 7C（分量拆分）完成；**仍无消费器超过 q95；7C 给出机制结论：全部收益来自 runtime 尾部。**

## Phase 7B：自适应风险 λ(slack, queue)

设计：`Score = Σ [p50 + λ·(p95 − p50)]`，λ 由 deadline slack（<5% → 1.0；<25% → 0.75；否则 0.4；无 deadline → 0.5）
和可选的队列压力项（ready/free 归一化，+0~0.25）决定。
- `predopt_h5_adapt`：仅 slack；`predopt_h5_adapt_q`：slack + 压力。

| 策略 | dev700 完成时间 | vs E2 | vs q95 |
|---|---|---|---|
| q95（静态 p95） | 179,125 | −17,229 | — |
| **adapt_q** | 184,996 | −11,358 [−12,859, −9,888] | +5,871 [5,331, 6,478] |
| adapt（仅 slack） | 197,228 | +874 [−347, +2,135]（≈E2） | +18,103 |

- 压力项是主要贡献（adapt_q 有效、adapt 无效）；但**自适应仍显著输给"一律 p95"**（q95 − adapt_q = −5,871，CI 不含 0；miss 也回退 +0.33pp）。
- 解释：本 workload 下选择性保守没有优势——统一按最坏情形惩罚预测链反而最好。

## Phase 7C：分量拆分（runtime vs load）

设计：`runtime p50+λ(p95−p50)` 与 `load p50+λ(p95−p50)` 分开控制。
- `predopt_h5_rt95`（runtime p95 + load p50）；`predopt_h5_ld95`（runtime p50 + load p95）。

| 策略 | dev700 完成时间 | 对比 |
|---|---|---|
| q95（runtime p95 + load p95） | 179,125 | — |
| **rt95（runtime p95 + load p50）** | **179,136** | **rt95 − q95 = +11 [−120, +135]（统计等价）** |
| E2 | 196,354 | rt95 − E2 = −17,218 |
| ld95（runtime p50 + load p95） | 211,444 | ld95 − E2 = +15,090 [13,259, 16,991]（显著更差） |

**结论：q95 的全部收益来自 runtime 尾部惩罚；load 尾部惩罚不贡献（q95 ≈ rt95），单独使用 load 尾部有害（ld95 比 E2 差 15.1s）。**

## Phase 7 综合（消费器家族全部结果，dev700）

| 家族 | 代表臂 | vs E2 | vs q95 |
|---|---|---|---|
| 表 p50（基线） | E2 | — | +17,229 |
| **逐步骤 p95 求和** | **q95** | **−17,229** | **0（冠军）** |
| 分量等价 | rt95 | −17,218 | +11（等价） |
| 场景采样 + CVaR（独立） | scen_k50/k100 | −12.8 ~ −13.2k | +4.0k |
| 场景采样 + CVaR（共单调） | comon_k50..200 | −7.8 ~ −10.5k | +6.8 ~ +9.5k |
| 自适应风险（slack+压力） | adapt_q | −11.4k | +5.9k |
| 自适应风险（仅 slack） | adapt | ≈0 | +18.1k |
| 纯分布均值 | scen_k0 | +13.8k | +31.0k |
| load 尾部 | ld95 | +15.1k | +32.3k |
| 机会约束 / 生存 / 缓存 / 内容（Phase 4–6） | cc/surv/… | 均未超过 q95 | — |

**统一解释：消费价值 = 对预测链的 runtime 尾部做"完全相关"式惩罚（每步 p95 直接相加），
乘以链长；任何把不确定性平均化（场景采样）、选择性（自适应）或转移到 load 维度（ld95）的做法都会减弱它。**

## 边界

- 调参只在 dev700；确认只在 confirm300（q95 已确认 −18,765 [−21,948, −15,932]）。
- 新增策略：`predopt_h5_{scen_k0/50/100, comon_k50/100/200, adapt, adapt_q, rt95, ld95}`；实现均在 `workload_v02_simulator.py`。
- 产物：`outputs/phase7b_dev700/`、`outputs/phase7c_dev700/`；统计 `.scratch/phase7bc_stats.py`。
