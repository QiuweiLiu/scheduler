# Forecast-aware — Phase 12：CVaR 族重写与公平对照（1,000 episodes，2026-09-15）

目的：按 GPT 代码审查的 **P1 清单**修复场景/CVaR 族，使其与逐步骤 p95 求和在**同一 artifacts、同一口径**下公平比较。

## 修复内容（P1 5–8）

1. **完整槽位 artifacts**：用 `--min-steps 5` 重新打包，再经 `scripts/normalize_full5_artifacts.py` 生成侧车：
   - `future_h*` = 截断到 `predicted_future_length`（恢复点消费臂依赖的隐式长度信号）；
   - `future_h*_full` = 全部 5 个槽（供长度分布采样）。
   （直接使用 full5 主链会把点消费臂的 r95 从 180.4k 退化到 197.1k —— 见"无效运行"一节。）
2. **加载器 bug 修复（本轮第二个重要发现）**：`load_future_artifacts` 原先只保留 `future_h*` 前缀字段，
   导致 **`length_probabilities` 与 `predicted_future_length` 从未进入调度器**：
   - 历史上所有"场景采样"臂的**长度采样是假的**（只能用 argmax 链长）；
   - `sjf_pred`（Length-SJF）实际退化为"当前成本 + 0"，其 Phase 9 数值（215,108）**作废**。
   修复为保留整行字段（row 完整 merge）。
3. **采样器重写**：lognormal 重建（sigma 由 p95/p50 定，clamp ∈[0.05,1.6]；p95 退化时用 p90），
   不再有 Q(0)=0 的人造下界与 9×(p95−p90) 的人造上尾；**真实长度采样**（从完整链 + length_probabilities）；
   comonotone 耦合同步作用于 runtime/load 发生/load 时长；N=128；目标改为凸组合 `(1−κ)E + κ·CVaR₀.₉`。
4. **无效运行记录**：`outputs/phase12_*`（全链主链 → 点消费臂失真）与 `outputs/phase12b_*`
   （归一化脚本漏 `rows.append` → 空 artifacts，5 臂全部退化为无未来信息）均标记作废，不进入任何结论。

## 结果（1,000 集，sidecar artifacts + 修复后加载器，配对 B=2000）

| 臂 | completion | miss |
|---|---|---|
| q95（点 p95 + load） | 180,362 | 7.25% |
| **r95（runtime p95 求和）** | **180,407** | 7.27% |
| comon128_k100（共单调 CVaR） | 194,548 | 8.33% |
| scen128_k100（独立 CVaR） | 199,692 | 9.08% |
| scen128_k50 | 210,355 | 10.38% |
| scen128_k0（纯均值） | 212,838 | 10.69% |

**关键配对**：

| 对比 | Δ completion | 结论 |
|---|---|---|
| **r95 − scen128_k100** | **−19,284 [−21,179, −17,600]** | 公平对照下 CVaR 仍输 19.3s |
| **r95 − comon128_k100** | **−14,141 [−15,437, −12,883]** | 共单调 CVaR 仍输 14.1s |
| r95 − scen128_k50 | −29,948 | |
| r95 − scen128_k0 | −32,430 | |
| comon − scen (k100) | −5,144 [−5,939, −4,370] | 共单调 > 独立采样（与早期观察一致） |
| scen k100 − k0 | −13,146 | 族内风险厌恶单调有效 |
| q95 − r95 | −46（CI 跨 0） | 交叉校验：sidecar 归一化忠实、加载器修复无副作用 |

## 结论

1. **公平对照下，场景/CVaR 族仍显著劣于逐步骤 p95 求和**（−14 ~ −19s）——本轮修复了截断、人造 CDF、N=32、
   假长度采样与目标函数后依然如此。C2（runtime 尾部的"完全相关式"惩罚优于分布化风险度量）通过最严格测试。
2. 族内规律一致：越保守越好（k100 > k50 > k0）；共单调耦合优于独立采样。
3. 但注意 CVaR 族仍可继续调参（α、κ 网格、N），论文中应表述为"在当前实现与网格下未超过 p95 求和"。
4. `sjf_pred` 的旧数值作废（构建于被丢弃的字段上），需按 GPT 建议重写为 lane-average 版再报告。

## 边界

- 单 workload、单预测器（J3:seed11）、纯仿真。
- model 分布采样不可行（resource head 不以 model 身份为条件，替代模型没有自己的资源分位数）——已在实现中说明；
  目前采样覆盖 length 分布 + 每步 runtime/load 分位数。
- 产物：`outputs/phase12c_*`（有效）、`outputs/sstar_predictor_artifacts_full5_sidecar2/`、`scripts/normalize_full5_artifacts.py`；
  脚本 `.scratch/phase12c_stats.py`。
