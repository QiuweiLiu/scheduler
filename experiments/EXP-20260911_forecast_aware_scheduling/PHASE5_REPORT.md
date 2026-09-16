# Forecast-aware scheduling — Phase 5：horizon 与消费方式（1,000 episodes，2026-09-11）

状态：完成。**两个新发现**：① H=10 已≈无界 oracle（H5 损失 10.2s，是最确定的收益来源）；② **风险敏感消费（p90）相对 E2 提升 10.5s**——资源分位数在正确消费方式下是会转化的。

## 0. 实验臂

| 臂 | 输入 | consumer | 说明 |
|---|---|---|---|
| trueopt_h5（已有） | 真值拓扑+真值资源 | greedy sum | H=5 上界 |
| **trueopt_h10 / trueopt_h20（新）** | 真值拓扑+真值资源 | greedy sum | horizon 扫描 |
| oracle（已有） | 真值拓扑+真值资源 | greedy sum（无界） | 全局上界 |
| E2（已有） | 预测 | 表 p50 | 基线 |
| **predopt_h5_risk（新）** | 预测（含资源分位数） | **p90 风险成本**（occ≥0.5 时加 p90 加载） | 风险消费 |
| aligned_predopt_h5_layer（新） | 预测 | cache-aware（表 p50） | 缓存感知消费 |
| length-only / survival（已有，Phase 4） | — | — | 对照 |

## 1. horizon（completion/queue/miss）

| 臂 | completion | queue | miss |
|---|---|---|---|
| trueopt_h5 | 183,845 | 110,661 | 7.30% |
| **trueopt_h10** | **173,614** | **86,798** | **6.15%** |
| trueopt_h20 | 173,614（=h10） | 86,798 | 6.15% |
| oracle（无界） | 172,733 | 78,938 | 6.08% |

配对：trueopt_h10 − trueopt_h5 = **−10,231 [−11,115, −9,363]**；oracle − trueopt_h10 = −881 [−1,048, −714]。
→ **H=10 基本吃满 horizon 收益**（H=10 与无界仅差 0.9s）；H5→H10 的 10.2s 是可部署预测器可争取的最大单项（需 H=10 标签/重训）。

## 2. 消费方式（预测输入）

| 臂 | completion | queue | miss |
|---|---|---|---|
| E2（表 p50） | 198,052 | 107,790 | 8.64% |
| **risk（预测 p90）** | **187,546** | **96,510** | **7.63%** |
| aligned layer（cache-aware 表 p50） | 198,702 | 108,025 | 8.74% |
| length-only（lane 均值） | 185,718 | 95,715 | 7.85% |
| survival | 197,329 | 106,652 | 8.40% |

配对：risk − E2 = **−10,505 [−11,539, −9,511]**（miss −1.01pp）；aligned_layer − E2 = +650（略差）；risk − survival = −9,783；risk − length-only = **+1,829（仍不如 lane 均值）**。

## 3. 结论（本阶段可支撑）

1. **horizon 是最大可部署杠杆**：H5→H10 oracle 等价收益 10.2s；H=10 饱和（H20 无额外收益）。
2. **资源分位数在"风险敏感消费"下确实转化**：predicted p90 作未来成本相对表 p50 提升 10.5s；此前"资源不转化"的结论只对 p50 替换与错误语义成立。
3. **身份成本仍然有害**：lane 均值（length-only）比 risk 还好 1.8s；最优组合可能是"长度截断 + lane 均值 + 风险倾向"，尚未实现。
4. cache-aware（表 p50）无增益（+0.7s），与"表已含位置信息"一致。

## 4. 边界与后续

- 边界同前（validation 1,000 集；J3:seed11；T_final 封存；无 S_* 拟合）。
- 建议下一步（按价值）：
  1. **H=10 预测器**（需重新立项/新标签/门禁）——把 10.2s 从 oracle 变成 deployable；
  2. **消费方式细化**：risk 的 λ/分位数扫描（p50+p90 混合、p95）、风险+长度+lane 均值的组合策略；
  3. 若走论文：主线=「长度/终止 + horizon + 风险消费」三件套，资源逐点 p50 与身份成本不转化。
- 产物：`outputs/phase5_oracle_horizon_1000/`、`outputs/phase5_consumers_1000/`、`outputs/phase5_smoke50/`；统计 `.scratch/phase5_stats.py`；新增策略 `trueopt_h10/h20`、`predopt_h5_risk`（仅新增）。
