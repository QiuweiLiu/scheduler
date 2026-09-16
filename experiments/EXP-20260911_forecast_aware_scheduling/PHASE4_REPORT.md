# Forecast-aware scheduling — Phase 4：内容消费通道（C1/C2 + survival）（1,000 episodes，2026-09-11）

状态：完成。**结论：内容身份信息对调度没有正价值（甚至有害）；消费 q(A) 分布仅有小幅增益；长度/终止结构仍是唯一主要价值来源。**

## 0. 实验设计（全部 1,000 episodes，paired bootstrap，0 failed）

| 臂 | 制品 | 策略 | 回答 |
|---|---|---|---|
| E2（参照） | J（变长、argmax 身份） | predopt_h5 | 既有基线 |
| length-only | J 步身份置 "unknown"（用 lane 均值计价） | predopt_h5 | 去掉模型身份是否有损失 |
| oracle-content（C1） | 真值身份、保留预测长度 | predopt_h5 | 完美内容是否有收益 |
| expected-cost（C2） | 固定 5 步 + 每步 `expected_cost=Σ P(model)·表成本` | predopt_h5_exp | 消费 q(A) 分布是否有收益 |
| survival | 固定 5 步 + 每步 `P(T≥h)` 权重 | predopt_h5_surv | 用长度分布替代 argmax 截断 |

## 1. 聚合（completion / queue / miss）

| 臂 | completion | queue | miss |
|---|---|---|---|
| E2 | 198,052 | 107,790 | 8.64% |
| **length-only** | **185,718** | **95,715** | **7.85%** |
| oracle-content | 212,274 | 122,120 | 10.71% |
| expected-cost | 210,637 | 131,169 | 10.31% |
| survival | 197,329 | 106,652 | 8.40% |
| J-fixed5（参照） | 212,984 | 135,927 | 10.68% |

## 2. 配对统计（A − B，95% CI）

| 对比 | completion Δ | queue Δ | miss Δ |
|---|---|---|---|
| **length-only − E2** | **−12,334 [−13,634, −11,077]** | −12,075 | −0.79pp |
| **oracle-content − E2（C1）** | **+14,222 [+12,850, +15,692]** | +14,330 | +2.07pp |
| expected-cost − E2 | +12,585（形态混淆） | +23,379 | +1.67pp |
| **expected-cost − J-fixed5（形态匹配）** | **−2,347 [−2,746, −1,960]** | −4,758 | −0.37pp |
| **survival − E2** | **−723 [−968, −475]** | −1,138 | −0.24pp |
| survival − J-fixed5 | −15,655 [−17,118, −14,248] | −29,275 | −2.28pp |
| **survival − length-only** | **+11,611 [+10,424, +12,887]** | +10,936 | +0.55pp |

## 3. 解读（回答用户的 Q1）

1. **内容精度不是"没有消费通道"的问题**：C1 把内容换成真值后**显著更差**（+14.2s）。更好的内容没有带来更好的调度 → 内容身份信息在本策略中无正价值。
2. **模型身份信息本身有害**：把身份换成 lane 均值（length-only）**显著更好**（−12.3s vs E2）。逐模型成本（来自稀疏模板统计）比 lane 均值更噪、更尖，伤害决策。
3. **消费 q(A) 分布只有小幅价值**：形态匹配下 expected-cost 比 argmax 身份好 2.3s，但仍远不如 lane 均值。
4. **长度/终止信息仍是唯一主要价值**：survival 权重（用 P(T≥h)）与 argmax 截断几乎等价（−0.7s），但比固定 5 步好 15.7s。
5. 因此 Phase 1 的答案是：**"内容无用"不是假象**（oracle 内容反而更差），而**"何时结束/持续多久"**才是调度可利用的信息；资源信息在 H=5 下同样没有价值。

## 4. 边界与后续

- 边界同前（validation 1,000 集、单种子 J3:seed11、T_final 封存、无 S_* 拟合）。
- 尚待执行（网页规划剩余）：Phase 2 cache-aware 对比（length-only / topology+cache / topology+resource+cache）；Phase 3 H=10（重新立项）。
- 论文主线建议（增强版）：**"未来预测的调度价值来自工作流延续性（是否继续/持续多久）的不确定性感知规划；内容身份与 naive 资源替换不转化（后者缺 execution-state alignment）"**。
- 产物：`outputs/c1c2_{lenonly,expected,oracle,surv}_1000/`；制品 `outputs/c1c2/j_*/prediction_artifacts/`；构建脚本 `scripts/build_c1c2_artifacts.py`；统计 `.scratch/c1c2_stats.py`。
