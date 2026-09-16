# 仓库审阅后的修复与重跑 v2（P0/P1/P2 + 审计修正，2026-09-17）

背景：GPT 对公开仓库（https://github.com/QiuweiLiu/scheduler）做了两轮审阅：
第一轮列出 5 P0 / 4 P1 / 4 P2；第二轮（审计）指出 2 个残留 P0（CP-SAT 的 executed 变量、指纹可绕过）与 3 处**过度声明**。
本报告是修正后的最终记录；所有数字均为配对 bootstrap（B=2000）。

## P0 修复

### P0-1 CP-RHO 参考实现：量纲 + 结构 + executed 变量（三轮修复）

1. **量纲**（第一轮审阅）：三个目标项统一到「毫秒 × OBJECTIVE_SCALE」；fallback 使用相同权重与毫秒量纲（`MEMORY_REFERENCE_MS=60000`）。
2. **结构**（我们发现）：每节点恰选一个候选且 `future_cost_ms` 不依赖 GPU ⇒ 未来项为常数、无法影响决策。
3. **executed 变量**（第二轮审计）：原实现用 `sum(is_earliest)>=1`，双 GPU 下多个候选可同时 start=0，被计费的动作与实际返回的动作可能不同。
   已改为**唯一** `executed` 布尔（`sum(executed)==1`、`executed<=selected`、executed ⇒ 最早开始），并**直接返回 executed 候选**。

**重跑（预指定=前 300 集；配对）**

| 臂 | episodes | completion | miss |
|---|---|---|---|
| r95（贪心 + 尾部求和） | 300 | **169,064** | 5.95% |
| cp_rho_h5（executed 修复） | 300 | 188,134 | 8.36% |
| cp_rho_h3（executed 修复） | 297 | 189,958 | 8.65% |
| cp_rho_h5（仅量纲修复，未来项仍为常数） | 300 | 204,153 | 10.20% |
| cp_rho_h5（旧目标） | 300 | 203,695 | 10.15% |

- r95 − cp_rho_h5(executed) = **−19,069 [−22,653, −15,911]**；r95 − cp_rho_h3 = −22,034 [−25,363, −19,053]；
- executed 修复相对 earliest 修复：**−2,937 [−4,852, −1,259]**（同 episode 子集）；
- **量纲修复单独几乎无效**（203,695 → 204,153）→ 主因是结构缺陷。

**求解器诊断（300 集 / 每集约 261 次决策）**：fallback 0%；平均 solve 47.6ms、p95 150.8ms；
**命中 0.25s 时间上限 15.4%**（最坏一集 91%）；平均 gap 10.3%（最坏 69%）；状态 OPTIMAL 56,448 / FEASIBLE 13,000。
→ 表述应限定为"在该时间预算与窗口化 V1 边界下，优化型参考仍落后贪心 + r95 约 17s"（**exploratory**，非全量 1,000 集）。

### P0-2 运行器指纹（含第二轮审计的绕过路径）

- 指纹现在覆盖：templates / episodes / episode-ids / artifact **manifest** + **artifacts 实际 gzip 载荷摘要**（`artifacts_payload_digest`）
  + 4 个语义代码文件 SHA256 → 扰动 artifacts（保留旧 manifest）不再能绕过；
- resume 时指纹不一致**拒绝**；**无指纹的 legacy 结果也拒绝**（除非显式 `--allow-fingerprint-change`）；
- 结果行内指纹不一致（混合行）同样拒绝。

### P0-3/P0-4/P0-5
normalizer 重算 manifest 哈希（已核验）；tail-shuffle 生成脚本与分析脚本公开（`scripts/perturb_future_artifacts.py`、`scripts/analysis/`）；`configs/requirements.txt`。

## P1 修复与重跑

### 统计推断（≥2000 + 双侧 + 聚类）

| 结果 | 数值 |
|---|---|
| T1 run 层方差分量（**未截断 MOM 对比量**） | −0.1389，聚类 bootstrap 95% CI **[−0.1557, −0.1233]** |
| T1 截断 ICC_run / ICC_video | 0 / 0.0008（MixedLM 未收敛，仅参考） |
| run 内平均协方差（base / rich-fixed） | −0.1222 / −0.0631 |
| T2 总 / 条件视频 / 位置调整 | −0.242 / −0.212 / −0.176 |
| **T2 首次出现** | **+0.688，CI [0.652, 0.717]** |
| **T2 后续出现** | **−0.261，CI [−0.298, −0.227]** |
| T2 计数–成本耦合 | Pearson −0.173 / Spearman −0.256 |
| T3 尾部 lift（q=0.8/0.9/0.95） | 0.930 / 0.701 / 0.311 |
| T3 双侧置换 p（(count+1)/(B+1)） | 0.538 / 0.201 / 0.247（均不显著） |

**措辞纪律（按审计修正）**：
- **不写**"run 层方差显著为负"——真实方差不能为负；应写"**未约束的 MOM run 层分量估计为负（CI 全负），未发现正的 run 层随机效应证据**"，并辅以直接协方差与截断 ICC。
- **首次调用 +0.688** 仅作**描述性发现**（含聚类 CI）；"冷启动导致"只能写 hypothesis，不写因果。
- T3 的置换未保留 video cluster；由于结论是"无证据"，可以写"未发现尾部共动证据"，**不可**写"证明负相关"。

### P1-8 fixed-argmax-L 场景对照（**retrospective sensitivity**，非 confirmatory）
1,000 集：r95 180,407 vs comon128-fixedL 195,484（+15,077 [13,727, 16,402]）vs scen128-fixedL 197,921（+17,514 [16,027, 19,135]）；
sampled-L 版为 +14,141 / +19,284 → **固定长度后场景族仍显著落后**，公平性边界关闭。

## P2
LICENSE(MIT)、`docs/PROVENANCE.md`、`scripts/reproduce_main.sh`、README 数据可用性措辞；置换 p 值改为 (count+1)/(B+1)。

## 更新后的结论（可写进论文的版本）

1. **未发现正的运行层共享因子**：未约束 MOM 分量 −0.139（CI [−0.156, −0.123]）、截断 ICC 0、直接运行内协方差 −0.122（rich-fixed −0.063）；
   尾部共动 lift 点估计 <1 但**双侧检验不显著**（p = 0.54/0.20/0.25）。
   **描述性结构发现**：首次调用的跨类型残差相关 **+0.69 [0.65, 0.72]**，后续调用 **−0.26 [−0.30, −0.23]**（与 cold-start / warm 组成效应一致，作 hypothesis）。
2. **消费规则的结论稳定且更强**：固定长度后场景/CVaR 族仍落后 15–17.5s；放大版 p50 落后 13s；打乱尾部落后 7s。
3. **优化型参考（exploratory）**：在 0.25s/决策预算、窗口化 V1 边界与 15% 时间上限命中下，修好后的未来感知 CP-SAT 仍落后贪心 + r95 约 19s（300 集预指定子集）。
4. 叙事保持 **forecast-error-aware ranking surrogate**。

## 产物
- 运行：`outputs/phase14_fixedl_1000/`、`outputs/phase15_cprho{3,5}_executed300/`、`outputs/phase14_cprho{3,5}_fixed/`、`outputs/phase14d_cprho{3,5}_futureaware/`
- 统计：`.../trace_dependence_report_v4.json`；脚本 `.scratch/phase14*_stats.py`、`.scratch/phase15_cprho_stats.py`
