# Forecast-aware scheduling — Phase 6：消费方式 + H10 重训 + cache-aware（1,000 episodes，2026-09-11）

状态：三阶段全部执行完毕。**结果：① 消费方式有重大进展（q95）；② H10 朴素重训为负迁移；③ cache-aware 资源消费为负。**

## Phase 1：消费方式（H5 预测器，6 臂）

| 臂 | completion | queue | miss | vs E2（配对） |
|---|---|---|---|---|
| E2（表 p50） | 198,052 | 107,790 | 8.64% | — |
| **q95（CVaR 代理）** | **180,362** | **89,612** | **7.25%** | **−17,690 [−19,238, −16,172]** |
| p90（λ=1.00） | 187,517 | 96,520 | 7.65% | −10,535 [−11,570, −9,519] |
| λ=0.50 | 212,128 | 121,265 | 10.63% | +14,077 |
| λ=0.00（p50 混合） | 214,049 | 123,020 | 10.88% | +15,997 |
| 机会约束（deadline p90） | 213,248 | 122,669 | 8.34% | +15,196（miss −0.30pp） |

- **q95 最好**：相对 E2 −8.9%、miss −1.39pp；比 p90 再低 7,155ms（[−7,836, −6,485]）。
- λ 单调性：p50→p90→p95 收益递增（风险厌恶越强越好，在本 workload/目标下）。
- 机会约束（按 p90 超期风险优先）未带来 completion 收益（miss 略好）。

## Phase 2：H10 重训（新实验 `EXP-20260911_p9d_j_h10_predictor`）

构建与训练（全部新登记，v3.1 契约、horizon=10）：
- v3 标签 H10：`results/processed/topology_predictor_p9d_v3_h10`（行数不变；槽位 74,124/11,083/8,329）；
- J 数据集 H10：`results/processed/j_series_dataset_h10`（93,536 监督实例；长度直方图 0..10）；
- 模型：backbone 3 seeds（269s）+ J3 3 seeds（295s，选 seed22，validation 805.8 最优）；artifacts 打包含 `future_h10`。

系统结果（同 workload，配对）：

| 对比 | completion Δ | 说明 |
|---|---|---|
| **H10 模型 + 10 步 q95 − H5 模型 + 5 步 q95** | **+14,587 [+13,273, +15,907]** | H10 更差 |
| **H10 模型 + 5 步 q95 − H5 模型 + 5 步 q95** | **+33,459 [+30,614, +36,438]** | **H10 模型前 5 步本身显著退化** |
| H10 λ=0 − H5 λ=0 | −2,062 [−2,383, −1,755] | 仅最保守设定略好 |

S_* 质量审计（前 5 步预测，直接证据）：

| 指标 | H5 模型 | H10 模型 |
|---|---|---|
| next-role acc | 0.912 | 0.901 |
| **next-family acc** | **0.803** | **0.640（−16.3pp）** |
| runtime pinball | 1,046 | **1,691（+62%）** |
| load duration p50 err | 888 | **2,181（+146%）** |
| length abs err | 0.408 | 0.406（持平） |

→ **结论：朴素延长标签窗口（同架构/同预算）造成前 5 步质量退化（共享槽位容量被 10 槽稀释），oracle H10 的收益没有转化；H10 需要不同训练配方（如后段槽位降权/分离 head/更长训练/两段式），属后续工作。** 另：H10 模型内部 10 步消费优于只用 5 步（194,949 vs 213,820），说明长窗口信息本身有帮助，但被模型退化抵消。

## Phase 3：cache-aware 资源消费（H5）

- 新臂 `aligned_predopt_h5_layer_res`（layered 前瞻 + 预测分位数 + 驻留模拟）：**220,430ms — 比 E2 差 +22,379 [20,316, 24,558]**（miss +2.76pp）。
- 与网页版判断一致：当前 cache 语义（缺 residency/reuse 数据、memory 块缺失）下 cache-aware 消费无价值，属后续工作（需数据采集）。

## 汇合：目前最佳 deployable 配置

**H5 预测器 + q95 消费 = 180,362ms（相对 E2 −8.9%，miss 7.25%）**；且此前已确认：相对无未来 −7.9%（长度结构），identity/内容与 p50 资源替换不转化。

## 边界

- validation 1,000 集；J3:seed11（H5）/seed22（H10）；T_final 封存；无 S_* 拟合；H10 为独立新实验，不与 H5 混用。
- 新增策略（仅新增）：`predopt_h{5,10}_lam{0,25,50,100}`、`predopt_h5_q95`、`predopt_h5_cc`、`aligned_predopt_h5_layer_res`；loader 修复为通用 `future_h*`（H10 必需）。
- 产物：`outputs/phase6_consumers_h5_1000/`、`phase6b_*`、`phase6c_cache_res_1000/`、`sstar_predictor_artifacts_h10[_sched]/`；统计 `.scratch/phase6_stats.py`。
