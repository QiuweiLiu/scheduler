# HANDOFF

## Goal
四条**联合基线**（faithful adaptation）对比 + 完全体证据链：
E2E 胜出 -> 固定 scheduler 换 predictor / 固定 predictor 换 scheduler 的 2x2 -> 四项消融。

## Done
- **P0 拓扑契约回归已修复**：v04 因果模板重建，640/640 过 v3.1 seriality gate，
  `chain_tail_is_answer` 640/640。机制在 `_resolve_edges()` 的 step 展开；**源数据无错**。
- **`load_templates` 有显式 `topology_view`** + 两条 fail-closed 规则。
- **调度机会量化**：修正图下 `Pr(feasible_actions>=2)=52.82%` —— 调度问题成立。
- **融合素材 0.2% -> 49.3%**（2,076 可省边界，最长 13 节点链）。
- **四条论文基线全部实现并过 fidelity**：
  - TIE-adapted 10/10（train-only current-node 分布 + `E+β·CVaR`）
  - Pythia-adapted 8/8（train-only profiler + `1/E[D_remaining]`，ω2=0 已声明）
  - LLMSched-adapted 12/12（BN + ε-greedy 双排序，结构信息在链式负载上为零）
  - Latency-Aware-adapted 12/12 + 9/9（**独立调度器**，Eq 3/4/5/7/8/11/12）
- SRTF+Aging 退役为 classical negative-control。
- F0 打包器 `attr_feature` bug 已修（sha `586ae65d...`），"F0 更保守 1.327x" 已撤回。

## Verified
- 全量 **276 个测试**，5 个失败套件**全部预先存在**（4 个缺数据文件 + `round_robin` bug）。
- 修正图 vs 错图（30 集）：所有策略慢 5.5–9.2%，量化了假并行的虚假收益。
- 独立调度器：132 个融合单元（旁路版 5），14 次预取，语义保持（16 job 全完成、0 失败）。
- 预测器侧全部结论不受影响，含 `F0 相对 J3 = -28.7%`。

## Open
- **`round_robin` 实跑 myopic**（缺 `return` 导致穿透到链尾 `else:`）—— 预先存在，**未修**。
  **任何引用 round_robin 的结果不成立。** 修法：块尾加一行 `return`。
- Latency-Aware 的 Scheduler 仍缺：**reclaim victim 选择**、**Eq (7) 跨候选时间线传播**
  （已在 `test_reclaim_victim_policy_is_still_pending` 显式记录）。
- **需要重新审计的 scheduler-facing 结果**（基于错图 v03）：
  6 臂 smoke、72,773 决策追踪、winner disagreement/Kendall τ、aging 激活与 +12%。
- **30 集正式对比未跑**（目前只有 3 集探针）。
- **`SchedulerTopologyContractGate` 未冻结**（GPT 要求的 P0 门禁）。
- **640 vs 648** 未从 manifest 确认。
- Latency-Aware 的预取测试用的是 **3-GPU 合成数据**（真实 workload 全是 2 GPU）——
  需补一个代表性测试。
- 2x2 的 `P_full` 定义未定；`P_no-prefix`、`F0-point-collapse`、`P_joint` 都不存在。

## Active
无长时任务在跑。

## Next
1. 跑 **30 集正式对比**（四条基线 + 经典基线，v04 修正图）。
2. 冻结 `SchedulerTopologyContractGate`；确认 640 vs 648。
3. 在 v04 上重跑受影响的 scheduler 结果，与 v03 对比（raw vs causal delta）。
4. 补齐 Latency-Aware 的 reclaim victim + 跨候选时间线传播（可选）。
5. 决定是否修 `round_robin`。
