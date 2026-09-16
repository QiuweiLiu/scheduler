# Phase 2 结果的网页版意见（2026-09-11）

对象：forecast-aware scheduling Phase 2（1,000 episodes）结果与下一步。结论：**核心结论基本成立，但需降级表述**。

## 1. 结论表述

- 可支持："未来行为（拓扑）预测提供了可利用的信息，并显著改善了前瞻调度。"
- 不可支持："J3 资源预测有效提升调度"（E3a 明确失败）。
- 更准确的说法："**未来结构预测是当前收益的主要来源；资源预测在当前域偏移与接口条件下尚未转化为调度收益。**"
- E2−E2-legacy 只能写成 **provider/artifact-level improvement**，不能归因于"J3 更强"（形态差异是主要混淆）。

## 2. 下一步优先级（建议）

1. **先做 形态匹配 + seed 稳健性**（论文前必须）：
   - 把 J provider 对齐 B05 规格：固定 H=5、输出 top-k 场景、类似 fallback、相同成本接口；**只改预测内容**；
   - 跑 J3 seed11/22/33，E2 matched vs legacy；
   - 若仍赢，归因成立。
2. **暂缓资源臂大改**。先做最小诊断：同一调度接口下对比 **oracle runtime / J runtime / static table** 三者：
   - 若 oracle runtime 都赢不过 static table → 问题在 scheduler/objective；
   - 若 oracle 能赢 → 问题才在 predictor/resource interface。
3. **oracle gap（14.7%）的最小实验：2×2 交叉**——{pred, oracle} topology × {pred, oracle} resource，定位 gap 来自 topology 还是 resource；不建议先提预测精度（gap 更可能来自调度器利用方式、H=5 窗口不匹配、oracle 含执行真值）。

## 3. 额外效度威胁

1. **零步回退是强混淆**（14.2% 节点无未来成本 → 可能成为"提前结束"信号）：需要 zero-step mask ablation 或统一 H=5 输出。
2. **模板复用**：确认 bootstrap 独立单元不是同模板近重复，否则 CI 偏乐观。
3. **artifact 与 workload 同源**：无泄漏但可能高估泛化 → 需要 S_* exploratory 或新的 sealed workload。
4. **收益可能来自"不确定性降低"而非 topology 本身**（J：prob=1 确定；B05：多场景）→ 建议固定接口下对比 deterministic J vs stochastic J(top-k)。

## 4. 总评

"目前最重要的不是继续提高 predictor，而是证明：**调度收益来自未来行为信息，而不是 artifact 差异或资源接口偏差**。" 该问题回答清楚后，论文主线比单纯追求更低 runtime 更强。
