# Phase 3 后网页版头脑风暴（2026-09-11）

问题：内容精度"无价值"是否因为消费通道缺失；如何更好利用预测精度；H=5 → 10/20。

## 总判断

**不能下结论"内容预测无价值"**。当前实验只验证了"未来长度/终止预测"的价值，没有真正验证"未来内容预测"的调度价值——因为 E2 的消费接口把内容压缩成了 `future node exists? + length`，而不是 `q(A) → resource/risk/cost → scheduling`。内容头处于"预测成功但没有消费通道"的状态。

## Q1：区分"内容无用"与"消费缺失"（不训练新模型）

- **C1：Oracle-content vs Pred-content**——固定相同未来长度、同一调度器、同一资源表，只把 q(A) 换成真实未来属性。
  - Oracle ≈ Pred → 内容本身没有调度价值；
  - Oracle 明显更好 → 预测质量或消费方式不足。
- **C2（更关键）：q(A) → cost distribution**——当前是 future node → 静态表 → cost；改为 q(A) → Σ p(attribute)·resource(attribute) → expected future cost（只用已有表）。
  - 四臂：length-only / length + argmax attribute / length + expected attribute cost / length + oracle attribute cost → 直接回答"属性信息有没有可消费价值"。

## Q2：消费机制排序

1. **Hazard/survival 未来成本（★★★★★）**：argmax 长度丢掉了概率信息；用 `E[C] = Σ_h P(T≥h)·c_h`（survival-weighted cost），替换 argmax 为 expected remaining horizon，无需训练。
2. **Cache-aware 未来模拟（★★★★）**：未来成本应模拟驻留（node1 冷加载、node2/3 热复用），而不是独立求和；最可能让 resource 信息产生价值的方向（已有 aligned cache-aware 路径可优先跑）。
3. **Risk-sensitive planning（★★★）**：cost = p50 + λ(p90−p50) 或 CVaR；适合 deadline 调度，但受资源预测质量限制。

## Q3：H=5 扩展排序

1. **政策侧 stochastic horizon（★★★★★）**：不要马上训 H=20；用已有长度分布做生存外推（`P(T>5)` 作为继续规划概率、滚动期望成本），<1 天。
2. **重训 H=10（★★★★）**：长期正确；需新标签/门禁/验证集；做 10，不要直接 20。
3. **自回归 rollout（★★）**：不推荐（exposure bias / 误差累积）。

## 推荐路线

- **Phase 1（~1 天，不训练）**：C1 oracle-content + C2 q(A)-expected-cost + survival horizon cost → 回答"内容有没有消费价值"。
- **Phase 2（~1 天）**：cache-aware 对比（length-only / topology+cache / topology+resource+cache）→ 回答"资源为什么没有转化"。
- **Phase 3（重新立项）**：H=10（H5 / H10 / survival rollout 对比）。

## 论文故事建议

不要继续追"预测器精度"。更强的故事："未来 agent 预测主要通过**工作流是否继续/持续多久/风险在哪**的不确定性感知规划改善调度；朴素的未来资源替换失败，是因为资源预测缺少**执行状态对齐**（execution-state alignment）。"
