# Phase 4 后网页版重新规划（2026-09-11）

背景：用户指出 C1 只固定了"内容→成本"的消费机制，因此只能回答"当前接口下预测内容 vs 真值内容"，不能回答"内容是否有价值但未被正确消费"。网页版接受该批评并给出两层实验结构。

## 1. 总体判断

- 用户批评**成立**：C1 把 `内容 → cost` 的映射机制固定死了。
- 需要把问题拆成三个潜变量：**① 未来结构/内容信息是否有价值；② 资源估计是否有价值；③ 调度器消费未来信息的方式是否合理**。三者不能混在一个实验里。
- 不做 `content×resource×horizon×consumer` 全组合（会爆炸）；用四个核心臂即可。

## 2. 第一层：信息价值（Axis A）

| 臂 | 内容 | 状态 |
|---|---|---|
| A0 | Predicted Future（现 E2） | 已有 |
| A1 | Oracle Topology H=5 + static resource | 已有 184,514ms |
| A2 | Oracle Topology H=5 + Oracle Resource | 已有 183,845ms（**A1≈A2 ⇒ 资源预测不是瓶颈**）|
| A3 | **Oracle Horizon H=10/20 + Oracle Resource** | **新增（待做）** |

分解：未来行为价值 = E2 vs no-future；预测内容损失 = E2 vs A1；资源预测损失 = A1 vs A2；**未来窗口限制 = A2 vs A3**。

## 3. 第二层：消费机制（最大遗漏）

固定输入 = `Oracle Topology H=5 + Oracle Resource`，比较 consumer：

- **C0**：当前 greedy sum（baseline）
- **C1**：**survival-weighted cost（已 +15.7s，优先级最高）**——直接利用 P(T)，避免 argmax 长度或单点信息
- **C2**：risk-sensitive（`p50+λ(p90−p50)` 或 CVaR；适合 deadline）
- **C3**：**cache-aware simulation（长期）**——现实缺陷：未来节点不独立执行，应是 `cost(node_i | history)` 而非 `cost(node_i)`

实验结构：第一层"信息有没有价值" → 第二层"消费方式有没有价值"。

## 4. 优先级与成本（网页版建议）

1. **Priority 1（立即，半天~1 天）**：`Oracle H=10 / H=20` horizon——H5→无界已有 −11.1s，是目前最大的确定收益来源；不训练；1,000 episodes × 几个 horizon 为分钟级；确认"长期未来信息是否值得"。
2. **Priority 2（~1–2 天）**：consumer ablation（固定 oracle topology + oracle resource；跑 sum / survival / risk / cache-aware），回答"调度器是否以正确方式消费未来信息"。
3. **Priority 3**：预测拓扑 + 新 consumer 组合。
4. **最后**才考虑重新训练 predictor。

## 5. 论文主线（网页版更新建议）

> "For agentic video workloads, the dominant scheduling signal is future workflow evolution rather than per-node resource estimation."

证据链：E2 > no future（未来信息有价值）→ H5 oracle > baseline（未来 topology 有价值）→ oracle resource ≈ table（资源预测不是主瓶颈）→ survival 提升（消费不确定 horizon 比预测单点资源重要）。

最大未知量不是 predictor，而是：**调度器是否以正确方式消费未来信息**。若这一步不解决，继续提高预测精度大概率不会转化。
