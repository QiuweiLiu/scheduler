# 新颖性扫描报告：PDRS 交集是否真的没人做（2026-10-05）

## 方法与范围

自主检索 8 组（agent workflow 调度、结构性不确定性、未来预测+GPU placement、
tool-call speculation、online-revealed DAG、SLO serving、LLM 调度理论），覆盖 2025–2026。

## 结论

**窄交集（per-instance 前缀 → 未揭示后缀概率信念 → node-level GPU placement/ordering，
在线重预测，且不读机器响应模型）在同行评审文献中未发现完全重合者；**
但周边非常拥挤，存在三类近邻，必须精确切割并披露。**最终裁定需正式 related-work audit。**

## 三类近邻与差异

### A. "已揭示前沿"派（最直接的对照）
- **HexAGenT**（arXiv 2605.16637, 2026-05）：online-revealed DAG，但"never plans the full
  workflow graph upfront... decisions over the currently revealed frontier"；horizon 由已揭示子图算。
- **Latency-Aware Orchestration**（arXiv 2609.03335, 2026-09）：workflow forecast + future model
  demand + placement/lifecycle/order 联合优化；但"A branch enters only after its control result
  resolves"——**显式不规划未揭示分支**。
- **FATE**（arXiv 2605.07238, 2026-05）：future-state-aware；已知 DAG + frontier 规划 + 下游状态评分。
- **TOPAS**（arXiv 2608.25523, 2026-08）：DAG + 一跳前瞻；"farther demand depends on unresolved
  branches"——刻意止步于一步。
- **差异**：我们预测的是**未揭示的后缀**（P(suffix|prefix)），它们只用已揭示/已知结构。

### B. "预测未来但用途不同"派
- **PASTE**（arXiv 2603.18897）：从历史模式预测**下一个 tool call**并投机执行（隐藏工具延迟）；
  用途=工具投机，非调度/放置。
- **Speculate While You Reason**（arXiv 2607.25816, 2026-07）：agent 自预测下一个 tool call（RL）；
  用途=投机解码式加速。
- **KVFlow**（NeurIPS 2025）：steps-to-execution 预测 → KV 驱逐/预取；用途=缓存。
- **SAGA**（arXiv 2605.00528, HPDC 2026）：AEG 转移概率 + 最可能后继预取；用途=KV/亲和/批处理。
- **差异**：它们把"下一步预测"用于缓存/工具/带宽；我们用于 **GPU 放置与排序**。

### C. "结构性不确定性 + 在线精化"派（最接近的方法论）
- **JITServe/GMAX**（NSDI 2026, arXiv 2504.20068）：structural uncertainty（DAG 未知）；
  **pattern-graph 增量匹配 + 在线精化**预测依赖；用途=SLO 带宽/批组成（goodput）。
- **LLMSched**（ICDCS 2025, arXiv 2504.03444；**我们的基线之一**）：BN 刻画时长+相关性、
  熵减探索；用途=JCT。
- **AgentIR**（krishmodi.com 项目页, 2026-02；**未见同行评审论文**）：编译 agent 代码为 IR +
  学习分支概率；"uses implied future demand to make online placement decisions" across
  heterogeneous vLLM instances。**精神上最接近我们的方案，但为工业项目页、非正式论文。**
- **差异**：JITServe 的消费是引擎内带宽/批；AgentIR 的预测来自静态编译 IR（结构来自代码而非
  从执行前缀在线推断的 belief），且未披露不依赖机器成本模型。

## 我们可守的四条差异（缺一不可）

1. **未揭示后缀**（不是已揭示前沿/已知 DAG）；
2. **node-level GPU placement + ordering** 的动作级消费（不是缓存/带宽/工具投机）；
3. **per-instance 在线后验重预测**（每揭示一步重算 belief；区别于 workflow-type 图/聚合历史）；
4. **方法不读机器响应成本模型**（区别于 Latency-Aware/FATE/HexAGenT 的 device cost model）。

## 风险与必做

- **风险**：AgentIR + HexAGenT + JITServe 的组合可能被认为覆盖大部分；交集很窄，必须逐条切割。
- **必做**：投稿前正式 related-work audit；论文用 "To our knowledge…"（禁 "first"）；
  AgentIR 作为最近邻披露（即使非正式发表）；把 HexAGenT 与 Latency-Aware 作为实验对照方向讨论。
- **建议的安全表述**：沿用第五轮 GPT 版本（instance-specific + unrevealed suffix + re-predict/
  replan + no machine-response cost model）。
