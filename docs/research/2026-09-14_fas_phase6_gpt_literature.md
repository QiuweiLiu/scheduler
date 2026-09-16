# FAS Phase 6 — GPT 文献调研回复 + 文献交叉验证 + 下一步计划

日期：2026-09-14（记录 Phase 6 三阶段结果发出后 GPT 的回复）
来源：ChatGPT Web 桥接会话 `https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e`
桥接恢复说明：Chrome for Testing (`E:\chrome-for-testing\chrome-win64\chrome.exe`) 必须带
`--proxy-server=http://127.0.0.1:7897`（本机直连 chatgpt.com 超时），profile `~/.chrome-chatgpt-bridge`。

---

## 1. GPT 回复要点（原文摘编）

### 1.1 对"消费方式"的判断

- 单步 p95 求和假设完全相关：`Q0.95(ΣXi) ≠ ΣQ0.95(Xi)`，没有利用长度分布、节点相关性、queue state。
  → Phase 7 重点应是"消费器"，不是继续提高预测精度。
- 优先级排序：
  1. **A 场景采样 + CVaR（★★★★★）**：用已有 runtime 分位数 + 长度分布 + 属性分布，采样 N=32 条轨迹，
     `Score = E(C) + κ·CVaR0.9(C)`，κ ∈ {0, 0.5, 1}；对照 E2 / q95 / scenario-CVaR。预计 <2 小时。
  2. **B 自适应风险系数（★★★★）**：`Score = Q0.5 + λ(s,q)·(Q0.95 − Q0.5)`，λ 随 slack/queue 变化
     （queue 高 → λ=1，空闲 → λ=0）；对照 static p95 / static p50 / adaptive。
  3. **D queue-aware rollout（★★★★）**：简化 MPC，对每个候选模拟未来 5 步（当前 queue + 预测链 → completion time）。
  4. **C 分量拆分（诊断）**：定位 p95 收益来自 runtime 还是 load。
- H10 负迁移应解释为 "horizon expansion introduces representation dilution"；下一步 H10-lite：
  保留 H5 backbone + 额外 horizon 辅助损失（multi-resolution）。
- 论文主张建议：
  "Future workflow uncertainty, rather than point resource prediction, is the dominant scheduling signal
  for video agents."（证据链：未来结构 > 无未来 17s；H10 oracle > H5 10s；oracle resource ≈ table；
  risk-aware consumption > naive expected cost。）
- 执行顺序建议：Step 0 先切 70/30 validation（dev/confirm，防过拟合）；Step 1 Phase 7-A；
  Step 2 adaptive risk；Step 3 轻量 MPC；Step 4 H10-lite。

### 1.2 GPT 给出的文献列表（需核实）

Clockwork (OSDI 2020)、Orca (OSDI 2022)、Sarathi-Serve (OSDI 2024)、DistServe (OSDI 2024)、
AlpaServe (MLSys 2024)、Tiresias (NSDI 2019)、Themis (ISCA 2020)、Gavel (OSDI 2020)、Pollux (OSDI 2021)。

**评估：偏旧且部分描述不准**（如 AlpaServe 并非"利用 workload distribution 做 placement"；Themis 是公平/撤销
而非 uncertainty）。这批论文的共同点（预测只是输入，贡献在决策目标）成立，但与我们"预测未来链 + 消费"
最贴题的近期工作未被覆盖。

---

## 2. 交叉验证后的贴题文献（含与我们的对应关系）

| 论文 | 会议/年份 | 消费机制 | 对我们的意义 |
|---|---|---|---|
| A Stochastic Approach for Scheduling AI Training Jobs in GPU-based Systems (STS) | IEEE TCC 2024 | 用**早停概率分布**做资源分配：到期前动态加资源、低功率起步，最小化期望能耗并满足 due date | 直接先例：终止/长度分布 → 决策；但目标是能耗+due date，不是 completion |
| Risk-Aware Proactive Scheduling via CVaR | AAAI 2023 | 用 CVaR 优化 robust makespan，组合 CVaR 最小化的分支限界 | 风险度量用于调度的直接先例；我们做的是"预测链上的场景 CVaR" |
| PAL: Variability-Aware Policy for Scheduling ML Workloads in GPU Clusters | 2024 (arXiv 2408.11919) | 用实测**性能变异度 profile** 做 placement/打包 | 变异度感知消费；与我们的分位数风险不同源但同类问题 |
| Dancer: Deadline-Aware Online Job Scheduling for Distributed Training | IEEE TCC 2025 | 动态调整 GPU 数量与类型；ILP + 在线算法最大化按时完成数 | 纯 deadline 目标、无未来链预测 |
| UniSched | IEEE TC 2024 | 估计器（时长）+ 选择器；支持不同停止准则（迭代/性能达标） | "停止准则"≈ 我们的提前终止；把 estimator 与 selector 解耦 |
| Cuckoo: Deadline-Aware Job Packing on Heterogeneous GPUs | SoCC 2025 | stage 粒度执行时间估计 + 干扰建模 → 最大权匹配/最大流打包 | 干扰/共置建模（我们的 queue 干扰问题的系统版） |
| Llumnix: Dynamic Scheduling for LLM Serving | OSDI 2024 | 运行时**重调度** + KV cache live migration；virtual usage 统一目标 | cache/residency 方向的系统先例（Phase 3 后续的参照） |
| ARES: Adaptive Rescheduling in PD-disaggregated LLM Inference | 2025 (arXiv 2510.13668) | 用 LLM 内部隐状态预测**剩余生成长度** → 多阶段重调度 | 与我们最像：预测"还剩多少"→ 重调度决策 |
| Online Scheduling for LLM Inference with KV Cache Constraints (MC-SF) | 2025 (arXiv 2502.07115) | 预测输出长度（**上界保守**）+ KV 内存可行性检查 + 预留 α=0.1 保护带 | 预测不确定性 → 保守可行域；与我们"风险消费"同构 |
| Efficient LLM Scheduling by Learning to Rank | NeurIPS 2024 | 不做精确长度，只预测**相对排序**以近似 SJF/SRTF | 备选路线：排序消费 vs 数值消费 |
| NexusSched（预测编排） | 2025 (arXiv 2509.23384) | 结构性在线性能模型 → SLO-aware 批调度 + 预测式路由 | 预测信号跨层消费的先例 |

**结论（对文献的整体判断）**：
- 集群调度侧：预测主要被消费为 ① 排序/优先级（SJF 类）、② deadline 可行性、③ 能耗/终止分布优化、
  ④ 变异度/干扰感知 packing；**很少有工作对"预测的未来链"做联合场景 + CVaR 优化**。
- LLM serving 侧：长度预测 → 批处理/内存可行性（保守上界 + 保护带）、重调度、排序。
- 我们的 Phase 7-A（场景采样 + CVaR）在"agent workflow 链预测"这一设定下具新颖性，且与上述两侧都有承接。

---

## 3. 采纳的下一步计划（待用户批准）

- **Step 0**：validation 1,000 集切 70% dev / 30% frozen confirm；后续所有消费器调参只在 dev，
  胜出配置在 confirm 只跑一次（防过拟合）。
- **Step 1（Phase 7-A）**：场景采样消费者（用 `--emit-distributions` 重打包 H5 artifacts）：
  N=32 场景、κ ∈ {0,0.5,1}、对照 E2 / q95。
- **Step 2（Phase 7-B）**：自适应风险系数 λ(slack, queue)。
- **Step 3（Phase 7-D）**：轻量 queue-aware rollout（简化 MPC）。
- **Step 4（诊断）**：C 分量拆分（runtime/load 分项）。
- **H10-lite（后续）**：H5 backbone + 后段 horizon 辅助损失（multi-resolution），修表示稀释问题。
- 论文主线：以"未来 workflow uncertainty 的消费"为核心，不写"预测 runtime 帮助调度"。

---

## 附：GPT 回复原文（存档）

见 `.scratch/export_conversation2.md`（完整导出，message 5）。
