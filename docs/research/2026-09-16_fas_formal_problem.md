# Forecast-aware Scheduling：形式化问题描述（v1，2026-09-16）

> 目的：把我们研究的问题写成严格数学形式，作为论文 §3 的骨架，并把命题 A–D 放进同一套记号。
> 记号尽量与实现一致（`workload_v02_simulator.py` 的语义）。

## 1. 记号

| 符号 | 含义 |
|---|---|
| $j \in \mathcal{J}$ | 一个 job（一次视频 Agent 请求） |
| $T_j = (V_j, E_j)$ | job $j$ 的工作流模板：DAG；$V_j$ 为节点集，$E_j$ 为依赖边 |
| $v \in V_j$ | 一个节点（工具/模型调用）；$\ell(v) \in \{\text{gpu}, \text{cpu}\}$ 为 lane；$m(v)$ 为模型身份 |
| $R_v$ | 节点真实 runtime（随机变量） |
| $L_v$ | 冷启动加载时间（仅当 $m(v)$ 不在 GPU 驻留集中才发生） |
| $a_j$ | 到达时间；$d_j$ | 截止期（可选） |
| $g \in \mathcal{G}$ | 一张 GPU；$C_g$ 为显存容量；$\mathrm{res}(g)$ 为驻留模型集合 |
| $s$ | 系统状态（空闲 GPU、驻留、就绪集合、队列长度、slack 等） |
| $C_j$ | job $j$ 的完成时间（flow time 的终点），$\mathrm{CT}_j = C_j - a_j$ |
| $\pi$ | 调度策略（在每个决策点选一个 (node, GPU) 动作） |
| $e \in \mathcal{E}$ | 一个 episode（多 job 到达序列 + 初始状态 + GPU 拓扑） |

## 2. 执行模型（事件的真实动力学）

工作流按依赖推进：节点 $v$ 在前驱全部完成后**就绪**。一个 GPU 同一时刻只执行一个节点；
执行节点需要其模型驻留，否则先付加载代价 $L_v$；显存不足时按实现规则驱逐。

**真实（隐藏）执行**：每个节点 $v$ 的 runtime 由环境根据模板的真实测量分布抽样；
调度器**不可见**真值，只能看到 (i) train-only 历史估计 $\hat{e}(v)$，以及 (ii) 预测器给出的未来预报（下节）。

## 3. 预测接口（我们新增的信息层）

对每个就绪节点 $v$（锚点），预测器给出其**未来行动链**的预报：

$$
\mathcal{F}(v) \;=\; \Big(\; \hat{L}(v),\;\; p_v(\ell),\;\; \big\{\,(\hat m_h, \hat\ell_h, \hat\pi_h),\ Q_{\tau}(R_h),\ Q_{\tau}(L_h),\ \mathrm{occ}_h\,\big\}_{h=1}^{H}\;\Big)
$$

- $\hat L(v) \in \{0,\dots,H\}$：预测**剩余步数**（$p_v$ 为其分布，$\hat L$ 为其 argmax）；
- 第 $h$ 步：预测的模型/角色/lane、runtime 分位数 $Q_\tau(R_h)$（$\tau \in \{0.5,0.9,0.95\}$）、
  加载发生概率 $\mathrm{occ}_h$ 与条件加载时长分位数 $Q_\tau(L_h)$。
- 约定：$\mathcal{F}(v)$ 描述的是"**$v$ 之后**"的链（不含 $v$ 自身）。

**关键**：$\mathcal{F}(v)$ 只给出**边际**分位数与长度分布，**不给出跨步的依赖结构**（copula 未知）。
这正是本文要研究的决策问题的根源。

## 4. 在线决策问题（真实目标）

决策时刻 $t$：至少一张 GPU 空闲且至少一个节点就绪。动作空间

$$
\mathcal{A}(s) = \{\, a = (v, g) \;:\; v \text{ ready},\ g \text{ free},\ v \text{ 显存可行} \,\}
$$

策略 $\pi$ 在 $t$ 选 $a_t \in \mathcal{A}(s_t)$，状态按第 2 节动力学转移。

**目标**（我们评测的）：最小化 episode 内 job 完成时间均值（并报告 deadline miss）：

$$
\min_{\pi}\ \ \mathbb{E}\Big[\ \frac{1}{|\mathcal{J}|}\sum_{j} \mathrm{CT}_j \ \Big]
$$

**理想（oracle）动作价值**：对候选 $a=(v,g)$，其"真实代价"是把 $v$ 放在 $g$ 上执行后，
**该 job 剩余全部工作**（含 $v$ 自身、后继节点、以及驻留/加载效应）的真实耗时：

$$
J^{\star}(a) \;=\; \mathbb{E}\Big[\; \underbrace{R_v + L_v\cdot\mathbb{1}\{m(v)\notin \mathrm{res}(g)\}}_{\text{当前节点}}
\;+\; \sum_{u \in \mathrm{suffix}(v)} \big( R_u + L_u\cdot\mathbb{1}\{\text{未驻留}\} \big) \;\Big|\; \text{residency 演化} \;\Big]
$$

（我们实现中的 `oracle` / `trueopt_*` 是它的近似：用模板真值逐层累加 + 简化驻留模拟；
它们**不含**排队/竞争/连锁效应，因此是"信息上界"，不是策略上界。）

## 5. 消费函数：我们要研究的对象

调度器实际使用的是**代理分数**：

$$
\mathrm{Score}(a) \;=\; \underbrace{\hat c_{\text{current}}(v,g)}_{\text{当前节点（表估计）}} \;+\; \Phi\big(\mathcal{F}(v),\, s\big)
$$

其中 $\Phi$ 称为**消费函数**（把结构化的未来预报折算为一个标量）。我们比较的实现族：

| 记号 | 定义 | 含义 |
|---|---|---|
| $\Phi_{\tau}^{\text{sum}}$ | $\sum_{h=1}^{\hat L} Q_\tau(R_h)$ | 逐步骤 $\tau$ 分位求和（$\tau{=}0.95$ 即 **r95**，当前冠军） |
| $\Phi_{\text{len}}$ | $\hat L \cdot \bar c_{\text{lane}}$ | 只用长度 |
| $\Phi^{\text{scen}}_{\kappa}$ | $(1-\kappa)\mathbb{E}[\Sigma] + \kappa\,\mathrm{CVaR}_{0.9}[\Sigma]$，$\Sigma$ 由独立采样联合分布得 | 分布化风险度量 |
| $\Phi^{\text{comon}}_{\kappa}$ | 同上但用共单调耦合 | 相关结构对照 |
| $\Phi^{\text{ada}}$ | $\sum_h \big[Q_{0.5}(R_h) + \lambda(s)(Q_{0.95}(R_h)-Q_{0.5}(R_h))\big]$ | 状态自适应风险 |
| $\Phi_{\text{cc}}$ | 先按 $\mathbb{P}(\text{finish}>d_j)$ 分档再排序 | 机会约束 |

**核心经验事实**（1,000 episodes，配对 CI）：

$$
\Phi^{\text{sum}}_{0.95} \ \succ\ \Phi^{\text{scen}}_{\kappa},\ \Phi^{\text{comon}}_{\kappa},\ \Phi^{\text{ada}},\ \Phi_{\text{cc}},\ \Phi_{\text{len}},\ \Phi^{\text{sum}}_{0.9},\ \Phi^{\text{sum}}_{0.5}
$$

（$\succ$ = 更小的 mean completion；r95 与 scale-matched p50 差 13 s，与 tail-shuffled 差 7 s。）

## 6. 理论问题（为什么有效）

记候选 $a$ 的未来总时长为 $S_a = \sum_{h=1}^{L_a} R_{ah}$（随机变量，跨步依赖结构未知）。

**Q1（精确性）**：何时 $\Phi^{\text{sum}}_{\tau}(a) = Q_\tau(S_a)$？
**Q2（排序）**：何时 $\arg\min_a \mathrm{Score}(a)$ 与 $\arg\min_a \mathbb{E}[\text{CT} \mid a]$ 一致？
**Q3（部分依赖）**：若真实 copula 介于独立与共单调之间，是否存在比 $\Phi^{\text{sum}}_{0.95}$ 更好的聚合？

**已有结论（经典）**：若 $(R_{a1},\dots,R_{aL_a})$ 共单调，则 $Q_\tau(S_a)=\sum_h Q_\tau(R_{ah})$（Dhaene 2002；Kaas 2002）；
更弱地，只需**上尾共单调**（Cheung 2009）或**尾部共单调**（Hua & Joe 2012），高置信水平下 VaR 仍可加。

**待写命题（论文骨架）**：
- **命题 A**：存在公共因子 $U \sim \mathrm{U}(0,1)$ 使 $R_{ah} = F_{ah}^{-1}(U)$（同一 job 的"快慢状态"作用于所有步）$\Rightarrow$ Q1 成立（等式）。
- **命题 B**：$\mathrm{ES}_\alpha$ 有 $\mathrm{ES}_\alpha(S_a) \le \sum_h \mathrm{ES}_\alpha(R_{ah})$（convex order 上界）；
  **但 VaR 无次可加性**，故 **r95 不是任意依赖下的上界**；任意依赖下的 VaR 上界需 $\alpha_h = 1-(1-\alpha)/H$（并集界）。
- **命题 C**：上尾共单调 $\Rightarrow$ 当 $\tau$ 高于阈值时 Q1 成立；**可用数据检验**：曲线 $q \mapsto Q_q(S)/\sum_h Q_q(R_h)$。
- **命题 D**：若只需 pairwise 排序，且在"同时就绪、单机两作业"下用交换论证，则按 $\Phi^{\text{sum}}$ 排序 = 按 $\mathbb{E}[S]$ 排序（**局部排序引理**，不声称全局最优）。

## 7. 实证协议

- 1,000 个 validation episodes（700 dev / 300 frozen confirm）；predicted/practice 只在 dev 调参；
- 配对 bootstrap（B=2000）比较策略；主指标 mean completion，次指标 deadline miss、p95、queue、驱逐数与利用率；
- 预测器冻结（J3:seed11）；接口一致（同 artifacts、node_id 免映射）；
- 对照：无未来（E0）/ 经典基线（FCFS/SJF/state-aware）/ 同信息点成本（E2）/ 真值信息臂（OracleInfo/OracleTopo）/ 消费者族 / 优化型框架（CP-SAT/MPC）。

## 8. 我们缺什么（待 GPT 审）

1. 记号与假设是否完整（是否漏掉 residency 状态、deadline、优先级等的形式化）；
2. 命题 A–D 的**严格陈述**与**证明套路**（能否直接引用 convex order / comonotonicity 的经典引理）；
3. Q2 在**多 GPU、非同时就绪**的现实设定下能证到什么程度（我们预期只能给局部引理 + 经验证据）；
4. 命题 C 的检验统计量（比值曲线的置信区间/检验方法）；
5. "依赖自适应聚合"与"upper-tail hybrid"如何写进同一形式框架（作为 $\Phi$ 的参数化族）。
