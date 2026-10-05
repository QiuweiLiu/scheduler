# 第二轮创新点讨论（文献扫描 + 架构挖掘）— 结论：Belief-conditioned Action Evaluation + Actionability 分解

## 来源

网页版 GPT（GPT-5.6 Sol + High）2026-10-05 第二轮讨论：用户提出"概率分布没用上"与
"放哪张卡是动作域问题"，并要求再次文献扫描 + 从现有架构挖掘创新点。

## 核心命题（推荐论文 thesis）

> **Prediction is useful only when it is action-aligned.**
> 未来预测必须进入**动作价值**，而不只是任务优先级：对在线展开的 agent workflow，
> 调度器应维护未揭示未来的 belief，并评价每个当前动作 a=(ready node, GPU) 把系统带向什么未来状态。

研究问题版本：**In online-revealed agent workflows, what future information is actionable,
through which scheduling decisions, and under what system conditions does partial clairvoyance
improve GPU scheduling?**

## 用户两个判断的回应

1. "概率分布没用上"——**属实，且是最有价值的切入点**。F0 把 belief 先压成 node-level 标量
   （p95 和 + 条件 p95 load）再调度；model_probabilities / runtime_probs[16] / termination /
   length 概率全部在进入 scheduler 前丢掉。
2. "放哪张卡是动作域"——方向对；更精确的术语是 **action-conditioned evaluation /
   action-aligned prediction**（动作域本已含 (node,GPU)；缺的是两个 GPU 动作拿到相同未来价值）。

## 本轮必须补的 related work（2025–2026）

- **SAGA (HPDC 2026)**：Agent Execution Graph 带 transition probabilities；概率未来用于 KV reuse /
  workflow-aware eviction / session affinity → **"我们有概率未来"不能再作为创新点**。
- **Scepsy (2026)**：不预测 DAG，用 aggregate LLM pipeline + fractional GPU 联合搜索
  （反方向；reviewer 会问"为什么必须预测未来节点"）。
- **KVFlow (NeurIPS 2025)**：steps-to-execution 预测未来 agent 激活 → KV eviction/prefetch。
- **Halo (2025)**：query-plan DAG + prefill/decode + cache reuse + GPU placement + adaptive batching。
- **理论语言**：Online Flow Time with Gradually Revealed Jobs (2026)（介于 clairvoyant 与非 clairvoyant
  之间的信息模型）；Online Scheduling with a Stochastic Signal (2026)（stochastic clairvoyance，
  部分随机信息可突破 non-clairvoyant 界）；ICML 2023 minimalistic predictions + online precedence。
- **Decoupling Readiness from Release (2026-09)**：把"是否/何时 release"本身作为动作（动作域扩展先例）。
- **POMDP 调度**（2026 IEEE TCAD）：belief-state 可作问题形式化，不可声称首创。

## 架构挖掘：六个候选（排序）

1. **Belief-conditioned Action Evaluation（9/10，推荐主方法）**：
   S_AC(i,g) = S_F0(i,g) + Φ(i,g)，Φ = (1−p_term)·Σ_m b_i(m)·min_h C_state(m,h | s'=T(s,(i,g)))；
   MVP 只用 load + eviction 成本（全部 ms，无 λ 调参）；不重训、不改 substrate。
   **更干净的 placement-only 先行版**：节点选择与 F0 完全一致，只重选 GPU
   → 直接证明"model distribution 的价值来自 placement 而非重排序"。
2. **Information × Actionability 分解（8.5/10，分析型贡献）**：
   同一份未来信息在不同动作域下的价值；含 **shuffled-probability 对照**
   （随机交换各节点的 model_probabilities，保持形状、破坏 prefix 信息）；
   prior → predicted → oracle belief 阶梯，接现有 Myopic→F0→Truth-H5→Oracle。
3. **Coupling-Robust Future-State Evaluation（7.5/10）**：把"无 joint"从缺陷变动机——
   在所有满足边际的 coupling 上做 robust action evaluation（小 OT/LP，无需重训）。
4. Future Action-Set Preservation（并入 #1 作机制）；5. Decision-Relevant Uncertainty Gating（稳健性）；
6. Prediction-Guided Preemption Option Value（暂不碰）。

## 最小实验（4 臂 + 2 对照，全部在冻结 DES/产物上跑）

1. F0（冻结）；2. **F0+placement-only belief**（只换 GPU）；3. Full action-conditioned（node+GPU）；
4. Oracle-next-model placement（天花板）；对照：shuffled model_probabilities、unconditional train-prior。
指标：mean_completion_ms + 机制指标（load_ms、evictions、future-model resident hit、共置/批准入、
干扰 ms、action flip rate）。第一阶段不碰 runtime_probs/role/batching/preemption。

**停止规则**：若最干净的 H=1 placement-only 无收益 → 停止 A*，论文主线转为
**information-actionability characterization**。

审核回复原文如下（未改动）。

---

炜，这一轮重新扫完 2025–2026，尤其把 SAGA、Scepsy、在线逐步揭示调度、POMDP 调度、Agent release scheduling 这些补进去后，我的判断比上一轮更收敛了：

**你现在最有价值的创新点，不应该叫“分位数调度”，也不应该简单叫“GPU-aware future cost”。更准确、更能站住的中心命题是：**

> **未来预测必须进入动作价值，而不仅进入任务优先级。**
>
> 对在线展开的 Agent workflow，调度器应维护对未揭示未来的 belief，并评价每个当前动作 \(a=(\text{ready node},\text{GPU})\) 会把系统带入什么未来执行状态。

而且你的第二个判断基本是对的，但我建议术语再精确一点：

**不是首先强调“动作域设计”，而是强调“action-conditioned evaluation / action-aligned prediction”。**

因为你当前动作域本来就已经有 `(node, GPU)`；真正缺的是 F0 对两个 GPU 动作赋予的未来价值完全相同。只有以后把“wait / preempt / prefetch / batch”等新增成可选动作时，才叫严格意义上的 action-space expansion。

---

## 一、先正面回应你的两个判断

第一个，“我的概率分布没用上”，**属实，而且这其实是现在最有价值的切入点之一。**

F0 当前做的是：

\[
S_{\mathrm{F0}}(i,g)
=
C_{\mathrm{now}}(i,g)
+
F_{\mathrm{p95}}(i)
\]

其中未来项已经压成一个 node-level scalar：

\[
F_{\mathrm{p95}}(i)
=
\sum_{k=1}^{5}
[p95(T_k)+I(P_{\rm load,k}\ge 0.5)p95(L_k)]
\]

所以以下信息全部在进入 scheduler 之前就丢掉了：

- `model_probabilities`
- `runtime_probs[16]`
- `termination_probability`
- `length_probabilities`
- role/family 概率
- “不同未来分支对不同 GPU 的后果不同”这一结构

换句话说，现在 predictor 明明输出了一个 belief，F0 却先把 belief 压成一个标量，再调度。

第二个，“任务放哪张卡应该是动作域问题”，方向也对。当前同一个 node：

\[
F_{\mathrm{p95}}(i,g_0)=F_{\mathrm{p95}}(i,g_1)
\]

因此 predictor 无法表达：

> “如果现在放到 GPU0，下一步高概率模型已经 resident；如果放到 GPU1，下一步可能要 load+evict。”

这意味着你的未来预测**只能改变 who first，不能改变 where**。

所以我会把论文问题从：

> 哪个 future statistic 最适合 scheduling？

改成：

> **哪些预测信息对哪些调度动作是 actionable 的？**

这个问题明显更强。

---

# 二、这一轮必须补进 related work 的工作

这一轮最重要的新发现其实是 **SAGA**。它让“我们有概率 future”本身不能再作为创新。

### 1. SAGA，HPDC 2026：对你威胁很大

SAGA 把整个 Agent workflow 当成 schedulable program，用 Agent Execution Graph：

\[
G=(V,E,P,\phi)
\]

其中 \(P\) 就是 transition probabilities。它还明确支持三种可观测性：framework 直接给 graph、从历史 trace 推断概率 graph、以及 cold-start fallback。概率 future 被用于预测 KV reuse、workflow-aware eviction 和 session affinity。citeturn312072search24turn312072search2

所以不能再说：

> “别人都假定 Agent workflow 已知，而我们有概率未来。”

至少 SAGA 已经不是这样。

但 SAGA 和你仍有一个关键差别：

**它的 probability 主要服务于 workflow-level cache continuity / affinity；我没有看到它对每个 `(ready node, GPU)` 动作，通过真实 GPU residency/load/interference/batching state transition 计算一个 probabilistic downstream action value。**

这正是你可以守的位置。

---

### 2. Scepsy，2026：另一个很重要的反方向

Scepsy专门处理 branch/fan-out/recur 导致的不可预测 Agent workflow，但它没有试图逐步预测 DAG，而是观察到“每个 LLM 占 workflow 总运行时间的比例比较稳定”，构造 Aggregate LLM Pipeline，然后联合搜索 fractional GPU allocation、TP 和 replica。citeturn175527academia35

它对你非常重要，因为 reviewer 可能会问：

> 为什么必须预测未来节点？为什么不能像 Scepsy 一样直接预测 aggregate demand？

因此你的论文最好证明：

**细粒度 future belief 的价值来自短时动作后果——residency、eviction、co-location、batching——这是 aggregate resource share 无法表达的。**

---

### 3. KVFlow，NeurIPS 2025

KVFlow 已经用 Agent Step Graph 的“steps-to-execution”预测未来 agent 激活，并据此决定 KV eviction 和 prefetch。citeturn547119search0

所以：

> “未来需求用于 cache/lifecycle”

也不是你的创新。

你需要强调：

> **uncertain future × current GPU assignment × measured substrate state transition**

而不是普通 future-aware cache。

---

### 4. Halo，2025

Halo 把 agentic workflow 建模为 query-plan DAG，然后联合考虑 prefill/decode、cache reuse、GPU placement、adaptive batching。citeturn547119academia12

所以“workflow DAG + GPU placement + batching”本身也不能当创新。

---

### 5. Online Flow Time with Gradually Revealed Jobs，2026

这个理论工作和你的问题非常匹配：一个 job 是一系列 operation，而后一个 operation 的 processing time 要等前一个完成以后才揭示。作者明确把它称作介于 clairvoyant 与 non-clairvoyant 之间的信息模型。citeturn880513academia25

更早的 ICML 2023 *Minimalistic Predictions to Schedule Jobs with Online Precedence Constraints* 更直接：依赖关系只有 predecessor 完成后才揭示，并研究“哪些额外预测信息能够帮助调度”。citeturn880513search0

这两篇反而给你的研究问题提供了很好的理论语言：

> 你的 Agent workflow 是一种 **online-revealed precedence scheduling**。

这比一直讲“Agent 很动态”学术上强很多。

---

### 6. Online Scheduling with a Stochastic Signal，2026

这篇提出 stochastic clairvoyance：任务在执行过程中随机时间提供部分信息，并证明这种部分随机 clairvoyance 可以突破纯 non-clairvoyant 的界。citeturn880513academia26

这非常适合支撑你的总命题：

> 未来信息不是 all-or-nothing，而是有“部分揭示程度”和“可利用形式”。

---

### 7. General Non-Clairvoyant KV-Cache Scheduling，2026

它研究输出长度未知、随着 decode 逐步揭示的 LLM scheduling，并给出 regime-aware routing，在 KV memory hard constraint 下处理逐步暴露的信息。citeturn289999academia24

它意味着你的 preemption/KV 一类扩展也不是无人区，但它研究的是**一个 inference request 的 execution uncertainty**，不是未展开 workflow suffix。

---

### 8. Decoupling Readiness from Release，2026-09

这篇对“动作域”特别有启发：它指出 ready 不等于必须立即 release，把“是否释放一个 ready turn”本身变成调度决策，并用 mean-CVaR 控制 agent workflow tail latency。citeturn150777academia36

这说明现在 agent scheduler 的 action 已经开始从：

> choose which ready request

扩展成：

> choose whether/when/how to materialize workload into serving system。

所以以后若你把 prefetch/preempt/release 加入 action space，是有文献语境的。

---

### 9. POMDP framing：能用，但不能当主要创新

本轮我没有检索到一个明确的“LLM Agent GPU workflow scheduler = belief-MDP”直接近邻，但 **POMDP scheduling 本身当然不是新东西**。2026 IEEE TCAD 已经有 POMDP + Active Inference 的 heterogeneous multicore task scheduling，用 belief 表示不可观测 task/system state。citeturn289999search0

所以你可以非常自然地写：

\[
b_t=P(G^\mathrm{future}\mid prefix_t)
\]

但不能声称：

> “首次将 POMDP 用于资源调度。”

最好把 belief-state 当作**问题形式化工具**。

---

### 10. 分布/风险这条线反而越来越拥挤

TIE 已经做 distribution-based length scheduling；而 ICML 2026 的 *Beyond Prediction* 甚至证明 prediction-driven SJF/SRPT 在 distribution shift、bursty arrivals、memory pressure 下可能脆弱，并提出 prediction-free tail-aware scheduler。citeturn150777search1

此外，2026 已有多 Agent GPU resource planning 用 chance constraints 处理 stochastic output length。citeturn289999search2

所以这进一步确认：

**“我们用了 distribution / CVaR / chance constraint”不能作为你的 headline。**

---

# 三、从 A–F 资产里真正可以挖出的创新点

这里我给六个，不只 A*。

| 候选 | 核心一句话 | 依赖资产 | 与最近邻的真正区别 | 成本 | 最大审稿风险 |
|---|---|---|---|---|---|
| **① Belief-conditioned Action Value** | 直接计算未揭示 future belief 下每个 `(node,GPU)` 的 downstream cost | A+B+C+D+F | FATE/Latency-Aware 有 future-state；SAGA 有概率 graph；你的差异必须是**概率 unrevealed suffix → 当前具体 GPU action 的 state-conditioned value** | 中 | 被说成 FATE/SAGA 的组合 |
| **② Actionability / Information×Action 交互** | 定量证明“同一预测在 ordering-only 和 placement-aware action 下价值不同” | C+D+E+F | 现有系统通常比较 scheduler，不测“信息价值被哪个动作域解锁” | 小 | 被说成只是 ablation |
| **③ Marginal-Coupling Robust Future Value** | predictor 只有 marginals 时，不假造 joint，而在所有合法 coupling 上做 robust action evaluation | A+B | 比普通 CVaR/DRO更贴合你真实 predictor contract；解决 joint-not-identifiable | 中 | DRO 是已知数学技巧 |
| **④ Future Action-Set Preservation** | 当前 placement 不只影响 cost，还决定下一步有几张卡还能合法共置/批处理 | A+B+D | FATE 有 future reachability，但你有**实测 co-location/batch admission surface + probabilistic future** | 中 | 与 FATE future device reachability 重叠 |
| **⑤ Decision-Relevant Uncertainty Gating** | 只在预测 uncertainty 足以改变最佳动作时使用复杂 future planner | A+D+E | 不是 p95-p50 gating，而是对 action ranking uncertainty 做判断 | 小-中 | 容易被说成 heuristic |
| **⑥ Prediction-Guided Preemption Option Value** | 根据未揭示 future 判断当前 decode 是否值得被打断/保留 | A+B(token phase)+D | 把 workflow future 接入真实 prefill/decode/recompute preemption | 大 | 与 tail-aware/preemption work 重叠且 action-domain 公平性复杂 |

我的判断是，①、②、③最值得保留。

④其实应该成为①里面的一个具体 mechanism，而不是独立论文贡献。

⑤可以作为优化/robustness。

⑥现在不建议先碰。

---

# 四、第一名：Belief-conditioned Action Value

这是我现在最推荐的主方法。

不要一开始做完整 H=5 rollout。

**MVP 就做 H=1。**

设当前可执行动作：

\[
a=(i,g)
\]

其中 \(i\) 是 ready node，\(g\) 是 GPU。

当前系统状态为 \(s_t\)。

执行动作以后：

\[
s'_a=T(s_t,a)
\]

这个 transition 你已经能从冻结 substrate 精确模拟：

- residency
- evict
- capacity
- active model
- load
- measured interference
- batching admission

然后 predictor 已有：

\[
b_i(m)=P(M_{t+1}=m\mid prefix_i)
\]

再加 termination：

\[
p_{\rm term}
\]

定义：

\[
\Phi(i,g)
=
(1-p_{\rm term})
\sum_m b_i(m)
\min_h C_{\rm state}(m,h\mid s'_a)
\]

这里 MVP 的 `C_state` **只需要 load + eviction**：

\[
C_{\rm state}
=
C_{\rm load}+C_{\rm evict}
\]

先不要碰 role、batch、interference。

于是：

\[
S_{\rm AC}(i,g)
=
S_{\rm F0}(i,g)+\Phi(i,g)
\]

全部都是 ms，甚至不用人为调 \(\lambda\)。

这已经把：

> model probability

真正转成：

> GPU placement consequence。

而且完全不用重训 predictor。

---

## 我甚至建议先做一个更干净的“placement-only”版本

这是非常重要的实验设计。

不要第一步就允许预测同时改变 node selection 和 GPU placement。

先做：

1. **完全按照 F0 决定下一步执行哪个 node \(i^*\)**；
2. 只重新选择这个 node 去哪个 GPU：

\[
g^*
=
\arg\min_g
[C_{\rm now}(i^*,g)+\Phi(i^*,g)]
\]

这意味着：

**每一次“先做哪个任务”的决定与 F0 完全一致。**

唯一变化：

> **放哪张卡。**

如果这个 arm 能提升，就形成极其干净的证据：

> model distribution 的价值不是来自重新排序，而来自 placement。

然后再跑 full version：

\[
(i^*,g^*)=\arg\min_{i,g}
S_{\rm AC}(i,g)
\]

这会成为第二阶段。

这个设计我非常喜欢，因为它直接利用了你的 F0 架构缺口。

---

# 五、第二名：不要再做一个 scheduler，而做“Actionability Principle”

我认为这个甚至可以成为论文的第二大贡献。

你的论文最有意思的问题其实可以变成：

> **同一份 future information，在 action space 不同时，到底值多少钱？**

这是一个非常好的 systems question。

做一个 information × actionability 的矩阵。

例如：

| Future info | node ordering only | action-conditioned placement |
|---|---:|---:|
| 无 future | Myopic | no-future placement control |
| node scalar p95 | F0 | — |
| model distribution | distribution ignored | **New** |
| truth next model | — | Oracle-placement |

这里有一个非常关键的 matched control：

### shuffled-probability control

把每个节点的 `model_probabilities` 随机交换给另一个节点，但保持总体 marginal distribution 不变。

于是：

- consumer 完全相同；
- distribution shape 完全相同；
- 只有**prefix-specific information**被破坏。

如果 New > shuffled，就能证明收益不是“多加一个 regularizer”。

再加：

- predicted model distribution
- unconditional train prior
- oracle one-hot next model

就得到：

\[
\text{prior}
\rightarrow
\text{predicted belief}
\rightarrow
\text{oracle belief}
\]

一个非常漂亮的信息质量阶梯。

这与你现有：

\[
Myopic \rightarrow F0 \rightarrow Truth-H5 \rightarrow Oracle
\]

能自然接起来。

而且 online scheduling 理论现在正好开始研究“information value”问题：2026 已经有工作把在线调度中的信息价值直接定义为它能减少多少 action uncertainty / Pareto action set。citeturn677082search0

所以你的 framing 是有理论背景的。

---

# 六、第三名：利用“没有 joint distribution”反而做一个创新

你现在有：

\[
P(M),P(R),P(T)
\]

但没有：

\[
P(M,R,T)
\]

这本来是 D 的致命问题。

但可以反过来利用。

比如共置 cost 依赖：

\[
c(m,r,g)
\]

而 predictor 给的是：

\[
p_M(m),p_R(r)
\]

不要写：

\[
p(m,r)=p_M(m)p_R(r)
\]

因为这是凭空假设 independence。

而定义所有满足这两个 marginal 的 joint：

\[
\Gamma(p_M,p_R)
\]

然后：

\[
\Phi_{\rm robust}(a)
=
\max_{\pi\in\Gamma(p_M,p_R)}
\sum_{m,r}
\pi(m,r)c(T(s,a),m,r)
\]

这实际上是一个很小的 optimal-transport / coupling LP。

优点很大：

1. 不需要 retrain；
2. 不需要假造 joint；
3. 正好使用你已有的 model/role probability；
4. 可以直接消费 measured co-location / batching table；
5. 没有 Wasserstein radius 这种难调超参。

这可以叫：

> **Coupling-Robust Future-State Evaluation**

但我不会把它做成第一贡献，因为 reviewer 可能说：

> “这是标准 robust optimization。”

最合适的位置是第一方法的 robustness extension。

---

# 七、你的“动作域”还能进一步挖出什么

一个特别值得写进论文 conceptual figure 的结构是：

\[
\text{Future belief}
\longrightarrow
\begin{cases}
\text{Ordering}\\
\text{Placement}\\
\text{Lifecycle}\\
\text{Co-location/Batching}\\
\text{Preemption}
\end{cases}
\]

你现在 F0 实际只连了第一根线：

\[
\text{Future prediction}\rightarrow \text{Ordering}
\]

而你的 substrate 已经具备后面四类真实 mechanism。

这其实就是你的 architecture moat。

别人可能有更大集群，但你已经有：

> **同一份 future information，在一套固定、实测、可审计 substrate 上，对不同 action classes 的边际价值可以被逐项隔离。**

这个实验能力非常少见。

所以我会考虑把论文核心 thesis 写成：

> **Prediction is useful only when it is action-aligned.**

然后你不是简单推出一个新 scheduler，而是回答：

1. future prediction 有没有价值？——已有 F0 vs Myopic；
2. 哪些 future fields 有价值？——p95 / model distribution / termination；
3. 它们对哪些 actions 有价值？——ordering vs placement；
4. action-conditioned consumer 能否释放原本浪费的信息？——New；
5. prediction 不准时怎么办？——prior/shuffle/oracle/robust coupling。

这个故事比“我们的 p95 scheduler 赢了五个 baseline”强很多。

---

# 八、我的最终排序

综合“新颖性 × 可行性 × 现有资产衔接度”：

**第一名：Belief-conditioned Action Evaluation，9/10。**

核心不是预测 DAG，而是：

\[
P(\text{unrevealed future}\mid prefix)
\rightarrow Q(s,(node,GPU))
\]

最小版本只用 `model_probabilities + termination + residency/load/evict`。

不重训、不改 substrate。

---

**第二名：Information × Actionability decomposition，8.5/10。**

它甚至不一定需要再设计复杂算法。

你现有 C/D/E/F 特别适合做这个。

这是非常好的 paper-level analysis contribution，而且能把你早期那些大量实验全部重新组织起来，而不是废掉。

---

**第三名：Coupling-robust future-state valuation，7.5/10。**

把“没有 joint distribution”从缺陷变成方法动机。

建议等第一名证明有增益后再实现。

---

B/C/D/E 原版，我都不建议升成主贡献。

尤其：

- adaptive p50/p95：像调参；
- chance constraint：已有大量工作；
- marginal CVaR：你缺 joint；
- p95-p50 gating：不是 epistemic confidence。

---

# 九、我建议你现在只跑的最小实验

先不要设计“大完全体”。

就新增 **4 个臂**：

1. **F0**：冻结原版。
2. **F0 + placement-only belief**：node 与 F0 完全相同，只改 GPU。
3. **Full action-conditioned belief**：node+GPU 联合。
4. **Oracle-next-model placement**：真实下一模型 one-hot，只作 ceiling。

再加两个几乎免费的 control：

- shuffled `model_probabilities`
- unconditional train-prior `model_probabilities`

主指标仍然 `mean_completion_ms`。

机制指标直接用已有：

- load_ms
- eviction count
- future model resident hit
- co-location/batch admission
- interference ms
- action flip rate

第一阶段甚至**不要用 runtime_probs、role probabilities、batching 或 preemption**。

先只回答：

> **把“下一模型的概率”从一个完全未使用的 predictor output 接到 GPU placement 后，有没有收益？**

如果有明显收益，再加入 measured co-location/batching 和 coupling-robust joint。

如果连这个最干净的 H=1 placement-only 都没有收益，我会建议立刻停止 A*，转而把论文主线做成 **information-actionability characterization**，而不是继续堆复杂 planner。

这是目前成本最低、信息量最高的一步。

最后给你一句我现在认为最适合作为整个项目研究问题的版本：

> **In online-revealed agent workflows, what future information is actionable, through which scheduling decisions, and under what system conditions does partial clairvoyance improve GPU scheduling?**

这句话比“预测未来 H5 做 GPU 调度”强得多，也比“分位数调度”强得多。memcite