# 事件链风险聚合：文献与理论定位（GPT，2026-09-16）

背景：我们的 r95（逐步骤 runtime p95 求和）在公平对照下优于场景采样/CVaR。
本文件给出该做法的经典理论对应、术语、必引论文、最近邻工作与论文表述建议。
**注意：论文正式引用前逐篇核对（GPT 已标注正式/arXiv 状态）。**

---

## 5. assistant
可以把你们现在的发现放进一个很成熟的理论框架里，但要注意：“逐步 p95 求和”与“共单调最坏情形”高度相关，但不能直接等同于真实联合分布就是共单调。 更稳的术语是 comonotonic VaR aggregation / tail-comonotonic surrogate。

先说经典理论。

PERT/项目调度。 Ringer, A Statistical Theory for PERT in Which Completion Times of Activities are Inter-Dependent, Management Science, 1971，正式发表，已经明确把活动时长依赖纳入 PERT。
PubsOnline
 Banerjee & Paul, On Path Correlation and PERT Bias, EJOR, 2008，正式发表，系统研究忽略 activity/path correlation 对项目完成时间估计的偏差。这里要纠正一个常见简化：对串行和，正相关不改变均值但会显著增大方差/尾部；对多条并行 path 的 max，正相关甚至可能降低 expected max，因此“正相关一定拉长平均项目工期”不是普遍定理。
科学直接
+1
 Yang, Risk Modeling of Dependence among Project Task Durations, Computer-Aided Civil and Infrastructure Engineering, 2007，正式发表，用 NORTA/copula 显式模拟任意活动相关结构，并展示忽略相关性会改变项目风险估计。
科学直接

随机调度 dependent processing times。 Boys, Glazebrook & McCrone, Single Machine Scheduling When Processing Times Are Correlated Normal Random Variables, EJOR, 1997，正式发表；直接研究 correlated processing times 下排序策略与 Smith-rule 类规则，说明加工时间依赖结构本身可以改变最优调度分析。
科学直接

最关键的是 comonotonicity。Dhaene et al., The Concept of Comonotonicity in Actuarial Science and Finance: Theory，Insurance: Mathematics and Economics, 2002，正式发表；以及同年 Applications，正式发表。它们证明：给定边际分布时，共单调耦合是极端正依赖情形，其和在 convex order 意义下最“危险”。
科学直接
+1
 Kaas et al., A Simple Geometric Proof that Comonotonic Risks Have the Convex-Largest Sum, ASTIN Bulletin, 2002，正式发表，进一步给出“共单调和是 convex-largest sum”的直接证明。
SSRN

对你们最直接的公式是：若 X
1
	​

,…,X
H
	​

 共单调，则同一分位点满足

VaR
α
	​

(
h
∑
	​

X
h
	​

)=
h
∑
	​

VaR
α
	​

(X
h
	​

).

Hua & Joe, Tail Comonotonicity: Properties, Constructions, and Asymptotic Additivity of Risk Measures, Insurance: Mathematics and Economics, 2012，正式发表，甚至证明只要求尾部趋向完全正依赖，也可得到 VaR/CTE 的渐近可加性。
IDEAS/RePEc
+1
 Cheung, Upper Comonotonicity, Insurance: Mathematics and Economics, 2009，正式发表，则说明超过阈值后的 upper-comonotonicity 已足以让 VaR/TVaR/ES 在相应高置信水平下可加。
科学直接

所以你们的

C
r95
	​

=
h
∑
	​

Q
0.95
	​

(R
h
	​

)

在数学上恰好等于“把各步骤 runtime 的边际分布以同一 rank 共单调耦合后，其总 runtime 的 VaR
.95
	​

”。但别写成“每一步最坏情况同时发生”：p95 不是最大值，更准确是“各步骤处于同一高分位状态”。

多期/嵌套风险。 Ruszczyński & Shapiro, Conditional Risk Mappings, Mathematics of Operations Research, 2006，正式发表，建立 conditional risk mapping 与多阶段动态规划；Ruszczyński, Risk-Averse Dynamic Programming for Markov Decision Processes, Mathematical Programming, 2010，正式发表，把风险度量嵌套进多阶段控制。
PubsOnline
+1
 这一线强调：多阶段风险通常不能简单“先把所有随机量平均成一个静态 CVaR”，而需要 stagewise conditional/nested risk 以保证 time consistency。与你们“链式事件不能随便独立采样平均掉”在思想上相通，但不是同一个定理。

“时间分散化”的直觉也一致：独立同分布风险求和时，标准差是 O(
H
	​

)；完全正相关时是 O(H)。金融里的 time diversification 文献长期讨论这种“跨期独立带来的风险平均化”，例如 Chung, Smith & Wu, Time Diversification: Definitions and Some Closed-Form Solutions, Journal of Banking & Finance, 2009，正式发表。
科学直接
 你们独立 scenario sampling 让坏运气跨步骤互相抵消，本质上就是引入了这种 diversification。

近期“事件链”方面，最接近的正式系统论文是 LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM Applications, ICDCS 2025，正式发表。它把 compound-LLM workflow 建模成不确定 DAG，用 Bayesian network 描述结构/时长不确定性，并利用阶段间相关信息和 mutual information 优先执行“能降低后续不确定性”的 stage；这是目前我找到的最接近“链内相关结构影响调度”的工作，但它没有比较 comonotonic VaR、独立 sampling CVaR 等聚合方式。
IEEE Xplore
+1

另外三篇近作：Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions, ICML 2026，正式发表/录用，把单请求输出长度分布通过 tail-risk score 进入 SJF，但只有单一随机标量，不涉及多步和的 dependence。
ICML
+1
 Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management (PBKV)，2026，仅 arXiv preprint，预测未来 agent invocation sequence，但消费目标是 cache reuse。
arXiv
 FATE: Future-State-Aware Scheduling for Heterogeneous LLM Workflows，2026，仅 arXiv preprint，显式考虑下游状态，但不是随机和的依赖风险聚合。
arXiv

还有一个很有意思的 2026 项目调度结果：Pajares Gutiérrez et al., The Role of Correlations Between Activity Durations in Project Schedule Risk Analysis, PMS 2026 Proceedings，正式同行评审会议论文。它发现 correlation 对平均项目时长影响未必大，却显著影响方差和风险敏感性，而且强烈受 network topology 调节。
UvaDoc
 这其实提醒你们：不要把“相关性”故事只写成均值效应，最好强调其通过调度排序/争用放大为系统性能差异。

对你们论文表述，我建议这样定：

r95 is a comonotonic upper-tail aggregation surrogate: it sums per-stage VaR
0.95
	​

, equivalent to the VaR
0.95
	​

 of the aggregate latency under a comonotonic coupling of the marginal stage-latency distributions.

最应该引用的三篇是 Dhaene et al. 2002 Theory、Kaas et al. 2002 convex-largest sum、Hua & Joe 2012 tail comonotonicity。
科学直接
+2
SSRN
+2

你们的贡献不应写成“发现了共单调理论”。更合理的是：

经典风险理论给出共单调聚合的极端依赖解释；我们的贡献是在动态 Agent workflow scheduling 中实证发现，这种 tail-aligned conservative coupling 比独立/场景化 CVaR 更适合调度排序，并通过 scale-matched 与 tail-shuffle 对照表明收益来自步骤特异的尾部对齐，而非简单放大。

“首个系统验证”可以谨慎写成 “to our knowledge, the first systematic evaluation of dependence-aware multi-step uncertainty aggregation for dynamic agent-workflow scheduling”。目前检索没有发现直接重叠，但投稿前还应再做一次最终查新。
