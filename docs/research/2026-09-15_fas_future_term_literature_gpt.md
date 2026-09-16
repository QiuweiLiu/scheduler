# FAS 定向文献检索：未来项与学习型调度器（GPT，2026-09-15）

来源：ChatGPT Web 会话（问题：近 2–3 年权威工作中，调度目标里的"未来项"设计；学习型调度器进展；我们是否需要重跑 BC/RL）。
**注意：文中论文仍需逐篇核验；TIE（ICML 2026）为目前最危险的 prior，投稿前必须正式核对。**

---

## 5. assistant
这次定向检索后，结论比前几轮更清楚：“预测未来再调度”本身已经不是空白，但“预测动态 Agent 的多步未来行动，并系统比较这些未来不确定性应如何折算成调度代价”仍有明显空间。 其中 ICML 2026 的 TIE 是你们最需要正面对照的新工作。

1. 2024–2026：预测/未来项如何进入调度

下面 10 篇都可核对；“正式”表示已正式会议/期刊发表或 ICML 官方录用列表可核验，“预印本”表示目前仅 arXiv。

论文	状态	未来信息如何被消费
Parrot: Efficient Serving of LLM-based Applications with Semantic Variable, OSDI 2024	正式	应用显式暴露多请求 dataflow/DAG，服务端用已知跨请求依赖做端到端优化；不是预测未知 future。
USENIX

Efficient LLM Scheduling by Learning to Rank, NeurIPS 2024	正式	不预测精确长度，而预测请求输出长度的相对 rank，用来近似 SJF；报告 chatbot latency 最高降低 2.8×。
NeurIPS 会议录

A Stochastic Approach for Scheduling AI Training Jobs in GPU-Based Systems, IEEE TCC 2024	正式	明确使用 early-termination 概率分布，优化期望能耗并满足 due date；这是与你们“未来是否继续”最接近的经典随机调度参照。
博阿大学

Imitation Learning Enabled Fast and Adaptive Task Scheduling in Cloud, FGCS 2024	正式	BC 初始化 DRL，再在线 imitation 更新，目的就是缓解纯 DRL sample inefficiency、慢收敛和振荡。
科学直通车

SOLA: Optimizing SLO Attainment for Large Language Model Serving with State-Aware Scheduling, MLSys 2025	正式	根据当前 request state + system state动态排序，而非预测下游 Agent action；代表强 reactive baseline。
MLSys Proceedings

SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads, NSDI 2025	正式	SlackFit 根据即时 slack 做 reactive model selection；证明强 state-aware 策略可以很有效，但没有未来控制流预测。
USENIX

Hybrid Learning and Optimization-Based Dynamic Scheduling for DL Workloads on Heterogeneous GPU Clusters (RLTune), SoCC 2025	正式	RL 负责 job priority，MILP 负责 placement；在 Philly/Helios/Alibaba traces 上做混合学习+优化，而不是让 RL 独自解决整个 combinatorial action。
DOI

Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions, ICML 2026	正式录用；arXiv版本可用	将输出长度建模成 heavy-tailed log-t 分布，再用 Tail Inflated Expectation 代替点长度进入 SJF；这是目前与你们“tail-aware future consumption”最危险的 prior。ICML 官方下载页已列出该论文。
arXiv
+1

Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management (PBKV), arXiv 2026	预印本	直接预测未来几步 agent invocation sequence，再按预测的 cache reuse potential 做 eviction/prefetch，并保守使用预测以抗误差。与你们“预测未来行动”高度接近，但目标是 KV-cache，不是节点级 GPU scheduling。
arXiv

FATE: Future-State-Aware Scheduling for Heterogeneous LLM Workflows, arXiv 2026	预印本	在候选 placement 中显式评估其对 downstream residency/locality/prefix reuse 等未来状态的影响，采用 horizon-aware planning；重点是“当前动作诱导的未来状态”，不是预测未知 agent action sequence。
arXiv

查新后的关键判断：我没有在这些正式系统/调度工作中找到与你们完全相同的实验：在同一多步 Agent future forecast上，系统比较 point estimate、逐步 quantile-sum、scenario sampling、CVaR、chance constraint、adaptive risk，并发现某种折算优于其他形式。TIE 最接近，但它研究的是单个 LLM 请求的输出长度分布 → SJF score，不是动态多模型 action chain。
arXiv

因此“首个”不要写成宽泛的：

first uncertainty-aware predictive scheduler

这已经不成立。

更安全且有力的是：

To our knowledge, this is the first systematic study of how multi-step future-action uncertainty should be consumed for scheduling dynamically unfolding agent workflows.

投稿前还应再做一次正式 related-work sweep，因此正文可用 “to our knowledge”。

2. 学习型 scheduler：现在的发展意味着什么

近年的趋势不是“RL 已经全面战胜 heuristic”。恰恰相反，较强工作越来越强调缩小 action space、混合优化器/heuristic、或用 imitation/offline data 稳定训练。RLTune 把 RL 限制在 priority，把 placement 交给 MILP；TopDRL 也是 DRL 选 job + heuristic 选 GPU，并明确把大 action space/难收敛作为问题。
DOI
+1
 ILETS 则直接指出纯 DRL 的 sample efficiency、初始性能和收敛稳定性问题，用 BC 来初始化和持续校正。
科学直通车

我没有找到近 2–3 年权威 LLM/GPU scheduler 已经做成：

future-agent-prediction → learned policy

这样的成熟路线。PBKV 使用学习预测，但后端 cache policy 仍是设计好的规则；FATE 是 planner；TIE 也是解析式 risk score，而不是 RL policy。
arXiv
+2
arXiv
+2

所以你们的旧结果“BC接近专家但不如启发式、PPO明显差”不是异常现象；但它不能直接作为当前新接口的正式 baseline，因为状态信息和 consumer 都已经变了。

3. 要不要重跑 BC/RL？

我的建议是：

BC：值得重跑；PPO：非必须。

原因不是期待 BC 赢 q95，而是堵住一个很自然的 reviewer question：

“既然 future artifacts 信息这么丰富，为什么不用 learned policy 自动学会怎样消费？”

最低可信 BC 就够回答这个问题。

建议当前接口下重新生成一套专家轨迹，专家必须和 BC 看到完全相同的 future artifacts。专家首选你们当前 MPC/CP-SAT 中性能最强且计算允许的一个，而不是因为“CP-SAT 名字高级”就固定 CP-SAT。若 q95 heuristic 本身比 optimizer 更强，可以同时做两个 teacher：BC-Opt 与 BC-q95，但主 baseline 一个即可。

训练量不用扩得夸张：你们旧版约 24.6 万决策已经是合理量级；新 BC 保持 ~200k–300k masked decisions、3 seeds 即可。必须按 episode/source-workflow 切训练/验证，不能随机切 decision，否则同一 workflow 的邻近状态会泄漏。指标不能只报 86% action accuracy，要同时报告 completion/miss、teacher gap、q95 gap 和 inference overhead。

评测必须 paired 到完全相同 episodes，比较至少：

E0 / E2 / q95 / expert / BC

若 BC≈expert 但仍输 q95，那么结论很有价值：

the gap is not imitation failure; the hand-designed future-risk surrogate is better aligned with the scheduling objective than the optimization teacher.

若 BC 超过 q95，那也很好：论文主线变成“future-action prediction enables both analytic and learned consumers”。

PPO 若时间有限我会不重跑。若要放正文，最低要求是 3–5 seeds、同 action mask、相同训练 episode distribution、固定 interaction budget、与 q95/BC/optimizer 同评测；否则旧 PPO 只能作为 exploratory negative result。当前已有文献足以解释为什么纯在线 RL 不应被视为必须胜出的 baseline。
科学直通车
+1

4. 对你们论文定位的直接影响

这轮查新后，我会把新颖性从：

“预测 Agent future 来调度”

稍微收窄，因为 PBKV 已经预测未来 agent invocation。
arXiv

更稳的定位是：

Prediction-conditioned GPU scheduling for dynamically unfolding multi-model agent workflows, with a systematic study of how multi-step future uncertainty should be converted into scheduling cost.

然后 Related Work 清楚分三层：

Parrot/Katz： future workflow 已知；
LTR/TIE/STS： future uncertainty 是单一标量或 termination distribution；
PBKV/FATE： 开始进入 agent future，但分别用于 cache management / downstream-state preservation；
你们： future 是多步 action sequence + per-step resource distribution + random termination，研究的是这种结构化 forecast 对节点级 GPU scheduling 的信息价值和消费函数。

尤其 TIE 的出现反而强化一个很好讲的差异：

TIE 证明“预测分布比点估计更值得消费”；你们进一步发现，在多步 Agent chain里，把每步不确定性组合成 joint decision score 本身成为新的系统问题——简单场景平均/CVaR 并不必然优于保守的 state-aligned tail aggregation。
arXiv

这条线目前比“再做一个 PPO”更值得投入。
