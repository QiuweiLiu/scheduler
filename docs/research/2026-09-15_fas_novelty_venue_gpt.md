# FAS 故事叙述与投稿评估（GPT，2026-09-15）

来源：ChatGPT Web 会话（同一系统工作，第二轮追问：故事怎么讲 / 与别人区别 / 解决什么问题 / 基线清单 / 投稿概率与审稿人反应）。
说明：引用论文（Parrot OSDI 2024、SOLA MLSys 2025、SuperServe NSDI 2025、Katz ATC 2025、LTR NeurIPS 2024、Tiresias、Pollux、STS TCC 2024）
仍需逐篇核验；投稿概率为 GPT 的主观匹配度估计，非官方录取率；中科院分区需按投稿当年最新版核对。

---

## 5. assistant
可以把整篇论文压成一句话：

Agent 调度的关键缺失信息不是“当前节点有多贵”，而是“这个请求接下来会做什么、还会走多远”；我们预测尚未生成的未来控制流，并证明这种未来状态能被风险感知调度器直接转化为端到端收益。

这就是卖点句。

叙述骨架建议是“三段式”。

问题。 多模型 Video Agent 的执行图不是提交时固定 DAG，而是在 planner/tool/model 交互中动态展开；传统 scheduler 只看到 ready node，因此 GPU 选择是局部的，无法知道某请求是否即将结束、之后会调用哪些模型以及未来尾部耗时，最终造成排队和 deadline 决策次优。Parrot 已证明 application-level workflow information 很重要，但要求应用显式暴露数据流；你们解决的是未来控制流尚不存在、必须在线预测的情况。
USENIX

洞察。 预测全部细节并不等于更好的调度：oracle resource≈静态表，真值内容甚至 +14.2s；真正重要的是 continuation/horizon，以及对预测链 runtime tail 的风险消费。H10 oracle 比 H5 再快 10.2s，而简单延长预测窗口会负迁移。

结果。 Future-action E2 比无未来快 17.1s；进一步采用逐节点 runtime-p95 风险成本，在冻结 confirm 上再快 18.8s；同时 oracle 分解解释“为什么有效、什么信息无效”。

Abstract 可按五句写：①动态 Agent workflow 给 scheduler 带来 latent-future problem；②提出 future-action predictor/interface/scheduler；③发现 continuation/horizon 而非精细 identity/resource 是主导信号；④提出 tail-aware future cost；⑤报告 −17.1s、−18.8s、H10 oracle −10.2s，并说明真实系统验证。

Prior 边界应这样讲：

工作	他们做什么	你们做什么	慎用表述
Parrot, OSDI’24	应用显式暴露 Semantic Variable/dataflow	预测尚未展开的未来 action chain	不能说首个 workflow-aware serving 
USENIX

SOLA, MLSys’25	依据当前 request/system state 动态调度	加入 prospective future-control-flow state	不能说首个 state-aware 
MLSys Proceedings

SuperServe, NSDI’25	SlackFit 根据当前 slack/reactive state 选模型	预测跨节点未来链后决策	不能说首个 slack/risk-aware 
USENIX

Katz, ATC’25	优化已知 diffusion+adapter workflow	workflow 由 agent 运行时动态产生	不能说首个 heterogeneous workflow serving 
USENIX

LTR, NeurIPS’24	预测请求相对输出长度，近似 SJF	预测 request 内部未来 action/termination/model sequence	不能说首次 prediction-driven scheduling 
NeurIPS 论文集

Tiresias/Pollux	前者在未知训练时长下用 LAS/Gittins；后者在线建模 goodput 调资源	预测 agent 内部控制流	别把二者写成 duration-predictor baseline 
USENIX
+1

STS, TCC’24	利用训练 early-termination 分布动态配资源	逐请求预测未来 action/termination，并服务节点级 GPU 调度	不能说首次利用 termination uncertainty 
Polimi Re Public

正式问题陈述建议写成：Dynamic agent execution hides near-future control flow from the resource manager. Decisions based only on currently-ready nodes ignore downstream execution externalities—future GPU demand, queue occupancy, model transitions and deadline risk. Existing systems either assume the workflow is known, optimize current execution state, or predict a scalar request/job length. We ask whether the latent future action sequence itself can serve as a scheduling state.

投稿必需基线，我会这样定。

最低集合：FCFS、SJF/LTR-style predicted remaining work、E0 current-node greedy、E2 future-action+static cost、q95 future-action、STS-style termination-aware heuristic、Oracle-H5/H10。Tiresias/Pollux 可做“adapted conceptual baseline”，但不是必须原样复现。

最加分的两个：Parrot-like known-DAG oracle（给 scheduler 真值未来图但仍用你们资源模型）和 SOLA/SlackFit-style state-aware baseline（只用当前 slack/queue，不看 future）。这样直接隔离“未来预测”相对于“已知图”和“强 reactive state”的价值。

真实系统里不必硬移植完整 Katz/Parrot；做 2-GPU prototype 时，FCFS/current-state/LTR-like/q95 足够。Orca/vLLM continuous batching针对单 LLM token serving，本身不是公平的端到端多模型 baseline。Orca 的核心是 iteration-level scheduling。
USENIX

投稿概率下面是主观匹配概率，不是官方 acceptance rate，前者=当前纯仿真，后者=补小规模真实 GPU replay/prototype。

期刊：TPDS 25%→45%，短板是单 workload/无实机；TCC 30%→50%，和 stochastic/resource scheduling 很匹配；IEEE TC 15%→30%，需要更强系统/算法普适性；FGCS 55%→70%、JPDC 45%→60%，当前形态已经现实；ACM TOCS 若你说的是系统刊则 10%→25%，要求明显更高；TACO <10%，偏 architecture/code optimization；ACM TOS 若指 Transactions on Storage，不建议投。中科院分区每年变化，不能把“1/2区”写死，应按投稿当年最新版核对。

会议：OSDI/SOSP <3%→5–8%；NSDI 4%→8–12%；EuroSys 6%→12–18%；ATC 10%→20–28%；MLSys 12%→25–35%，是高档里最匹配；SC 5%→10–15%，需要更大规模/HPC叙事；HPDC 20%→35%；ICDCS 20%→35%；IPDPS 15%→28%；SoCC/Middleware 20%→35–40%；ICPP 35%→50%；CCGrid/Cluster 45%→60%。真正把概率抬档的是“真实 GPU 闭环 + 多压力区间”，不是再多做 predictor 小改动。

如果我是审稿人，最可能三组攻击是：

R1（最狠）：“这是 simulator artifact；17–19s 是否会在真实 GPU 上存在？”
防御：2-GPU replay，报告真实 completion、策略排序、sim-vs-real error。没有这个，OSDI/EuroSys 很难防。

R2：“只有一个 Video-Agent workload 和一个 predictor，所谓 future-action value 是否普适？”
防御：arrival×deadline×model-mix pressure sweep，再做 artifact corruption/sensitivity；至少证明趋势跨 5/6 cells。单 workload 本身仍是无法完全消除的限制。

R3：“q95-sum 看起来只是调参出来的 heuristic，为什么是贡献？”
防御：保留 frozen 300 confirm；与 p50/p90/CVaR/chance/state-aware/STS 对比；再做 runtime-tail correlation/stress 分析，证明收益随 contention/deadline pressure呈规律变化。否则应把 q95 降为 implementation choice，而不是算法创新。

如果只能再做 1–2 个实验：

第一名：2-GPU真实 replay/prototype。 一项实验同时补 external validity、simulator fidelity、系统完整性，是档次提升最大的一步。

第二名：6-cell pressure sweep（E0/E2/q95/Oracle即可）。 成本很低，却直接回答“只在一个 workload point 有效吗”。

做完这两项，我会从“稳妥投 FGCS/JPDC/ICPP”提升到“可以认真准备 MLSys，ATC/EuroSys 可冲”；如果真实系统仍能保持 future-action 与 q95 的优势，这篇论文的故事就基本闭环了。
