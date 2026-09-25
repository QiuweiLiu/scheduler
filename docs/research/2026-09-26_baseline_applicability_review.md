---
research_backend: "ChatGPT Web"
requested_model: "GPT-5.6 Sol"
requested_reasoning: "High"
model_verification_basis: "high_ui_mapping"
conversation_id: "6ab6b895-43b8-83ec-982a-4c13ac404115"
conversation_generation: "1"
conversation_url: "https://chatgpt.com/g/g-p-6a45fd9c56e48191a5e1a006582fcdef-diao-du/c/6ab6b895-43b8-83ec-982a-4c13ac404115"
verified_before_submit: true
verified_after_response: true
timestamp: "2026-09-26T02:18+08:00"
status: "valid"
---

# Research Report

炜，我把原论文、你仓库当前的 Latency-Aware 实现说明，以及 Agentix / Kairos / Maestro / SAGA / Parrot 重新对了一遍。先给最重要的结论：

我的建议不是简单“删掉 Latency-Aware”。而是：修正它的信息边界 → 保留为 secondary baseline，并明确标记 fusion = N/A；同时补一个真正匹配动态 ReAct 的 primary baseline。若只能保留四条 primary baseline，我会用 Agentix 替换 Latency-Aware；若可以多跑一条，则保留 Latency-Aware + 新增 Agentix。

而且我发现一个比你当前问题更重要的点：仓库现在把 Latency-Aware 的 graph visibility 写成了 logical_graph_assumed_known，这与原论文并不一致。 原论文明确说 scheduler 看到的是 workflow 的 resolved portion，一个 branch 只有在 control result resolve 后才进入 planning window，near-ready 也要求 branch、role、model、eligible devices 已经确定；论文甚至直接说这个窗口“不假设 unresolved branches”。因此，拿完整 realized template 穿过 planner 决策点做 fusion，不只是“对你们方法不公平”，也不符合 Latency-Aware 自己的原始信息契约。
arXiv

这意味着我建议你在正式测量前重新打开 Latency-Aware 的 freeze，做一次 correctness fix。

Q1. Latency-Aware 还算不算公平、有意义的 baseline？
Verdict：保留，但降级定位；修正信息契约之后再跑。

[VERIFIED] Latency-Aware 原论文不是 full-DAG clairvoyant scheduler。它在时刻 
𝑡
t 观察的是已经 resolve 的 
𝐿
𝑡
𝑤
L
t
w
	​

，并明确规定“branch enters only after its control result resolves”。它构造的 planning window 只有 ready + 已经确定身份的 near-ready 节点。
arXiv

因此你们正确的适配应该是：

resolved_window_only

而不是现在 freeze 文件中的：

logical_graph_assumed_known

各组件在你们 workload 上的适用情况
Latency-Aware 部件	Verdict	原因
Device-aware Predictor	适用	[VERIFIED] run time / peak memory / loading cost 都是当前或已 resolve activation 的合法预测量，不要求知道未来 planner 决定。
arXiv

Placement / memory admission	适用	[VERIFIED] 已 ready 节点仍然需要异构 GPU placement、显存可行性和 model residency 决策。
Lexicographic scheduler	适用	[VERIFIED] ready-work coverage / completion / lifecycle cost 仍然有定义。
Model lifecycle / load / reclaim	适用	[VERIFIED] 对当前 ready deployment 完全合法。
Near-ready prefetch	部分适用	[VERIFIED] 只有当 successor 的 branch、role、model 已经 resolve，但 predecessor 还在运行时才合法。planner 尚未决定下一个工具时不能 prefetch 某个真实 successor。
arXiv

Cross-workflow lifecycle sharing	原则上适用	[INFERENCE] 如果不同 workflow 同时出现已 resolve 的相同 deployment，可以共享 lifecycle 行为；需要你们实际统计机会率。
Constructor fusion	结构性 N/A	[VERIFIED，基于你提供的 workload 统计] 没有合法的连续 same-deployment、无中间控制决策的链；跨 planner 决策点读取后继属于未来泄漏。

特别是 fusion。

原论文 Eq. (3) 要求：

𝑠
𝑢
𝑐
𝑐
(
𝑢
)
=
{
𝑣
}
,
𝑝
𝑟
𝑒
𝑑
(
𝑣
)
=
{
𝑢
}
,
𝑑
(
𝑢
)
=
𝑑
(
𝑣
)
succ(u)={v},pred(v)={u},d(u)=d(v)

但这个条件是在已 resolved 的 
𝐿
𝑡
L
t
	​

 上判断，而不是在“这一 episode 最后实际会走哪条路”的离线模板上判断。论文 QMSum 的三个 Qwen3-8B aggregation 节点之所以可以 fuse，就是因为那三步已经是固定、连续的逻辑链。
arXiv

而你们是：

planner → 运行时决定 tool → planner → 运行时决定 tool → ...

所以：

[VERIFIED] 你们不是“fusion 命中率低”，而是主 workload 的语义本身不产生合法 fusion opportunity。

这两种说法区别很大。

另外，这也不是一个无关紧要的机制。Latency-Aware 自己的消融报告显示，禁用 fusion 会让 p95 增加 8.5%–44.9%，makespan 增加 3.0%–8.1%；它是原系统尤其重要的 tail-latency mechanism。
Anonymous View

所以不能把一个根本没有 fusion 机会的运行结果写成：

“We faithfully reproduce the complete Latency-Aware system.”

更准确的是：

“We faithfully instantiate all workload-applicable components; fusion has zero legal opportunities under the target control-flow semantics.”

Q2. 审稿人会怎么质疑？应该选 i / ii / iii 哪一个？

我的判断是：

主张：i + ii + 补充新 baseline，而不是单独 iii。

也就是：

保留 Latency-Aware，但明确说明 fusion structurally inapplicable；同时将其定位为 applicable-components evaluation，再增加 Agentix。

审稿人最可能有三个问题。

第一种质疑：

“Latency-Aware 的核心贡献之一根本没有被 workload exercise，你为什么拿它的结果代表 Latency-Aware？”

这是合理质疑。

解决方法不是删掉它，而是在 table/implementation notes 中明确：

Fusion opportunities = 0

最好直接报告：

# legal fusible edges = 0
# fused chains length > 1 = 0
# boundaries removed = 0

这样审稿人不会怀疑你偷偷把 fusion 实现坏了。

第二种质疑更严重：

“为什么这个 baseline 能看到 planner 尚未作出的决定？”

如果沿用当前 repo 的 realized-template visibility，这个问题非常难解释。

而且现在已经不能用：

“Latency-Aware 本来就假定 known logical graph”

来挡了。

[VERIFIED] 原论文明确否定了这种解释。 它说的是 resolved portion，并且明确说 unresolved branch 不进入 planning window。
arXiv

所以我认为这是一个应该 break freeze 的 correctness issue，不是为了调性能而 retune。

第三种质疑是：

“你的 workload 天生压制了竞争方法的主要机制。”

这个问题靠 disclosure 只能缓解，不能完全消除。

所以最好的实验结构是：

Primary competitors

LLMSched
Pythia
TIE
Agentix

然后：

Additional systems comparison

Latency-Aware-Orchestration — fusion N/A on target workload

如果计算量允许，我甚至不删 Latency-Aware，直接做五条。

这样你的叙事反而更漂亮：

某些现有 workflow-serving 方法依赖 resolved/static future structure；而我们的目标场景是 future structure 本身尚未展开。

这正好对应你们论文真正想解决的问题。

Q3. 如果补/换 baseline，我推荐什么？

结合你现在已经有 LLMSched / Pythia / TIE，我给出的实际顺序如下。

排名	Baseline	出处	信息需求	调度接口	我对它的定位
1	Agentix / Autellix	NSDI 2026	不需要未来 topology	attained service → ready-call priority；ATLAS 用已完成父节点构造动态 critical-path service	最佳替代 Latency-Aware
2	Maestro	ICDCS 2026	当前 stage + role/tool intent + 历史 conditional future profile	predicted remaining work + readiness/memory → SRTF + placement	最佳 prediction-driven 补充 baseline
3	Kairos	arXiv 2025	runtime workflow trace + 历史 latency/memory distributions	remaining-latency priority + memory-aware dispatch	很贴动态 workflow，但系统环境稍弱匹配
4	SAGA	HPDC 2026	framework AEG 或历史概率 AEG	workflow/session affinity + AEG-predicted cache reuse + fairness	语义非常贴 ReAct，但你们 simulator 缺 KV abstraction
1. Agentix：我最建议你补的

正式论文现在叫 Agentix: An Efficient Serving Engine for LLM Agents as General Programs，NSDI 2026；早期 arXiv 名称是 Autellix。USENIX 已正式收录。
USENIX

它几乎精准命中你现在遇到的问题。

[VERIFIED] Agentix 明确写：

dynamic general-purpose agentic programs；
non-clairvoyant；
不知道未来 execution graph；
execution DAG 随程序执行动态构造；
PLAS 只用同一 program 已完成调用的累计 service；
ATLAS 对 dynamic DAG 用已完成 parent 的 service 构造 non-clairvoyant critical-path estimate。
USENIX
+2

它甚至在 appendix 明说：

full prior knowledge of execution is unrealistic；

而 SRPT 如果读取完整总调用量和 decode steps，只能作为 clairvoyant simulator reference。
USENIX

这与你们的：

planner 尚未决定下一步

完全兼容。

在你 simulator 里的接口甚至很简单：

𝑝
𝑟
𝑖
𝑜
𝑟
𝑖
𝑡
𝑦
(
𝑗
)
=
attained service of workflow 
𝑗
priority(j)=attained service of workflow j

如果有并行 DAG，就实现 ATLAS：

𝑝
(
𝑣
)
=
max
⁡
𝑢
∈
𝑝
𝑟
𝑒
𝑑
(
𝑣
)
[
𝑝
(
𝑢
)
+
𝑡
𝑢
]
p(v)=
u∈pred(v)
max
	​

[p(u)+t
u
	​

]

然后异构 GPU placement 仍由该 baseline 自己的合法 placement policy 完成。

需要注意：

[INFERENCE] Agentix 原文主要优化 LLM-engine scheduling/data locality，不是专门为 heterogeneous GPU selection 设计的。

所以迁移时应该写成 Agentix-adapted，别给它凭空增加未来 GPU predictor。

但作为一个**“完全不知道未来还能不能打得很好”的强对照**，价值很大。

2. Maestro：其实和你们研究问题非常近

这个我强烈建议你认真考虑。

Maestro: Workload-Aware Cross-Cluster Scheduling for LLM-Based Multi-Agent Systems，作者页面和 IEEE DOI 信息显示为 ICDCS 2026。
Yangrenyu
+1

它不读取 realized future，而是预测：

current agent role
workflow position
tool-use probability
output length
KV footprint
remaining workflow time

然后做 workflow-aware SRTF、model readiness 和 memory-aware placement。
arXiv

最关键的是：

[VERIFIED] 它专门处理 branching 和 iterative behavior。

其未来剩余时间不是把真实 suffix 偷进来，而是：

𝑇
^
rem
=
𝑇
^
current
+
𝑇
^
future
T
^
rem
	​

=
T
^
current
	​

+
T
^
future
	​


其中 future term 是历史执行在当前 state 下的 conditional median；state 包含 role、stage template、invocation index、tool-intent。
arXiv

这和你们非常值得比：

Maestro：用历史统计压缩未来。
你们：显式预测未来 topology / content / resource distribution 再用于调度。

这会是一个非常干净的 scientific comparison。

唯一 caveat：

[VERIFIED] Maestro 原 prototype 是 homogeneous A100 nodes，论文自己把 heterogeneous accelerators 列为 future deployment extension。 
arXiv

因此在你们 2 个异构 GPU simulator 上要标：

Maestro-adapted: device-conditioned execution profiles replace its homogeneous-node profiles.

我认为这是合理 adaptation，不破坏核心机制。

3. Kairos

Kairos 是 Low-latency Multi-Agent Serving with Shared LLMs and Excessive Loads in the Public Cloud, arXiv:2508.06948。

它其实比我一开始印象中更适合你。

[VERIFIED] Kairos 明确批评 static full-workflow assumptions 无法捕获 autonomously planning multi-agent applications 的 dynamic and uncertain call relationships。 它在线收集执行信息、推断 workflow，并用 remaining execution latency 排优先级，再按 memory demand dispatch。
arXiv

论文甚至统计了：

dynamic branching
sequential execution
dynamic feedback

三类 workflow。
arXiv
+1

所以信息边界是合法的。

不过：

[VERIFIED] 原实验主要是共享 Llama3-8B + A40，而不是你们这种 heterogeneous multi-model GPU pool。

因此我把它排在 Agentix / Maestro 后面。

4. SAGA

SAGA 是 HPDC 2026 正式论文。
开放评论

它非常有意思，因为它真的写了 ReAct。

AEG：

𝐺
=
(
𝑉
,
𝐸
,
𝑃
,
𝜙
)
G=(V,E,P,ϕ)

其中 
𝑃
P 就是 transition probability。

对 ReAct，论文直接建模为：

𝑃
(
𝑣
𝑖
→
𝑣
𝑖
+
1
)
≈
1
−
𝑝
t
e
r
m
P(v
i
	​

→v
i+1
	​

)≈1−p
term
	​


如果 framework 没直接给 AEG，它会从历史 request stream 推断 tool-pattern 和 transition probability。
arXiv
+1

从研究思想上，这跟你们“概率化未来 Agent 图”非常接近。

但我暂时不建议现在把它作为第五条正式 baseline。

原因不是论文不好，而是：

[VERIFIED] SAGA 的核心收益是 KV-cache retention / WA-LRU / session affinity。

而你仓库现在的 simulator 明确没有完整 KV/prefix-cache serving abstraction。

所以如果只移植 AEG priority：

你又会重复现在 Latency-Aware 的问题——论文核心机制没有真正被 exercise。

因此它更适合作为 related work，或者以后 simulator 扩 KV 后再做。

Q4. 能不能改 workload，让 Latency-Aware 的 fusion 重新成立？

可以，但不要为了让 baseline 好看而修改主 workload。

这里需要区分两件事。

方案 A：真的加入固定连续同模型段

例如：

planner → tool_A → tool_A2 → planner

并且 planner 在进入 tool_A 前已经一次性确定 A 和 A2。

那么：

[VERIFIED] Latency-Aware fusion 合法。

因为 A→A2 已经是 resolved、one-to-one、same deployment。

但问题是：

[INFERENCE] 如果你为了 benchmark 人工创造这种 topology，它回答的是一个不同的问题。

你原本研究的是：

runtime-decision-driven agent workflow

变成：

partially preplanned workflow

所以不应该替换主 benchmark。

但我非常支持额外增加一个 control-flow visibility sensitivity experiment：

0
%
,
25
%
,
50
%
,
75
%
,
100
%
0%,25%,50%,75%,100%

的 downstream path 提前 resolve。

然后看：

Agentix
Latency-Aware
你们的方法

随 future visibility 增加怎么变化。

这个实验甚至可能很有论文价值：

当未来结构已经确定时，结构融合型方法开始发挥作用；当 future latent/unresolved 时，prediction-based scheduling 才体现价值。

方案 B：给节点增加 resolved_at 标记

这个我更推荐。

不是：

“模板里最终会走到这个节点，所以我说它 resolved。”

而是记录真实运行时：

𝑡
resolved
(
𝑣
)
t
resolved
	​

(v)

只有：

𝑡
≥
𝑡
resolved
(
𝑣
)
t≥t
resolved
	​

(v)

scheduler 才能看到节点。

这样所有 baseline 都共享统一 visibility contract。

这会彻底消灭现在的争议。

例如：

planner 仍在推理 → successor unknown

planner 输出 tool call → successor resolved

tool 等 GPU → scheduler 可以看到它

如果还有已经确定的下一步 → near-ready/preload 合法

这是我认为你们 simulator 以后应该长期保留的 abstraction。

“给分支集合”不等于 resolved

这里特别注意。

如果 planner 当前只知道：

𝑃
(
𝑡
𝑜
𝑜
𝑙
=
𝐴
)
=
0.6
,
𝑃
(
𝑡
𝑜
𝑜
𝑙
=
𝐵
)
=
0.3
,
𝑃
(
𝑎
𝑛
𝑠
𝑤
𝑒
𝑟
)
=
0.1
P(tool=A)=0.6,P(tool=B)=0.3,P(answer)=0.1

那么对你们自己的 probabilistic predictor 很有用。

但是：

[VERIFIED] 这不满足 Latency-Aware 的 near-ready/fusion 定义。

它要求 branch / role / model 已确定。

所以不能让 Latency-Aware 从 A/B 中偷偷挑最终真实 A。

否则仍然是 leak。

Q5. 你列出的 FCFS / SJF / slack / STS / Parrot / RL 怎么排？

如果目标明确为：

decision-driven Agent workflow + heterogeneous GPU

我的实用排序是：

第一梯队：Agentix-style attained-service / termination-aware + state-aware predicted remaining-work。

这两类信息契约最干净，而且分别代表两个重要方向：

不预测未来，也能调度
利用统计预测未来，但不读取未来真值

所以我尤其建议：

Agentix-ATLAS + Maestro-style predicted remaining time

成为两个很强的 reference point。

第二梯队：FCFS + predicted-SJF/slack。

FCFS 必须有。它不是为了强，而是实验锚点。

SJF/laxity 也值得保留，但必须写清：

𝑅
^
𝑗
R
^
j
	​


来自 predictor / history，而不是实际 suffix。

一旦 SJF 读取：

这个 episode 最终还剩多少节点、多少 ms

那它就是 Oracle-SRPT。

Agentix 自己也是这么处理的：true remaining runtime 只作为 clairvoyant reference。
USENIX

第三梯队：Parrot-like known-DAG。

Parrot 是 OSDI 2024 正式论文，通过 Semantic Variables 将 application-level dependency/dataflow 暴露给服务层。
微软
+1

它很重要，但在你这里应定位为：

structural-information reference

而不是完全公平的 primary competitor。

如果给它 full realized template：

Oracle-ish / privileged-graph baseline

如果只给当前 resolved DAG：

公平，但 Parrot 的 DAG 优势会被大幅削弱。

这两种可以分别标成：

Parrot-resolved
Parrot-full-DAG (informational oracle)

千万别混。

第四梯队：BC / RL。

可以做，但我不建议现在为了修 Latency-Aware 临时塞一个 RL baseline。

因为 RL 会额外引入：

training budget
reward design
state encoding
generalization split
hyperparameter budget

审稿人很容易从“baseline 公平吗”跳到另一个泥潭。

你如果已有 PPO/RL 实验，可以作为 learned-policy comparison / appendix，但不是最需要补的 system baseline。

因此我建议最终 baseline 阵容

如果你愿意多跑一条，我最推荐：

系统	定位
LLMSched-adapted	uncertainty-aware compound workflow
Pythia-adapted	workflow semantic predictability
TIE-adapted	uncertainty-aware execution-cost prediction
Agentix-adapted	non-clairvoyant dynamic-agent scheduling
Latency-Aware-adapted	heterogeneous lifecycle/placement；fusion N/A
Your full method	explicit future topology/resource prediction

如果你必须严格只有四条 paper baselines：

把 Latency-Aware 从 primary 四条中换成 Agentix。

Latency-Aware 移到 appendix / additional comparison。

如果还有能力补第二条：

再补 Maestro-adapted。

原因是 Agentix 回答：

“完全不预测未来够不够？”

Maestro 回答：

“只用 role/tool-intent + 历史 remaining-work statistics 够不够？”

而你们回答：

“显式建模尚未展开未来的 topology/resource uncertainty 是否进一步有价值？”

这三层对照非常干净。

我会立即修改的一个 repo 决策

你仓库当前 latency_aware_baseline_freeze_v1.json 里这段：

graph_visibility: logical_graph_assumed_known

以及：

“Eq (3) and Eq (5) are defined on a KNOWN logical graph”

建议撤销。

改成类似：

graph_visibility: resolved_logical_window_only

并加入 gate：

𝑣
𝑖
𝑠
𝑖
𝑏
𝑙
𝑒
(
𝑣
,
𝑡
)
  
⟺
  
𝑡
≥
𝑟
𝑒
𝑠
𝑜
𝑙
𝑣
𝑒
𝑑
_
𝑎
𝑡
(
𝑣
)
visible(v,t)⟺t≥resolved_at(v)

然后重新统计：

legal_fusible_edges
multi_node_fused_chains
near_ready_visible_nodes
prefetch_eligible_nodes

我预计 fusion 前两项会归零；但后两项不能在没实际统计前说一定为零。

这应该写成：

[VERIFIED] information-contract correction，而不是 hyperparameter retuning。

可直接放论文里的 disclosure

我建议用这版，比较稳妥：

Latency-Aware-Orchestration applicability. We adapt Latency-Aware-Orchestration under its original resolved-workflow information boundary: only activations whose control-flow branch, role, and deployment have been resolved at runtime are exposed to the scheduler. Our target workloads are decision-driven ReAct-style workflows in which each tool invocation, continuation, or termination is determined by a preceding planner at runtime. Consequently, the workload contains no legally visible consecutive same-deployment activation pairs before an intervening control decision, and the Constructor's chain-fusion optimization has zero eligible opportunities. We therefore retain the baseline's applicable device-aware cost prediction, model-lifecycle management, admission/placement, and scheduling mechanisms, while reporting fusion as not applicable (N/A) rather than exposing the realized future workflow. Results for this baseline should thus be interpreted as the performance of its applicable mechanisms on dynamically unfolding agent workflows, not as an evaluation of its fusion optimization.

如果你想再防 reviewer 一层，可以紧接一句：

To avoid structurally disadvantaging this baseline from serving as the sole representative of dynamic agent scheduling, we additionally evaluate Agentix, whose non-clairvoyant scheduler explicitly assumes no prior knowledge of the program's execution DAG.

这句话有 Agentix 原论文直接支撑。
USENIX
+1

最后把我的决定压成一句话：

[VERIFIED + INFERENCE] 不要再让 Latency-Aware 读取完整 template；按论文恢复 resolved-only visibility，接受 fusion=0，并把它从“完整核心对手”降为“适用组件对手”。最值得补的是 Agentix；若还能补一条，则是 Maestro。不要为了让 fusion 触发去改主 workload，只可以另做 resolved-visibility / fixed-chain sensitivity experiment。

而且这其实是个好消息：你现在暴露出来的不是“我们的 workload 太奇怪”，恰恰是一个很明确的系统边界——已有方法利用的是已经暴露的未来结构，而你真正研究的是尚未 resolve 的 future structure 如何被预测并用于调度。 这个差异比单纯“我们预测 DAG”更有研究定位价值。
arXiv
+2
