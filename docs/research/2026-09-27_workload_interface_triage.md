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
timestamp: "2026-09-27T00:14+08:00"
status: "valid"
---

# Research Report

…
显示更多
思考了 5m 31s
ChatGPT 说：

炜，我现在的判定不是“workload 有问题”或者“baseline 实现有问题”二选一，而是：

[VERIFIED + INFERENCE] 两者都有，而且顺序上必须先处理接口公平性，再设计新的 workload stress regime。
现在这张正式表可以保留，但它不适合单独支撑“未来预测很有价值”这个核心叙事。更准确的含义是：在当前 node-atomic、serial-causal、统一 GPU substrate 和各自声明 adaptation 下，F0 比这些 paper-derived end-to-end baselines 好。 它不能证明“未来信息本身”比 current-only 信息有价值。

我在仓库里还找到了一个很关键的事实：你们其实在 8 月已经做过一次类似 stress pilot，当时把 arrival overlap 拉高后，action width 和 Oracle gap 都明显增大，但 PredOpt-H5 仍然和 Myopic 几乎完全相同。这个旧结果和你现在看到的现象高度一致：当前问题不只是 workload 不够挤，还包括 prediction → decision coupling 本身未必足够强。

1. 到底是 workload，还是接口/公平性？
Workload 侧：确实偏弱，而且是可验证的

[VERIFIED] 当前正式 projection 的 640/640 个 workflow 都通过 serial_causal_chain gate。也就是说 scheduler-visible 执行拓扑全部是串行链，没有真实的 intra-job 并行分支。

这直接意味着：

Agentix 的 ATLAS 场景不存在，formal 只能退化为 PLAS；
Latency-Aware 的 fusion 在合法 visibility 下 N/A；
“未来 DAG 宽度/并行结构”几乎没有 scheduling value；
未来预测实际主要是在预测“后面还有多少节点、什么类型、什么模型”，而不是决定并行 DAG 的资源布局。

所以如果论文核心故事仍然写成“dynamic future DAG scheduling”，这个 workload 本身确实支撑不足。

[VERIFIED] episode generator 的 offered load 只有：

0.50
,
0.70
,
0.85
,
0.95
,
1.05
0.50,0.70,0.85,0.95,1.05

而且 job 数虽然有 16/32/64，但 arrival window 会随着总 GPU work 一起缩放：

𝑤
𝑖
𝑛
𝑑
𝑜
𝑤
=
total GPU work
𝜌
⋅
𝑁
G
P
U
window=
ρ⋅N
GPU
	​

total GPU work
	​


因此“64 jobs”并不自动意味着更高 contention；它很多时候只是让 episode 更长，而保持类似平均 offered load。

[VERIFIED] deadline 也是统一的：

𝑎
𝑟
𝑟
𝑖
𝑣
𝑎
𝑙
+
{
1.5
,
2
,
3
}
×
𝑡
𝑜
𝑡
𝑎
𝑙
_
𝑝
90.
arrival+{1.5,2,3}×total_p90.

仓库已有 pressure audit 之前就发现这些 deadline 普遍偏松；旧 stress pilot 也验证了“只收紧 deadline”主要改变 deadline-miss，而不改变 completion/queue，因为大多数 policy 根本不拿 deadline 排序。

所以 deadline 不是当前 mean completion 问题的首要元凶。

更值得注意的是 synthetic hard priority

[VERIFIED — 我认为这是当前 workload 最值得先审的一项]

episode builder 直接写：

Python
运行
service_class = "normal" if job_index % 4 else "priority"

即每 4 个 job 人工指定一个 hard-priority job。

而几乎所有 policy 的排序 key 第一项都是：

纯文本
hard service class

Pythia、Agentix、F0、myopic 等都不能跨这个 class 重排。

这相当于你人为把 ready set 先切成两层，然后 baseline 只能在同层内部竞争。

这会进一步压缩：

真正的 job-ordering decision space

而这个 priority class 不是从 VideoSeek trace 的真实 QoS 标签产生的。

[INFERENCE] 如果你的论文并不研究 QoS class，我会认真考虑让 “all-normal” 成为标准 benchmark，至少必须有 all-normal sensitivity。

这一点可能比把 deadline 从 2× 改到 1.2× 更重要。

接口侧：现在确实存在一个明显 confound

[VERIFIED]

当前 choose_action() 的 candidate 不是：

纯文本
ready job/node

而是：

纯文本
ready job/node × every free GPU

也就是一个联合：

(
𝑛
𝑜
𝑑
𝑒
,
 
𝐺
𝑃
𝑈
)
(node, GPU)

动作。

而不同 scheduler 对 GPU placement 的处理方式不一样。

例如 Myopic/F0 明确计算：

𝑐
𝑢
𝑟
𝑟
𝑒
𝑛
𝑡
=
𝑟
𝑢
𝑛
𝑡
𝑖
𝑚
𝑒
𝑝
50
+
{
0
	
𝑚
𝑜
𝑑
𝑒
𝑙
 
𝑟
𝑒
𝑠
𝑖
𝑑
𝑒
𝑛
𝑡


𝑙
𝑜
𝑎
𝑑
𝑝
50
	
𝑜
𝑡
ℎ
𝑒
𝑟
𝑤
𝑖
𝑠
𝑒
current=runtime
p50
	​

+{
0
load
p50
	​

	​

model resident
otherwise
	​


因此天然偏好当前已有 model 的 GPU。

但 Agentix formal PLAS 的 key 基本是：

纯文本
hard_class
attained_service
ready_time
...
gpu.index

Pythia 类似：

纯文本
hard_class
-S_eff
ready_time
...
gpu.index

也就是说，它们选定某个 job 后，并没有 locality-aware GPU placement，最后实际上靠 gpu.index tie-break。

LLMSched 也没有把 per-GPU residency 放进主 JCT/entropy key；TIE 则显式带 resident 信息；Latency-Aware 有自己的 placement/admission logic。

所以你现在比较的不只是：

F0 scheduler vs Agentix scheduler

而是同时比较：

F0 job ranking + locality-aware placement

vs

Agentix job ranking + index-based placement

这确实可能放大 eviction/load，并进一步转化成后面的 queueing。

你给的 eviction：

纯文本
myopic 565
Pythia 758
Agentix 690

与这个代码事实至少是方向一致的。

[INFERENCE] 因此目前不能把所有 queue-time 差距都归因于 Agentix/Pythia 的 job-ranking rule。

这是我认为在改 workload 之前必须先排掉的公平性问题。

2. 哪些 workload 维度最可疑？优先级怎么排？

我会按下面的优先级查。

优先级	维度	当前判断	最关键诊断
P0	真正的竞争宽度	[VERIFIED/INFERENCE] 很可能过窄	competitive_jobs_after_hard_class
P0	placement/locality confound	[VERIFIED] 存在	GPU-choice regret / shared-placement rerun
P1	arrival overlap / offered load	[VERIFIED] 偏低	按 pressure/burst 分层 F0−myopic
P1	serial-only topology	[VERIFIED] 640/640	无法靠当前数据修复
P1	program service heterogeneity	[UNVERIFIED] 需要统计	total-GPU-service CV、node-count CV
P1	future-information ceiling	[UNVERIFIED] 最重要	myopic / truth-H5 / oracle
P2	model lifecycle heterogeneity	[INFERENCE] 有影响但非唯一原因	avoidable loads、model-transition entropy
P2	deadline	[VERIFIED] 对当前 mean JCT 不是主要矛盾	deadline-only stress 已基本证明

这里特别注意：

不要再只报告 candidate_count。

因为如果：

纯文本
1 ready node × 2 free GPUs

candidate_count=2，

但实际上没有任何 job-ordering choice。

你真正需要的是三个宽度：

𝑁
r
e
a
d
y
 
j
o
b
s
N
ready jobs
	​

𝑁
c
o
m
p
e
t
i
t
i
v
e
 
j
o
b
s
 
a
f
t
e
r
 
h
a
r
d
 
c
l
a
s
s
N
competitive jobs after hard class
	​

𝑁
f
e
a
s
i
b
l
e
 
p
l
a
c
e
m
e
n
t
s
N
feasible placements
	​


三个必须分开。

3. 在改 workload 前，我最建议先做的一个“未来价值分解”

这是现在最重要的诊断，比继续看五条 baseline 都重要。

直接在完全相同的 workload、完全相同 placement/consumer下跑四个点：

Myopic
→
F0 predicted-H5
→
SameShape Truth-H5
→
Full Oracle
Myopic→F0 predicted-H5→SameShape Truth-H5→Full Oracle

然后看三个 gap。

情况 A

如果：

𝑀
𝑦
𝑜
𝑝
𝑖
𝑐
≈
𝐹
0
≈
𝑇
𝑟
𝑢
𝑡
ℎ
𝐻
5
≈
𝑂
𝑟
𝑎
𝑐
𝑙
𝑒
Myopic≈F0≈TruthH5≈Oracle

[INFERENCE] workload 几乎没有 future-information value。

这时必须增强 workload。

情况 B

如果：

𝑀
𝑦
𝑜
𝑝
𝑖
𝑐
≈
𝐹
0
Myopic≈F0

但：

𝑇
𝑟
𝑢
𝑡
ℎ
𝐻
5
≪
𝑀
𝑦
𝑜
𝑝
𝑖
𝑐
TruthH5≪Myopic

那么：

[INFERENCE] workload 有未来价值，是 predictor 没把这个价值恢复出来。

情况 C

如果：

𝐹
0
≈
𝑇
𝑟
𝑢
𝑡
ℎ
𝐻
5
≈
𝑀
𝑦
𝑜
𝑝
𝑖
𝑐
F0≈TruthH5≈Myopic

但：

𝑂
𝑟
𝑎
𝑐
𝑙
𝑒
≪
𝑀
𝑦
𝑜
𝑝
𝑖
𝑐
Oracle≪Myopic

那么：

[INFERENCE] H=5 / 当前 future representation / consumer action space 不够。

不是 predictor accuracy 本身。

这会把你现在最大的疑问一次切开。

我甚至建议报告一个 diagnostic：

Future Opportunity
=
𝑀
m
y
o
p
i
c
−
𝑀
t
r
u
t
h
H
5
Future Opportunity=M
myopic
	​

−M
truthH5
	​


以及：

Realized Fraction
=
𝑀
m
y
o
p
i
c
−
𝑀
F
0
𝑀
m
y
o
p
i
c
−
𝑀
t
r
u
t
h
H
5
.
Realized Fraction=
M
myopic
	​

−M
truthH5
	​

M
myopic
	​

−M
F0
	​

	​

.

如果 denominator 接近 0，就明确写“该 workload 下 finite-horizon future information lacks actionable value”。

4. 我建议的 4 个 workload 改造

重点：不要覆盖现在 confirm300。

现在这批结果已经看过了。应该保留为：

Natural / Original regime

然后重新定义：

High-contention / Stress regime

规则只允许用 train/development workload statistics 确定，再冻结。

如果你们还有 sealed final test，最终新版 benchmark 应在那里一次性评估；否则新版只能称 post-hoc stress study，不能假装又是独立 confirm。

方案一：arrival compression + higher offered load

[VERIFIED precedent + INFERENCE recommendation]

这是第一优先。

不换 templates，不改未来结构，只把：

𝑡
𝑖
′
=
𝑡
0
+
𝛼
(
𝑡
𝑖
−
𝑡
0
)
t
i
′
	​

=t
0
	​

+α(t
i
	​

−t
0
	​

)

例如固定：

𝛼
∈
{
1
,
0.67
,
0.5
}
α∈{1,0.67,0.5}

或者新 target load：

𝜌
∈
{
0.9
,
1.2
,
1.5
}
.
ρ∈{0.9,1.2,1.5}.

这样保留原 arrival relative shape，也完全保留“未来尚未展开”。

仓库旧 stress pilot 已经证明 arrival overlap 增强能把 action width 大幅提高，并显著拉大 Oracle gap。

预期受益：

所有 job-ordering policy；
LLMSched/Pythia/Agentix 尤其需要多个同时 ready program；
F0 也终于有更多“未来信息改变选择”的机会。

风险：

post-hoc stress fitting。

解决方式：

α / load levels 从 development workload structural metrics 冻结，不根据哪条方法赢来选。

比如目标只规定：

纯文本
competitive-job decision rate 50–70%
GPU utilization 80–90%

不要规定：

纯文本
F0 必须赢 myopic 5%
方案二：去掉 synthetic hard service class，做 all-normal

[VERIFIED + INFERENCE] 我强烈建议至少做。

因为现在 25% priority 是 builder 人工加的。

如果这个 QoS class 不是研究对象，我倾向：

stress/main scientific comparison 全部 normal

原 workload 保留 hard-class 版本作为 QoS sensitivity。

这不会改变任何 future structure，只会解开人为锁死的 reorder opportunity。

预期：

Pythia/Agentix/LLMSched 等 job-level policy 得到更多真实竞争；
F0/myopic 也同样获得机会；
公平。

风险很低，因为不是为了某条 baseline 变强，而是在移除非研究核心的 synthetic constraint。

方案三：真实 template 的 short/medium/long balanced mix

[INFERENCE]

不要随机 uniform 抽 template，而是在 training templates 上按：

total GPU service
total GPU service

或者 node count 分成：

纯文本
short / medium / long

比如 train-only tertiles。

每个 episode 固定：

纯文本
1/3 short
1/3 medium
1/3 long

仍然全部来自真实 trace。

这样会制造更有意义的 job-level ordering tradeoff。

预期：

Agentix 的 attained service 有东西可以区分；
Pythia 的 completion distance 更有意义；
LLMSched 的 JCT/uncertainty ranking 更可能改变顺序；
F0 的 predicted remaining work 也更有价值。

建议同时报告：

纯文本
CV(total GPU service)
CV(node count)
Corr(attained_service, true_remaining_service)

尤其最后一个能直接解释 Agentix 为什么好/坏。

方案四：按 model-lifecycle intensity 分层，而不是造新模型

[INFERENCE]

不要人为增加虚构 model。

从 train templates 算：

纯文本
model-switch count
load / total-GPU-runtime fraction
model entropy

再分：

纯文本
low-churn
high-churn

两个 workload regime。

这样可以回答：

future information 的价值是不是只来自提前知道 model sequence / lifecycle？

而且不破坏 trace 真实性。

但这一项一定要等 placement confound 解决之后做，否则你只是把 myopic 的 resident advantage 放大。

不建议做的“第五种”

[VERIFIED/INFERENCE] 不要给当前真实 projection 人工加 parallel branches。

因为 640/640 已经验证是 serial causal control flow。

如果未来你确实要把论文核心提升到：

future DAG topology scheduling

那应该找/构建第二个真实有并行 sub-agent 的 workload，而不是把 VideoSeek trace 人工改成 DAG。

那会成为第二数据集/扩展实验，而不是当前 workload 的 stress transform。

5. 主对比到底怎么组织？

我的建议是：两者都要，但角色对调。

主科学证据：controlled same-interface comparison

[INFERENCE]

论文核心 claim：

future prediction 是否有 scheduling value

不能主要靠“五篇不同系统论文 baseline 都输给我”。

因为那些方法优化目标、信息来源和原始 serving abstraction 都不同。Agentix 重点解决 program/request HOL blocking，且原系统依赖 program-level scheduling/preemption；LLMSched直接针对 compound workflow structural uncertainty；Pythia利用 workflow predictability；TIE则本质是 uncertainty-aware output-length SJF。
USENIX
+3

更强的证据应该是：

同 substrate、同 placement、同资源估计、同 candidate set，只改变 future information / decision rule。

我建议主表做：

Arm	future info	consumer/placement
Myopic	none	common
F0-no-future	future masked	same F0 consumer
F0	predicted H5	same F0 consumer
Truth-H5	true H5	same F0 consumer
Oracle	privileged	upper bound

再做一个 rule table：

同样 current-visible state + common placement 下，换不同 job-ranking rule。

这两张才是你的核心因果证据。

五条 paper baseline

继续保留。

但我会把标题写成：

End-to-end faithful adaptations under the unified node-atomic substrate

而不是：

SOTA baselines.

它们是重要的 system-level reference，但不是未来预测价值的唯一证明。

6. myopic ≈ F0 应不应该报告？

[VERIFIED observation / INFERENCE interpretation] 必须报告。

不能把它当坏数据藏起来。

它有两种可能含义，而前面的 Truth-H5/Oracle ladder 会决定是哪一种。

如果 Truth-H5 也 ≈ Myopic：

“在自然负载下，future information is mostly non-actionable.”

这是一个很真实的系统结论。

你甚至可以把论文故事变得更好：

Future prediction is not universally useful; its value appears only when contention creates decision opportunity.

然后高-pressure regime 展示价值随竞争度增加。

这比“任何情况下未来预测都提升”更可信。

但如果 Truth-H5 明显优于 Myopic：

那 F0≈myopic 就不是 workload 解释，而是你当前 predictor/consumer 还没兑现 oracle opportunity。

这种情况下不应该靠加 stress workload 掩盖。

7. 改 workload 前的接口排查，我建议严格按这个顺序

这是我认为你现在最应该实际执行的一组实验：

[VERIFIED/INFERENCE] 先修/隔离 placement。
把 action 临时拆成两阶段 diagnostic：

policy chooses job/node
→
common placement chooses GPU
policy chooses job/node→common placement chooses GPU

common placement 只用当前合法信息：

memory fit；
resident-first；
最小 incremental current load/cost；
deterministic tie-break。

不读 future。

跑：

纯文本
baseline-native-placement
baseline-common-placement

的 paired delta。

如果 Pythia/Agentix 大幅回升，当前正式表有显著 placement confound。

[VERIFIED recommendation] 做 GPU-choice regret shadow audit。
对每个实际 selected job/node，计算：

𝑐
ℎ
𝑜
𝑠
𝑒
𝑛
𝐺
𝑃
𝑈
_
𝑐
𝑢
𝑟
𝑟
𝑒
𝑛
𝑡
𝐶
𝑜
𝑠
𝑡
−
𝑏
𝑒
𝑠
𝑡
𝐹
𝑟
𝑒
𝑒
𝐺
𝑃
𝑈
_
𝑐
𝑢
𝑟
𝑟
𝑒
𝑛
𝑡
𝐶
𝑜
𝑠
𝑡
.
chosenGPU_currentCost−bestFreeGPU_currentCost.

再统计：

纯文本
avoidable non-resident placement rate
avoidable load ms
avoidable eviction count

这个甚至不用修改 trajectory 就能先定位。

[VERIFIED] 把 candidate width 改成 job-level width。
每次记录：

纯文本
ready_job_count
competitive_job_count_after_hard_class
feasible_GPU_count_per_job

不再把 (same node, GPU0) 和 (same node, GPU1) 当两次 scheduling opportunity。

[INFERENCE] 跑 all-normal shadow。
完全相同 episode，只把 synthetic service_class 统一成 normal。
如果 policy disagreement/action width 明显上升，就证明 hard class 在遮蔽调度机制。

[INFERENCE] 跑 future-opportunity ladder：

纯文本
Myopic
F0
SameShape-Truth-H5
Full Oracle

这是决定“改 workload”还是“改 predictor/consumer”的最终 gate。

[INFERENCE] 最后才做 workload stress。
先只用 anchors：

纯文本
Myopic
F0
Truth-H5
Oracle

检查 workload structural activation；冻结 stress definition 后，再跑五条论文基线。

我的整体建议可以压缩成一句：

现在不要因为五条 paper baseline 输得太惨，就直接去“造一个更难 workload”。你们已经有足够证据说明当前 natural workload 的 scheduling opportunity 偏窄，但代码里也确实存在 joint job×GPU action 造成的 placement confound，而且 synthetic hard-priority class 进一步压缩了重排空间。先用 common-placement + all-normal + Myopic/F0/Truth-H5/Oracle 这三组诊断把问题切开；如果 Truth-H5/Oracle 显示未来确实有价值，再用 train/dev 定义 high-contention stress workload。原 confirm300 不删、不改，作为 natural regime 保留。

这样最后论文反而可以形成一个更强的结果结构：Natural regime：future value limited；High-contention regime：future information becomes actionable；End-to-end paper adaptations：在统一 substrate 下的系统级参照。 这会比现在“所有论文基线都被我碾压”更容易让审稿人相信。
