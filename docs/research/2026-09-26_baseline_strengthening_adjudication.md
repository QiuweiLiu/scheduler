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
timestamp: "2026-09-26T12:04+08:00"
status: "valid"
---

# Research Report

…
显示更多
思考了 3m 41s
ChatGPT 说：

炜，这次我会把标准收得更严：不是“论文里有的都补”，而是“只补那些在你们 simulator 里存在同语义状态、能够 leak-free 映射的核心机制”。按这个标准，我的总裁决是：

Pythia aging：补齐。Pythia S_unblock：补齐，但明确是 queue-demand proxy adaptation。Agentix K 队列/防饿死：不要塞进主 baseline，最多作为 non-preemptive sensitivity variant。Agentix token/KV 抢占：不补。LLMSched sample_tasks(r)：不补。

原因很关键：Pythia 的 aging 和 S_unblock 都能用你现在已经存在的“等待时间、历史 PFA、当前实时 model demand”表达；Agentix 的 K 队列则是为可抢占的 decoding quantum设计的，拿掉抢占之后只剩一个离散化外壳，已经不再是同一个机制。Pythia Algorithm 3 确实把 S_completion 和 DownstreamIdleRisk 合成为 base priority，并在 worker 侧加入按等待时间增长的 aging。
arXiv
+1
 Agentix 则明确把 K 队列、time quantum、耗尽后 demotion 和 program-level anti-starvation 放在它的 preemptive scheduler 中。
USENIX
+1

下面逐 Q1–Q7。

Q1. 补齐机制，还是只披露 omission？
Verdict：[INFERENCE] 对“能在同一语义层复现”的核心调度机制，应补；对缺少底层 primitive 的机制，应保留 omission。

这比“全部 omission”更可辩护。

审稿人真正容易质疑的不是：

“为什么你自己给 baseline 写了代码？”

因为跨系统 benchmark 几乎必然要重新适配。

真正危险的是：

“你为什么只实现了对手论文里最容易的那一半，而漏掉会让它变强的机制？”

尤其 Pythia 很明显。原论文 Algorithm 3 不是只有：

𝑆
completion
S
completion
	​


而是：

base_priority
=
𝜔
1
𝑆
completion
+
𝜔
2
𝑆
unblock
.
base_priority=ω
1
	​

S
completion
	​

+ω
2
	​

S
unblock
	​

.

同时 local worker 会根据 accumulated wait time 动态加 aging。
arXiv
+1

所以你现在的 omega2=0 + no aging 确实是一个偏弱 adaptation。

这里补齐是加分项，不是坏事。

但另一个极端也危险：

“为了号称 full reproduction，我们自己发明了一个并不存在于原 substrate 的等价机制。”

这就是 Agentix preemption 的风险。

因此我建议以后统一用这个原则：

Representable core mechanism → implement.
Missing semantic primitive → disclose, do not emulate.

这样不会变成“我们替对手设计了一个新系统”。

Q2. 四个可补项怎么做？

先给总表。

机制	Verdict	最小实现	必要 sentinel	成本
Pythia aging	补齐	Pythia-specific effective priority	wait 单调性、同 prefix、class 隔离	0.5–1 天
Pythia S_unblock	补齐	PFA future-role hitting distance + live model-demand proxy	identical-prefix/different-future、train-only、queue causality	1–2 天
Agentix K 队列	不进主 baseline	可做 discrete non-preemptive sensitivity	bin monotonicity、future invariance	~1 天
Agentix anti-starvation	仅随上面 sensitivity	
𝑊
/
𝑇
≥
𝛽
W/T≥β promotion	threshold/reset/program aggregation	0.5–1 天
LLMSched sample_tasks(r)	不补	—	omission test/documentation	<0.5 天文档；真做 2–4+ 天
2.1 Pythia aging
Verdict：[VERIFIED] 补齐。

原论文说得非常直接：

每个 scheduling window 开头，local worker 重新计算 effective priority，并向 global base_priority 加入一个随 accumulated waiting time 增加的 aging factor。
arXiv

而当前仓库 workload_v02_simulator.py 的 Pythia branch 只做：

纯文本
hard_class
-S_completion
ready_time

并明确承认 ready_time 只是 tie-break。

所以这是真 omission。

不过我不建议直接复用现在的 srtf_aging_key() 数学公式。

为什么不能直接复用？

[VERIFIED repo] 你现有 SRTF aging 是：

𝑅
𝑗
−
𝑤
𝑎
𝑖
𝑡
𝑗
R
j
	​

−wait
j
	​


两项单位都是 ms。

但 Pythia：

𝑆
completion
=
1
𝐷
S
completion
	​

=
D
1
	​


是无量纲 score。

直接：

𝑆
completion
+
𝑤
𝑎
𝑖
𝑡
𝑚
𝑠
S
completion
	​

+wait
ms
	​


当然不行。

最小实现

复用你已有的：

纯文本
wait_age_ms = now - ready_since

但新建：

Python
pythia_effective_priority(
    base_priority,
    wait_ms,
    aging_scale_ms,
    aging_weight
)

例如 conceptual form：

𝑆
eff
=
𝑆
base
+
𝜆
age
𝑊
𝜏
age
.
S
eff
	​

=S
base
	​

+λ
age
	​

τ
age
	​

W
	​

.

然后 scheduler key：

纯文本
hard_service_class
-effective_priority
ready_time
node_id
gpu

这里我把 
𝜆
,
𝜏
λ,τ 标为：

[UNVERIFIED] 我在论文正文可验证部分没有看到具体 aging 系数公式/默认值。

所以不要拍脑袋在 test/confirm 上调。

优先级顺序应该是：

如果作者 release/code 给默认值 → 直接用；
找不到 → 预注册一个很小的 dev-only grid；
参数选完 freeze，再碰 confirm。

必要 sentinel：

[VERIFIED target] wait 增长时 effective priority 单调增加；
相同 wait 下 S_completion 更大的仍更优；
future suffix 任意改动不影响 aging；
hard service class 若保留，aging 永远不能跨 class；
wait=0 时精确退化到当前 Pythia base score。
2.2 Pythia S_unblock / DownstreamIdleRisk
Verdict：[INFERENCE] 我现在建议补。

这一项比 aging 更重要，因为它属于 Algorithm 3 的另一半。

原算法实际上是：

纯文本
future_agents = GetFutureAgents(r.regex)

for future agent a:
    if queue[a.model] < IDLE_THRESHOLD:
        risk += 1 / E[D_a]

然后：

base= \omega_1S_\text{completion} + \omega_2S_\text{unblock}. \] :chatgpt-content-reference{index="4"} 而且论文解释得很清楚：如果一个未来 agent 对应的 model 快闲下来了，那么当前 upstream request 越接近那个 model，boost 越大。:chatgpt-content-reference{index="5"} ### 你提出的方向是对的，但我会改一个地方 你说： > “PFA 推算下一步可能是什么 role → model” **不要只看下一步。** Pythia 原版考虑的是： > **all relevant future agents + expected distance** 因此你们已经有 PFA，最自然的适配是预计算： \[ E[D(a)\mid r]

即从当前 role 
𝑟
r 到未来 role 
𝑎
a 的 expected first-hitting distance。

然后把 role 映射到 model。

建议模块

比如：

纯文本
pythia_unblock.py

冻结三个 train-only table：

纯文本
role_transition_probability
role_to_model_distribution
expected_hitting_distance[current_role][future_role]

运行时只读取：

纯文本
candidate.current_role
current live model demand
frozen PFA tables

绝对不能读：

纯文本
job.template.successors
future node id
realized next tool
future_artifacts
实时 Rqueues 怎么映射？

这里必须诚实。

[VERIFIED] 原论文使用的是 model-replica queue depth。 
arXiv

你们没有 model-server replica queue。

所以推荐一个非常保守的 proxy：

𝑄
𝑚
(
𝑡
)
=
𝑁
𝑚
ready
(
𝑡
)
+
𝑁
𝑚
running
(
𝑡
)
Q
m
	​

(t)=N
m
ready
	​

(t)+N
m
running
	​

(t)

也就是当前 scheduler-visible 的：

这个 model 已 ready / 正运行的 request 数量。

如果：

𝑄
𝑚
(
𝑡
)
=
0
Q
m
	​

(t)=0

定义为 idle-risk.

不要用：

纯文本
model currently resident?

代替 idle。

因为 resident ≠ 有 work。

也不要看 future planned demand，那就循环了。

这个 adaptation 应写成：

model-demand proxy for replica queue depth

而不是声称复制了原 Pythia 的 Rqueues.

PFA 怎么算 risk？

我倾向：

𝑆
unblock
=
∑
𝑎
∈
𝐹
(
𝑟
)
1
[
𝑄
𝑚
(
𝑎
)
<
𝜃
]
⋅
𝑃
(
reach 
𝑎
∣
𝑟
)
𝐸
[
𝐷
𝑎
∣
𝑟
]
.
S
unblock
	​

=
a∈F(r)
∑
	​

1[Q
m(a)
	​

<θ]⋅
E[D
a
	​

∣r]
P(reach a∣r)
	​

.

这里比原 pseudocode 多了一个 reach probability。

为什么？

因为你现在的 PFA 是概率图，不是作者生成的 bounded regex 实例。

如果一个 role 只有 5% 可能出现，却和 100% 会出现的 role 一样贡献 1/D，会系统性过估。

所以：

[INFERENCE] 这是更合理的 PFA faithful adaptation，但不是论文逐字公式。

如果想更贴论文，可以用：

retained bounded PFA 中所有 reachable future roles，每个只用 
1
/
𝐸
[
𝐷
]
1/E[D]。

然后概率已经通过 expected distance 隐含进去。

我建议先做后者，改动更少、解释更直接。

non-leaky sentinel 必须很硬

这一项我至少要求 5 个：

A. identical-prefix / different-realized-future

两个 workflow：

纯文本
observed prefix 完全一样
current role 完全一样
realized future A / B 完全不同

Pythia 的：

纯文本
S_unblock
最终 action

必须完全一致。

这是最重要的一条。

B. future-template mutation

只改未执行 suffix 的：

纯文本
role
model
runtime
length

分数不变。

C. PFA mutation

改 train-only PFA 的 transition probability，分数应该改变。

证明你真的消费了 predictor，而不是 trace future。

D. current queue causality

若 PFA 预测 model M reachable：

纯文本
Q_M : busy -> idle

必须增加 S_unblock。

改变完全 unrelated model 的 queue 不应影响。

E. train-only mapping

validation/test 里的 role→model 统计不允许改变 frozen mapping。

做到这五条，我会认为：

[INFERENCE] non-leak contract 足够强。

2.3 Agentix K-level queues
Verdict：[VERIFIED mechanism + INFERENCE applicability] 不建议补到主 baseline。

这是这轮最需要避免“看起来更完整、其实更不忠实”的地方。

Agentix 为什么有 K queues？

不是单纯为了：

“把 PLAS 离散一点”。

原论文明确说 continuous priority + preemption 可能退化成 worst-case round-robin，并产生频繁 KV swaps，所以才：

priority 分到 K 档；
queue 有 time quantum；
call 在 quantum 内执行；
quantum 用完 demote；
高优先级 queue 可抢低优先级 call。
USENIX
+1

因此：

“service 涨了 → 下一次自然进低档”

只覆盖了原机制的一小部分。

它不是原来的：

in-call quantum exhaustion → demotion。

关键问题

你们 node 是 atomic。

当前 node 执行：

纯文本
start → finish

中间没有 decode iteration。

所以不存在：

纯文本
quantum exhausted but call unfinished

这个状态。

因此 K queues 失去了它在 Agentix 中的主要语义。

如果你特别想做

可以做：

Agentix-Discrete-NoPreempt

但是必须单独列为：

sensitivity / adaptation diagnostic

不能替换主 Agentix PLAS。

实现：

𝑏
𝑖
𝑛
=
𝑓
(
𝑎
𝑡
𝑡
𝑎
𝑖
𝑛
𝑒
𝑑
_
𝑠
𝑒
𝑟
𝑣
𝑖
𝑐
𝑒
)
bin=f(attained_service)

key：

纯文本
hard_class
bin
ready_time
...

但它回答的是：

“PLAS continuous vs coarse discretization 是否影响结果？”

不是：

“完整 Agentix MLFQ 是否更强？”

这两个不能混。

成本很低，大概半天到一天。

2.4 Agentix anti-starvation
Verdict：[VERIFIED mechanism] 原机制重要；[INFERENCE] 在没有 MLFQ 时不要单独塞进 primary。

原论文：

𝑊
𝑝
+
𝑊
𝑐
𝑇
𝑝
+
𝑇
𝑐
≥
𝛽
T
p
	​

+T
c
	​

W
p
	​

+W
c
	​

	​

≥β

则把 call promotion 到 
𝑄
1
Q
1
	​

。promotion 后只 reset 当前 call 的 
𝑊
𝑐
,
𝑇
𝑐
W
c
	​

,T
c
	​

，program-level history 保留，因此同一 program 的 threads 会倾向一起提升。
USENIX
+1

这个机制是与 K queues 绑定的：

promote to Q1.

如果你没有 Q1，就只能发明：

纯文本
starving=True → override PLAS

这已经变成你自己的 policy。

所以：

不要只给 continuous PLAS 加一个 mysterious starvation bonus。

如果你做 Discrete-NoPreempt sensitivity，可以一起实现：

纯文本
if W_total/T_total >= beta:
    bin = Q1

这是有意义的。

但仍标：

non-preemptive adaptation of Agentix queueing/anti-starvation.

2.5 LLMSched sample_tasks(r)
Verdict：[VERIFIED] 不补。

这里你的判断我同意。

LLMSched Algorithm 1 确实在 exploration 被选中时：

𝑡
=
𝑠
𝑢
.
𝑠
𝑎
𝑚
𝑝
𝑙
𝑒
_
𝑡
𝑎
𝑠
𝑘
𝑠
(
𝑟
)
t=s
u
	​

.sample_tasks(r)

而 exploitation 使用 all_tasks()；论文解释这么做是因为 exploration 若一次执行整个 stage，会拖慢短 job，所以只抽样一部分 task 来获取 uncertainty information。
arXiv
+1

但你们执行单元已经是一个 indivisible node。

没有：

纯文本
stage contains N exchangeable tasks

这个抽象。

为了实现它，你必须把：

纯文本
node

拆成：

纯文本
subtasks

这会改变：

workload DAG；
resource accounting；
service time；
ready semantics；
GPU action space；
甚至所有 baseline 的比较单位。

[INFERENCE] 代价/风险明显大于收益。

正确做法就是写：

task-level sampling is not representable because the unified execution unit corresponds to an indivisible activation rather than a multi-task stage.

这类 omission 很安全。

Q3. Agentix 抢占：敏感性臂还是不做？
Verdict：彻底不要把现有“整节点重算抢占”叫 Agentix preemption。

如果有余力：

最多做一个明确的 non-paper sensitivity arm。

原因已经非常清楚。

[VERIFIED repo] 你们现有 preemption：

纯文本
中断 node
丢弃进度
重新执行完整 compute

而 Agentix 原版是 decode-level/time-quantum scheduling，并专门讨论 KV block GPU↔CPU swapping 与多步调度优化。
USENIX
+1

两者完全不是一个 cost model。

所以绝对不能：

纯文本
agentix_preemptive = reuse maybe_preempt()

然后说：

full Agentix.

如果要做 sensitivity，我建议名字就直接：

Agentix-PLAS + restart-preemption (non-paper sensitivity)

目的只回答：

在我们的 node-restart substrate 里，允许抢占是否影响趋势？

不要放主表。

Q4. 公共 hard service class 要不要让 baseline 覆盖？
Verdict：[INFERENCE] 保留，而且所有 baseline 一致。

你现在：

纯文本
service_class=priority

永远在 algorithm-specific score 前。

我认为这反而是正确的 benchmark contract。

原因是如果：

Pythia 可以覆盖 class；
Agentix 可以覆盖 class；
你的 scheduler 不能；
LLMSched 又有另一种解释；

比较就混入：

谁允许破坏系统 service-class policy

而不是：

谁的 scheduling 更好。

所以统一：

key
=
(
hard class
,
  
baseline internal priority
,
.
.
.
)
key=(hard class,baseline internal priority,...)

最清楚。

但是要披露：

all baseline-specific priorities operate within the common service class.

这不是论文机制。

它是 benchmark substrate contract。

我建议补一个很便宜的 sensitivity：

all-normal workload

所有 service_class=normal。

如果排序和结论基本一致，就能证明 hard class 没掩盖 baseline mechanism。

不用做大规模，confirm/dev 子集就够。

Q5. PFA 推 S_unblock 是否 non-leaky？
Verdict：[INFERENCE] 是，只要 PFA 完全 train-only，且 runtime 不读取 realized suffix。

这里需要把“预测未来”和“偷看未来”区分开。

Pythia 本身就故意预测 future agents。

它通过历史 workflow pattern / regex 来得到：

纯文本
future agents
expected distance

这是它的论文机制，不是 leakage。Pythia 强调动态 workflow 可以通过历史重复模式/regular-expression representation 获得结构可预测性。
arXiv

所以：

纯文本
train history
→ PFA
→ probabilistic future roles

完全合法。

非法的是：

纯文本
this episode's template
→ actual future node

只要前面那五个 sentinel 都过，你可以非常有底气地写：

Pythia receives historical probabilistic workflow knowledge, not realized future structure.

Q6. 补完之后，最小实验怎么设计？
Verdict：不要因为“对手变强”而加一大堆额外 benchmark；主表直接换成最完整、最公平的 leak-free variant。

这点很重要。

如果补完以后：

纯文本
我们领先 10%
→ 只领先 3%

这不是坏事。

反而论文更可信。

如果补完以后：

纯文本
Pythia 超过我们

那你就提前发现了研究问题，而不是投稿之后让 reviewer 帮你发现。

我建议的最小实验集只有三层
A. 主表：只放“最终冻结的最完整公平 variant”

比如：

纯文本
LLMSched-adapted
Pythia-full-priority-adapted
TIE-adapted
Latency-Aware-applicable
Agentix-PLAS-adapted
Ours

Pythia full 指：

𝑆
completion
+
𝑆
unblock
+
𝑎
𝑔
𝑖
𝑛
𝑔
S
completion
	​

+S
unblock
	​

+aging

在你们可表达的范围内。

Agentix 仍是：

PLAS non-preemptive adaptation

不要为了“对称”硬加 fake MLFQ。

B. 一张小 ablation 表证明 baseline 没被你故意削弱

Pythia：

纯文本
completion only
+ aging
+ S_unblock
+ both

这其实是一个很漂亮的 2×2：

aging
∈
{
0
,
1
}
×
𝑆
unblock
∈
{
0
,
1
}
.
aging∈{0,1}×S
unblock
	​

∈{0,1}.

这个 2×2 我强烈推荐。

它不仅证明：

我们把原论文两个 missing mechanism 都补了，

还能告诉你它们到底有没有用。

Agentix 不需要 2×2。

主表 PLAS。

附录最多：

纯文本
continuous PLAS
discrete-no-preempt
discrete-no-preempt + anti-starvation
C. 一个 oracle/privileged reference

保留你们已经有的 truth/oracle 系列。

但明确：

informational upper bound, not a competitor.

用于回答：

性能差距是 predictor 的问题，还是 scheduler 的问题？

不建议现在做 0→100% future visibility 曲线

[INFERENCE] 这更适合 Latency fusion / known-DAG vs latent-future 的研究问题。

对于这轮：

“baseline 有没有被削弱？”

不是必须。

如果文章后面真的把核心 claim 定位成：

value of unresolved-future prediction

再把它作为机制实验加入。

现在别把实验树重新炸开。

Q7. 怎么避免“你们给 baseline 写强/写弱”的质疑？

这部分我认为比代码本身还重要。

Verdict：[INFERENCE] baseline 可以独立实现，但必须采用“先定义、再实现、再冻结、最后看结果”的流程。

我建议以后每一条 paper baseline 都有一个固定的 fidelity manifest，至少包含这些字段：

纯文本
paper
paper equation / algorithm lines
information available to original method
our simulator mapping
faithful components
adapted components
unrepresentable components
hyperparameters
training-only statistics
runtime-visible state
prohibited inputs
sentinel tests
freeze commit

核心原则只有一句：

主 baseline 是“最接近论文、且在我们的 substrate 中可合法表示的版本”，不是“dev 上表现最好的版本”。

这是避免 cherry-pick 的关键。

参数规则

例如 Pythia：

纯文本
ω1
ω2
aging λ
aging τ
idle threshold

如果论文明确提供值：

paper value。

如果没有：

只能在 dev 上按预注册 grid 选一次。

然后 freeze。

不能：

纯文本
正式跑完发现 Pythia 太强
→ 改参数

也不能：

纯文本
Pythia 太弱
→ 再给它调到差不多

两种都不行。

Agentix 的：

纯文本
K
queue boundaries
quantum
β

如果你们不实现它，就不存在这个问题。

Gate 规则

我建议把 baseline freeze 分成四类 gate：

Fidelity gate

toy case 手算必须和论文 mechanism 一致。

Leakage gate

identical-prefix / different-future。

Interface gate

只能读取 manifest 声明过的 state。

Non-degeneracy gate

新增机制确实在真实 workload 上会触发。

这个最后非常重要。

比如补完 Pythia 后至少统计：

纯文本
aging changed winner: x%
S_unblock nonzero: y%
S_unblock changed winner: z%

否则你辛苦补了机制，结果在 workload 上根本不起作用。

这和 Latency fusion 的教训一样。

建议执行顺序

我会按下面这个顺序做，而且做到第 4 步就可以重新 formal freeze：

Pythia aging。 最简单、论文直接要求、现有 wait state 完全够用。不要复用 srtf_aging_key 数学，只复用 wait-time plumbing。
arXiv

Pythia S_unblock。 新建独立模块：train-only PFA → reachable future roles / expected distance → frozen role→model mapping；runtime 只读当前 model demand proxy。补完整 identical-prefix sentinel。
arXiv

Pythia 2×2 dev diagnostic。 completion-only / aging / unblock / both。用于确认实现真的有作用，不用于事后选“最好版本”；主 variant 应预先规定为 both。

重新 freeze Pythia。 Manifest 写清 queue-depth proxy、权重来源、aging 参数来源、hard-class-first。

修 Agentix disclosure，而不是继续补 scheduler。 正式版固定 PLAS, non-preemptive；把 MLFQ / quantum / anti-starvation / KV-aware preemption 都列成 unavailable serving-granularity mechanisms。Agentix 官方 scheduler 的这些机制确实依赖 decoding-level preemption。
USENIX
+1

可选 Agentix sensitivity。 只有还有时间才做 discrete-no-preempt + anti-starvation，并永远不替代 primary PLAS。

LLMSched 不动。 sample_tasks(r) 继续 omission；它要求 stage 内存在可部分执行的 task 集，而当前 node abstraction 不提供这个 primitive。
arXiv

所以我这次的最终建议比上一轮更具体：

你提出“不要怕给 baseline 写独立 scheduler”的方向是对的，但只对 Pythia 成立得很漂亮。Pythia 的 aging 和 S_unblock 都应该补，因为状态语义能合法映射；补完以后它应该取代当前 completion-only Pythia 成为正式主 baseline。Agentix 则相反——它缺失的招牌机制恰恰依赖你们没有的 token/KV preemption primitive，硬补会比 omission 更不忠实。因此 Agentix 主 baseline 保持 PLAS non-preemptive，K 队列/防饿死最多做敏感性，preemption 不做。LLMSched task sampling 也不做。

来源上，我这轮对 Pythia 使用 arXiv v2 原论文 Algorithm 3 / §4.2.2，对 Agentix 使用 USENIX NSDI 2026 正式 prepublication，对 LLMSched 使用 ICDCS 2025/arXiv 原文 Algorithm 1；仓库实现则直接读取了你当前公开仓库的 pythia_profiler.py、workload_v02_simulator.py、两套 fidelity tests，没有依赖之前会话的结论。
arXiv
+2
