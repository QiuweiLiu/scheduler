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
timestamp: "2026-09-26T15:36+08:00"
status: "valid"
---

# Research Report

…
显示更多
思考了 4m
ChatGPT 说：

炜，我复核了 359862e → e9af513 的真实 diff，以及 pythia_methods.py、pythia_profiler.py、live simulator branch、24 项 fidelity tests 和 freeze manifest；论文侧重新对了 Pythia Algorithm 3 和 §4.2.2。

先给总裁决：

当前还不能 freeze。没有发现 realized-future leakage，但有 1 个明确的 P0 fidelity bug：S_unblock 用错了距离。 此外还有几个 P1：candidate-dependent 的“均值归一化”不是原公式的等价缩放、freeze manifest 自相矛盾、参数状态仍写着“pending dev-grid”，以及缺少真正的 identical-prefix live-action sentinel。

修完这些以后，我认为这条 Pythia 可以正式 freeze，而且不需要大改 substrate。

1. base priority、aging 和 S_unblock 是否忠实？
Base priority 公式本身：PASS

[VERIFIED]

你现在：

𝑆
base
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
S
base
	​

=ω
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


和 Algorithm 3 完全一致。论文明确给出这个组合。
arXiv

S_completion 继续用你们已经冻结的 convention：

𝑆
completion
=
1
1
+
𝑉
(
𝑟
)
S
completion
	​

=
1+V(r)
1
	​


其中 
𝑉
(
𝑟
)
V(r) 是 current role 后面的 future-role count expectation。这个 +1 是你们对论文 
1
/
𝐸
[
𝐷
remain
]
1/E[D
remain
	​

] 的一致 distance convention，我上一轮已经接受，当前没有新问题。

但 S_unblock 当前实现有一个 P0

[VERIFIED — P0]

论文 Algorithm 3 对每一个 future agent 
𝑎
a 算的是：

𝐸
[
𝐷
𝑎
]
=
E
x
p
e
c
t
e
d
D
i
s
t
(
current agent
,
𝑎
)
E[D
a
	​

]=ExpectedDist(current agent,a)

然后：

𝑆
unblock
=
∑
𝑎
:
  
𝑞
𝑢
𝑒
𝑢
𝑒
(
𝑎
.
𝑚
𝑜
𝑑
𝑒
𝑙
)
<
𝑡
ℎ
𝑟
𝑒
𝑠
ℎ
𝑜
𝑙
𝑑
1
𝐸
[
𝐷
𝑎
]
.
S
unblock
	​

=
a:queue(a.model)<threshold
∑
	​

E[D
a
	​

]
1
	​

.

也就是：

当前 agent 距离这个将来可能被饿死的 agent 有多远。

论文文字也明确解释：“one step away” 应该比 “ten steps away” 得到更大的 boost。
arXiv

但你们 pythia_methods.py:66–72 当前写的是：

Python
total += s_completion(
    expected_remaining_steps(profiler, future)
)

expected_remaining_steps(future) 的含义却是：

future role 
𝑎
a 距离 workflow terminal 还有多远。

这是另一个量。

最简单的反例：

纯文本
A -> B -> C -> END

从 A 看：

论文应该是：

𝐷
(
𝐴
,
𝐵
)
=
1
,
𝐷
(
𝐴
,
𝐶
)
=
2
D(A,B)=1,D(A,C)=2

所以：

𝐵
:
1
,
𝐶
:
0.5.
B:1,C:0.5.

也就是 B 更值得赶紧 unblock，因为它更近。

但你们现在：

𝑉
(
𝐵
)
=
1
,
𝑉
(
𝐶
)
=
0
V(B)=1,V(C)=0

于是：

𝐵
=
1
/
(
1
+
1
)
=
0.5
B=1/(1+1)=0.5
𝐶
=
1
/
(
1
+
0
)
=
1.
C=1/(1+0)=1.

结果完全反过来了：越远的 terminal-side agent 反而可能越高。

所以这一项必须修，正式结果不能基于现在的 S_unblock。

正确的最小实现

我建议新增一个明确不同的 quantity：

Python
expected_distance_to_role(
    profiler,
    current_role,
    target_role,
    horizon
)

不要复用 expected_remaining_steps()。

对于你们概率 PFA，比较干净的 adaptation 是：

𝐸
[
𝐷
𝑎
∣
𝑎
 hit within 
𝐻
]
=
∑
𝑑
=
1
𝐻
𝑑
 
𝑃
(
first hit 
𝑎
 at 
𝑑
)
∑
𝑑
=
1
𝐻
𝑃
(
first hit 
𝑎
 at 
𝑑
)
.
E[D
a
	​

∣a hit within H]=
∑
d=1
H
	​

P(first hit a at d)
∑
d=1
H
	​

dP(first hit a at d)
	​

.

如果 H 内 hit probability=0，就不计这个 target。

这样同时：

利用了 train-only transition probability；
没读 realized future；
正确表达 current → future-agent distance；
循环也可控。

[INFERENCE] 这不是 Pythia regex 的逐字实现，但在你们 PFA abstraction 下明显比现在的 terminal-distance substitution 忠实。

2. S_unblock 取均值是否可接受？
Verdict：建议改回 sum，不要取 candidate-wise mean。

[VERIFIED paper / INFERENCE adaptation]

论文就是 raw sum：

S_\text{unblock} = \sum_a 1/E[D_a]. \] :chatgpt-content-reference{index="2"} 你现在做： \[ \frac{1}{|\mathcal F(r)|} \sum_a 1/E[D_a]

不能称为简单的“尺度归一化”。

因为 denominator：

∣
𝐹
(
𝑟
)
∣
∣F(r)∣

随 candidate 改变。

所以它会改变 ranking，而不仅仅是乘一个常数。

例如：

A 有 1 个近距离 idle future model：

𝑆
𝐴
=
1.
S
A
	​

=1.

B 有同样那个近模型，同时还有 9 个当前不 idle 的 future roles：

原论文：

𝑆
𝐴
=
𝑆
𝐵
=
1.
S
A
	​

=S
B
	​

=1.

你们 mean：

𝑆
𝐴
=
1
,
𝑆
𝐵
=
0.1.
S
A
	​

=1,S
B
	​

=0.1.

B 被额外惩罚了。

这不是 omega2 可以吸收掉的固定缩放。

推荐

保留：

𝑆
unblock
=
∑
𝑎
1
𝐸
[
𝐷
𝑎
]
.
S
unblock
	​

=
a
∑
	​

E[D
a
	​

]
1
	​

.

然后让：

𝜔
2
ω
2
	​


负责相对尺度。

这正是 Algorithm 3 已经提供 
𝜔
1
,
𝜔
2
ω
1
	​

,ω
2
	​

 的意义之一。

如果特别想归一化，也只能用全局固定常数，例如 train-only PFA 上预先冻结的：

𝐶
unblock
C
unblock
	​


然后：

𝑆
unblock
′
=
𝑆
unblock
/
𝐶
.
S
unblock
′
	​

=S
unblock
	​

/C.

因为所有 candidate 都除同一个常数，这才等价于重新定义 
𝜔
2
ω
2
	​

。

所以：

[P1] candidate-wise mean 建议删除。

3. queue-demand proxy 是否正确？全局还是 workflow 内？
全局 per-model：正确

[VERIFIED]

一定应该把不同 workflow 合并。

Pythia 原文读取的是：

real-time queue depths of all model replicas across the cluster

然后避免某一 specialized model worker 因为上游没有及时推进而闲置。
arXiv

所以当前：

Python
for job in jobs:
    for ready/running gpu node:
        demand[model] += 1

方向是对的。

如果只统计同一 workflow，会失真：

纯文本
workflow A 的未来需要 model X
workflow B 已经有很多 X 请求

这时 X 根本不处于 starvation/idle risk。

Pythia 应该看整个 cluster。

因此：

global per-model，而不是 per-workflow。

ready + running 作为 queue proxy：可接受 adaptation

[INFERENCE — ACCEPT WITH DISCLOSURE]

原论文有 replica queue，你们没有。

所以：

𝐷
𝑚
=
𝑁
𝑚
ready
+
𝑁
𝑚
running
D
m
	​

=N
m
ready
	​

+N
m
running
	​


然后：

纯文本
D_m == 0 -> zero visible demand

是一个合理、保守的 proxy。

而且你们没用：

realized future demand；
predicted future arrivals；
residency 当 busy；
future suffix。

这一点信息边界很好。

但我建议文案不要叫：

model_is_idle

更准确叫：

model_has_zero_visible_demand

因为“没有 visible demand”不严格等于“存在一个 idle model replica”。

还有一个小但重要的 fidelity nuance：

[INFERENCE — P1-ish disclosure] 原论文的 Rqueues[a.model] 隐含着这个 model 有 replica/worker；而你们如果某 model 根本没 resident，demand=0 也会把它当“idle-risk”。

最干净的 proxy其实是：

idle-risk
(
𝑚
)
=
replica-present
(
𝑚
)
∧
𝐷
𝑚
=
0.
idle-risk(m)=replica-present(m)∧D
m
	​

=0.

这里 residency 不是拿来判断“忙不忙”，而只是判断：

有没有一个 model-serving capacity 可以被“闲置”。

不过如果你们不想引入 residency coupling，也可以保留 zero-demand proxy；只是必须写：

zero-visible-demand proxy, not literal replica idleness.

我不会把这一点升级 P0。

4. role → model 多数映射是否忠实/泄漏？
Leakage：PASS

[VERIFIED]

pythia_profiler.py:163–170 明确只在：

Python
if tpl.split == "train"

的数据上累计：

纯文本
role -> model counts

然后 majority + deterministic tie-break。

tests 也已经检查：

validation 不改变 mapping；
profiler 只计 train run。

所以这里没有 future leakage。

Fidelity：可以接受，但需要一个 purity 诊断

[INFERENCE]

原 Pythia 的 future agent 不是只有 agent role，它在 regex/config 层知道：

纯文本
future agent -> model

Algorithm 3 直接访问：

R_{queues}[a.model]. \] :chatgpt-content-reference{index="4"} 你们因为 role ontology 没直接带 model，所以用： \[ \arg\max_m P(m\mid role)

是合理 adaptation。

但它是否“足够好”取决于：

𝑃
(
𝑚
\*
∣
𝑟
𝑜
𝑙
𝑒
)
P(m
\*
∣role)

有多集中。

建议 freeze artifact 增加：

纯文本
role
majority_model
majority_fraction
model_entropy
support

如果基本都是：

纯文本
0.98 / 1.00

那这个 adaptation 几乎就是 deterministic mapping。

如果存在：

纯文本
planner -> model_A 55%
planner -> model_B 45%

那 majority 就比较粗暴。

这种情况下更合理的是：

𝑆
unblock
(
𝑎
)
=
∑
𝑚
𝑃
(
𝑚
∣
𝑎
)
  
𝐼
[
𝑚
 zero-demand
]
1
𝐸
[
𝐷
𝑎
]
.
S
unblock
	​

(a)=
m
∑
	​

P(m∣a)I[m zero-demand]
E[D
a
	​

]
1
	​

.

仍然完全 train-only，而且更符合 uncertainty setting。

所以：

[VERIFIED] 无泄漏。
[UNVERIFIED] mapping fidelity 需要先看 purity。

额外注意：GPU-only future agent

当前 model_counts 对所有 role 都统计 model，而 unblock_score() 没有 lane 信息。

如果 PFA 中存在 CPU/API/control-only role：

[UNVERIFIED] 它的 model demand 永远不会出现在 GPU demand map 中，于是可能永远被视为“idle”，产生错误 boost。

所以 freeze 前建议钉死：

S_unblock 只对对应 GPU/model-serving future roles 生效。

如果 v04.1 的 PFA states 全是 GPU-serving agent，这一项自然 PASS；但最好 gate 显式证明。

5. aging 形式和 tau=30000 ms 是否可接受？
形式：可以

[INFERENCE — ACCEPT AS ADAPTATION]

论文只明确说：

effective priority = global base priority + aging factor scaled by accumulated wait time.

没有给 Algorithm 3 式的具体 aging 函数。
arXiv

所以：

𝑆
eff
=
𝑆
base
+
𝜆
𝑊
𝜏
S
eff
	​

=S
base
	​

+λ
τ
W
	​


有几个优点：

无量纲；
wait=0 退化回 base；
waiting 单调增加 priority；
能防 starvation。

这是很合理的 implementation choice。

但不要写成：

exact Pythia aging formula.

应该写：

dimensionless linear adaptation of Pythia's wait-time aging.

tau=30s：本身可以，但 provenance 必须冻结清楚

[INFERENCE]

真正的问题不是 30s 是不是“神奇正确”。

论文没给这个值，就不存在唯一正确值。

问题是你们代码 pythia_methods.py:42–44 现在还写：

pre-registered defaults pending a single dev-grid selection, then frozen

而 freeze manifest 又写：

frozen constants.

这是互相矛盾的。

所以现在必须二选一：

方案 A，我更推荐：

直接声明：

纯文本
lambda = 1
tau = 30s
omega1 = 1
omega2 = ...

是 pre-registered, data-independent adaptation constants。

从现在开始不调。

方案 B：

明确还没 freeze，在 development-only set 做一次预注册 grid：

纯文本
tau ∈ {10s,30s,60s}
omega2 ∈ {...}

然后 freeze。

但你们已经看过三个 confirm 集的 2×2 结果。

[UNVERIFIED] 我不知道这三个 confirm 集是否属于以后正式报告的数据。

如果属于 confirm/final，不应该再根据它们调 tau/omega2。

所以从实验洁净度看，现在更建议：

修 correctness bug，但不要根据当前 confirm performance 调 30s。

6. 上轮要求的 sentinel 齐了吗？
Verdict：没有。

你现在新增的测试很好，但主要是函数级。

已经有：

[VERIFIED] aging 随 wait 单调；
[VERIFIED] aging dimensionless；
[VERIFIED] zero-wait identity；
[VERIFIED] idle downstream → unblock 增加；
[VERIFIED] unrelated model 不影响；
[VERIFIED] PFA mutation 改变结果；
[VERIFIED] role→model train-only；
[VERIFIED] END/start-role exclusion。

但我上一轮最重要的两个 non-leak sentinel 仍然没出现。

至少补下面 4 个。

Identical-prefix / different-realized-future live-action invariant。

构造 Template A/B：

纯文本
当前 ready prefix 相同
current role/model 相同

未来真实 suffix:
A -> X
B -> Y

PFA 相同。

在 branch 尚未 resolve 时：

纯文本
base priority 相同
effective priority 相同
choose_action winner 相同

必须全部相同。

这是最强的 leakage gate。

Future-template mutation invariant。

只改未执行节点的：

纯文本
role
model
runtime
length
successor

live Pythia decision 不变。

Cross-workflow global-demand gate。

Workflow A 当前候选未来预测会用 M。

加入 workflow B：

纯文本
B has visible ready/running demand on M

A 的 S_unblock 必须下降。

若 B demand 是 unrelated M2，不变。

这样才能钉死“全局 cluster queue proxy”。

Current→future distance direction gate。

这项现在最重要：

纯文本
A -> B -> C

B/C model 都 zero-demand。

应验证：

𝑐
𝑜
𝑛
𝑡
𝑟
𝑖
𝑏
𝑢
𝑡
𝑖
𝑜
𝑛
(
𝐵
)
>
𝑐
𝑜
𝑛
𝑡
𝑟
𝑖
𝑏
𝑢
𝑡
𝑖
𝑜
𝑛
(
𝐶
)
.
contribution(B)>contribution(C).

具体应类似：

纯文本
D(A,B)=1
D(A,C)=2

而不能用：

纯文本
remaining(B)
remaining(C)

这个 test 当前代码会失败，所以它正好是正确的 regression test。

另外我建议顺手加：

raw-sum cardinality gate

两个同距离 idle future agents：

𝑆
unblock
(
2
 
𝑎
𝑔
𝑒
𝑛
𝑡
𝑠
)
>
𝑆
unblock
(
1
 
𝑎
𝑔
𝑒
𝑛
𝑡
)
.
S
unblock
	​

(2 agents)>S
unblock
	​

(1 agent).

这能防止以后又变回 candidate-wise mean。

7. 还有哪些 P0/P1？

我的审计分类是：

等级	问题	Verdict
P0	S_unblock 用 remaining-to-terminal(future) 代替 distance(current,future)	必须修
P1	S_unblock 用 candidate-wise mean，而论文是 raw sum	建议改 raw sum
P1	freeze manifest 仍写 S_unblock omitted / omega2=0 / aging not migrated	必须清理
P1	code 写“pending dev-grid”，manifest 又写 frozen constants	freeze 前必须统一
P1	缺 identical-prefix/different-future live-action sentinel	必补
P1/UNVERIFIED	role→model majority 没有 purity/non-GPU-role 审计	补诊断即可
Adaptation	zero visible demand ≠ literal replica idleness	可接受，正确命名/披露

尤其 manifest 这里是明确错误。

当前 pythia_baseline_freeze_v1.json 同时包含：

纯文本
adapted:
S_unblock approximated...
aging expressed...

但后面又写：

纯文本
deviation:
S_unblock not migrated

以及：

纯文本
omissions:
S_unblock omitted, omega2=0
aging NOT migrated

这些旧字段必须删除/更新。

否则最终 artifact 本身就是 contradictory provenance。

8. 现在可以 freeze 吗？
Verdict：[VERIFIED] 不能。

不是大工程。

我要求的 minimum fix 很明确：

把 S_unblock 改成真正的：
1
/
𝐸
[
𝐷
(
current
,
𝑎
)
]
1/E[D(current,a)]

而不是 S_completion(V(a))。

改回 raw sum：
∑
𝑎
1
/
𝐸
[
𝐷
𝑎
]
a
∑
	​

1/E[D
a
	​

]

如果需要 scale，用固定 
𝜔
2
ω
2
	​

，不要 candidate-wise mean。

zero-visible-demand proxy 可以保留，全局跨 workflow；把它明确标为 queue-depth approximation。

加上述至少四个 sentinel，尤其：

current→future distance；
identical-prefix/different-future live action。

清干净 freeze manifest 的旧 omega2=0 / omitted aging / S_unblock omitted 文案。

明确四个参数到底是：

已冻结；
还是仍待 dev-grid。

不能两者同时写。

完成后，我预期裁决会变成：

Pythia Algorithm-3-priority-adapted: ACCEPT / FREEZE

而不是“completion-only”。

最后直接回答你最关心的几个点：

[VERIFIED] global ready+running per-model demand 应跨所有 workflow 合并，这是对的；没有发现这一部分读取 future truth。

[VERIFIED] train-only role→model majority 没有 leakage；但应报告映射 purity，必要时升级成 
𝑃
(
𝑚
𝑜
𝑑
𝑒
𝑙
∣
𝑟
𝑜
𝑙
𝑒
)
P(model∣role)。

[INFERENCE] S + λ(wait/τ) 是合理的 aging adaptation，30s 可以冻结，但不能一边写“frozen”一边保留“pending dev-grid”。

[VERIFIED] 现在真正阻塞 freeze 的核心不是 aging，也不是 queue proxy，而是 S_unblock 的距离量定义错了。 Pythia 的原机制奖励“离 idle downstream model 越近”，而 e9af513 现在实际上在奖励“那个 future role 离 terminal 越近”。这两个必须分开。
arXiv
+1
