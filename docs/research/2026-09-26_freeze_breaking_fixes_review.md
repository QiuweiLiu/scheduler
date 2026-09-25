---
research_backend: "ChatGPT Web"
requested_model: "GPT-5.6 Sol"
requested_reasoning: "High"
model_verification_basis: "high_ui_mapping"
conversation_id: "6a9822da-e278-83e9-9c1a-675923acda0e"
conversation_generation: "1"
conversation_url: "https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e"
verified_before_submit: true
verified_after_response: true
timestamp: "2026-09-26T00:56+08:00"
status: "valid"
---

# Research Report

我按 2fbb44a...54a2d25 的实际 diff、c06a170、当前 freeze manifests、.project/DECISIONS.md、相关 tests，以及四篇论文重新走了一遍。

2fbb44a → 54a2d25 实际 diff

54a2d25 commit

c06a170 re-pin commit

Executive Summary

结论：NEEDS CHANGE。现在可以跑 30 集作为工程诊断 smoke，但我不批准把它视为 freeze-qualified 的正式前置 smoke，更不批准启动 300×5。

原因不是 LLMSched/Pythia 修坏了——这两项的核心 P0 基本修对了。真正的阻塞是：

P0 — Latency-Aware 的 realized-future structure leak 实际没有修掉，只是被 freeze manifest 重新解释成 logical_graph_assumed_known；这个解释与论文原文冲突。

P1 — TIE 的 load adaptation 存在上一轮被截断的条件统计问题，现代码确实仍有。

P1 — Pythia 现在 completion-distance 本身正确，但代码声称论文的 waiting-time aging“由 hard priority 承担”，实际并没有。

P1 — LLMSched whole-job expectation 已修好，但 duration interval 仍没有完全使用同一 known-present information state。

实验控制仍应补一次 formal-mode fail-closed gate；c06 的重新 pin 不是新的独立 freeze 签字。

你本地报告的 38/38、13/13、45/45、121/121 我标记为 UNVERIFIED execution——我没有运行代码；但这些新增测试及对应 sentinel 我已在公开仓库中实际检查，属于 VERIFIED test coverage exists。

一、逐项 verdict
项目	Verdict	核心结论
LLMSched	NEEDS CHANGE（P1，不是 P0）	原核心 P0 已修；conditional math 正确；interval 仍是较宽的弱条件 envelope
Pythia	ACCEPT as Pythia-completion-adapted，但有 P1 文案/aging 缺口	毫秒→步数修复正确；不能称完整 Algorithm 3
TIE	NEEDS CHANGE（P1）	核心 TIE formula 没问题；nonresident load 使用了 unconditional mean
Latency-Aware	REJECT（P0）	load 与 prefetch-order 修了，但 unresolved future graph leak 没修
Runner / freeze	NEEDS CHANGE before formal	survivor-mean P0 修了，但 formal provenance gates 还不够 fail-closed
1. LLMSched

路径：

src/tracing/analysis/llmsched_bn.py
src/tracing/analysis/workload_v02_simulator.py
tests/test_llmsched_bn_v2.py

1.1 Whole-job remaining 修复

VERIFIED — ACCEPT。

新的：

expected_job_remaining_ms(profiler, evidence, current_stage)

确实不再只走 descendants(current_stage)，而是遍历全部 stage_order 中：

未完成；

非 current stage；

的 stage。

每个 future stage 都通过：

conditional_state_probs(query_stage, current_stage, evidence)

计算。

这正好解决上一轮两个 P0：

与当前候选没有 BN descendant 路径、但实际上仍属于 job remaining 的 stage 不再被漏掉；

EXPLOIT 的 current / future 不再处于两个不同 information states。

对于总剩余工作的期望，逐 stage 相加是合法的，即使 stage 之间相关，也有：

E[
i
∑
	​

D
i
	​

∣E,X

=A]=
i
∑
	​

E[D
i
	​

∣E,X

=A]

不要求独立。

这与 LLMSched 的核心思想一致：BN 根据已观察 stage 更新未完成 stage 的 posterior，然后 exploitation 用估计的 job remaining 做 SRTF。论文明确把结构与时长不确定性建进 BN，并在执行证据到来后更新剩余工作估计。
arXiv

1.2 conditional_state_probs() 数学上对不对？

VERIFIED — 正确。

代码实际算的是：

P(Q=q∣E,C

=A)=
∑
q
′
	​

∑
c

=A
	​

P(C=c,Q=q
′
∣E)
∑
c

=A
	​

P(C=c,Q=q∣E)
	​


实现过程：

posterior_joint([condition_stage, query_stage], evidence)
→ 删除 condition_stage 的 ABSENT slice
→ 沿 condition axis 求和
→ 按剩余总 mass 归一化

这正是条件概率的定义。

而且它没有错误地删除 query stage 自身的 ABSENT 状态——这一点很重要，因为：

当前 stage 已知存在，并不意味着任意未来 stage 也必然存在。

没有发现这里的 normalization、axis 或条件化方向错误。

1.3 新发现：duration interval 仍不完全一致

VERIFIED — P1。

job_duration_interval_ms(..., known_present=...) 当前只对：

name in known_present

的那个 stage 自身调用 absorb()。

对其它 unresolved stages 仍然调用：

posterior_state_probs(profiler, name, evidence)

而不是：

P(Y∣E,X
ready
	​


=ABSENT)

所以现在存在一个很细的不同：

EXPLOIT expectation：所有 future stage 都条件在 ready stage 存在；

interval grouping：ready stage 本身知道存在，但其它 stage 没有传播这个条件。

如果 X exists 会显著改变其它 stage 是否出现，那么 interval 仍然比 scheduler 当前 information state 下的真正 interval 更宽。

它不是 leakage，也不是 optimistic correctness bug；恰恰通常是保守的，会让本来可以分开的 job interval 仍然 overlap。因此我不把它升为 P0。

另外，当前 docstring 把这个 interval 称作精确 SUPPORT 也稍强：逐 stage min/max 相加在存在相关性/互斥结构时可以组合出不可能同时出现的极值，更准确应叫：

conservative remaining-duration envelope

最小修法有两个选择：

正式把它定义为 conservative envelope，并增加测试证明它只会 over-approximate，不会漏 support；

或让所有 unresolved stage 都使用同一 known-present 条件 posterior。

我偏向第二个，和上一轮要求最一致。

2. Pythia

路径：

src/tracing/analysis/pythia_profiler.py
src/tracing/analysis/workload_v02_simulator.py
tests/test_pythia_fidelity.py
.../pythia_baseline_freeze_v1.json

2.1 毫秒 → remaining distance

VERIFIED — ACCEPT。

这是这轮修得最干净的一项。

旧版实际做的是：

V(r)=E[future milliseconds]

再在 consumer 里加入：

current runtime + load

本质上已经变成 duration-weighted SRTF。

新代码变成：

V(r)=E[# future non-END role transitions]

遇到：

<end>

不计一个 role。

consumer：

S
completion
	​

=
1+V(r)
1
	​


并彻底没有 ms term。

论文 Algorithm 3 明确写的是：

E[D
remain
	​

]=ExpectedRemainingDistance(r.regex)
S
completion
	​

=
E[D
remain
	​

]
1
	​


而且文本用“离 terminal 一步”和“十步”来解释这个量，明确是结构距离，不是 runtime。
OpenReview

2.2 1/(1+V) 会不会 off-by-one？

INFERENCE — 我接受。

因为你们把 V 明确定义成：

current role 之后还会出现多少个 non-END role。

例如：

A → B → C → END

则：

V(A)=2
V(B)=1
V(C)=0

于是：

D(A)=3
D(B)=2
D(C)=1

最终 verifier 的 remaining distance 应该是正数，否则论文的 1/D 会发散。因此这个 +1 是一个合理、内部自洽的 distance convention。

建议 freeze manifest 明确写一句：

V counts future roles after the current request; therefore D_remaining = 1 + V.

这样 reviewer 不会以为你们擅自给论文公式加了 1。

2.3 omega1=1, omega2=0 必须继续披露

VERIFIED — 必须。

Algorithm 3 不只有 completion：

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


其中 S_unblock 根据 future agent 对应模型的实时 queue idleness 加权。
OpenReview

所以当前实现忠实的是：

Pythia completion-priority component

而不是：

full Pythia scheduler。

你们现在 arm 名 pythia_completion 很好，freeze manifest 中：

S_unblock omitted
omega2 = 0

也应该永久保留。

2.4 新 P1：waiting-time aging 没有真正实现

VERIFIED — P1。

这是本轮新发现。

workload_v02_simulator.py 注释写：

paper's own wait-time aging is left to the scheduler's hard priority.

但代码里的 hard priority 是节点进入 ready queue 时固定的：

Python
运行
priority = 0 if service_class == "priority" else 1

之后 Pythia key：

hard service priority
-S_completion
ready_time
...

ready_time 只是三级 tie-break，不是 aging。

而论文明确说本地 worker 会周期性根据 accumulated waiting time 重新计算 effective priority，加一个 aging factor。
OpenReview

所以：

hard service class ≠ Pythia waiting-time aging

这不会让 distance 修复失效，但当前代码注释不正确。

处理方式二选一：

如果你们要声称 faithful Algorithm-3 scheduling adaptation：实现 within-class waiting aging；

如果只想保留 Pythia-completion-adapted：freeze manifest 新增 omission：

local worker waiting-time aging is not migrated; ready_time is only a deterministic tie-break.

我认为第二种即可，不需要为这一点重造 vLLM local batch scheduler。

3. Latency-Aware：这里仍然有 P0

路径：

src/tracing/analysis/latency_aware_predictor.py
src/tracing/analysis/latency_aware_fusion.py
src/tracing/analysis/latency_aware_lifecycle.py
src/tracing/analysis/latency_aware_scheduler.py
src/tracing/analysis/workload_v02_simulator.py

3.1 Load head

VERIFIED — FIXED。

此前 request-tiered：

(model, lane, request tier)

不符合论文的 deployment/device loading cost。

2fbb44a 后 load 独立按 deployment key 建模，这个方向正确。

没有发现新的 load/run double counting。

3.2 Prefetch 移到 dispatch 后

VERIFIED — FIXED for currently-ready work。

当前 event loop 顺序已经是：

release/ready
→ dispatch runnable GPU work
→ 找仍 idle 的 GPU
→ latency-aware prefetch

因此在当前 scheduling tick 已经 ready、可以 dispatch 的 work 不会再被 prefetch 抢 GPU。

这和论文非常一致。论文明确说 near-ready prefetch 只能利用 residual capacity，而且：

near-ready prefetch is considered only when no ready action can advance.

Anonymous View

3.3 但对“未来到达 job”不能声称不延迟

VERIFIED / INFERENCE。

假设：

t=100: 当前没有 ready work → GPU 开始 300ms prefetch
t=150: 新 job arrival

已经 commit 的 prefetch 并没有抢占机制，所以新到 job 可以等到 prefetch 结束。

这不构成论文语义违例。论文是 tick-local planning + small-prefix commitment；它保证的是：

不为了 near-ready preparation 延迟当前 selected plan 中已经 ready 的工作。

它没有承诺能预知未来 arrival。

论文最好写：

Prefetch is admitted only from residual capacity after all currently runnable work has been placed. A committed prefetch may still overlap with a later, previously unseen arrival and is not preempted.

不要写：

prefetch never delays ready work.

3.4 真正 P0：logical_graph_assumed_known 是错误的信息契约

VERIFIED — REJECT / P0。

这是我这轮最重要的发现。

.project/DECISIONS.md 和 freeze manifest 现在说：

GRAPH_VISIBILITY_CONTRACT = "logical_graph_assumed_known"

并把 full realized template 描述成论文自身的前提。

但论文原文不是这样。

论文定义：

L
t
w
	​


为时刻 t 已经 resolved 的 workflow portion，planning window 只包含 ready 和 immediate near-ready units。

原文非常明确：

A branch enters only after its control result resolves.

并解释 repair activation 在 test 选择对应分支以前仍是 unknown；near-ready 的前提是 branch、role、model、eligible devices 已经确定。
Anonymous View

所以论文假设的是：

resolved logical window known

不是：

complete realized future logical graph known。

而当前仓库：

latency_aware_fusion.py

Python
运行
maximal_fusible_chains(template)

直接遍历完整：

template.nodes
node.successors

latency_aware_lifecycle.py

Python
运行
running = job.template.by_id[node_id]
for succ_id in running.successors:
    succ = job.template.by_id[succ_id]

也直接从完整 trace-derived template 读取 running node 的 realized successor。

更严重的是 fusion：

members and all(future member state == "pending")

只检查后续 member 仍 pending，却没有检查：

它们现在是否已经 resolved / visible。

因此两个 runtime prefix 完全相同、但真实未来分支不同的 workflow，在 branch result 尚未产生前，Latency-Aware 已可能：

看到不同 successor；

选择不同 prefetch；

形成不同 fusible chain。

这就是典型的 truth-structure leakage。

论文自己反而强调 agent graph 是增量展开、unresolved branch 不能进入 planning window。
Anonymous View

所以旧审计里的这个 P0：

realized unexecuted suffix leak

没有被 54a2d25 修掉。它只是被 manifest 改成“允许”。

而这次 reinterpretation 本身是不成立的。

最小修法

不用重写整套 Latency scheduler。

需要一个：

resolved_graph_view(job, now)

或等价 visible-node contract。

然后：

maximal_fusible_chains(resolved_view)
near_ready_deployments(resolved_view)

而不能吃完整 job.template。

最重要的 regression sentinel：

Template A:
visible prefix = P
unresolved future = X

Template B:
visible prefix = P
unresolved future = Y

branch resolve 前必须：

fusion candidates A == B
prefetch candidates A == B
ranking/action A == B

branch resolve 后才允许分叉。

如果当前 simulator 很难准确表示 “resolved”，最保守的替代方案是：

禁止 fusion/prefetch 跨越尚未 ready/显式 revealed 的 node。

虽然比论文弱，但不会泄漏。

4. TIE：补全上一轮被截断的确切问题

路径：

src/tracing/analysis/tie_methods.py
tests/test_tie_fidelity.py

首先说清：

VERIFIED — TIE 的核心部分现在总体是不错的。

代码已经做到：

bank train-only；

key 只有 (model_id, lane)，没偷 sequence_index；

runtime bank 用 runtime_ms - load_ms，避免 compute+load 双计；

E[X] + β CVaR_0.9[X]；

adaptive beta；

waiting decay；

consumer 不再悄悄 fallback 到 p50/p90。

TIE 原论文核心就是用 uncertainty distribution 的 expectation + tail/CVaR 替换 SJF 的 point output length，并按 queue pressure 自适应 β；另有 waiting-time decay。
arXiv
+1

被截断的那个问题到底是什么？
TIE load adaptation 用了错误的条件分布。

VERIFIED — P1。

当前 bank：

Python
运行
if load_ms is not None:
    loads_by_key[key].append(load_ms)

包括：

0

然后：

Python
运行
load_mean_ms = mean(all loads)

当前测试甚至专门断言：

[0, 10] → load_mean = 5

这个 unconditional statistic 自身没有算错。

问题出在消费时：

Python
运行
load = 0 if resident else tie_load_estimate(...)

当 live candidate 已经知道 nonresident 时，系统已经知道：

L=load required

此时应该估计：

E[T
load
	​

∣load required]

而不是：

E[T
load
	​

]

后者把历史里“模型当时已经 resident，因此 load=0”的样本也平均进去了。

举最简单的例子：

训练：

resident observation:     load=0
nonresident observation:  load=10

当前：

E[L]=5

实时现在知道模型 nonresident。

合理 estimate：

E[L∣nonresident]=10

不是 5。

所以你们上一轮提到的“窄条件统计问题”就是：

live residency state 与历史 load estimator 的 conditioning set 不一致。

它确实很窄，因为：

TIE 本身没有错；

runtime/CVaR 没错；

只影响你们为了统一 GPU simulator 新增的 model-load adaptation。

建议修法

bank 存：

load_observation_count
load_required_count
load_occurrence_rate
load_conditional_mean_ms

如果数据语义明确：

load_ms == 0 ↔ no load/resident
load_ms > 0  ↔ actual cold load

那么：

μ
load,cold
	​

=mean(load_ms∣load_ms>0)

consumer：

resident    → 0
nonresident → load_conditional_mean_ms

不要再乘历史 load_occurrence_rate，因为 live scheduler 已经观察到当前 residency，是否需要 load 不再是不确定事件。

更好的版本是若模板存在明确 load_occurred/residency indicator，就用显式 indicator，避免把真实的 0-duration load 和 “没发生 load”混淆。

新的 sentinel 应变成：

training loads [0, 10]

occurrence_rate = 0.5
cold_load_mean = 10

resident candidate surcharge = 0
nonresident candidate surcharge = 10

而不是当前的：

unconditional mean = 5

这应该在正式实验前修，因为它可能改变 TIE 排序。

5. 有没有其它 leakage / double count / 维量纲问题？

总体结果：

VERIFIED — LLMSched/Pythia 本轮没有引入新的 truth-duration leakage 或维量纲混用。

LLMSched：

current intrinsic duration
+
conditional whole-job remaining

current 没被 remaining 重复计入，额外 load 也删除了。

Pythia：

现在 ranking scalar 是纯 dimensionless completion score，不再混：

steps + milliseconds

TIE：

compute bank 已经明确从 wall runtime 中扣了 load，再在 live nonresident 时另加 load，因此 decomposition 思路正确；问题只是 load mean 的 conditioning。

Latency：

prediction duration/memory 当前实际走 train-derived predictor，而不是直接使用 node.runtime_ms/workspace truth。我没有看到本轮新增 duration truth leak。

但有一个 P2 defense-in-depth：

FusedChain 结构体里仍保存：

summed_runtime_ms = sum(m.runtime_ms)

当前真正 scheduler scoring 没有消费它，所以目前不是 active leak。但既然这是未来 truth，而且 baseline 已经因为 structure visibility 出过问题，我建议删掉这个字段或改成明确的 audit-only field，避免以后被顺手调用。

6. Runner：旧 experiment P0 修了，但 formal gate 还不够

scripts/four_joint_baselines.py

已修

VERIFIED：

--smoke 已注册；

failed jobs 非 0 会 fail；

completed_jobs != expected jobs 会 fail；

metric 非 finite 会 fail；

projection SHA 有 hard assertion。

所以旧的：

survivor mean 把失败 job 悄悄排除

这个 P0 已关掉。

仍建议正式前补

VERIFIED — experiment-control P1；对 300×5 我会作为 blocking gate。

当前 runner 更多是“记录”：

split manifest SHA
episode file SHA
git head

而不是全部 hard-assert frozen value。

我没有看到它在正式 run 启动前硬要求：

BaselineFidelityManifest.pass == true
TopologyContractGate.pass == true
episode_file_sha == frozen_episode_sha
split_manifest_sha == frozen_split_sha
F0 base/overlay hashes == frozen hashes
freeze heads/status == approved version

尤其 git_head() 失败时还能返回：

"unknown"

正式实验不该允许。

我的建议是加：

--formal

formal mode 下这些任何一项不匹配直接退出。

30集工程 smoke 可以宽一些。

7. Freeze 清单只重新 pin、不重新签字，够不够？

VERIFIED — 不够。

c06a170 的性质我确认过：它是 provenance re-pin，没有新的 baseline semantic code。

因此：

re-pin ≠ independent review

旧 approval quote 是针对旧 implementation 的。

而且这次 freeze-breaking 改的是：

LLMSched scoring semantics；

Pythia priority semantics；

Latency action timing / information-contract claim。

都不是注释修正。

所以正确流程应该是：

freeze-breaking code
→ tests
→ independent semantic audit
→ exact-head freeze declaration
→ smoke
→ formal measurement

而不是：

old approval
→ change code
→ copy old approval onto new SHA

更重要的是，我这次没有办法给新 head 签 PASS，因为 Latency 仍有 P0。

所以当前：

freeze.status=APPROVED

最多代表仓库自己的 manifest 状态，不代表这次独立审查认可。

新发现问题优先级
P0

P0-1 — Latency-Aware unresolved future structure leakage

latency_aware_fusion.py 和 latency_aware_lifecycle.py 读取完整 realized job.template；logical_graph_assumed_known 与论文的 resolved-L
t
	​

 contract 冲突。论文明确只允许已解析 branch 进入 window。
Anonymous View

这是正式实验的硬 blocker。

P1

P1-1 — TIE nonresident load 应使用 conditional cold-load mean，而不是 resident+nonresident unconditional mean。

P1-2 — Pythia wait aging 实际未实现。
hard service priority 是固定 0/1，不是论文的 accumulated-wait aging。应该实现或明确列为 omission。
OpenReview

P1-3 — LLMSched interval 没有向其它 stage 传播 current_stage != ABSENT 条件。
当前结果是 conservative envelope，不宜称 exact posterior support。

P1-4 — formal runner 尚未把所有 freeze/hash/gate 做 hard assertion。

P2

P2-1 — Latency FusedChain.summed_runtime_ms 保留了 unused truth runtime 字段。 删除更安全。

P2-2 — formal 模式不能允许 git_head="unknown"。

原审计 P0 逐条回放
上一轮问题	本轮状态
LLMSched 只算 descendants，不是 whole-job remaining	FIXED / VERIFIED
LLMSched future posterior 未条件 ready stage exists	FIXED for expectation / VERIFIED
Pythia remaining 用 milliseconds 而不是 steps	FIXED / VERIFIED
Pythia consumer 又加 current runtime/load	FIXED / VERIFIED
Latency request-tiered load head	FIXED / VERIFIED
Latency prefetch 在 ready dispatch 前	FIXED / VERIFIED
Latency 读取 unresolved realized future graph	NOT FIXED / P0
runner survivor mean / incomplete completion	FIXED / VERIFIED
TIE narrow load adaptation	NOT FIXED / 本次已补全

所以这不是“修复失败”，而是：

大部分实现 bug 修对了，但最关键的 Latency information-state P0 被错误地重新解释成了合法前提。

进入正式测量前的最小项

我不会要求再做大规模工程，只需要先完成这几个：

Latency：加入 resolved/visible graph view；fusion + near-ready 只能看 resolved portion；加 identical-prefix/different-future sentinel。

TIE：load bank 改为 conditional cold-load estimate；加 [0,10] sentinel。

Pythia：二选一——实现 waiting aging，或在 freeze manifest 明确列为 omitted；同时删除代码中“hard priority 承担 aging”的错误说法。

LLMSched：最好让 interval 使用 same known-present conditioning；若暂不改，实现/文案改成 conservative envelope 并增加 over-approximation test。

Runner：formal mode hard assert freeze manifests/gates/episode+split+resource hashes，不允许 "unknown" HEAD。

然后重新生成 manifests，提交一个新 SHA。

我建议按那个新 SHA重新走一次 freeze 声明。

再跑 30 集 freeze-qualified smoke。

30 集无结构/失败/coverage 异常后，再启动 300×5。

因此目前的最终 gate 是：

30 集 diagnostic smoke：可以。
30 集作为正式 freeze smoke：暂不批准。
300×5：暂不批准。

其中真正必须先动代码的第一优先级只有两个：Latency P0 和 TIE conditional-load P1。其余三项基本都是把信息契约与正式运行 gate 收干净。

开始新对话
