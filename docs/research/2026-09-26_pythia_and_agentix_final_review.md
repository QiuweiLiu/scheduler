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
timestamp: "2026-09-26T21:39+08:00"
status: "valid"
---

# Research Report

…
显示更多
思考了 5m 48s
ChatGPT 说：

炜，这一轮已经可以把两件事分开定性了：

A / Pythia：算法实现可以 ACCEPT；只剩一个纯 provenance 的机械问题，修完即可正式 FREEZE，不需要再改算法、不需要重跑实验。

B / Agentix-Discrete-NoPreempt：作为明确标注的非论文敏感性变体可以保留；不应该再补“整节点重算式抢占”。不过如果准备把这个 sensitivity 写进论文，我建议再修一个小的 waiting-time 语义问题，并把主 PLAS 臂的 disclosure/service accounting 补完整。

下面逐项说。

A. Pythia d01d202 终审
A1. loop / first-return：通过

[VERIFIED — CLOSED]

expected_distance_to_role() 已经删除 target == current -> 0，现在从 step=1 开始传播 probability mass；target 命中后不继续传播，所以对同 role 求的是正的 first-return time：

𝐸
[
𝑇
𝐴
+
∣
𝑇
𝐴
+
≤
𝐻
]
.
E[T
A
+
	​

∣T
A
+
	​

≤H].

reachable_future_roles() 也不再排除起点 role，因此 PFA 中的重复 role 能重新作为 future role 出现。

你新增的测试：

纯文本
A -> A / B
D+(A,A)=1

以及无返回路径时 None，与代码一致。

[INFERENCE] 对你们这种“role=PFA state”的建模，这是正确的 loop adaptation。Pythia 本来就通过重复 role / bounded repetition 表达循环，因此不应把第二次 planner 当成“距离 0 的当前 planner”。

这一项不再阻塞。

A2. future-suffix sentinel：这次真正修对了

[VERIFIED — CLOSED]

之前的 test 确实是假 sentinel；现在已经实际构造了一个 Template，然后修改未执行 n1 的：

role；
model；
runtime；

再比较 mutation 前后的 Pythia score。

更重要的是，我又顺着 live scheduler 检查了一遍实际读取路径：

纯文本
current_role()
    -> 只读取当前 ready candidate

visible_model_demand()
    -> 先判断 state in {ready,running}
    -> pending suffix 在查 node 之前就 continue

PFA / role_model_dist
    -> train-only frozen profiler

[VERIFIED] 当前 live Pythia branch 没有读取 pending/unexecuted suffix 的路径。

所以虽然这个 gate 仍然主要是 score-level，而不是完整 choose_action twin-template test，但配合实际代码路径，我认为 non-leak contract 已经闭环，不再要求你继续加测试。

A3. train-only mapping：通过

[VERIFIED — CLOSED]

现在 test 已扩展到：

纯文本
role_model
role_model_dist
role_lane

而不再只是 majority role_model。

因此我上一轮提出的这个 P1 已关闭。

A4. 常量 provenance：现在表述合理

[VERIFIED]

当前代码已经改成：

data-independent adaptation constants fixed BEFORE formal measurement

而不是没有证据支撑的 “pre-registered from the start”。

这比上一版严谨。

我仍然维持上一轮结论：

[INFERENCE] 不要再做 dev grid。

现在去调：

𝜔
2
,
𝜆
,
𝜏
ω
2
	​

,λ,τ

反而比固定 1,1,1,30s 更容易产生 baseline performance-tuning 的质疑。保持不动更干净。

A5. S_unblock decision-inactive 不阻止 freeze

[VERIFIED fact / INFERENCE interpretation]

你现在如实记录：

纯文本
score-active:
585 / 727 = 80.5%

decision-active:
0 / 614 flips

这其实比只报告“makespan 没变”更有信息量。

它意味着：

机制真的被计算、也在大量候选上非零，但是在当前 workload + 固定 
𝜔
2
=
1
ω
2
	​

=1 下不足以改变 winner ordering。

这不能叫 dead code。

更准确是：

operationally inactive on the audited sample.

[INFERENCE] 这完全不影响 fidelity freeze。

而且不要为了让它产生 action flip 去加大 omega2。那会把“忠实适配”变成“调到 baseline 更强”。

正式 300-episode 时可以顺手 shadow-report action-flip rate，但只作 telemetry，不能据此再调参数。

A6. 我发现唯一还剩的机械问题：freeze SHA 字段不一致

这个比较隐蔽。

[VERIFIED — provenance P1]

d01d202 自己当然不能把自身 SHA 写进自己的文件；所以你后面的 7926b5a 做 re-pin 是合理的。我也检查了 7926b5a：

纯文本
freeze.head =
d01d202f2625c37cd54586a67aa233ab7d6ffa81

BaselineFidelityManifest 也已经是：

纯文本
freeze_head =
d01d202f2625c37cd54586a67aa233ab7d6ffa81

这一部分正确。

但是 pythia_baseline_freeze_v1.json 中仍然保留：

纯文本
head_full_sha =
54a2d25088832049afb0cf860cdabfe5152c8c3d

也就是说现在同一个 freeze object 同时说：

纯文本
head          = d01d202...
head_full_sha = 54a2d250...

两个字段互相矛盾。

而 BaselineFidelityManifest PASS 没抓到它，说明当前 manifest gate 只检查了 freeze_head 等字段，没有做：

ℎ
𝑒
𝑎
𝑑
=
ℎ
𝑒
𝑎
𝑑
_
𝑓
𝑢
𝑙
𝑙
_
𝑠
ℎ
𝑎
=
𝑚
𝑎
𝑛
𝑖
𝑓
𝑒
𝑠
𝑡
.
𝑓
𝑟
𝑒
𝑒
𝑧
𝑒
_
ℎ
𝑒
𝑎
𝑑
.
head=head_full_sha=manifest.freeze_head.
我的处理建议

只做一个 provenance-only commit：

纯文本
head_full_sha -> d01d202f2625c37cd54586a67aa233ab7d6ffa81

然后 gate 加一个一致性检查。

不需要重新跑 Pythia 实验，不需要改变任何参数。

所以 A 的最终裁决是：

[VERIFIED/INFERENCE] PYTHIA ALGORITHM: ACCEPT
FREEZE: APPROVED subject only to correcting the stale head_full_sha; no algorithmic blocker remains.

如果把“freeze”严格理解成仓库 artifact 也必须自洽，那当前 48194e9 里的 artifact 还差这一行；修完我不会再要求算法复审。

B. Agentix-Discrete-NoPreempt
B1. 你对原 Agentix preemption 的理解基本正确，而且不应该套现有 restart-preemption
Verdict：不做“Agentix + 整节点重算式抢占”

[VERIFIED] Agentix 的 scheduler 是 decoding-level preemptive scheduler：call 获得 queue quantum，用尽 quantum 后降级；论文讨论的 preemption memory overhead 是 GPU↔CPU 的 KV-cache block swapping，不是取消整次 LLM call 后从头重算。Agentix 还专门通过每 N 个 decoding steps 才执行一次 scheduler，以及把多个 KV blocks 合并为 bulk transfer，降低 swap 频率和开销。
USENIX
+1

[VERIFIED] 论文 ablation 报告其 bulk swap kernel 最多减少约 18× swap operations、降低约 3–7× swap time；offline 4000-program 实验中，普通 MLFQ 因大量 active requests、GPU-CPU swapping 和 memory contention 最终 OOM。
USENIX
+2

有一点措辞我要稍微收紧：

“不重载模型”

[INFERENCE] 这是合理理解，因为它是在已加载模型的 LLM engine 内 preempt request，论文明确讨论的是 KV state 的 GPU/CPU swapping，而不是 model reload；但论文正文没有用一句字面上的 “we never reload the model”。

“从断点续跑”

[INFERENCE] 也基本正确——保留/换回 KV cache 就是在保存 decode state，而非重新执行全部 token；但最好在论文中表述成 “preserves request decoding state through KV-cache swapping”，比“断点续跑”更严谨。

而你们 substrate 的：

纯文本
cancel whole node
discard progress
later recompute whole node

成本模型完全不同。

所以我的建议很明确：

不要再做一个名为 Agentix 的 restart-preemption sensitivity。

如果将来你确实想研究：

node-level restart preemption 对我们 simulator 有什么作用？

可以叫：

纯文本
Restart-Preemption sensitivity

但不要冠 Agentix。

目前没必要加，会把 baseline section 搞得更乱。

B2. discrete 变体的方向和标注总体可接受
Queue discretization：通过，作为 sensitivity

[VERIFIED]

当前：

Python
edges = (0, 30s, 90s, 210s)

queue_index() 单调，低 attained service → 高优先级 queue。

这保留了 Agentix：

arriving call 按 program attained service 被放进相应 priority bin

这一思想。原论文确实将 continuous priority 分成 
𝐾
K 个 queue range，新 call 并非像传统 MLFQ 一样全部从 Q1 开始。
USENIX
+1

但我建议把：

“跨调用降级”

改称：

cross-call re-binning

因为论文原本其实同时存在两个动作：

纯文本
call arrival:
    按 program service 直接选择初始 queue

call running:
    quantum exhausted -> demote same call

你们现在实现的是前者跨后续 calls 的自然变化，并不是后者的替代品。

AGENTIX_DISCRETE_DEVIATION 已经说明 within-call quantum demotion 不存在，所以这只是措辞优化，不是代码 bug。

anti-starvation 的核心 ratio 方向正确

[VERIFIED paper]

Agentix 原式是：

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

然后 promote 到 Q1；论文还说明 promotion 后只重置 current call 的 
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

，保留 program history。
USENIX
+2

你们 sensitivity：

Python
wait / service >= beta
→ queue 0

方向正确。

但是这里发现一个小的 implementation semantic gap。

当前 job.queue_ms 不是纯 Agentix LLM-engine wait

[VERIFIED code]

simulator 中：

Python
job.queue_ms += ...

既在 GPU node dispatch 时更新，也在 non-GPU node start 时更新。

而 discrete mode 使用：

Python
wait_ms =
    job.queue_ms
    + current_ready_wait

因此它可能包括：

CPU/API/control node 的历史等待。

Agentix 的 
𝑊
𝑝
W
p
	​

 定义则是 program 在 LLM engine scheduler queue 的 waiting time。论文 process table 明确把 waiting time 定义为 LLM engine scheduler queue 时间。
USENIX

所以：

[VERIFIED code + INFERENCE severity] 作为 non-paper sensitivity，这不是致命 bug；但如果你想把它称为“Agentix program-level anti-starvation adaptation”，最好改成 GPU/LLM-serving queue wait only。

最干净的做法是 sensitivity 自己维护：

纯文本
agentix_gpu_wait_ms[job]

只在 GPU node 获得 dispatch 时累计：

𝑛
𝑜
𝑤
−
𝑟
𝑒
𝑎
𝑑
𝑦
_
𝑠
𝑖
𝑛
𝑐
𝑒
.
now−ready_since.

不用改公共 substrate。

如果不想再改代码，也至少在 deviation 里加：

anti-starvation uses the simulator's cumulative job wait proxy, which can include non-LLM nodes.

我更偏向前者，因为改动很小。

B2. queue edges / beta 常量是否可辩护？
作为 sensitivity：可以，但不要叫论文参数

[VERIFIED/INFERENCE]

我核了正式 NSDI 论文：论文定义了 symbolic 
𝐾
K、queue intervals 和 
𝛽
β，并说明改变 
𝛽
β 会产生 fairness / mean-response-time tradeoff；我没有在正文里找到你们 (30s,90s,210s) 或 β=1 这样的官方数值。
USENIX
+1

因此：

纯文本
K=4
edges=(0,30k,90k,210k)
beta=1

只能叫：

fixed sensitivity parameters

而不能叫：

Agentix official configuration.

如果你有真正早于结果的 registration artifact，“pre-registered”可以保留；否则和 Pythia 一样，我建议改成：

data-independent sensitivity constants fixed before measurement.

[INFERENCE] 不需要为了“优化 sensitivity”去 grid search。

因为它本来就不是 primary competitor。

建议正式报告时再补两个 cheap telemetry：

纯文本
queue occupancy Q1/Q2/Q3/Q4
anti-starvation promotion rate

能证明四档不是全都退化到一个 queue。

B3. Agentix 主 PLAS 臂还缺什么 disclosure？

这里有两个层次。

第一层：纯 disclosure，应该明确写上

[VERIFIED] 当前 AGENTIX_DEVIATION 仍不够完整。

现在只明确说：

batching 未复现；
cross-engine routing 未复现；
KV locality 未复现。

还应该加：

formal main arm is continuous PLAS and non-preemptive；
K queues / decoding quantum / in-call demotion / anti-starvation are not part of the formal arm；
hard benchmark service class precedes PLAS；
placement / memory admission / residency use the unified substrate；
当前 candidate feasibility 会使用 shared current-node resource estimates，所以现有那句：

“never a predicted value”

仍然太绝对。

应该改成：

no prediction of future workflow structure or remaining work; current ready-node resource estimates are inherited from the common substrate.

第二层：有一个不只是 disclosure 的 service-accounting 问题

这个我认为在你正式 freeze Agentix 主臂以前应该解决。

[VERIFIED code] 现在 PLAS：

Python
sum(
    intrinsic_runtime_of(template)(node_id)
    for node_id in job.completed
)

即：

所有 completed nodes 的 runtime_ms 都计入 attained service。

但是 Agentix 原论文 PLAS 定义的是：

completed LLM calls 在 LLM engine model executor 上的 cumulative execution time。
USENIX

而你们 simulator 明确存在 non-GPU nodes；同时：

纯文本
job.completed

并没有过滤 lane / LLM-serving node。

另外 runtime_ms 在你们 substrate 中也不一定等同纯 model-executor compute，GPU 路径本身还区分 compute_ms 和 load/lifecycle overhead。

所以：

[VERIFIED code / INFERENCE fidelity] Agentix formal PLAS 目前存在“service semantics 太宽”的风险。

它不是 future leak，但可能把：

纯文本
CPU/control/API duration
model lifecycle/load

算成 Agentix attained model service。

我建议在正式 Agentix freeze 前定义专门的 observation：

𝑆
𝑝
=
∑
completed LLM/GPU calls
observed model-executor service
.
S
p
	​

=
completed LLM/GPU calls
∑
	​

observed model-executor service.

至少应：

只计 GPU/LLM-serving nodes；
排除 queue wait；
排除 model load / eviction；
只在 call 完成后写入 observation store。

这会比直接读 template runtime_ms 更符合 Agentix 原文，也让 non-clairvoyance structural fail-closed。

这不是本轮 discrete sensitivity 必须修才能保留的问题，但它是主 PLAS 正式 baseline 下一次 freeze 前我会要求关闭的问题。

最后的两个 verdict

A：

[VERIFIED/INFERENCE] Pythia：算法 ACCEPT。没有剩余算法/泄漏 blocker。

正式 repository FREEZE 只差一个机械 provenance fix：head_full_sha 仍是旧的 54a2d25...，应改为 d01d202f...，并让 gate 检查 head == head_full_sha == freeze_head。不需要重跑任何实验。

B：

[VERIFIED/INFERENCE] Agentix-Discrete-NoPreempt：ACCEPT AS SENSITIVITY。

不做 restart-preemption Agentix arm；你对 KV-based preemption 与你们 restart-preemption 不同的判断是对的。建议把“跨调用降级”改叫“cross-call re-binning”，并最好把 anti-starvation 的 wait accumulator 改成 GPU/LLM-only wait。参数作为固定 sensitivity constants 可接受，不要调参追性能。

而Agentix formal PLAS 主臂我暂时还不会签完整 freeze，主要不是因为 discrete variant，而是它仍需把“all completed node runtime”收紧成“completed LLM/model-executor service”，同时补齐上述 disclosure。这个修复比实现 KV preemption 小得多，也更值得优先做。
