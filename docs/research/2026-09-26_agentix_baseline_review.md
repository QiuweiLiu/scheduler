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
timestamp: "2026-09-26T10:55+08:00"
status: "valid"
---

# Research Report

…
显示更多
思考了 5m 32s
ChatGPT 说：

炜，我按 c06a170 → 122e184 的实际 commit diff、agentix_methods.py、workload_v02_simulator.py、test_agentix_fidelity.py，以及 USENIX 的 NSDI 2026 正式版 Agentix 原文重新核了一遍。原论文部分我以 USENIX 正式论文为主，而不是依赖之前会话里的转述。你要求不跑代码，所以你报告的 10/10、372 tests 和 16/16 我记为 UNVERIFIED execution；但测试代码和实现本身我已经核过。

先给总裁决：

NEEDS CHANGE，但不是因为 future leakage。

[VERIFIED] 当前默认 PLAS 路径在“未来工作流信息”意义上确实是 non-clairvoyant，没有发现读取未执行 suffix、future artifact 或预测未来 topology 的路径。

但是，当前实现更准确的名字是：

Agentix-PLAS-priority-adapted

而不是“完整 Agentix scheduler adaptation”。

主要有三个需要在正式 freeze 前处理的问题：ATLAS 的“等价重算”说法不成立、Agentix 核心的 preemptive multi-level scheduler/anti-starvation 没迁移、attained-service 的时间来源目前也没有严格对应论文的 model-executor service。

这三个里，前两个是我认为最重要的。

1. PLAS / ATLAS 是否忠实？ATLAS 重算 critical path 是否等价？
PLAS：基本正确

[VERIFIED — ACCEPT] agentix_methods.py:46–54

Python
运行
sum(duration_of(node_id) for node_id in job.completed)

与 Agentix Eq. (1) 的核心定义是一致的：单线程 program 的 priority 是同一 program 此前完成 LLM calls 的累计执行时间。Agentix 明确是 non-clairvoyant，PLAS 基于 prior completed calls，而不是 total/remaining runtime。
USENIX
+1

因此：

completed smaller → priority higher

这个方向完全对。

不过，service-time 的具体映射还有一个问题，我后面单独讲。

ATLAS：你现在的“重算 completed DAG critical path 等价于论文 scalar”这一说法不成立

这里我要修正。

agentix_methods.py:57–76 当前是：

Python
运行
for node_id in completed:
    best = max(path[pred])
    path[node] = best + duration(node)


return max(path.values())

也就是根据已经完成的真实 dependency graph精确重算：

max
⁡
completed paths
∑
𝑡
𝑖
completed paths
max
	​

∑t
i
	​


这在数学意义上是“已完成图上的 exact critical path”。

但 Agentix 实际实现不是这么做的。

[VERIFIED] 原论文 Algorithm 1 是：每个新来的 call 读取当前 program scalar：

𝑐
.
𝑠
𝑒
𝑟
𝑣
𝑖
𝑐
𝑒
=
𝑝
𝑡
[
𝑐
.
𝑝
𝑖
𝑑
]
.
𝑠
𝑒
𝑟
𝑣
𝑖
𝑐
𝑒
c.service=pt[c.pid].service

完成以后更新：

𝑝
𝑡
.
𝑠
𝑒
𝑟
𝑣
𝑖
𝑐
𝑒
=
max
⁡
(
𝑝
𝑡
.
𝑠
𝑒
𝑟
𝑣
𝑖
𝑐
𝑒
,
  
𝑐
.
𝑠
𝑒
𝑟
𝑣
𝑖
𝑐
𝑒
+
𝑐
.
𝑚
𝑜
𝑑
𝑒
𝑙
_
𝑡
𝑖
𝑚
𝑒
)
pt.service=max(pt.service,c.service+c.model_time)

论文特意强调它不需要 tracking dependencies between LLM calls；每个 active call 在 arrival 时继承当前 program scalar。
USENIX
+1

所以两者在一般动态 DAG 上并不严格等价。

一个简单反例：

纯文本
        b(100)
       /
a(1)
       \
        c(1) -> e(1)

假设 b 先结束。

Agentix scalar：

纯文本
a complete          scalar = 1
b,c arrive          both inherit 1

b complete          scalar = 101
c complete          scalar remains 101

e arrives           e inherits 101
e complete          scalar = 102

而真实 completed-DAG longest path：

纯文本
a-b = 101
a-c-e = 3
max = 101

你的算法得到 101，Agentix scalar 得到 102。

所以：

[VERIFIED] agentix_methods.py:60–62 的 “Recomputing it from the completed set is equivalent” 应删除。

它现在实现的是一个更“图感知”的 exact-completed-critical-path variant，而不是论文那个 dependency-free ATLAS process-table mechanism。

不过对你当前实验有一个很重要的缓解：

[VERIFIED] v04.1 是串行链，因此 PLAS = exact completed critical path = Agentix scalar ATLAS。

所以这个问题不会污染当前默认 PLAS 正式结果。

我的建议是：

当前正式 baseline 固定 mode="plas"；不要把现有 atlas 宣称为 faithful Agentix ATLAS。

如果以后真的要支持并行 Agent workload，再把 ATLAS 改成论文真正的 inherited scalar + completion update。

2. (hard_priority, attained_service, ready_time, …) 是否忠实？缺没缺 preemption？

这里结论比较明确。

Attained service 排序本身忠实

[VERIFIED] workload_v02_simulator.py:3777–3793

核心：

Python
运行
return (
    float(item[0]),
    attained,
    float(item[1]),
    ...
)

其中 attained 越小越先执行。

这捕获了 PLAS 最核心的 LAS 思想。

但 hard_priority 放在 attained 前面不是 Agentix 原文

[VERIFIED — DEVIATION]

Agentix 的 program scheduling priority 来自 attained service → discretized priority queue；论文没有你们这个：

纯文本
service_class=priority
    永远压过 normal

的硬二元 class。

你们在 release_ready()：

workload_v02_simulator.py:4721–4730

先生成：

Python
运行
priority = 0 if service_class == "priority" else 1

然后 Agentix 把它放在 PLAS 前。

所以当前准确描述应该是：

hard service class first; Agentix PLAS orders programs within a class

如果这个 hard class 是你们统一 simulator 的系统约束，我认为可以保留，但必须写进 adaptation disclosure。

否则 reviewer 会认为 Agentix 排序被你们改了。

更大的问题：Agentix 的 preemptive scheduler 没有实现

这是当前最大 fidelity gap。

[VERIFIED] Agentix 不是单纯 min(ready_calls, attained_service)。

论文完整 Algorithm 1 包括：

continuous PLAS/ATLAS service → K 个离散 priority queues；
每个 queue 有 time quantum；
quantum 用完 → demote；
queue 内 FCFS；
不同 queue 形成 preemptive scheduling；
program-level anti-starvation：
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

时 promote 到最高 queue。
USENIX
+1

论文明确把 “dynamically preempts calls” 写成 program-aware scheduler 的组成部分。
USENIX

而你当前：

workload_v02_simulator.py:4833–4839

Python
运行
if policy not in {"myopic_preempt", "batch_myopic_preempt"}:
    return False

所以 agentix 永远不会走 preemption。

同时没有：

K-level queue discretization；
quantum；
demotion；
program-level wait/service anti-starvation。

因此：

[VERIFIED] 当前不是完整 Agentix scheduler 的 faithful adaptation，而是 Agentix PLAS priority component 的 non-preemptive adaptation。

这个区别不能只算“engine batching 没迁移”；它是 §4.2 Scheduler 本身。

如果你们统一 simulator 明确规定“node-atomic、所有 paper baseline 都不能 call-level preempt”，我可以接受不实现，但论文里必须明写。

3. non-clairvoyance 是否闭环？有没有泄漏？

这里反而整体很好。

Future-workflow leakage：没有发现

[VERIFIED — PASS]

Agentix branch 3754–3793：

不访问 future_artifacts；
不调用 future_truth_cost；
不访问 successor suffix；
不访问 predicted topology；
attained service 只从 job.completed 出发。

critical_path_service_ms() 虽然访问：

Python
运行
template.by_id[node_id]
node.predecessors

但循环集合只有 job.completed。

因此它读取的 dependency 都属于已经发生过的 completed portion。

这是 causal 的。

Agentix 原论文也明确：

initial execution DAG unknown，系统随着运行逐步构建 IR；

不知道 program arrivals、future structure、workload distribution。
USENIX

所以这一条我给你：

[VERIFIED] future-structure non-clairvoyance PASS。

但是“never a predicted value”这个注释是错的

你现在 agentix_methods.py:25–28 和 simulator 3763–3765 都说：

never a predicted value

这个表述太强。

因为在进入 Agentix branch 以前：

workload_v02_simulator.py:2941–2964

所有 policy 都经过：

Python
运行
estimate_row = estimate(node, train_stats)


predicted_fit =
    estimate_row["memory_p95_mb"] <= gpu.capacity_mb

然后：

Python
运行
pool = fitting or candidates

因此 Agentix 的当前 ready node GPU admissibility确实受到 train-derived memory prediction 影响。

这是完全合理的——当前 call 已经存在，你需要判断它能不能放进 GPU。

但准确说法应该是：

Agentix uses no prediction of future program structure or remaining work; current ready-call resource estimates are inherited from the unified GPU substrate.

而不是：

never uses prediction.

这是 disclosure/gate 需要修的地方，不是 leakage。

还有一个“形式安全性”问题

[VERIFIED] intrinsic_runtime_of()：

agentix_methods.py:95–107

直接：

Python
运行
return template.by_id[node_id].runtime_ms

而 runtime_ms 是模板中从一开始就存在的 execution-truth 字段。

现在为什么没有泄漏？

因为调用它的上游只遍历：

Python
运行
job.completed

所以逻辑上 causal。

但这不是一个强的 structural guarantee。

你们 simulator 已经有一个更好的东西：

workload_v02_simulator.py:4821–4828

在真正 node_finish 后写：

Python
运行
job.observed_intrinsic_ms[member_id]

因此我建议 Agentix 也改成只读 observation store，而不是：

Python
运行
template.runtime_ms

这样 future runtime 在物理接口层就不可访问。

也就是：

当前没有实际 leakage，但接口还可以 fail-closed 得更彻底。

4. AGENTIX_DEVIATION 披露够不够？

[VERIFIED — 不够。

现在只披露了：

batching 未迁移；
cross-engine routing 未迁移；
KV locality 未迁移。

这些都对。

但至少还应该补四件事：

必须披露	原因
preemptive multi-level scheduler / quantum / demotion 未迁移	这是 Agentix §4.2 核心 scheduler，不只是 serving infrastructure
program-level anti-starvation 未迁移	Algorithm 1 明确组成部分
hard service class precedes PLAS	原 Agentix 没有这个 lexicographic precedence
current ready-call admission/placement 使用统一 simulator 的 train-derived resource estimates	所以不能写 “never uses predicted values”

如果保留当前 ATLAS，还要再写：

ATLAS option recomputes the longest path over completed nodes rather than reproducing Agentix's inherited process-table scalar.

但我更建议修掉/降级这个 ATLAS option，不要把这个 deviation 带进论文。

因此你当前这一句：

Only the priority rule is ported

其实方向是对的。

问题是后面列 omission 时没有把最重要的 scheduler omission 写出来。

5. 当前 workload 串行，应该默认 ATLAS 吗？
不应该。继续默认 PLAS。

[VERIFIED] 原文明确把 PLAS 定义给 single-threaded programs，把 ATLAS 定义给 multi-threaded dynamic DAG。 
USENIX

而你们当前 v04.1 是串行 control chain。

因此：

纯文本
default = PLAS

是论文最直接的对应关系。

现在 PLAS 和 ATLAS 数字相同，不代表应该默认更复杂的 ATLAS。

恰恰相反：

PLAS 不需要 graph；
PLAS 完全符合当前 workload；
PLAS 更容易证明 non-clairvoyant；
当前 ATLAS 实现还与论文 scalar 有偏差。

所以我的建议是：

正式实验只跑 agentix/plas。ATLAS 保留 unit-test / future extension，不进入当前 formal arm。

论文里一句：

Since all target workflows are serial at the scheduler-visible execution layer, we use Agentix's PLAS policy; ATLAS is designed for multi-threaded dynamic DAGs and is equivalent to PLAS on this workload.

这就很干净。

6. 还应该补哪些 gate / disclosure？

你现在 10 项 gate 对“未来 runtime 不影响 score”已经很好，但还不足以支持：

Agentix is structurally non-clairvoyant and faithfully adapted.

我建议正式 freeze 前至少补下面这 6 个 sentinel：

Identical-prefix / different-future action invariance。 两个 job 当前 completed + ready prefix 完全一样，但 future 节点 identity / runtime / length / model 全部不同；choose_action("agentix") 的当前选择必须完全一致。你现在只检查 score，最好把整个 live action 钉住。

No future artifacts dependency。 同一 decision 输入两个完全不同的 future_artifacts，Agentix action 必须相同。这能证明 Agentix branch 没被以后重构时不小心接到 predictor。

Observed-service-only gate。 completed node 已记录 observed service 后，再修改 template.runtime_ms，Agentix priority 不得改变。这个测试按你当前代码会失败，正好促使你把 service 来源从 template truth 改成 completed observation。

Current-resource prediction scope gate。 改未来节点的 memory/model 不能影响 action；改当前 ready 节点合法的 memory estimate 可以改变 GPU feasibility。这样把“允许的 current prediction”和“禁止的 future prediction”边界写死。

CPU/non-engine node exclusion gate。 Agentix 原文统计的是 completed LLM calls 在 model executor 上的 service。你现在 completed_service_ms() 对 job.completed 全加，没有 lane/type 过滤。如果 workload 有 CPU/control node，它们也会被计入。这一点目前对真实 v04.1 的影响我标 UNVERIFIED，但 gate 应明确你们的 adaptation ontology究竟哪些节点算 Agentix service。

PLAS ordering gate。 两个同 hard-class programs：A attained=100、B attained=10，即使 A 当前 ready 更早，必须选 B。再单独一个 test 证明 hard service class 可以覆盖 PLAS，并把这个行为登记为 adaptation。

此外，如果真要声称 ATLAS faithful，就必须再增加一个 scalar-inheritance sentinel，而不是当前 “parallel DAG exact CP = 40” 这个测试。现在 tests/test_agentix_fidelity.py:111–123 实际测试的是你自己的 exact-DAG algorithm，不是 Agentix Algorithm 1 的 inherited scalar。

还有一个我认为值得在 freeze 前修的 service-time 问题

这个不在你六问里，但我审核代码时看到了。

[VERIFIED code + INFERENCE mapping] intrinsic_runtime_of() 当前累计的是：

Python
运行
node.runtime_ms

但你们 simulator 自己定义：

Python
运行
compute_ms = runtime_ms - load_ms

实际 dispatch 又是：

Python
运行
duration = node.compute_ms + eviction_cost + live_load

见 simulator 5142–5148。

也就是说，模板里的 runtime_ms 并不等同于“本次模拟中实际占用 model executor 的 service”。

而 Agentix 原论文定义的是：

cumulative execution time of completed calls on the LLM engine's model executor。
USENIX
+1

所以如果一个 trace 中原来的：

纯文本
runtime = 100
load = 30
compute = 70

但当前 simulator 中模型已 resident：

纯文本
实际 GPU inference compute = 70
live load = 0

你现在 PLAS 仍然加：

纯文本
100

这会把原 trace 的 cold-load 历史算进 Agentix service。

我不认为这是最严重的 future leak——它不是 leak——但它是service semantics fidelity 问题。

最干净的定义应该冻结成：

Agentix service = completed node's observed model-execution service, excluding queue wait and model lifecycle overhead.

然后 finish 时把这个量写进 observation store。

这一点比单纯把 runtime_ms 换成 observed_intrinsic_ms 更彻底，因为后者看起来仍然可能包含原 trace load。

最终 freeze verdict

我现在不会给 122e184 的 Agentix baseline 签 “faithful/frozen”。

我会给：

NON-CLAIRVOYANCE: PASS
PLAS PRIORITY FIDELITY: PASS
FULL AGENTIX FIDELITY: NEEDS CHANGE

原因不是未来泄漏，而是 fidelity scope。

最小修改路线其实不大：正式 arm 固定 PLAS；删掉“ATLAS 重算等价”的声明；把 AGENTIX_DEVIATION 补上 non-preemptive / no MLFQ / no anti-starvation / hard-class-first；明确 current resource prediction 合法；把 PLAS attained service 改成 completed-observed model service；再补 identical-prefix action sentinel。

如果你们坚持“我们只迁移 priority rule，不迁移 call-level serving scheduler”，那甚至不必强行实现 Agentix 的 MLFQ/preemption。但论文和 manifest 要始终称它：

Agentix-PLAS-priority-adapted (non-preemptive)

而不要写成“full Agentix-adapted system”。

这样我认为审稿上是可辩护的。相反，为了追求“更像论文”硬把现有 simulator 的 整节点重算式 preemption 套给 Agentix，我反而不推荐——Agentix 的 preemption 是 LLM decoding/KV-aware 的时间片机制，你们现成的 recompute preemption 不是同一个东西，可能制造新的失真。
USENIX
+1

来源选择上，这一轮 Agentix 机制全部以 USENIX NSDI 2026 正式论文/官方 prepublication 为依据；代码结论来自 GitHub connector 直接读取 122e1847123e57085ab76fa33b8e2df0e9c74b8c，不是依赖仓库里的研究笔记或之前会话转述。
USENIX
+1
