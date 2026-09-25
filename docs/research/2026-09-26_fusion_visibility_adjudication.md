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
timestamp: "2026-09-26T01:43+08:00"
status: "valid"
---

# Research Report

结论先给：严格按 arXiv:2609.03335v1，我会撤回上一轮“fusion 必须受 near-ready 窗口约束”的判定。A 过严；但你现在描述的 B 也还差一个关键条件。最忠实的是 C：在运行时已经 resolved 的 L
t
	​

 上，按 Eq.(3) 做一对一链的最大传递融合；不要求 successor 事先 near-ready，但绝不能仅凭 realized template 的一对一拓扑把尚未 resolved 的未来节点拉进 L
t
	​

。

如果你们 v04.1 的纯串行 workflow 确实是逻辑拓扑、role/model/config 在 job 到达时就已确定，那么在这个 workload 上 C 与你说的 B 等价，B 可以接受并应恢复 fusion。如果这些未来节点只是因为离线 realized trace/template 告诉了我们它们最终出现，那么直接 B 仍然是结构泄漏。

另外我核了公开 commit 54a2d25：公开的 54a 实际并不是你这里描述的 A。latency_aware_fusion.py::maximal_fusible_chains(template) 仍直接遍历整份 template，并没有 resolved_graph_view；simulator 也是按 template 缓存 maximal chains。只有 latency_aware_lifecycle.py 的 near-ready prefetch 明确限制为 RUNNING 节点的 successor。因此如果 A 是你们之后本地尚未推送的修改，我无法从公开仓库验证那部分。54a2d25 的 fusion 实现
 54a2d25 的 lifecycle 实现

1. READY 的 u 能不能和尚未 running 的 v 融合？

VERIFIED：可以，只要 v 已经属于 resolved graph L
t
	​

，不要求 v 在融合前已经 near-ready。

论文 §3.2 给出的正式顺序是：

L
t
	​

Fuse
	​

G
t
0
	​

ready / near-ready
	​

F
t
	​

,N
t
	​

	​

W
t
	​

=G
t
0
	​

[F
t
	​

∪N
t
	​

].

原文先说 Constructor 把 resolved L
t
	​

 转成 fusion-normalized G
t
0
	​

，然后才定义 F
t
	​

,N
t
	​

 以及 planning window。§3.4 的 Eq.(3) 也明确写的是 succ_{L_t} / pred_{L_t}，并明确说 contracting maximal eligible chains 得到 G
t
0
	​

=Fuse(L
t
	​

)。
arXiv
+1

所以 §3.4 开头那句 “Given the planning window … Constructor … first contracts …” 从文字上略有歧义，但不能覆盖前面更明确的数学定义。Eq.(3) 的 fusion domain 是 resolved L
t
	​

，不是 F
t
	​

∪N
t
	​

。

这也就是说，在：

A → B → C

中，如果 A/B/C 都已经是 resolved logical activations，而且 deployment/config 相容：

Fuse(L_t):  [A,B,C] → c

随后才判断 fused unit c 是否 ready。

不是：

A ready
→ 等 A running
→ B near-ready
→ 才允许考虑 A+B

后一种就是你们的 A，它把顺序反过来了。

2. A 为什么与论文自己的 fusion 示例冲突？

VERIFIED：如果必须“前驱 running 才能 fusion successor”，论文的典型串行 same-model chain 就基本失去 fusion 机会。

论文不仅理论上定义 Eq.(3)，还直接给了 QMSum 示例：

三个连续的 Qwen3-8B aggregation nodes 被构造成同一个 chain c=(v
1
	​

,v
2
	​

,v
3
	​

)。

它强调的是“一次 grant、同一 replica lease，三个 generation 仍分别执行”。
arXiv

背景部分甚至用“三次独立 acquire vs 一个 fused three-node sequence”来说明 repeated-admission 问题，并明确说 fused chain “competes at its head”。
arXiv

这与 A 的运行逻辑不兼容：

v1 尚未运行
→ v2 不算 near-ready
→ 不能形成 v1+v2

等 v1 已经 running 后，又已经错过“chain competes at its head”的 admission 时机。

所以：

VERIFIED：A 不是 faithful reading；它把 near-ready 的 lifecycle/prefetch 可见性规则错误施加到了 fusion normalization 上。

论文没有说“串行 chain 无法 fusion”这种局限，相反，它明确把三节点顺序 chain 当作 fusion 的代表例子。

3. 那 B 是否就是正确答案？

这里需要非常精确：

B 的核心方向正确，但“只凭 Eq.(3) 一对一边做 realized-template 闭包”还不够。

论文对 L
t
	​

 有一个独立的限定：

scheduler observes the resolved portion L
t
w
	​


而且：

branch 只有在 control result resolve 后才进入。

§2.1 也明确区分了“未来真实会发生”与“当前已经知道”：repair activation 在 test 选择 branch 前仍然 unknown。
arXiv

所以不能推出：

succ(u)={v}⟹v is resolved.

这是两个不同性质：

succ={v}, pred={u}：fusion legality；

v ∈ L_t：information visibility。

举个最危险的例子：

test
 ├─ success → finish
 └─ fail    → repair

realized trace 最终选择了：

test → repair

事后 template 里 test.successors == {repair}。

如果 B 因为现在看到“一对一”就提前把 repair 拉进 closure，那么你实际偷看了 test 的未来控制结果。论文明确不允许这种情况。

所以我建议实现 C：

Python
运行
resolved = resolved_graph_view(runtime_state)

chains = maximal_fusible_chains(resolved)

其中：

resolved_graph_view

负责论文 §3.2 的 L
t
	​

；

maximal_fusible_chains

只负责 Eq.(3)。

两层不要合并成：

Python
运行
one_to_one_edge == resolved
对你们 v04.1 的具体裁决

你说 v04.1：

每节点恰好 0/1 successor，纯串行。

这本身还不能证明所有 future nodes resolved。

需要再问一个客观问题：

对一个 job，在它到达 scheduler 的时候，A→B→C 的 logical identities、role、model、serving configuration 是否由 workflow definition 确定，而不是事后从该 run 的 realized trace 得到？

如果答案是“是”：

VERIFIED/ACCEPT：你们描述的 B 在 v04.1 上可以采用。

因为：

L_t = entire deterministic serial logical graph

所以沿一对一边 closure 没有偷看未来控制结果。

如果答案是“不是，template 是 realized execution 的事后记录”：

REJECT：裸 B 仍泄漏。

必须额外有 resolved_at_t / workflow-definition visibility 标记。

因此我的正式 verdict 是：

C = faithful；在 fully pre-resolved serial workload 上，C 退化为 B。A 不 faithful。

4. sentinel 应该测什么？

你现在提出的“分叉/汇合不暴露后继”也需要稍微改一下。

Eq.(3) 说的是：

one-to-one condition keeps branches and joins at chain boundaries.

含义是：

branch/join 阻止 fusion 跨边界，不等于 branch/join 后的所有节点必须不可见。

例如一个静态 fork：

       B
      /
A ───<
      \
       C

如果这是 workflow definition 固定的并行 fork，那么 B、C 完全可以是 resolved 和 visible；只是：

succ(A) = {B,C}

因此 A 不能和任何一个 fuse。

真正应该加的 sentinel 是这三个。

Sentinel 1 — resolved-prefix noninterference，最重要。

构造两个 workflow：

T1: resolved prefix 相同 + unresolved suffix X
T2: resolved prefix 相同 + unresolved suffix Y

在 control result 尚未 resolve 时：

resolved_graph_view(T1,state)
==
resolved_graph_view(T2,state)

并且：

fusion(T1,state)
==
fusion(T2,state)

这才是真正防止 truth-structure leakage 的测试。

所以我上一轮所谓：

“两个不同 template 必须结果相同”

如果没有“相同 resolved state，只在 unresolved suffix 不同”这个限定，就是说得太宽了。任意两个不同已 resolved graphs 当然允许得到不同 fusion。

Sentinel 2 — static serial chain must fuse at head。

A → B → C

全部预先 resolved、same deployment：

tick 0:
A ready
B,C not ready

仍应：

chain = (A,B,C)

这正是防止重新退化成 A。

Sentinel 3 — Eq.(3) boundary。

已 resolved 的：

A → {B,C}

不能 A+B；

以及：

{A,B} → C

不能 A+C。

所以“分叉/汇合停止 fusion”是对的；“分叉/汇合后继必定不可见”不是论文要求。

5. 论文 workload 到底偏并行还是偏串行？

VERIFIED：论文不是一个以纯串行为主体的 workload，也不是纯并行；它刻意覆盖了两者。无法从论文数据支持“多数节点是并行”或“多数节点是串行”这种数量判断。

三个 scenario：

GSM8K：5 个 heterogeneous solvers 并行，然后 majority aggregation + refine；

MBPP：generation/test/diagnosis/review/repair/validation，包含 sequential 和 multi-parent dependencies；

QMSum：两条并行的 3-stage rolling-summary chains，之后是一条 3-stage aggregation chain。论文总结说三者覆盖 fork-join、fan-in、tool-interleaved dependencies 和 repeated-model chains。
arXiv

所以 fusion 的论文机会恰恰来自：

DAG 中的局部线性 same-deployment chain

而不是要求整个 workflow 串行。

QMSum 就是最典型：

lane1: A1 → A2 → A3
                   \
                    → G1 → G2 → G3
                   /
lane2: B1 → B2 → B3

外层有并行，局部却有非常适合 fusion 的线性 chain。

而且 §4.5 的消融说明 fusion 不是装饰机制：禁用 fusion 后，跨三种 arrival process，p95 增加 8.5–44.9%，makespan 增加 3.0–8.1%。
arXiv

因此你们 A 把纯串行 workload 的 fusion 从约 39 次打成 0，很可能不是“我们的 workload 与论文不适合”，而是 visibility gate 把 Eq.(3) 的合法机会一并杀掉了。

对 54a 冻结声明还有一个需要修正的地方

VERIFIED：54a freeze manifest 里的

graph_visibility = "logical_graph_assumed_known"

以及：

“paper's Eq (3)/(5) are defined on a KNOWN logical graph”

表述过强。

论文实际上明确说的是：

resolved portion L
t
	​

	​


而不是 entire logical graph known upfront；它还特别规定 unresolved branch 不能提前进入窗口。
arXiv

所以如果你们准备 freeze-breaking 改 B/C，这个 contract 建议同步改成类似：

GRAPH_VISIBILITY_CONTRACT =
    "runtime_resolved_logical_graph"

或更明确：

"resolved_logical_graph_only; fusion_before_ready_near_ready_window"

不要继续写 logical_graph_assumed_known。

最终裁决

A：REJECT / needs change。

原因不是性能变差，而是论文形式定义直接显示 fusion 在 near-ready classification 之前发生。要求 successor 的 predecessor 已经 running，给 Eq.(3) 增加了论文没有的限制。

裸 B：CONDITIONAL ACCEPT。

如果“传递闭包”的边和节点来自当前已经 resolved 的 logical graph，它就是正确方向；如果只是对完整 realized template 做一对一 closure，则仍可能偷看动态 branch / future activation。

C：ACCEPT，最忠实。

1. 从 runtime state 构造 L_t = currently resolved logical graph
2. 在 L_t 上应用 Eq.(3) maximal one-to-one same-deployment fusion
3. 得到 G_t^0
4. 再在 G_t^0 上定义 F_t / N_t
5. W_t = G_t^0[F_t ∪ N_t]
6. prefetch 仍只属于 N_t，并继续保持“不能延迟 ready work”

这五步与论文 §3.2–§3.4 的数学顺序是一致的。
arXiv
+1

所以，如果你现在问我一句最实用的话：

不要保留 A。把它改成 C；如果你能证明 v04.1 的完整串行 topology/deployment/config 在 job 到达时就是 logically resolved，那么 C 在你们这个 workload 上可以直接实现成你说的 B，并恢复那约 39 次 fusion。

这也意味着上一轮我把“resolved graph”误收缩成“ready + RUNNING-predecessor near-ready graph”的那部分整改建议，应该撤销。

你已达到此对话的长度上限，你可以开始新聊天以继续对话。

开始新对话
