# FAS 结果评审 + 基线代码审查（GPT，2026-09-15）

来源：ChatGPT Web 会话（输入：Phase 8/9/10 结果汇总 + 三个新基线实现代码；输出：结果评价、代码问题、C2 机制缺口）。
本文件的操作项已同步到 `PLAN.md` / `EXPERIMENT_GATE.json`。

---

## 5. assistant
先给总判断：最新结果整体更强了，但 C2“runtime tail 风险本身导致收益”目前仍有一个关键混淆没排除：q95 可能只是把 future cost 的权重整体放大了。 代码里另外有两个基线语义问题，尤其 state_aware 目前不能作为强 state-aware baseline。

一、结果评价

1）最强结果：pressure sweep。 q95 相对 E2 在 6/6 cells 显著改善，而且随负载从约 8.6s→14.0s→28.9s 单调放大。这非常像一个真实 scheduling effect，而不是单一 workload point 偶然命中。它最稳妥支持：

conservative future-cost consumption becomes increasingly valuable under contention.

但不要马上写成“tail uncertainty 在 deadline 紧张时更重要”，因为 tight/loose 几乎没差。现在证据更支持“contention-sensitive future workload penalty”，而不是“deadline-risk-specific”。

2）q95 优于 trueopt/known-DAG 是可信的，但命名必须改。

物理上完全可能，因为你们的 trueopt_h5 不是“最优策略”，而是：

oracle future information + 某个固定 point-cost scheduling heuristic。

q95 则是：

imperfect future information + 一个更保守的 risk surrogate。

更好的 surrogate 完全可能在 mean completion 上击败拿真值但目标函数较差的 heuristic。并且 q95 的 miss 还略差于 known-DAG，说明它也没有 Pareto dominate。

论文千万不要写：

prediction outperforms oracle.

应该写：

Our risk-aware policy outperforms point-cost policies even when the latter are given oracle H=5 topology/resource information, indicating that decision-rule design can dominate information accuracy.

并把 oracle 改名为：

OracleInfo-H5-Point

OracleTopo-H5-Table

真正的 oracle upper bound 应保留给“知道完整未来并按理想目标优化”的策略。

必须补一个组合臂：
oracle topology H5 + 与 q95 完全相同的 risk consumer。

但注意“真值 runtime”只有一个 realization，没有 p95。最干净的是：

oracle topology + J3 predicted per-step q95

predicted topology + J3 q95（现 q95）

这样只替换 topology truth，消费者完全一致。

如果 oracle-topology+q95 > q95，合理；
若反而更差，说明预测长度/结构本身在充当 implicit regularizer，需要进一步解释。

二、三个新 baseline 的代码审查

1. FCFS：基本正确，但名称有问题
Python
运行
(priority, ready_time, ...)

所以它不是纯 FCFS，而是Priority-FCFS。

如果 priority 是系统既定不可违反的业务优先级，并且所有策略都把它放第一位，这是公平的；论文注明：

all policies obey the same exogenous priority class.

否则审稿人会说 FCFS 被改造了。

GPU 选择最后只按 gpu.index，完全不看 residency/cache，这是合理的弱 baseline，但不能拿它证明 cache-aware 优势。

2. sjf_pred：没有代码崩溃 bug，但算法语义过弱

最大问题：

Python
运行
per_step = current_node_runtime + current_node_load
remaining = L * per_step

即假设所有未来步骤都和当前节点一样贵。

这不是严格意义的 LTR-style remaining-work prediction，而更像：

Length-aware SJF heuristic

而且每个未来步骤都加 load_p50，隐含“每一步都 cold load”，会系统性高估。

建议最低修改成：

R=current+
h=1
∑
L
	​

c
ˉ
lane
	​


其中 \bar c_lane 是 train-only lane-level average，不能使用 J 的未来 content。这样仍是简单 predicted-length baseline，但语义干净。

名称建议改为 Length-SJF，不要宣传成复现 LTR。

3. state_aware：目前有实质性排序问题

现在：

Python
运行
(priority, miss_flag, current, -urgency, ...)

因为 min：

① miss_flag=0 永远优先于 miss_flag=1。

也就是说已经处于负 slack 的 job 被排到最后。除非你明确采用“放弃不可救任务”的 admission policy，否则这很难解释。

② current 在 urgency 前。

所以只要两个 current cost 不完全相同，urgency 几乎不起作用。

当前策略实质上是：

current-cost SJF + 一个很弱的 slack tie-breaker。

不能称为 SOLA/SlackFit-style strong state-aware baseline。

最低修改建议直接使用 laxity-first：

Python
运行
score = (
    priority,
    slack,       # 越小越优先，负 slack 自动最紧急
    current,
    ready_time,
    ...
)

或者归一化：

laxity=(deadline−now−current)/window

按 laxity 升序。

若希望“不可救 job 降级”，必须把它作为明确的 drop/late-job policy 单独定义，不能悄悄藏在 baseline 中。

4. pool = fitting or candidates：全策略共有，所以不是臂间不公平，但有系统级风险

最大问题不是 fairness，而是：

Python
运行
fitting = memory_p95 <= gpu.capacity

若一个节点本身超过 GPU physical capacity，fitting 为空后又回到 candidates，理论上允许选择物理不可执行节点。

如果 simulator 后面会明确 reject/evict/spill 并保证不超容量，需要写清楚；否则应该：

Python
运行
if not fitting:
    no_dispatch

而不是 fallback。

另外这里比较的是总 capacity，不是当前 available memory；如果后续层负责 eviction/residency，这是可以的，否则也是 bug。

三、敏感性实验够不够支撑 C2？

还不够，差一个非常关键的 scale control。

目前 tail shrink：

p95→p50 越多，性能越差。

这有两种解释：

A. 真的是 tail uncertainty information 有价值；

B. q95 数值更大，因此只是把“未来链长度/未来 workload”在 score 中权重放大了。

目前你们已经知道 length-only 本身就很强，所以 B 非常 plausible。

最重要的新对照不是再做 CVaR，而是：

Scale-matched p50

令：

C=k∑p50
h
	​


其中 k 用 train/dev בלבד 固定成使其平均 future cost 与 q95 相同的常数。

比较：

raw p50

scaled-p50

p90

p95

若：

q95≫scaled-p50

才能说收益来自state-dependent tail structure，而不仅是未来权重更大。

再补一个更强的：

Tail-shuffle

保留每个 episode 的 p95-p50 gap 分布，但把这些 gap 在节点/链之间随机打乱。

如果 actual q95 明显优于 shuffled-q95：

tail magnitude 必须与具体 future state 对齐才有价值。

这会把 C2 从“经验 heuristic”真正推进成机制证据。

最后，当前“content corruption 无影响”不能作为强证据。因为 q95 consumer 几乎直接读取 resource.runtime p95，content/lane 主要只影响 load，而你们已经证明 load≈无贡献；因此翻转 lane 本来就很难改变结果。它只能表述为：

content fields have no measurable marginal effect under the q95 resource-consumption path.

不能写成：

content prediction generally has no scheduling value.

如果补完 scaled-p50 + tail-shuffle + oracle-topology/q95 三个小实验，C2 的因果解释会扎实很多，而且每个都只是新的消费者/重打包，成本很低。
