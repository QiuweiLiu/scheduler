# GPT 规划：决策级可观测性 + 属性分布×统计量混合（提交 eb88a7b 之后）

- 日期：2026-09-20
- 会话：https://chatgpt.com/g/g-p-6a45fd9c56e48191a5e1a006582fcdef-diao-du/c/6aae601b-a35c-83ea-b604-04a54934d43a
- 请求：`.scratch/gpt_plan_request.md`
- 项目负责人硬前置：**先能看见每种方式的分数 / 排序 / 耗时**，再比较
- 项目负责人澄清：**"所需时间" = 被调度任务的实际执行耗时**（判断谁估得准）

## 实施顺序（GPT 正式定版）

```
修 P0 → decision tracing → shadow scoring regression → AttrMix-Atom → AttrMix-Hist16
      → 300 集单轨迹机制实验 → 人工/统计审查 → 再冻结 full-unfreeze 目标
```

## 它认为的最大风险（不是代码）

> 属性不确定性**根本不是 runtime uncertainty 的主要来源**。
> 已有证据：stats 只有 78 个 coarse groups；即使属性完全正确，组内 estimate/truth 仍在 0.5×~2.4×；
> summarize 单一动作内部差 64 倍；还有 legacy load double-count 与 eviction omission。
>
> 所以很可能：AttrMix-Atom 相比 Argmax 几乎没变化；AttrMix-Hist16 虽产生更宽的分布，
> 但主要在表达组内 aleatoric noise，而不是属性概率带来的信息。
>
> **这不是坏结果** —— 它能把问题讲清楚：
> `runtime uncertainty = 属性头可改善的 attribute uncertainty + 属性头解释不了的 within-group execution uncertainty + load/eviction/residency system-state cost`
> 而我们现在终于能用**同一个 decision trace** 量化这三层到底哪一层真正改变 scheduler ordering。

---

==========================================================================================
这次我建议把目标拆成两个完全不同的阶段：

先做“可观测性 + 机制实验”，确保同一个决策态里，所有方法的 candidate score、rank、选择和计算耗时都能看见；然后再做**“属性分布 + 统计量混合”**。这两步都不需要训练。只有看清机制之后，再决定是否把哪种 consumer 带进 full-unfreeze。

项目负责人要求“先能看见，再比较”，我认为这是正确顺序。

A. 决策级可观测性：实现规格

这里我建议不要继续往普通 event log 里塞字段，而是增加一个独立的：

decision_trace_v1.jsonl.gz

一行 = 一个真实 scheduler decision state。

这样一个决策里所有 candidate 和所有方法天然绑定在一起，不会出现后处理时匹配错候选的问题。

A1. 一条 decision record

建议 schema：

JSON
{
  "schema_version": "scheduler-decision-trace-v1",

  "experiment_id": "...",
  "episode_id": "...",
  "decision_id": 137,
  "time_ms": 28341.2,

  "trajectory_policy_id": "A0",
  "trajectory_predictor_artifact_id": "...",

  "pool_build_wall_ns": 18200,

  "candidate_count": 7,
  "competitive_priority": 0.0,
  "competitive_candidate_count": 4,

  "state_hash": "...",
  "candidate_set_hash": "...",

  "actual_choice": {
    "candidate_id": "...",
    "job_instance_id": "...",
    "node_id": "...",
    "gpu_index": 0
  },

  "method_timings": {
    "j3_q95": {
      "score_compute_wall_ns": 32800,
      "rank_select_wall_ns": 5100,
      "total_wall_ns": 37900
    }
  },

  "candidates": []
}

这里的 competitive_candidate_count 特别重要：

scheduler 第一排序键是 priority，所以真正的“排序竞争”应该主要分析：

C
d
	​

={i:priority
i
	​

=
j
min
	​

priority
j
	​

}.

不同 priority 的 candidate 本来就不会因为 runtime score 被翻过去。

A2. 每个 candidate 必须保存的字段

建议 candidate identity 至少用：

(job_instance_id, node_id, gpu_index)

因为同一个 node 在不同 GPU 上可能是两个不同 candidate。

字段：

JSON
{
  "candidate_id": "...",

  "job_instance_id": "...",
  "node_id": "...",
  "model_id": "...",
  "gpu_index": 0,

  "priority": 0.0,

  "model_resident": true,

  "current_runtime_p50_ms": 412.3,
  "current_load_ms": 0.0,
  "current_cost_ms": 412.3,

  "legacy_tiebreak_1": 17,
  "legacy_tiebreak_2": 2,
  "legacy_tiebreak_3": "...",

  "scores": {
    "...": {}
  }
}

这里 current_cost_ms 必须显式落盘，因为我们现在已经知道：

total=current+future

而且 future scale 与 current 的相对权重本身会影响 ordering。

A3. 每种方法都保存同一套 score record

例如：

JSON
"j3_q95": {
  "future_cost_ms": 37885.0,
  "total_cost_ms": 38297.3,

  "future_rank": 2,
  "total_rank_within_priority": 1,
  "scheduler_key_rank": 1,

  "scheduler_key": [
    0.0,
    38297.3,
    37885.0,
    17,
    2,
    0,
    "...",
    0
  ],

  "would_choose": true
}

这里我坚持把 完整 scheduler_key 保存下来。

不要以后再根据若干字段“重构” scheduler key。

否则将来 simulator 改 tie-break 时，旧 trace 就无法精确复现为什么当时选了谁。

A4. 本轮 shadow scoring 应该同时打哪些方法？

我建议第一次机制 trace 至少注册这些：

method_id	future score 来源	用途
j3_q95_v1	J3 q95	当前历史基准
r1b_q95_v1	R1b q95	新 distribution 同 legacy consumer
r1b_mean_v1	R1b mean	风险中性
r1b_stepcvar95_v1	R1b marginal step CVaR95	tail consumer
attr_argmax_stats_v1	现有 _step_estimate_cost()	老“属性+统计量”基线
attrmix_stats_atom_v1	新属性概率混合，组内用旧 point stats	隔离属性不确定性
attrmix_stats_hist16_v1	属性概率 × 组内 empirical distribution	属性 + 组内噪声
truth_h5_v1	truth successor reference	机制 reference

这八个全部只做：

shadow score

真正执行 trajectory 可以仍只由：

j3_q95_v1

决定。

A5. “所需时间”必须拆成两种完全不同的时间

项目负责人说的“耗时”很容易产生歧义，因此我建议两种都保存。

第一种：方法本身的计算耗时

用：

Python
运行
time.perf_counter_ns()

并明确计时边界。

对每个 method：

Python
运行
t0 = perf_counter_ns()

for candidate in competitive_pool:
    score = method.score(candidate)

t1 = perf_counter_ns()

rank_candidates(...)
winner = ...

t2 = perf_counter_ns()

落盘：

score_compute_wall_ns = t1 - t0
rank_select_wall_ns   = t2 - t1
total_wall_ns         = t2 - t0

日志序列化绝对不要包含在这个计时里。

否则你测到的是 gzip/json 性能，不是调度方法性能。

项目负责人如果想看：

“每种方式打完一个 decision 要花多久”

看这个。

第二种：被调度任务本身的执行时间

这不是 wall-clock scoring time。

建议单独写：

execution_trace_v1.jsonl.gz

通过 decision_id 连接。

例如：

JSON
{
  "episode_id": "...",
  "decision_id": 137,

  "chosen_candidate_id": "...",

  "truth_compute_ms": 6300.0,
  "truth_load_ms": 1200.0,
  "truth_eviction_ms": 800.0,

  "truth_runtime_ms": 7500.0,
  "realized_service_ms": 8300.0
}

必须明确：

truth_runtime=compute+load

而 simulator 中真实 service：

realized_service=compute+load+eviction.

这样以后不会把模型预测 target 和 scheduler 真正付出的系统成本混为一谈。

A6. 是否必须重跑 300 集？
判定：必须重新跑 scheduler，但不用重跑 predictor。

你判断正确。

现有结果无法补算真实 decision trace，因为没有保存：

当时 candidate pool；

GPU residency；

candidate→GPU pairing；

priority；

current load 是否为 0；

各方法的 scheduler key。

而且六个 arm 之前走的是不同 trajectory，一旦前面一次选择不同，后面的候选池也会不同。

所以：

不能从现有 per-episode 结果倒推真实 candidate ranking
	​

但第一轮不需要再跑 300×8

我更推荐：

300 episodes×1 actual trajectory+8 shadow methods
	​


即仍走 A0/J3 trajectory。

在每个真实 A0 decision state，所有方法同时评分。

这样最大的优点是：

current, candidate pool, GPU state, priority

对所有方法完全相同。

这才是干净的机制比较。

之后如果发现某个新方法真的明显改变 decision ordering，再跑它自己的 actual trajectory 测 end-to-end scheduling。

A7. 分数/排序对比报表

除了原始 JSONL，我建议自动生成三个 report。

Report 1：decision-level wide table

例如：

candidate	current	J3 future	J3 total/rank	R1b q95 total/rank	AttrMix total/rank	Truth total/rank
X	100	10	110 / #1	104 / #3	126 / #2	119 / #2
Y	10	100	110 / #2	38 / #1	31 / #1	25 / #1
Z	75	50	125 / #3	90 / #2	145 / #3	160 / #3

如果 winner 不一致，直接标：

J3 chooses X
R1b chooses Y
AttrMix chooses Y
Truth chooses Y

这就是负责人最想看的东西。

Report 2：method-pair mechanism summary

例如：

X vs Y	future Kendall	total Kendall	pair flip	top1 disagreement	margin crossing
J3 vs R1b	.91	.63	18%	27%	21%
Argmax vs AttrMix	.85	.71	14%	19%	16%

重点同时报告：

τ
F
	​


和：

τ
C+F
	​

.

因为我们现在的核心假设正是：

future ranking 可能变化不大，但由于 scale/spacing 与 fixed current 相加，total ranking 会显著变化。

Report 3：最值得人工看的翻转案例

自动筛：

top-1 disagreement
largest margin crossing
largest local truth regret
high attribute entropy
largest rank flip

每类展示前 20 个 decision。

这样负责人不用翻几万个 JSON record。

A8. 回归 guard：必须证明插桩不改变行为

这是这一步的硬 gate。

同 episode、同 policy、同 artifact：

trace OFF
vs
trace ON + shadow scoring

要求：

chosen_candidate sequence

完全相同。

建议生成：

chosen_sequence_sha256

然后要求：

trace_off_hash == trace_on_hash

同时：

mean_completion_ms
makespan
deadline_miss
evictions
completed_jobs
failed_jobs

全部 bitwise 相同或浮点 exact-equality。

额外要求：

每个 decision 的 actual-method winner
==
原 simulator min(pool,key=...)

如果不相同，直接 fail。

因此这一步可以做到：

零训练 + 零 predictor rerun + 零调度语义修改
	​


只重新运行 simulator。

B. “属性分布 + 统计量混合”怎么构造？

这里我会改变一个容易踩坑的想法：

不要把 role × action_family × model_class 三个独立 softmax 做全笛卡尔积。

原因是：

P(R,A,M)

=P(R)P(A)P(M)

并没有任何证据支持属性条件独立。

这样做会产生很多训练中根本不可能的组合。

B1. 混合什么？

原则非常简单：

只混合真正进入 statistics lookup key 的不确定属性。

你现在统计量消费的是：

(model, lane, sequence_index).

所以 v1 runtime mixture 不需要把 role 和 action_family 塞进去。

它们可以：

落盘；

算 entropy；

做机制分析；

但不参与 v1 runtime mixture。

这是刻意约束模型自由度。

最理想情况：artifact 有完整 model-id distribution

若有：

p
h
	​

(m)=P(model=m∣x,h),

而：

lane=l(m),

则直接：

w
m
	​

=p
h
	​

(m).

每个 m 对应：

g=(m,l(m),s
h
	​

).

然后用现有 hierarchical lookup 找 stats。

这是最干净的。

如果只有 model_class distribution

假设只有：

P(model_class=c)

且 lane_for(c) 可以确定 lane。

那么：

P(lane=l)=
c:lane(c)=l
∑
	​

P(c).

此时我建议只在：

lane

层面混合。

lookup 使用：

* | lane | sequence_index

然后 fallback：

* | lane | *
→ global

不要做：

“model_class probability × 当前 argmax model_id”

因为那是在凭空制造一个不存在的 joint distribution。

如果 artifact 只有 top-1 probability

这一点要在实现前 preflight。

比如只有：

role = execute
role_probability = 0.71

但没有：

role_probs = {...所有类别...}

那它不够构造 attribute distribution。

绝对不要自己补：

execute=0.71
others均分0.29

这是没有依据的。

这时正确做法是：

从冻结 checkpoint 重新 pack 一个 attribute-distribution sidecar。

不是训练，只是 inference/serialization。

例如：

attribute_distribution_sidecar_v1.jsonl.gz

存：

JSON
{
  "node_id": "...",
  "future_h5": [{
    "steps": [{
      "slot": 1,
      "model_class_probs": {...},
      "role_probs": {...},
      "action_family_probs": {...}
    }]
  }]
}
B2. sequence_index 怎么处理？

我不把它当随机变量。

它应该是：

deterministic structural context
	​


每个预测 future slot 已经有：

h=1,…,H.

如果 artifact 中的 sequence_index 与 train statistics 里的 sequence_index 语义完全一致，就直接使用。

如果只是：

future horizon slot number

而训练 stats 是：

真正 workflow absolute sequence index

那不能混用。

因此加一个 preflight：

sequence_index_semantics

必须明确为：

compatible

否则 stats lookup 强制退回：

model | lane | *

或：

* | lane | *

不要拿一个“长得像 index”的值硬塞进去。

还要记录：

exact_position_hit_rate
position_fallback_rate
lane_fallback_rate
global_fallback_rate

这个很重要。

B3. 第一层混合：只表示 attribute uncertainty

为了科学归因，我建议先做一个非常简单的版本：

AttrMix-Atom
	​


对于每个可能 lookup state g，现有 stats 给：

r
g
	​

=runtime
p50,g
	​

l
g
	​

=load
p50,g
	​

.

attribute mixture weight：

w
g
	​

.

得到一个原子混合：

P(T
h
	​

=r
g
	​

)=w
g
	​

.

注意：

runtime distribution 和 load surcharge 分开。

runtime mixture：

P(T
h
	​

)=
g
∑
	​

w
g
	​

δ
r
g
	​

	​

.

legacy load：

L
h
	​

=
g
∑
	​

w
g
	​

l
g
	​

.

然后可以从这个 atomic distribution 得：

E[T
h
	​

]
q
.50
	​

(T
h
	​

),q
.95
	​

(T
h
	​

)

以及：

CVaR
.95
	​

(T
h
	​

).

scheduler step score，例如 mean consumer：

C
h
	​

=E[T
h
	​

]+L
h
	​

.

这个版本只改变：

argmax attribute → probability-weighted attribute

没有额外引入组内 aleatoric distribution。

所以它是最重要的机制 control。

B4. 组内噪声确实不能忽略

你现在已经量化：

p10≈0.50×,p90≈2.40×.

所以：

属性知道得再准，也不代表 runtime 就确定了。

因此第二个版本应该是：

AttrMix-Hist16
	​


但我建议作为第二个 shadow method，而不是第一版本直接揉在一起。

训练统计 bank

完全使用 training split。

沿用现有 mass-balanced16 edges。

对每个 lookup group g，统计：

P
g
	​

(B
k
	​

)=
n
g
	​

n
g,k
	​

	​

.

即：

JSON
{
  "lookup_key": "...",
  "n": 382,

  "runtime_bin_probs": [
    ...
  ],

  "runtime_p50_ms": ...,
  "load_p50_ms": ...
}

同时建立现有 hierarchical fallback 同层级：

model | lane | position
model | lane | *
*     | lane | position   （若当前 stats 有）
*     | lane | *
global

不新加平滑超参。

也就是说：

group 存在 → 用 empirical histogram；

group 不存在 → 沿用当前 hierarchical fallback；

不因为 group 小就临时 invent 一个 threshold。

这样自由度最低，也最容易解释。

最终 step distribution

对于 attribute latent state g：

P(T
h
	​

∈B
k
	​

∣g).

于是：

P(T
h
	​

∈B
k
	​

)=
g
∑
	​

w
g
	​

P(T
h
	​

∈B
k
	​

∣g)
	​


这就是完整的：

attribute epistemic uncertainty + within-group empirical variability。

并且天然得到：

runtime_probs[16]

所以能直接复用你当前 resource-v2 的 canonical view 代码：

mean
q50
q90
q95
CVaR95
B5. 它如何接当前两层 interface？

我建议不要修改 R1b artifact。

Attribute-stat mixture 是另一类 producer。

定义：

resource_source_type =
"attribute_stats_mixture"

manifest：

JSON
{
  "artifact_schema_version": "resource-v2",

  "resource_source_type": "attribute_stats_mixture",

  "attribute_source_artifact_id": "...",
  "stats_bank_id": "...",
  "stats_bank_sha256": "...",

  "mixture_axis": "model_id"
}

step 仍然输出完全相同：

runtime_probs
runtime_ms_quantiles
runtime_mean_ms
cvar95_ms
bin_schema_id

于是 scheduler 根本不需要知道：

这个 distribution 来自 neural resource head 还是 attribute-stat mixture。

这就是现有两层 interface 最大的价值。

B6. 但我更建议第一轮不要重新 pack

机制阶段可以直接：

attribute probs
+
stats bank
→ shadow scorer

on-the-fly 算。

这样先回答：

这种方法究竟给出什么 score/rank？

如果后面决定正式进入 scheduler benchmark，再 pack 成 resource-v2 artifact。

避免现在为了机制实验扩 schema。

B7. 与 R1b 是什么关系？
我的裁决：并列的第二个预测/消费路线，不是 R1b 的 replacement。

本轮应形成：

ArgmaxStats→AttrMixAtom→AttrMixHist16
	​


这一条轴，回答：

属性概率有价值吗？
组内 noise distribution 又增加了什么？

另一条：

J3→R1b
	​


回答：

learned resource head 有什么价值？

两条路线并排比较。

本次 300 集机制实验不选 winner。

C. 文件级实施计划

我建议按四个阶段。

阶段	做什么	产物	Gate
C0	修阻塞 P0 + trace skeleton	validator/tests	所有旧 regression PASS
C1	decision trace + shadow score	decision_trace_v1.jsonl.gz	trace ON/OFF 行为完全一致
C2	AttrMix-Atom + AttrMix-Hist16	stats bank + shadow methods	probs/weights/lookup 全部 fail-closed
C3	300 episode A0 trajectory mechanism run	trace + reports	完整性 PASS，输出预注册 CI
C0. 先修哪些？

你上轮五个 P0，我建议全部在正式 mechanism run 前修掉：

1 canonical-view NaN/missing validator
2 overlay probability finite/range check
3 loader 实际重算 SHA
4 runner A1-A0 verdict overwrite
5 exact episode-set pairing assertion

因为机制 trace 里还会同时 shadow R1b/CVaR 等 resource-v2 方法。

但是已知 double-load / eviction 缺失这轮不要修

这是一个非常重要的范围控制。

老 stats family 存在：

runtime+load

重复 loading，以及 eviction omission。

这些应该：

记录、标注，但本次不改变。

否则：

Argmax → AttrMix

的比较会同时夹杂 cost-semantics 修复。

本轮 manifest/report 明确写：

cost_semantics =
legacy_runtime_plus_load_no_eviction_v1

以后另开 experiment 修。

C1. simulator 新增函数

我建议先把现有 scoring 从 min(..., key=...) 里面抽出来。

例如：

Python
运行
def sameshape_score_components(
    candidate: Candidate,
    *,
    future_cost_ms: float,
) -> SchedulerScore:

返回：

priority
current_runtime_ms
current_load_ms
current_cost_ms
future_cost_ms
total_cost_ms
scheduler_key

然后：

Python
运行
def score_shadow_methods(
    decision_context: DecisionContext,
    methods: Sequence[ShadowScoringMethod],
) -> ShadowDecisionScores:

以及：

Python
运行
def emit_decision_trace(
    writer: DecisionTraceWriter,
    decision_context: DecisionContext,
    scores: ShadowDecisionScores,
    chosen_candidate_id: str,
) -> None:

最好不要让 shadow scorer 操作 GPU/job state。

全部纯函数读取。

C2. 统一 method interface
Python
运行
class ShadowScoringMethod(Protocol):
    method_id: str

    def future_cost(
        self,
        *,
        node_id: str,
        candidate: Candidate,
        context: DecisionContext,
    ) -> FutureScoreResult:
        ...

FutureScoreResult：

Python
运行
@dataclass(frozen=True)
class FutureScoreResult:
    future_cost_ms: float

    runtime_statistic: str
    runtime_probs: tuple[float, ...] | None

    lookup_level: str | None
    attribute_entropy: float | None

这样：

J3；

R1b；

AttrMix；

truth

都走同一 tracing path。

C3. 属性 stats bank

新脚本：

scripts/build_attribute_runtime_stats_bank.py

核心函数：

Python
运行
def build_runtime_histogram_bank(
    train_rows,
    bin_edges_ms,
) -> RuntimeStatsBank:

以及：

Python
运行
def lookup_runtime_distribution(
    bank: RuntimeStatsBank,
    *,
    model_id: str | None,
    lane: str,
    sequence_index: int | None,
) -> GroupRuntimeDistribution:

必须复用现有 hierarchical fallback 规则，不 invent 第二套 lookup hierarchy。

C4. attribute mixture 函数

建议独立文件：

src/tracing/analysis/attribute_stats_mixture.py

避免继续把 4000+ 行 simulator 塞大。

核心：

Python
运行
def build_lookup_mixture(
    step: Mapping[str, Any],
    stats_bank: RuntimeStatsBank,
) -> AttributeLookupMixture:

然后：

Python
运行
def mix_atomic_runtime_distribution(
    mixture: AttributeLookupMixture,
) -> DiscreteDistribution:

以及：

Python
运行
def mix_empirical_runtime_distribution(
    mixture: AttributeLookupMixture,
    stats_bank: RuntimeStatsBank,
) -> Distribution16:

最后：

Python
运行
def canonical_runtime_views(
    probs: Sequence[float],
    reps: Sequence[float],
    alpha: float = 0.95,
) -> RuntimeViews:

最好直接复用 resource-v2 已经审过的 canonical-view implementation，避免又写第三份 quantile/CVaR。

是否需要改 predictor artifact schema？
第一阶段：不需要。

先检查现有 artifact 是否真的有完整 probability vector。

必须加 preflight：

attribute_distribution_available
probability_vector_length
prob_sum_error
category_schema_id

若只有 top1 + role_probability：

fail
	​


然后再生成一个 frozen-predictor sidecar。

不要修改已有 artifact，更不要猜余下类别概率。

机制实验的预注册假设

这轮不要设置：

Kendall > 0.8 PASS

这种无依据阈值。

冻结问题和方向即可。

H1：Attribute uncertainty actually changes the score

比较：

AttrArgmaxvsAttrMixAtom.

报告：

future Kendall；

total Kendall；

pairwise flip；

top-1 disagreement；

score ratio；

95% episode-cluster bootstrap CI。

如果接近完全一致：

attribute probabilities 对 scheduler 当前接口几乎没有额外价值。

这是有效结果，不是失败。

H2：Attribute uncertainty is most relevant when entropy is high

预注册：

high-entropy decision states 应有更高的 Argmax-vs-Mix top1 disagreement / flip rate。

不要提前定 entropy threshold。

可以使用：

train-frozen quartiles；

或直接报告 entropy vs disagreement 的 episode-level association。

H3：Within-group variability adds information beyond attributes

比较：

AttrMixAtomvsAttrMixHist16.

如果 group noise 重要，应观察到：

tail-sensitive scores 变化；

total ranking 变化；

特别是在 high within-group dispersion lookup groups 中更明显。

仍然报告 CI，不人为规定必须多大。

H4：Future scale 与 fixed-current interaction

所有主要 pair 同时报告：

τ
F
	​


和：

τ
C+F
	​

.

核心机制量：

Δτ=τ
F
	​

−τ
C+F
	​

.

如果 future ordering 很接近，但 total ordering 明显更低：

直接证明 future score scale/spacing 与 fixed current 的耦合正在改变调度选择。

这是这轮我最希望看到的结果之一。

Observability 本身的硬 gate

这个可以有真正的 PASS/FAIL：

300/300 episodes traced
0 missing decisions
0 missing candidates
0 duplicate candidate ids
0 non-finite scores
0 malformed scheduler keys
all shadow methods evaluated on identical candidate sets
trace ON/OFF chosen_sequence_sha256 identical
trace ON/OFF per-episode metrics identical
all method timing fields finite and >= 0

只有这些全部 PASS，机制 report 才能用。

明确不做什么

这轮我建议范围严格锁住：

不训练任何模型；

不 full-unfreeze；

不修改 checkpoint；

不用这 300 episode 构造 loss；

不打开 J test；

不修 legacy double-load；

不加入 eviction-aware 新 scheduler；

不做 role×family×model_class 独立笛卡尔积；

不根据 smoke 结果重新缩放 q95；

不把 AttrMix 做成新的 winner；

不做 joint horizon distribution，因为你们没有 joint；

不把 shadow-scoring 的效果直接宣称为 end-to-end scheduling improvement。

最后一条尤其重要：

Shadow scoring 告诉你“如果这个方法面对同一个状态，会怎么评分、怎么排序”。

它不能完全替代：

“如果真的让这个方法连续控制 scheduler，最后会怎样”。

因为一旦第一次选不同，后续 trajectory 就分叉了。

我认为这条路线最大的风险

不是代码。

而是：

属性不确定性根本不是 runtime uncertainty 的主要来源
	​


你已经知道：

stats 只有 78 个 coarse groups；

即使属性完全正确，group 内 estimate/truth 仍大约从 0.5× 到 2.4×；

summarize 这种单一动作内部就能差 64 倍；

还有 legacy load double counting 和 eviction omission。

所以很可能发生：

AttrMix-Atom 相比 Argmax 几乎没变化；
AttrMix-Hist16 虽然产生更宽的 distribution，却主要是在表达组内 aleatoric noise，而不是属性概率带来的信息。

这不是坏结果。

它反而能把论文里的问题讲得非常清楚：

runtime uncertainty=
可由属性头改善
attribute uncertainty
	​

	​

+
属性头解释不了
within-group execution uncertainty
	​

	​

+
load / eviction / residency
system-state cost
	​

	​


而你们现在终于可以用同一个 decision trace 量化这三层到底哪一层真正改变 scheduler ordering。

因此我建议当前实施顺序正式改成：

修P0→decision tracing→shadow scoring regression→AttrMix-Atom→AttrMix-Hist16→300集单轨迹机制实验→人工/统计审查→再冻结full-unfreeze目标
	​


full-unfreeze 的代码框架可以并行写，但不要在这轮机制结果出来之前启动最终训练。这样项目负责人先能真正“看到每一种方式为什么选这个节点”，你们后续再决定该优化 distribution、attribute uncertainty、还是 scheduler-facing score，就不会再盲猜。
