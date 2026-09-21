# GPT 规划：放开历史资源输入 + 重训整个预测器 + 重跑 risk/CP

- 日期：2026-09-21
- 请求：`.scratch/gpt_plan_unfrozen_input.md` + 因果边界澄清 `.scratch/gpt_causal_boundary_clarification.md`
- 项目负责人给的硬边界（逐字）：
  > 历史信息（包括属性和运行用时）都可以用；当前节点的类别可以用；
  > **但是耗时不可以用**；未来节点的所有信息**绝对不可以用**

## 总方向

> 这次不要把这五件事揉成一个大实验：
> 加入历史耗时 / 换资源头 / decision-focused training / risk 改造 / CP 改造。
> 最干净的主线是：**先只验证历史资源信息有没有价值 → 冻结新预测器 →
> 统一 future coefficient 重跑 risk/CP → Truth-Q 解释预测信息与真值信息的差距**

**先定死一个结论**：直接把 `runtime_ms` 加进 `_history_token()` **会泄漏当前节点耗时**
（因为 `events[:event_index+1]` 的最后一个是 current）。所以**不能从 forbidden list 里删掉 `runtime_ms`**。

**正确契约**：
- `categorical visible(i) ⟺ i ≤ current`
- `resource visible(i) ⟺ i < current`

---

==========================================================================================
我先给总方向：这次不要把“加入历史耗时”“换资源头”“decision-focused training”“risk 改造”“CP 改造”一次性揉成一个大实验。最干净的主线应该是：

先只验证历史资源信息有没有价值→冻结新预测器→统一 future coefficient 重跑 risk/CP→Truth-Q 解释预测信息与真值信息的差距
	​


其中一个结论可以先定死：直接把 runtime_ms 加进现有 _history_token() 会泄漏当前节点耗时。 当前 make_feature() 用 events[:event_index+1]，最后一个 token 就是 current，而项目原 causal contract 明确规定当前 ready node 的 runtime/load/memory/status 在派发前不可见。因此不能简单从 forbidden list 里删掉 runtime_ms。

更好的办法也不是把整个 history 改成 events[:event_index]——那会把当前节点本来合法可见的类别信息也删掉。应该保持现有 categorical prefix 不变，只给严格早于 current 的已完成 token开放资源通道。

需求 1：放开输入契约 + 全预测器重训
1.1 字段怎么放

我建议第一版只引入最有解释力、在线时确实可观测的历史信息。

原始字段	v2 是否进入模型	表示	原因
runtime_ms	是，仅 completed history	log1p + train-only robust z-score + missing mask	最直接的历史执行速度信号
local_runtime_ms	否，第一版不放	—	仓库当前没有足够清楚的独立语义证明；很可能与 runtime 高度重叠，先避免重复特征
load_ms	是，仅 completed history	log1p + robust z + missing mask	历史冷启动/cache 行为对未来可能有价值
memory_mb / peak memory	条件放	log1p + robust z + missing mask	只有 Stage-0 证明该字段确实是运行后在线可观测、且训练/部署同语义时才启用
status	是，仅 completed history	categorical embedding	success/failure/retry 历史是合法因果信号
当前节点以上所有字段	全部禁止	mask=0 / UNOBSERVED	派发时未知

我不会第一轮加入“过去 K 步均值”“最近同类动作平均耗时”等人工 aggregation。

原因是 GRU 本身已经在编码历史序列。先让：

categorical history+observed resource history

成为唯一实验变量，最容易归因。

1.2 不要直接喂原始毫秒

直接用 raw ms 有两个问题：

第一，runtime 的跨度太大，0.1ms metadata 到 10s LLM 相差几个数量级。

第二，模型容易学习：

“这台机器上 Qwen 大概就是 5 秒”

而不是：

“这个工作流最近资源状态有什么变化”。

跨机器、跨 GPU 或 workload scale 时会很脆。

所以数值变换固定为：

x
′
=log(1+max(0,x))

然后用仅 train split估计：

z=
max(IQR
train
	​

/1.349,ϵ)
x
′
−median
train
	​

	​


再固定：

z←clip(z,−6,6).

模型输入只看到 z 和 missing mask。

manifest 必须保存：

numeric_scaler_version
train_split_sha256
median_log1p
iqr_log1p
clip_range

validation/test 绝对不能重新估 scaler。

1.3 因果边界：这是本轮最重要的 gate
VERIFIED

现有：

Python
运行
history = events[:event_index + 1]
current = history[-1]

因此直接给 _history_token() 加：

Python
运行
runtime_ms=event["runtime_ms"]

一定会把 current outcome 放进去。

正确契约是：

categorical visible(i)⟺i≤current
	​


但：

resource visible(i)⟺i<current
	​


也就是继续保持：

Python
运行
events[:event_index + 1]

的 categorical history。

但是资源通道：

Python
运行
resource_observed = position < event_index

current token：

runtime/load/memory masks = 0
status = UNOBSERVED
推荐的 token

概念上变成：

Python
运行
{
    # 原来字段，current 仍然合法
    "position": ...,
    "event_type": ...,
    "node_type": ...,
    "role": ...,
    "raw_action": ...,
    "action_family": ...,
    "model_id": ...,

    # 新增 causal history channel
    "history_resource": {
        "observed": true/false,
        "runtime_z": ...,
        "runtime_present": 1,
        "load_z": ...,
        "load_present": 1,
        "memory_z": ...,
        "memory_present": 0,
        "status_class": "success"
    }
}

但我甚至建议继续保留原始 forbidden names：

runtime_ms
load_ms
memory_mb
status

在 FORBIDDEN_MODEL_INPUT_KEYS 中。

不要把它们全局解禁。

模型输入使用：

hist_runtime_z
hist_load_z
hist_memory_z
hist_status_class

这种 derived allowlist。

这样如果有人未来又粗暴写：

Python
运行
current["runtime_ms"] = ...

旧递归 guard 仍然会立刻报错。

1.4 必须增加的因果 guard

除了现有 recursive forbidden-key audit，再加：

Python
运行
audit_history_resource_causality(...)

要求：

token_source_index < current_event_index
    iff
resource_observed == True

并且强制：

current token:
runtime_present == 0
load_present == 0
memory_present == 0
status == UNOBSERVED

我还强烈建议做一个反事实 mutation test。

同一个 sample：

先正常生成 feature A。

然后把 current event：

runtime_ms = 999999999
load_ms = 999999
memory_mb = 999999
status = "FAILED_SENTINEL"

生成 feature B。

必须：

model_input
A
	​

=model_input
B
	​

	​


byte-identical。

然后改过去一个 completed event 的 runtime：

feature 必须只在对应 historical-resource channel 变化。

这比单纯检查 key 更强。

1.5 模型怎么改

不需要注意力，也不需要单独的 K-step branch。

第一版尽量小。

当前每个 token：

e
t
	​

=∑categorical embeddings+position.

改成：

e
t
	​

=e
t
cat
	​

+e
t
pos
	​

+m
t
	​

f
θ
	​

(r
t
	​

)+e
t
status
	​


其中：

r
t
	​

=[runtime
z
	​

,load
z
	​

,memory
z
	​

,runtimeMask,loadMask,memoryMask].

m_t=1 表示至少有一个 historical resource field 可见。

推荐：

Python
运行
hist_resource_proj:
    Linear(6, hist_dim)
    SiLU
    Linear(hist_dim, hist_dim)

最后乘：

Python
运行
observed_mask

避免全 0 输入仍通过 bias 产生“伪 current resource”。

然后仍然进原 GRU。

也就是说：

categorical history ─┐
numeric past-resource ├→ token embedding → GRU → repr_vec
status embedding ─────┘
                                      ↓
                   structure / behavior / attributes
                                      ↓
                                    q(A)
                                      ↓
                               resource head

其余 J3 结构不需要重写。

1.6 baseline 怎么设计才真正能归因

不要做一个模糊的“新模型 vs J3”。

至少保留三个角色：

arm	输入	架构/训练	用途
J3-frozen	old	原冻结 checkpoint	历史 anchor
F0-NoHistRes	old categorical，resource 全 mask	新模型、新头、全参数训练	控制“重新训练/新资源头”的影响
F1-HistRes	categorical + causal past-resource	与 F0 完全相同	唯一多了历史资源信息

真正回答项目负责人：

“过去耗时有没有用？”

看：

F1−F0
	​


不是 F1-J3。

J3 只是外部历史基线。

初始化

J3 checkpoint 不改。

F0/F1：

所有与 J3 同 shape 参数从 J3 初始化；

新 historical-resource branch 用同 seed 初始化；

新 16-bin runtime head 用同 seed 初始化；

F0/F1 对应 seed 的初始化、batch order、bootstrap 完全一致；

全部参数 requires_grad=True。

所以“整个预测器重训”得到满足，但 J3 artifact 完全冻结。

1.7 资源头

我建议继续 16-bin。

原因是它已经证明：

distribution fitting 比旧 3 quantile 更强。

而且现有：

runtime_probs
mean
p50/p90/p95
CVaR

scheduler interface 都已经建好了。

bin edges 不重新搜索。

直接冻结现有：

mass_balanced_16

因为 train 数据没变。

这样不会又把“加历史信息”和“重划 bins”混在一起。

1.8 损失函数：我这里会修正之前的建议

这轮不建议直接上：

L
dist
	​

+0.25L
horizon
	​

+0.25L
decision
	​

.

原因不是这些 loss 不合理，而是这次科学问题变了：

历史资源信息是否能提高完整预测器？

如果同时引入新的 decision loss，结果改善后就无法知道：

是历史资源输入；

还是 decision-aware training。

所以主实验先固定：

L=L
S
	​

+L
C
	​

+L
B
	​

+L
R
	​

	​


其中：

L
R
	​

=L
runtime−dist
	​

+L
load
	​

+L
memory
	​

.

runtime：

L
runtime−dist
	​

=NLL+0.5RPS+0.25L
pseudoHuber(mean)
	​

.

如果你现在已经有预先固定定义的 bandCE，可以两臂共同保留：

+0.25L
bandCE
	​

.

如果还没实现，不要为了这一轮新增。

NLL + RPS 在加入历史耗时后仍然完全合理；输入变化不改变分布 target。

L_decision

这一轮明确：

不训练
	​


因为 300 集不能作为训练信号，而另外构造 train-only decision dataset 会把研究问题进一步扩大。

如果 F1 预测更好但调度仍失败，再单独开 Phase-D。

1.9 训练协议

继续复用原 J-series 严谨协议：

3 seeds: 11 / 22 / 33
30 epochs
batch=256（6GB 能跑就保持）
无 early stopping
固定 validation video bootstrap B=1000

checkpoint：

先满足原 interface NI：

endpoint	原冻结 δ
future length MAE	+0.02
termination BCE	+0.02
content accuracy	−0.02
next-role acc	−0.01
next-family acc	−0.02

在 NI-feasible epoch 中：

argminRuntimeQScore.

这条不用重新发明。

1.10 Predictor 预注册判据

第一层是完整性：

RuntimeQScore≤845.0

即至少不能退到旧 J3 以下。

同时：

CalibrationError≤0.05

以及：

quantile crossing=0.

第二层回答历史资源输入：

Δ
hist
	​

=RuntimeQScore
F1
	​

−RuntimeQScore
F0
	​

.

paired video-cluster bootstrap。

若：

CI
upper
	​

(Δ
hist
	​

)<0

才称：

historical resource telemetry improves runtime prediction。

否则不这么写。

RPS/NLL/log-MAE/band ratio 都是 secondary。

调度侧判据
δNI=485ms 继续沿用吗？

我的答案是：

是
	​


前提是：

同一个 v03；

同一个 mean_completion_ms；

同一个 paired episode protocol。

这个 δ 是以前结果出来前已经冻结的工程 practical margin，没有理由因为换 predictor 再改。

但这 300 集已经看过，所以它现在只能是：

development gate

不能作为最终 confirmatory evidence。

对于新 predictor + 原 sameshape_h5_p95：

Δ=F1−J3.

Non-inferior：

CI
upper
	​

<+485ms.

Strong success：

CI
upper
	​

<0

且：

point≤−485ms.

同时一定报告：

F1−F0

以回答输入契约价值。

J test 继续完全不碰，直到 predictor、risk、CP 配置全部冻结。

需求 2：Risk v2

当前仓库 risk 确实是：

score=−U
norm
	​

+R
norm
	​

−A
norm
	​


其中：

R=(transition+memory+eviction)/3.

三个权重当前都是 1.0。

我建议唯一改动

增加：

F
norm
	​

=MinMax
C
	​

(F
j
	​

)

其中默认：

F
j
	​

=
h
∑
	​

q
0.95,j,h
	​

	​


来自最终选定的新 predictor。

Risk-v2：

score
j
	​

=−U
j
	​

+R
j
	​

−A
j
	​

+F
j
	​

	​


这里所有变量都已经归一到 [0,1]。

未来越贵，score 越大，因此越不优先。

权重不要调

固定：

w_u = 1
w_r = 1
w_a = 1
w_f = 1

理由不是“1 最优”，而是：

原三个 normalized component 本来就是等权；新增 future 也以等权进入，可以完全避免 validation 上调参。

整体除以 4 没必要，因为不改变排序。

risk 必须有三臂
risk_v1_no_future
risk_v2_pred_future
risk_v2_truth_future   # diagnostic

第三个 future：

F
truth
=limited_future_truth_cost(H=5)

只是 truth-informed coefficient，不叫 oracle。

这样可以回答：

risk
pred
	​

−risk
noFuture
	​


“未来信息有没有帮助 risk”

以及：

risk
truth
	​

−risk
pred
	​


“信息误差还有多少影响”。

不要同时 sweep mean/CVaR/q95。

这轮只固定 q95。

需求 2：CP 重跑

现在的 CP 历史结果不能当干净负结果，这一点你的总结是对的。

我会新建一个：

cp_rho_v2_clean

而不是覆盖原 cp_rho。

首先清理比较契约

为了做：

2×2

CP-v2 必须和 greedy 使用相同：

candidate feasibility
hard priority tier
current-cost definition
future coefficient
future_weight

我建议 CP-v2 先过滤：

priority=p
min
	​


即和 sameshape 一样把 priority 作为硬规则，而不是现在 CP 里的 soft flow weight。

同时：

priority_weight = 0
memory_weight = 0
future_weight = 1

memory 仍可作为feasibility，但不要再以额外 objective penalty 改变实验因素。

现有 legacy current-cost semantics 本轮也不修，包括已经知道的 load double-count 问题。

否则又多一个变量。

干净 2×2

固定：

F
pred
=
h
∑
	​

q
.95,h
	​

F
truth
=limitedFutureTruthCost
H5
	​

.

四臂：

	predicted future	truth-informed future
Greedy	G-P	G-T
CP-v2	CP-P	CP-T

G 的 key：

(priority, current+F, F,…)

CP 两臂唯一不同也必须只是 F。

然后报告三个核心 contrast：

Information effect
G
	​

=G
T
	​

−G
P
	​

Information effect
CP
	​

=CP
T
	​

−CP
P
	​

Framework effect
Pred
	​

=CP
P
	​

−G
P
	​


和：

Framework effect
Truth
	​

=CP
T
	​

−G
T
	​

.

还可以报告 interaction：

(CP
T
	​

−CP
P
	​

)−(G
T
	​

−G
P
	​

).

这才是真正回答：

是 prediction 限制更大，还是 greedy framework 限制更大？

CP 时间预算怎么定

我不建议现在说：

“就用 0.5s”。

正确方法是先校准求解器，而不是拿 scheduler completion 调时间参数。

这不算 outcome tuning。

固定 train-only calibration set

例如：

500 个 train-only decision states

要求有 ≥2 competitive candidate。

对每个 state 分别跑：

50 ms
100 ms
250 ms
500 ms
1000 ms

num_workers=1 保证重复性。

记录：

status
incumbent/objective
best_bound
relative_gap
first_action
solve_ms
fallback
candidate_count
预算选择规则提前冻结

选择最小的时间预算，使：

nonFallbackRate≥95%

且相对于下一个更长预算：

FirstActionAgreement≥99%.

如果到 1s 仍达不到，就不继续加时间。

转而简化 CP model。

同时根据 calibration 实测调用数外推，要求：

300-episode 单个 CP arm 的 projected wall time 不超过原先 cp_rho_h5_r95 约 1.6h/1000 episodes 所对应的量级，目标控制在约半小时级。

这比拍一个 0.25/0.5 秒数字科学得多。

如果仍太慢

按顺序只做两件事：

先：

hard priority filtering
	​


它既减规模，又使 CP/greedy 比较公平。

仍不够再考虑候选 cap。

但 K 不能拍脑袋。

从 train-only decision states 计算 competitive unique-node count 分布，取能够覆盖 ≥99% train states 的最小整数 K。

如果实际需要截断，必须报告：

candidate_truncation_rate

不要默默删 candidate。

CP 求解质量必须怎么报

每个 arm 至少保存：

OPTIMAL %
FEASIBLE %
UNKNOWN / INFEASIBLE %
fallback %
time-limit-hit %
solve_ms p50/p90/p95
relative_gap p50/p90/p95
first-action stability
candidate count
planned nodes

并保存：

objective, bestBound, gap

每次调用。

以后不能再写：

“CP 比 greedy 差，所以优化没用。”

除非 solver quality 本身已经证明足够。

“预测信息 vs 真值信息”怎么对比着看

我建议分成三层。

第一层：预测器本身

生成：

predictor_truth_comparison.csv

逐 future slot：

sample_id
slot
truth_runtime_ms
pred_mean_ms
pred_p50_ms
pred_p90_ms
pred_p95_ms
pred_bin
truth_bin
pred/truth
role
family
model

再按：

runtime band
role
family
model class

聚合。

第二层：scheduler decision trace

继续用现有 tracing：

同一个 candidate 显示：

J3 future
F0 future
F1 future
truth-H5 future
risk-v2 score
CP-v2 first-action status
rank
winner

因此负责人可以真正“一行对着看”。

第三层：Truth-Q

这个才回答：

“哪个动作在真实最终 objective 下实际上最好？”

不需要 29 万 rollout。

抽样方案

固定 300 个 dev episodes。

每 episode 最多抽 3 个 competitive decision：

early 1/3 一个；

middle 1/3 一个；

late 1/3 一个。

每个区间通过固定 hash 从 eligible decisions 中选择，不按结果选。

最多：

900 decisions.

平均约 4 candidate，对应大约：

3600

个 counterfactual rollout。

continuation policy 固定为：

冻结 J3 sameshape-H5-p95
	​


所有候选都一样。

所以：

Q(s,a)=J(force a,then frozen A0 continuation).
Truth-Q 必报

对于每个 method m：

Top1Agreement
m
	​

=P(a
m
	​

=a
Q
∗
	​

).

local regret：

Regret
m
	​

=Q(s,a
m
	​

)−
a
min
	​

Q(s,a).

此外必须报告：

P(a
truthH5
	​


=a
TruthQ
	​

)

和：

Regret
H5
	​

.

以及：

J3
F0
F1
risk_v2_pred
risk_v2_truth
G-P
G-T
CP-P
CP-T

和 Truth-Q winner 的一致率。

这会把项目负责人的问题最终回答得很完整：

predictor 预测得怎样 → scheduler 怎么用 → truth suffix 如何 → 真正 action outcome 又如何。

需求 3：实际实施顺序

我建议锁成 8 个 stage。

Stage	内容	产物/Gate	预计成本
0	新 causal input contract	leakage audit + scaler manifest	CPU 1–5 min
1	dataset-v2 hist-resource	F0/F1 数据 hash、样本数完全匹配旧集	CPU 数分钟
2	model-v2 + 单测	current-resource mutation guard PASS	<1 min
3	F0/F1 3 seeds×30ep	6 个 checkpoints	GPU 约 10–20 min
4	predictor evaluation + pack	prediction/truth report + resource-v2 artifacts	2–5 min
5	CP budget calibration + Risk-v2 smoke	solver budget frozen	CPU ~5–20 min
6	300 paired risk + 2×2	scheduler contrasts + trace	risk 数分钟；CP 约 30–60 min 总量级
7	sampled Truth-Q	~900 states / ~3600 rollout	约 10–30 min

按你们之前 RTX3060 的 J-series 实测，完整旧实验 18 个 30-epoch run 也才约 28 GPU-min；这次 6 个正式 run 加一个很小 numeric branch，不应该成为算力瓶颈。

真正耗时的是 CP，而不是训练。

可以并行什么

Stage 0–2 是阻塞项。

完成 causal contract 后：

F0/F1 predictor training；

Risk-v2 consumer；

CP-v2 solver calibration infrastructure；

Truth-Q runner；

代码可以并行开发。

但正式 scheduler run 必须等：

最终 predictor artifact + CP budget
	​


都冻结。

明确不做

本轮不要：

给 current token 喂 runtime/load/memory/status；

改整个 categorical prefix 为 events[:event_index]；

喂 raw milliseconds；

第一轮使用 local_runtime_ms；

新增手工 K-step runtime aggregate；

修改 mass-balanced 16-bin；

引入 L_decision；

用 300 smoke/trace 生成训练信号；

调 risk 权重；

sweep q95/mean/CVaR；

根据 scheduler completion 调 CP time limit；

修 legacy load double-count / eviction semantics；

做 joint runtime distribution；

把 truth-H5 称为 oracle；

打开 J test。

这份范围我建议直接写进实验 manifest。

最大风险点

我认为第一风险不是模型训练失败，而是：

过去的资源耗时对未来资源条件信息增量很小
	​


因为当前 categorical context 已经知道：

model；

action family；

role；

workflow prefix。

而你们又已经观察到同一个 summarize 内部存在约 64× 波动。这意味着很多 runtime variability 可能来自：

当前输入内容、服务随机性、系统状态，

而不是过去节点运行了多久。

如果最终：

F1≈F0

这是完全可解释、也值得报告的负结果：

historical execution telemetry adds little predictive information beyond workflow semantics under the current video-agent workload.

第二种很有价值的结果是：

F1 predictor更好但scheduler不改善。

那就进一步坐实：

prediction bottleneck 已经不是主要问题，consumer/framework 才是问题。

第三种：

CP−P≈G−P,CP−T≈G−T

且 CP gap 小、fallback 低。

这才是相对可信的：

在当前 rolling ready-window 下，复杂优化框架没有带来足够额外价值。

这些都比“又训练一个模型分数高一点”更有论文意义。

我能接受的提交前验收清单

你可以直接按这个自检：

J3_seed11 文件及 SHA 完全未改。

新数据写到全新 schema/output root，旧 dataset 不覆盖。

categorical history 与旧数据逐 token 一致。

current categorical token仍保留。

current resource mask 永远全 0。

所有可见 resource token 的 source index 严格 < current_event_index。

current-resource sentinel mutation 后 model_input byte-identical。

past-resource mutation 只改变对应 historical-resource channel。

raw runtime_ms/load_ms/memory/status 仍在 forbidden raw-key list。

scaler 只由 train split 拟合，并有 SHA/manifest。

F0/F1 行数、sample ID、labels、split hash 完全一致。

F0/F1 architecture 除 resource-history mask 外完全一致。

3 个 matched seeds 使用相同 initialization/data order。

所有 encoder/attribute/resource 参数确实可训练。

runtime 16-bin spec 与 Phase R 相同。

30 epochs fixed；不 early-stop。

原 interface NI gate 仍运行。

RuntimeQScore / calibration / crossing gate 实现固定。

F1−F0 paired video bootstrap 已实现。

packer 能输出完整 distribution + canonical views。

risk-v1 行为逐位 regression 不变。

risk-v2 只有新增 + future_norm 一个语义差异。

risk 权重固定 1/1/1/1，无 sweep。

CP-P/G-P 使用完全相同的 predicted future coefficient。

CP-T/G-T 使用完全相同的 truth-H5 coefficient。

CP-v2 candidate tier/current-cost/future-weight 对齐 greedy 设计。

CP budget 只由 train-only solver-quality calibration 决定，不读取 scheduler completion。

每次 CP 调用记录 objective/bound/gap/status/fallback/solve_ms。

2×2 contrasts 在运行前写进 manifest。

Truth-Q sampling IDs 在 rollout 前冻结。

Truth-Q continuation policy 固定为 frozen A0。

300 smoke/decision trace 没有进入任何训练 loss。

J test 仍完全未读取。

如果这 33 条全部 PASS，我会认为这套实现已经到了值得上传 GitHub让我做正式代码审查的状态。

最重要的一个设计决定我再强调一次：这轮先不要加 L_decision。 你们已经被 R1b prediction↑ / scheduling↓ 教训过一次；现在最有科学价值的是先把“过去耗时这个新信息到底有没有增量”单独测干净，再让 risk/CP 使用同一份 future coefficient 去回答“信息”和“决策框架”各自贡献多少。

====================================================================================================TURN 7  role=unknown  chars=2905====================================================================================================
# 项目负责人明确了因果边界（硬约束，请按此规划）

我上一条规划请求里把"因果边界怎么切"列为需要你裁决的首要问题。**项目负责人现在直接给了答案**，请以此为硬约束。

## 他的原话（逐字）

> 我先明确一个点
> 历史信息（包括属性和运行用时）都可以用
> 当前节点的类别可以用
> **但是耗时不可以用**
> 未来节点的所有信息**绝对不可以用**

## 我据此整理的规则（请你核验是否忠实、是否有遗漏）

对一个位于 event_index 的 anchor：

| 字段 | 历史事件（严格 < event_index） | **当前节点（== event_index）** | 未来节点（> event_index） |
|---|---|---|---|
| 属性（event_type / node_type / role / raw_action / action_family / model_id） | ✅ 可用 | ✅ **可用** | ❌ 绝对禁止 |
| runtime_ms | ✅ **可用** | ❌ **禁止** | ❌ 绝对禁止 |
| load_ms | ✅ 可用 | ❌ 禁止 | ❌ 绝对禁止 |
| 显存（peak_allocated_mb / peak_reserved_mb） | ✅ 可用 | ❌ 禁止 | ❌ 绝对禁止 |
| status_class（执行结果） | ✅ 可用 | ❌ 禁止 | ❌ 绝对禁止 |

**核心**：历史可以带"真实发生了什么"（含耗时与显存），当前节点只能带"它是什么"，未来一个字都不能带。

## 我已经实测的数据事实（供你规划引用）

**1. 全部数据在本机，不需要远端原始 trace。**
- experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/artifacts/node_table.jsonl.gz（0.39 MB）
  **15,481 行逐节点真值**，字段含 node_id / run_id / **chain_position** / step_id /
  runtime_ms / load_ms / peak_allocated_mb / peak_reserved_mb / status_class / role /
  model_class / exec_class / action_family / is_retry / merged_nested_call
- results/processed/j_series_dataset_v1/{j_train,j_validation,j_test}.jsonl.gz
- results/processed/topology_predictor_p9d_v3/{features,labels}_*.jsonl.gz
- **data/raw 是空的（0 文件）** —— 原始 trace 确实只在远端，但上面的表已经够用

**2. join 关系已实测确认**
- (run_id, chain_position) 在 node_table 中**零重复**；1,360 runs
- **13,754 / 13,754 = 100%** 的 J train 行能 join 到自己的 run
- **位置对应关系**：history[i] ↔ chain_position i−1（history[0] 是 run 容器，不在表里）
  - 偏移在 11,429 行中**完全一致**为 +1
  - 用 node_id 交叉验证：**89.1% 匹配（303/340）**
- **剩余 10.9% 不匹配**：我推测是 merged_nested_call（嵌套调用被合并）或 retry 导致
  **事件顺序 ≠ 链顺序**。**请你给出更稳的 join 规则**（按身份而非按位置？）

**3. 一条真实样例**（这就是"带真实耗时的历史"）

hist 0  run       （容器，不在 node_table）
hist 1  api_call  planner  Qwen3-4B      truth=24254.715 ms  load=4499.139  peak=7921.552
hist 2  action    tool     cpu-metadata  truth=0.104 ms      load=0.0       peak=None
hist 3  api_call  planner  Qwen3-4B      truth=7336.314 ms   load=0.0       peak=7924.981  ← 当前节点


**4. 泄漏风险已实证**：当前节点自己的真值（上面 7336.314）**就在 node_table 里**，
所以按位置 join 时**必须显式排除 chain_position == anchor_position 那一行**。请给出**自动守卫**。

## 请你补充回答的问题

Q1. 我上面的字段表**是否忠实于他的原话**？有没有我漏掉的边界（例如 is_retry、merged_nested_call、
    nested_calls 里的子事件耗时、node_id 本身）？
    特别请裁决：**过去事件内部的 nested_calls（子调用）的耗时算不算"历史"？** 可以带吗？

Q2. **"运行用时"的表示形式**（这是我上一条规划请求里的核心问题，边界明确后更关键了）：
    - 原始毫秒？log？分位数分桶？相对该组中位数的比值？
    - 还是"上 K 步同类动作的均值/中位数"这类**聚合**形式？
    - **请特别说明：直接用原始毫秒会不会让模型依赖绝对尺度、在域外崩掉？**
    - 我倾向：既然历史耗时是"已观测"的，直接给数值最诚实；但需要你判断是否需要归一化以及怎么归一化
      （按什么分组归一？会不会把信息又抹掉？）

Q3. **10.9% 的位置不匹配怎么处理？** 请给出可实现的稳定 join 规则 + 覆盖率门槛。

Q4. 边界明确后，**我上一条规划请求的其余部分（需求 1/2/3）是否需要调整**？

Constraints 不变：
- 不改冻结的 J3_seed11；J test 继续封存
- 不用已看过的 300 集 smoke / 决策追踪构造训练信号
- 只有每步边际分布，无联合分布
- 不做事后缩放校正

Required Deliverables:
- Q1 的字段表核验结果（含 nested_calls 的裁决）
- Q2 的表示形式建议（含是否需要归一化及理由）
- Q3 的 join 规则 + 覆盖率门槛
- Q4 的调整说明


====================================================================================================TURN 8  role=unknown  chars=4====================================================================================================
正在思考
