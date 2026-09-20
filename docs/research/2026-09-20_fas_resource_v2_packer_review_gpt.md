# GPT 审查：resource-v2 打包器与 artifact 契约（提交 78f000a）

- 日期：2026-09-20
- 会话：https://chatgpt.com/g/g-p-6a45fd9c56e48191a5e1a006582fcdef-diao-du/c/6aae601b-a35c-83ea-b604-04a54934d43a
- 模型：GPT-5.6 Sol + High，思考 10 分 2 秒
- 被审提交：https://github.com/QiuweiLiu/scheduler @ 78f000a
- 提交简报：.scratch/repo_review_packer_brief.md
- 原始转储：.scratch/conv_6aae601b_dump.md

## 总裁决

模型与 feature replay 地基**正确，不需要返工 Phase R**；两层 schema 思想**忠实**。
但 artifact 契约目前仍是「能用的研究脚本」，**未达到可放心做正式 paired smoke 的 fail-closed 程度**。
三个最重要的修复（都不是算法问题）：**严格完备性断言 / resource-v2 overlay loader / 打包后分布-视图不变性审计**。
消费臂**可以开始写**，正式 runner **必须等 P0 修完**。

Q2(a) 受控对比、Q2(b) res_hidden 重放等价性、Q2(c) probs/quantiles 不会新旧混用：**三项全部 VERIFIED**。
Q2(d) CVaR 公式对离散代表值分布**正确**，但 B2 必须命名为
「CVaR of the discretized representative-valued predictive distribution」。

---

==========================================================================================
我直接按提交 78f000a 审了源码、两个 manifest、bins.json、现有 simulator consumer 和相关测试。结论先给：

这版实现忠实于上一轮“两层 schema：完整分布是 source of truth + 固定泛函是 canonical views”的核心思想；R1b / R3a-U 的 res_hidden 重放路径也没有发现错配。Q2(a)(b)(c) 从代码路径上都通过。

但我不建议现在直接跑 300 集。不是模型算错，而是工程合同还缺几个 P0 防护：当前 packer 可以静默漏节点/漏 step，resource-v2 目录也不能直接交给现有 load_future_artifacts()，而且没有 post-pack 一致性 validator。消费臂本身可以开始写，但正式 runner 必须等这些修掉。

Q1. 两层 schema 是否忠实？
判定：VERIFIED — 核心忠实，但还没达到“冻结接口合同”级别。

逐项对照：

上轮建议	当前实现	判定
artifact_schema_version	manifest schema_version	✅ 语义有，建议统一命名
producer_checkpoint_sha	checkpoint_sha256	⚠️ 有值但名字模糊，应明确是 J3 backbone
resource_head_version	step 有 resource_head_id；manifest 只有 arm	⚠️ manifest 应显式增加
bin_schema_id	mass_balanced_16	✅，但 ID 不含内容 hash
bin_edges_ms	每 step 重复	✅ 数据有；位置不理想
bin_representatives_ms	每 step 重复	✅
runtime_probs[H,K]	每 step runtime_probs[K]	✅ 等价表示
q50/q90/q95	runtime_ms_quantiles	✅
mean	runtime_mean_ms	✅
CVaR95	cvar95_ms	✅，但语义需收紧
quantile_rule	有	✅
overflow_rule	有	✅
calibration_version	有	✅
runtime_semantics	conditional_on_active	✅ 很重要
consumer_compat	没有	❌
producer/head/bin/artifact hashes	checkpoint SHA 外基本没有	❌
CVaR alpha	只隐含在字符串和字段名里	⚠️ 应显式 cvar_alpha=0.95

我会调整字段层级。

artifact/manifest 级应该放：

artifact_schema_version
artifact_id
artifact_sha256
producer_commit_sha
producer_checkpoint_sha256
resource_head_id
resource_head_sha256
bin_schema_id
bin_spec_sha256
bin_edges_ms
bin_representatives_ms
runtime_semantics
quantile_rule
overflow_rule
calibration_version
cvar_alpha
cvar_semantics
consumer_compat
horizon

step 级只需要：

runtime_probs
runtime_ms_quantiles
runtime_mean_ms
cvar95_ms

最多额外保留一个：

bin_schema_id

方便脱离 manifest 检查。

现在 bin_edges_ms 和 16 个 representatives 每个 step 都重复一次，这是你从 ~6 MB 膨胀到 ~12 MB 的主要原因之一，不是逻辑错误。

还有一个接口细节：bins.json 的最后 edge 是 Python JSON 写出的 Infinity。Python 自己能读，但它不是严格 RFC JSON。建议 schema v2 冻结前改成：

JSON
"upper_edge_ms": null,
"overflow": true

或 manifest 中单独定义 overflow。

Q2. P0/P1 代码缺陷
Q2(a) “复制非 resource 字段”是否真的受控？
VERIFIED：对这个提交使用的具体 base pack，是受控的。

我专门追了 base pack 的 manifest：

outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts/
b05_artifact_manifest.json

它明确写：

producer =
scripts/pack_j_predictor_artifacts.py
(J3:seed11, --emit-distributions)

而这个 producer 在 pack_j_predictor_artifacts.py:172-176 把 runtime/load 派生值全部放进：

Python
运行
step["resource"]

外部的：

role
action_family
model_id
execution_lane
length
termination
topology...

来自结构/属性/行为头，不依赖 runtime head。

它同时在 pack_j_predictor_artifacts.py:179-185 明确只生成一个 scenario，manifest notes 也写了：

single-scenario deterministic chain (probability 1.0)

所以本次：

base → R1b / R3a-U 只替换 resource runtime 部分

这个因果对照成立。

但是 packer 本身没有把这个前提写成 assertion，见下面 P0。

Q2(b) R1b / R3a 的 hook 重放是否等价？
VERIFIED：等价，没有发现静默错配。

这是这次审计里最重要的通过项。

J3 原路径在 j_series_common.py:372-390：

Python
运行
x = cat[repr_vec, slot_vec, attr_feature]
z = tanh(res_hidden(x))
runtime = head_runtime(z)

eval 时 dropout=0。

R1b 原训练缓存的是 head_runtime 的输入 z：

j_series_resource_dist_train_eval.py:130-193

而且原脚本明确做了：

Python
运行
direct = outputs["resource"]["runtime_log_quantiles"]
replay = model.head_runtime(captured_z)

逐元素 equivalence check。

现在 packer 在 pack_resource_v2_artifacts.py:155 hook 的是：

Python
运行
model.res_hidden

得到 x，然后 R1b 做：

Python
运行
new_head(torch.tanh(model.res_hidden(x)))

也就是精确恢复：

x→z→R1b

正确。

R3a 更直接。R3a 原训练在 j_series_resource_dist_r3a.py:70-115 就是缓存 res_hidden 输入 x；训练/评价路径使用：

Python
运行
head(torch.tanh(res_hidden(x)))

保存 checkpoint 时还同时保存：

Python
运行
head
res_hidden

packer 对 R3a-U/F 使用的正是这套 state。

所以：

R1b 与 R3a 两条 pack inference path 和它们各自训练/验证时的 forward 定义一致。

这一点不需要重训。

Q2(c) probs 与 quantiles 会不会新旧混用？
VERIFIED：对被成功重写的 step，不会。

pack_resource_v2_artifacts.py:223-236：

Python
运行
p = probs[i, t]
views = derive_views(p, reps)

resource["runtime_ms_quantiles"] = ...
resource["runtime_mean_ms"] = ...
resource["cvar95_ms"] = ...
resource["runtime_probs"] = ...

而 derive_views() 在 :97 同一次 CDF 上算：

Python
运行
mean
q50
q90
q95
CVaR95

所以不存在：

probability 来自 R1b，但 q95 还沿用 J3

这种路径。

而且 j_series_resource_dist.py:621-623 的离线评价也使用同样的：

Python
运行
cdf = cumsum(probs)
mean = probs @ reps
quantile = first cdf >= tau

问题不是生成逻辑，而是没有 post-pack assertion。

我发现的 P0 / P1
级别	位置	问题	是否阻塞
P0	pack_resource_v2_artifacts.py:209-218	missing node / missing scenario 会 continue，step 超 H 会 break，全部静默发生	阻塞正式 smoke
P0	workload_v02_simulator.py:158-180	load_future_artifacts() 强制要求 H1/H3/H5；你的 resource-v2 目录只有 H5，不能直接作为 artifact root	阻塞 runner
P0	packer:219-238	没有 post-pack validator 验证 probs→q/mean/CVaR 一致、所有 step 都已升级	阻塞正式 smoke
P0	manifest :243-259	没有 immutable artifact/head/base/bin SHA，runner 无法真正冻结 predictor_artifact_id	阻塞正式 smoke 归档
P0-new	新 consumer	若 mean/CVaR 改了 runtime，同时顺手改 load consumer，就会混淆 consumer effect	实现时必须避免
P1	read_pack:72-79	duplicate node_id 会静默覆盖	可并行修
P1	packer:231-239	全局 bin metadata 每 step 重复	可并行
P1	manifest	checkpoint_sha256 不清楚是 backbone 还是 head	可并行
P1	manifest	consumer_compat 缺失	smoke 前最好补
P1	manifest	absolute F:\... 路径不可移植	可并行
P1	bin schema	mass_balanced_16 不足以唯一标识具体 edges	加 bin_spec_sha256
P1	workload_v02_simulator.py:923	_q95_step_cost docstring 写成 “CVaR-proxy”，概念不准确	可并行
P1	bins.json	Infinity 非严格 JSON	schema 冻结前改
P1	cvar95_ms	名字容易被读成连续真实分布 CVaR	增加明确 semantics

我这里特别强调：

没有发现“R1b / R3a pack 实际跑错网络”的 P0。

P0 是“现在的代码太容易静默生成一个部分正确 artifact”，不是已经发现当前模型结果污染。

P0-1 应该怎么修

把这三处：

Python
运行
if record is None:
    continue

if not scenarios:
    continue

if t >= probs.shape[1]:
    break

全部变成 fail-closed。

至少：

Python
运行
assert set(anchor_ids) == set(base_pack)
assert len(scenarios) == 1
assert len(steps) <= probs.shape[1]

结束后：

Python
运行
assert written == len(rows) == len(base_pack)
assert steps_written == base_step_count

当前 manifest 已经是：

9575 nodes
35362 steps

把它们变成后置条件，而不是日志。

P0-2：当前 resource-v2 目录不能直接喂 loader

这个比较隐蔽。

现有：

load_future_artifacts() 明确要求：

b05_node_h1.jsonl.gz
b05_future_h3.jsonl.gz
b05_future_h5.jsonl.gz

但：

outputs/resource_v2_artifacts/r1b/

只有：

b05_future_h5.jsonl.gz
resource_v2_manifest.json

所以 runner 不应该：

Python
运行
load_future_artifacts(resource_v2_root)

我建议不要复制 H1/H3 文件，增加一个 overlay loader：

Python
运行
def load_resource_v2_overlay(
    base_root: Path,
    resource_v2_h5: Path,
) -> dict[str, dict[str, Any]]:

语义：

base = load_future_artifacts(base_root)

单独读 resource-v2 H5

assert node set 完全相同

对每个 node 只替换 future_h5

assert 所有非-resource 内容与 base H5 相同

跑 resource-v2 validator

这样最符合“只换资源头”。

Q2(d). CVaR 实现对不对？
VERIFIED：公式对“16 个 representative 构成的离散分布”是正确的。

pack_resource_v2_artifacts.py:82-95 正确处理了 crossing bin 的部分质量。

即：

CVaR
0.95
	​

=
0.05
1
	​

k
∑
	​

w
k
	​

m
k
	​


其中仅取 CDF 95% 以上的 mass。

这是合法的 discrete expected-shortfall。

但它不等于“真实连续 runtime 分布的精确 CVaR”。

overflow 的情况

最后一格：

[24053.05ms,∞)

训练样本 242 个，占：

0.5006%

代表值：

m
overflow
	​

=25020.71ms.

j_series_resource_dist.py:232-243 的 representative 是该 bin 训练样本的经验均值。

因此：

对 train empirical mean

overflow 这一格压缩成 mean：

一阶矩偏差=0
	​


这是好的。

但 deployment 时若真实 overflow 条件均值是 μ
o
	​

，预测给它的概率是 p
o
	​

，mean 的 tail contribution bias：

p
o
	​

(25020.7−μ
o
	​

)

而 CVaR95 中，如果整个 overflow 落在 top 5%：

0.05
p
o
	​

	​

(25020.7−μ
o
	​

)

例如：

p
o
	​

=0.005,μ
o
	​

=30000ms

则 CVaR 每 step 大约会低估：

0.05
0.005
	​

(30000−25020.7)≈498ms.

而且因为这个 bin 上界是 ∞，理论最坏误差无界。

所以 B2 应该被称为：

CVaR of the discretized representative-valued predictive distribution

不是未经限定的“真实 runtime CVaR”。

Q3. 两个新消费臂的精确规格

这里一定要注意一个代码事实：

现有 _q95_step_cost() 不只是 runtime q95。

它实际上是：

runtime
p95
	​

+1
GPU
	​

1
P(load)≥0.5
	​

load
p95
	​

.

所以如果 B1 改成 runtime mean 时顺便改变 load 规则，B1−A1 就不再只是“消费泛函变化”。

先抽出原 load 规则

建议：

Python
运行
def _legacy_p95_load_cost(
    step: Mapping[str, Any],
) -> float:
    ...

完全复制当前 _q95_step_cost:930-938 的 load 部分。

然后现有 q95 重构成：

Python
运行
def _q95_step_cost(
    step: Mapping[str, Any],
    train_stats: Mapping[str, Mapping[str, Any]],
) -> float:
    runtime = _strict_runtime_q95_or_legacy_fallback(...)
    return runtime + _legacy_p95_load_cost(step)
B1
Python
运行
def _mean_step_cost(
    step: Mapping[str, Any],
    train_stats: Mapping[str, Mapping[str, Any]],
) -> float:

严格语义：

C
h
	​

=E[T
h
	​

∣active]+L
p95,h
legacy
	​

.

runtime_mean_ms 缺失/NaN/负值时直接 raise。

不要像 legacy q95 那样 fallback 到 _step_estimate_cost()。

因为实验 arm 应该 fail-closed，否则一个缺字段就偷偷变成另一套 consumer。

我建议 policy key：

sameshape_h5_condmean

正式 consumer ID：

sum_conditional_runtime_mean_plus_legacy_load_v1

这里不用直接写 sum_mean，因为你的 artifact 已经明确：

runtime_semantics = conditional_on_active

所以它严格来说是：

emitted predicted chain 上各 step 的 conditional mean 之和，

不是把 topology/termination uncertainty 也积分进去后的完整 horizon expectation。

B2
Python
运行
def _cvar95_step_cost(
    step: Mapping[str, Any],
    train_stats: Mapping[str, Mapping[str, Any]],
) -> float:

定义：

C
h
	​

=CVaR
0.95
disc
	​

(T
h
	​

∣active)+L
p95,h
legacy
	​

.

policy key：

sameshape_h5_stepcvar95

consumer ID：

sum_marginal_step_cvar95_plus_legacy_load_v1
论文必须写成

“sum of marginal step-wise CVaR
0.95
	​

 scores under the discretized per-step predictive distributions”

或中文：

各预测未来步边际 CVaR95 的加和风险分数

绝对不要写：

CVaR
0.95
	​

(
h
∑
	​

T
h
	​

).

你现在没有 joint distribution。

因此：

B2=
h
∑
	​

CVaR
.95
	​

(T
h
	​

)
	​


不是：

CVaR
.95
	​

(
h
∑
	​

T
h
	​

)
	​

.
sameshape_future_cost() 扩展

当前 workload_v02_simulator.py:952-974 可以改成：

Python
运行
_SAMESHAPE_PREDICTED_STATS = (
    "p50",
    "p95",
    "condmean",
    "stepcvar95",
)

然后：

Python
运行
if stat == "p50":
    ...
elif stat == "p95":
    ...
elif stat == "condmean":
    return sum(_mean_step_cost(...) for step in steps[:horizon])
elif stat == "stepcvar95":
    return sum(_cvar95_step_cost(...) for step in steps[:horizon])
六臂的 immutable identity

Runner 不要只写 policy="p95"。

应该记录三个概念：

Arm	predictor_artifact_id	consumer_policy_id	simulator policy
A0	j3_seed11@<base_sha256>	legacy_sum_q95_v1	sameshape_h5_p95
A1	r1b_seed11@<artifact_sha256>	legacy_sum_q95_v1	sameshape_h5_p95
A2	r3a_u_seed11@<artifact_sha256>	legacy_sum_q95_v1	sameshape_h5_p95
B1	same R1b	sum_conditional_mean_v1	sameshape_h5_condmean
B2	same R1b	sum_marginal_step_cvar95_v1	sameshape_h5_stepcvar95
O	oracle_truth_v03	samekey_true_future_h5_v1	sameshape_h5_truth
Q4. O 臂怎么做？
判定：VERIFIED：不要用 legacy trueopt_h5。仓库里其实已经有更合适的现成 O。

你现在已经有：

sameshape_h5_truth

见 workload_v02_simulator.py:944-974。

它的注释就是：

identical current cost, chain/score key order; only future statistic differs

而 tests/test_sameshape_consumer.py 已经对这个 same-shape family 做测试。

Phase 20B 也已经把它当成：

oracle-truth（同一消费形状 + 真实未来）

跑过。

所以 300 集里 O 不需要重跑预测器，也不用造新的模型 artifact。

但必须把它准确命名为：

joint-future-truth headroom
	​


而不是：

resource-head oracle
	​


因为 sameshape_h5_truth 的 future cost 来自：

Python
运行
limited_future_truth_cost(...)

它沿着真实 successor walk。

而 A0/A1 沿着 predicted identity-free chain。

因此 O 同时消除了：

future runtime error；

future topology/length error。

所以可以说

R1b 距离“完美未来信息”还剩多少系统 headroom。

不能说：

R1b resource head 距离 perfect runtime 还剩多少。

如果以后真要 pure-resource oracle，就需要一个 slot-aligned truth sidecar；当前 artifact 因为 predicted future step 没有真实 node identity，不存在无歧义的 resource-only oracle mapping。

这次 smoke 没必要为此继续扩实验。

Q5. 300 集 runner 精确规格

我直接检查了 v03 workload。

1000 个 episode 实际有：

135

个 cell：

3 arrival×5 load×3 GPU topology×3 initial state.

具体边际是：

arrival: alibaba_replay / poisson / burst
load:    0.50 / 0.70 / 0.85 / 0.95 / 1.05
GPU:     32760+32760 / 24576+24576 / mixed
state:   cold / warm / skewed

deadline multiplier 在当前 workload 中与 state 配对：

cold   -> 1.5
warm   -> 2.0
skewed -> 3.0
300 episode 抽样

不要简单拿 validation_000000:000299。

预注册：

每个 135 cell 先固定取 2 个：

270

再选 30 个 cell，各取第三个 episode。

这 30 个 extras 必须满足：

每种 arrival +10

每个 load +6

每种 GPU topology +10

每种 state/deadline +10

同一 cell 内按 episode_id 排序。

cell 选择用固定 seed/hash；生成一次 resource_v2_smoke_v1_episode_ids.txt 后永久冻结并记 SHA256。

于是 300 集边际恰好：

arrival: 100 / 100 / 100
load:     60 / 60 / 60 / 60 / 60
GPU:     100 / 100 / 100
state:   100 / 100 / 100

这个比随机抽 300 干净得多。

Runner 不要调用现有 run() 一次跑六 policy

因为 workload_v02_simulator.py:4026+ 的：

Python
运行
run(...)

没有 per-policy artifact binding。

你六个 arm 用的是不同 artifact。

正确 runner 应直接调用：

Python
运行
simulate_episode(
    episode,
    templates,
    policy,
    future_artifacts=arm_artifacts,
    train_stats=train_stats,
    collect_events=...,
)

其现有签名在 :3347-3360 已经支持。

建议 runner registry：

Python
运行
ARMS = {
    "A0": {...},
    "A1": {...},
    "A2": {...},
    "B1": {...},
    "B2": {...},
    "O":  {...},
}

每个 episode × 每个 arm 单独调用，因此天然 paired。

预注册 contrasts

Primary 沿用 Phase20 已经冻结的：

不要新造总分。

固定 contrasts：

A1−A0

= predictor effect。

A2−A1

= 解冻 mapping 的附加效应。

B1−A1

= mean consumer effect。

B2−A1

= step-CVaR consumer effect。

A1−O

= remaining joint-future-information headroom。

全部：

paired episode bootstrap
B = 2000
cluster = episode_id
non-inferiority margin 从哪里来？

仓库目前没有已经冻结的 NI margin。

所以不能假装它已经存在。

我的推荐是从“此前已冻结的系统收益”派生

Phase20B 已经在 dev700 看到：

p95 − E0 = −4853 ms

把“最多允许丢回原收益的 10%”作为 practical margin：

δ
NI
	​

=0.1×4853=
485 ms
	​


优点是：

完全来自本次 Phase R 之前的数据；

与原 scheduler 已确认的实际 benefit 挂钩；

不是看到新结果后拍出来的。

这是一个治理选择（INFERENCE），不是数学定理。项目负责人应在运行 300 集之前签字冻结 10% 这个比例。

建议定义：

non-inferior
CI
upper
95
	​

(Δ)<+485ms
material improvement
CI
upper
95
	​

(Δ)<0

且

Δ≤−485ms.
practically saturated against O

若：

CI
upper
95
	​

(A1−O)≤485ms

则可以说：

在当前 scheduler 下，与 joint-future-truth ceiling 的剩余性能空间小于预注册 practical margin。

注意因为 O 是 joint truth，若 gap 大，不能自动归罪于 resource head。

必须保存的 runner 字段

run-level 至少：

experiment_id
git_commit
simulator_sha256
workload_file_sha256
template_file_sha256
episode_selection_version
episode_list_sha256
bootstrap_seed
bootstrap_B

arm_id
predictor_artifact_id
predictor_artifact_sha256
consumer_policy_id
simulator_policy_key

producer_checkpoint_sha256
resource_head_id
resource_head_sha256
bin_schema_id
bin_spec_sha256
resource_schema_version

每 episode：

episode_id
scenario_cell
arrival_pattern
realized_offered_compute_load
gpu_topology_mb
initial_state
deadline_multiplier

arm_id
mean_completion_ms
p95_completion_ms
makespan_ms
mean_job_queue_ms
deadline_miss_rate
gpu_evictions
gpu_utilization
failed_jobs
completed_jobs

artifact preflight 还要存：

node_count
step_count
missing_nodes
duplicate_nodes
multi_scenario_rows
prob_sum_max_abs_error
canonical_view_max_abs_error
nonresource_mismatch_count
bin_schema_mismatch_count

这些必须全部：

0 / within tolerance

才允许启动 300 episode。

另外 R3a-U 已经违反原 calibration integrity gate，所以 A2 要标：

eligible_for_model_selection = false
role = diagnostic_only

即使它 scheduler 偶然最好，也不能偷偷把它变成最终 winner。

Q6. 现在能不能开始写消费臂？
判定：可以写；不能正式跑。

我会给 Codex 的执行顺序是：

先并行完成两件事：

写 condmean / stepcvar95 consumer；

修 pack/loader/validator 的 P0。

但正式生成 smoke 结果之前必须全部满足：

[PASS] anchor node set == base-pack node set
[PASS] exactly one scenario per H5 row
[PASS] every emitted step upgraded
[PASS] no silent fallback
[PASS] probs -> q50/q90/q95/mean/CVaR rederive match
[PASS] non-resource fields unchanged
[PASS] artifact/head/bin/base SHA frozen
[PASS] overlay loader works
[PASS] sameshape_h5_p95 regression test remains bit-for-bit identical

然后再跑 300×6。

所以我对提交 78f000a 的最终裁决是：

模型与 feature replay 地基是对的，不需要返工 Phase R；接口思想也是对的。但 artifact contract 目前还是“能用的研究脚本”，还没到“可以放心做正式 paired smoke”的 fail-closed 程度。

最重要的三个修复不是算法，而是：

strict completeness assertions
	​

resource-v2 overlay loader
	​

post-pack distribution/view invariance audit
	​


修完这三个
