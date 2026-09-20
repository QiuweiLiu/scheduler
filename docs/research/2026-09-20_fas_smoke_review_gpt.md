# GPT 审查：resource-v2 六臂 smoke（提交 eb88a7b）

- 日期：2026-09-20
- 会话：https://chatgpt.com/g/g-p-6a45fd9c56e48191a5e1a006582fcdef-diao-du/c/6aae601b-a35c-83ea-b604-04a54934d43a
- 被审提交：https://github.com/QiuweiLiu/scheduler @ **eb88a7b**（完整 SHA `eb88a7bfb3f77c5ac6c45174d95c239fcc1eac2e`）
- 读取状态：**13/13 全部成功**（raw 通道被 `DisabledError` 挡住，改用 GitHub 连接器直接按 commit 读取；
  另外它自己下载并解析了完整的 1,800 行 `per_episode.jsonl`）
- 独立复算：A1−A0 = **+796.5545 ms**、B2−A1 = **−427.6650 ms**、O−A1 = **−2339.6176 ms** —— 与仓库一致

## 一句话结论（它给的最终表述）

> 这次结果**没有**证明"越保守越好"，而是证明了：**提高概率预测质量不足以保证下游收益**，
> 因为 predictor replacement 同时改变了 scheduler 实际消费的 future-cost statistic 的
> **尺度、当前/未来相对权重、以及可能的局部候选排序**。
> 下一轮 full-unfreeze 应**保留 calibrated distribution 目标**，同时**显式学习一个 decision-aligned
> scheduler score**，并用**真实 scheduler 结果**作为最终 gate。

---

==========================================================================================
可以。先把读取状态说清楚。

你给的 13 条 raw.githubusercontent.com 直链，我这里的普通网页抓取通道都返回了 DisabledError，所以我没有假装通过 raw URL 读到它们。随后我改用 GitHub 连接器，直接按 QiuweiLiu/scheduler@eb88a7b 读取同一批文件；13/13 全部成功，commit 也核对为完整 SHA eb88a7bfb3f77c5ac6c45174d95c239fcc1eac2e。你上传的 bundle 中也记录了同一 commit 和文件清单。

review_bundle_eb88a7b

逐个状态如下。

scripts/pack_resource_v2_artifacts.py：raw 通道失败；GitHub commit 文件读取 成功，27,165 B。

scripts/resource_v2_preflight.py：raw 失败；GitHub 成功，5,203 B。

scripts/resource_v2_scheduler_smoke.py：raw 失败；GitHub 成功，22,122 B。

tests/test_sameshape_consumer.py：raw 失败；GitHub 成功，16,929 B。

src/tracing/analysis/workload_v02_simulator.py：raw 失败；GitHub 成功，184,163 B；我重点审了你指定的新符号及其 scheduler 调用路径。

.../REPORT.md：raw 失败；GitHub 成功。

.../artifacts/contrasts.json：raw 失败；GitHub 成功。

.../artifacts/mechanism_conservatism.json：raw 失败；GitHub 成功。

.../artifacts/episode_selection.json：raw 失败；GitHub 成功。

.../artifacts/overlay_preflight.json：raw 失败；GitHub 成功。

.../artifacts/manifest_r1b.json：raw 失败；GitHub 成功。

.../artifacts/validation_r1b.json：raw 失败；GitHub 成功。

docs/research/2026-09-20_fas_resource_v2_packer_review_gpt.md：raw 失败；GitHub 成功。

此外我还直接读取并解析了你没要求强制读取的完整 per_episode.jsonl：1,800 行、300 episode × 6 arm，以及 run_meta.json、manifest_j3.json、manifest_r3a_u.json。我自己重算出的 A1−A0=+796.5545 ms、B2−A1=-427.6650 ms、O−A1=-2339.6176 ms 与仓库结果一致。仓库提交的 contrasts 数值见 

review_bundle_eb88a7b

。

下面是正式审查。

Q1. 代码正确性
判定：VERIFIED：本次 300×6 结果没有发现会使结果失效的代码污染；但下一轮之前仍有 5 个 P0。

上一轮五个基础 P0 基本已经修掉。当前 artifact 实测节点 9,575、step 35,362，节点集、scenario、升级完整性、非 resource 漂移、bin schema、load 字段都通过，R1b canonical-view 重推误差约 3.64×10
−12
。

review_bundle_eb88a7b

但我发现的新问题如下。

级别	文件:行	问题	本次结果	下一轮
P0	pack_resource_v2_artifacts.py:393-397	缺失/NaN canonical view 可能被 max(..., nan) 静默吞掉	当前 artifact 完整，未触发	阻塞
P0	workload_v02_simulator.py:380-383	overlay 的 probs 检查允许 NaN/Inf/bool/负概率部分漏过	当前 softmax artifact 正常	阻塞
P0	workload_v02_simulator.py:310-313	loader 只是读取 manifest 中 SHA，没有重算 artifact/base SHA	当前未发现篡改	阻塞正式复现
P0	resource_v2_scheduler_smoke.py:392-415	第二轮 A1-A0 写入覆盖第一轮 contrast，导致正式 verdict 丢失	点估计/CI 不受影响	阻塞
P0	resource_v2_scheduler_smoke.py:364-415	pairing 用 episode set 的交集；某臂少 episode 会静默缩样本	当前恰好 300×6	阻塞
P1	packer 370-371 vs overlay 370	preserved load field 的严格程度不一致	无影响	并行修
P1	packer vs overlay	canonical view tolerance 分别 1e-9 / 1e-6	当前远低于两者	并行
P1	runner 329-348	bootstrap 没维持原 135-cell 分层	不推翻 smoke	建议 stratified bootstrap
P1	runner 348	prob_le_zero 容易被误读成后验概率/p-value	无	改名
P1	runner 202-221	n_extra 参数实际未控制 extra 数量	默认 30 时无影响	并行
P1	tests 223+	没显式测试合法整数/0	无	补测
Q1(a) 两个 validator 有分歧吗？

VERIFIED：有。

最值得修的是 post-packer：

Python
运行
view_err = max(
    view_err,
    abs(float(quantiles.get(key, np.nan)) - want)
)

如果 key 根本没有，表达式产生 NaN；Python 的 max(0.0, nan) 可以继续留下 0.0。因此：

packer validator 可能把一个缺 canonical view 的 artifact 判为正常。

overlay loader 反而显式通过 optional_number() 把缺值变成 inf，所以会拒绝。

也就是说确实存在：

validate_post_pack PASS → load_resource_v2_overlay FAIL

的理论路径。

另外 packer 对 PRESERVED_RESOURCE_KEYS 是无条件要求存在；overlay 是“base 原来有才要求保留”。后者更正确。这属于“packer 可能过严”。

建议最终不要维护两套 validator，把：

validate_resource_v2_step()
validate_resource_v2_record()

抽成公共函数，packer 与 loader 调同一实现。

Q1(b) _required_runtime_field() 会误伤合法 0 / int 吗？

VERIFIED：不会。实现正确。

workload_v02_simulator.py:1205-1224 是：

Python
运行
if not isinstance(value, (int, float)) or isinstance(value, bool):
    raise

if not finite:
    raise

if value < 0.0:
    raise

因此：

0 ✅

0.0 ✅

integer runtime ✅

positive float ✅

bool ❌

NaN/Inf ❌

negative ❌

这正是我希望的 fail-closed 语义。相关代码与设计说明见 

review_bundle_eb88a7b

。

只是测试建议增加：

Python
运行
0
0.0
1

确保以后没人把 <0 改成 <=0。

Q1(c) bootstrap / contrasts / verdict

VERIFIED：paired bootstrap 核心是对的，但 report verdict 有 bug。

代码实际先构造每个 episode：

Δ
e
	​

=M
left,e
	​

−M
right,e
	​


再对这 300 个 paired delta bootstrap mean。

这是对的。

我从完整 per_episode.jsonl 重算得到：

A1-A0  +796.5545
A2-A1  +294.8472
B1-A1   +45.3816
B2-A1  -427.6650
O-A1  -2339.6176

全部一致。

但是有两个问题。

第一，A1-A0 的 verdict 被覆盖了。第一次 loop 在 :400 写：

Python
运行
stats["verdict"] = ...

第二个 *-A0 loop 又在 :407-415 用没有 verdict 的 stats 覆盖同 key。这就是为什么最终 contrasts.json 中 A1-A0 没有 verdict，而 A2-A1 等有。

review_bundle_eb88a7b

第二，现有 verdict 本身缺一个“worse”分支：

Python
运行
material_improvement
non_inferior
else inconclusive

对于：

+796.6[471.0,1104.7]

它只会变成 inconclusive。

正确逻辑建议：

Python
运行
if ci_high < 0 and point <= -margin:
    material_improvement
elif ci_high < margin:
    non_inferior
elif ci_low > margin:
    materially_inferior
elif ci_low > 0:
    statistically_worse
else:
    inconclusive

这里尤其注意：

你目前可以说：

A1 显著比 A0 差；点估计 +796.6 ms 超过 485 ms practical margin；并且不满足 non-inferiority。

但不能说：

95% 置信下退化幅度超过 485 ms。

因为：

CI
low
	​

=471<485.

prob_le_zero 数学计算本身没错，它只是：

2000 个 bootstrap replicate 中，mean delta≤0 的比例。

不要把字段展示成 P(Δ≤0)，因为它不是 Bayesian posterior probability，也不是标准 p-value。

改成：

bootstrap_fraction_mean_le_zero

即可。

Q1(d) balanced_extra_cells()

VERIFIED：当前 v03 确实满足四组精确边际。

代码检查了 3 arrival × 5 load × 3 GPU × 3 state 且共 135 个 cell；由于 cells 是 dict keys，所以 135 个都是 unique，这已经等价于完整 factorial。

实际结果也精确得到：

arrival：100/100/100；

load：60×5；

GPU：100/100/100；

state：100/100/100。

review_bundle_eb88a7b

隐藏假设只有：

scenario_cell 永远严格是四段 arrival|load|gpu|state；

v03 始终是完整 3×5×3×3 factorial；

字符串排序被当作固定设计次序。

这些都被当前代码显式/隐式锁住，因此当前实验没问题。

但 select_episodes(..., n_extra=...) 中的 n_extra 实际没有传进 construction，是 P1 API bug。

Q2. “调度器消费的是保守性，不是分布精度”成立吗？
判定：INFERENCE，当前措辞过强。

这里我和你现在的怀疑一致。

当前数据真正证明的是：

① scheduler-facing score 的 scale/distribution 改变了：VERIFIED

平均未来 score：

B2=48020>A0=37885>A1=36560>A2=32702>B1=15079.

review_bundle_eb88a7b

同时 R1b/J3 p95 ratio：

median=0.938,

但分布很宽：

p10=0.775,p90=1.444,

而只有 61.5% 节点 <1。

review_bundle_eb88a7b

所以绝对不是：

R1b≈0.938×J3

这种统一“降低保守性”。

这是一个异质性的 score transformation。

② candidate pairwise ordering 是否改变：UNVERIFIED

mechanism_conservatism.json 没有测：

同一个 decision state 内 A0/A1 Kendall；

pairwise flip rate；

top-1 candidate disagreement；

decision margin crossing。

所以现在不能说：

“排序没变，只是 scale 变了。”

相反，0.775～1.444 这种高度非均匀变换，很可能改变 candidate ordering。

但要测才能下结论。

③ uncertainty calibration 改善：VERIFIED，但它不是已证明的 scheduler mechanism

R1b calibration 比 J3 好，这个预测结论成立。

但 scheduler 根本不直接读 calibration error。

它只读：

f(P)=q95,E[T],CVaR,…

所以“calibration 好”只有在改变 consumer-facing scalar 后才可能影响决策。

④ “越保守越好”：被当前数据否定为一般规律

你自己的 B1 就是反例：

B1/A1 p95 scale≈0.372

但 B1 仍略优于 A1。

review_bundle_eb88a7b

更关键的是，我直接从完整 1,800 行 per_episode 重算三个 p95 arm 的 episode 排序：

A0 < A1 < A2 : 83 / 300
A0 < A2 < A1 : 69
A1 < A0 < A2 : 49
A1 < A2 < A0 : 32
A2 < A0 < A1 : 32
A2 < A1 < A0 : 35

六种 permutation 全都出现。

所以：

Spearman=-0.8

其实只是5 个 arm aggregate means 上的描述性相关。

三个 p95 arm 的平均排序一致，不代表 episode-level mechanism 被证明。

更关键的代码机制：score scale 实际改变了 scheduler 的“隐式权重”

看 workload_v02_simulator.py:3017-3038，决策 key 里包括：

Python
运行
current + future,
future,
...

所以一旦把 future score 从：

15k→36k→48k

变化，它不仅仅“更保守”。

它还改变了：

future cost 相对 current cost 的隐式权重
	​


这实际上比“保守性”更准确。

所以 B2 变好可能来自：

tail information 更合适；

scale 更合适；

当前/未来 cost trade-off 被重新加权；

candidate ordering 改了；

目前不能拆开。

我建议论文采用的机制表述

英文可以写：

Replacing the resource predictor changed not only distributional accuracy but also the geometry of the scheduler-facing future-cost scores. Under the fixed legacy ∑q
0.95
	​

 consumer, R1b induced heterogeneous per-node score changes rather than a uniform rescaling. At the aggregate arm level, larger future-cost scores coincided with better completion time among the three p95 variants, and using R1b’s more tail-sensitive marginal-CVaR score recovered part of the scheduling loss. However, this relationship was neither monotonic across all consumers nor consistent across episodes. We therefore interpret the result as evidence of a prediction–consumption mismatch—particularly a mismatch in the scale and local ordering geometry of the consumed decision statistic—rather than evidence that more conservative predictions are intrinsically better for scheduling.

中文版：

新的资源分布头虽然提高了概率预测质量，但同时改变了调度器实际消费的未来代价分数的尺度、相对权重以及可能的候选局部排序。当前结果支持“预测分布质量与下游决策统计量存在错位”，但不支持“预测越保守，调度越好”这一一般性结论。

这版我认为可以放论文。

Q3. A1−A0 还能不能称“预测器效应”？
判定：VERIFIED：可以称 predictor replacement effect；不能称 prediction-quality effect。

A0 与 A1：

scheduler code 相同；

consumer ID 都是 legacy_sum_q95_v1；

workload 相同；

load rule 相同；

唯一替换的是 predictor artifact。

因此：

A1−A0

是一个非常干净的：

end-to-end predictor replacement effect under a fixed consumer

结果就是：

+796.6ms.

但它不是：

“21% pinball improvement 的因果效应”。

因为 predictor 替换同时改变了：

parameterization；

quantile extraction；

marginal score scale；

local ordering；

uncertainty shape。

所以 REPORT 中“not a clean predictor-quality contrast”是对的。

review_bundle_eb88a7b

是否必须把两个 p95 按完全相同口径重算？

不必须。

它们语义上已经都是：

Q
0.95
	​

(T∣x).

一个是 direct quantile head，一个是 discrete-CDF inversion，这个差异本来就是 predictor design 的一部分。

但如果你的研究问题变成：

“为什么 A1 会更差，是 scale 还是 ranking？”

那么需要新增机制实验。

我最推荐的零训练最小实验

先不要再动模型。

记录每个 scheduler decision：

episode_id
decision_id
candidate_node_ids

A0_current
A0_future
A0_total

A1_current
A1_future
A1_total

chosen_A0
chosen_A1

truth_future_cost

计算：

Kendall
τ
	​

(A0,A1)

within decision state；

以及：

pairwise_flip_rate
top1_disagreement_rate
margin_crossing_rate
score-ratio distribution
decision disagreement -> downstream delta

这一步就能回答：

score scale changed?

和：

local ordering changed?

如果还要进一步分开 scale 与 ordering

做一个纯诊断、train-only 的 monotone quantile mapping：

g=F
J3
−1
	​

∘F
R1b
	​

.

把：

q95
R1b
→g(q95
R1b
)

使 R1b 的 marginal scheduler-score distribution 与 J3 对齐，但因为 g 单调，R1b 自己的 ranking 完全不变。

然后比较：

A0 = J3 native
A1 = R1b native
A1-M = R1b rank preserved + marginal score matched to J3

如果 A1-M 接近 A0：

score scale/distribution 是主因。

如果 A1-M 仍接近 A1：

conditional/local ranking geometry 是主因。

这只用于机制诊断，不作为最终部署校正，也不是“p50 乘系数”。

不需要重训任何现有模型。

Q4. 整体解冻后该怎么训？
Q4(a) pinball/NLL 被否定了吗？
VERIFIED：没有。

本次否定的是：

“predictive score 更好 ⇒ scheduler 自动更好”。

不是：

“proper probabilistic loss 没价值”。

R1b 的：

RuntimeQScore −21%；

calibration 改善；

log-MAE 改善；

都是真实进展。

review_bundle_eb88a7b

所以完整分布仍然应该用 proper loss 训练。

Q4(b) 直接优化 Σq95 / ΣCVaR95？
INFERENCE：不建议作为唯一目标。

原因正是这次 smoke：

Σq95 的 scale 本身已经成为 scheduler 的隐式参数。

而：

∑CVaR
h
	​


也不是：

CVaR(∑T
h
	​

).

代码现在对这个边界写得是正确的。

review_bundle_eb88a7b

如果你直接为了调度把整个 distribution 扭到一个 heuristic score 上，很可能得到：

调度更好，但 distribution 不再 calibrated。

这会损失你 Phase R 最有价值的成果。

我推荐的 Full-Unfreeze 目标

最好的做法是把“概率预测”和“调度消费量”拆开。

保留：

P(T
h
	​

∣x)

作为 calibrated distribution。

额外增加一个 scheduler-facing head：

s
h
sched
	​

≥0.

明确命名：

scheduler_runtime_score_ms

不要叫 q95、CVaR 或 mean。

最终：

S
j
	​

=
h=1
∑
H
	​

s
j,h
sched
	​

+legacy load.
总损失

结构/内容/行为仍保留：

L=L
structure
	​

+L
content
	​

+L
behavior
	​

+L
R
	​

.

资源部分：

L
R
	​

=L
dist
	​

+0.25L
horizon
	​

+0.25L
decision
	​

	​


建议所有三项先除以初始 train-only moving mean 归一化，避免数值单位支配梯度。

1. L_dist

继续使用当前已经证明有效的：

L
dist
	​

=NLL+0.5RPS+0.25L
pseudoHuber(mean)
	​

+0.25CE
band
	​

.

它负责：

distribution 仍然可信。

2. L_horizon

让新的 scheduler score 学实际 H5 future cost：

S
^
j
	​

=
h
∑
	​

s
j,h
sched
	​

S
j
∗
	​

=realized H5 future runtime cost.

损失：

L
horizon
	​

=SmoothL1[log(1+
S
^
j
	​

),log(1+S
j
∗
	​

)].

这直接解决你现在的：

单步预测不错，但聚合 consumer 不一定有正确尺度。

3. L_decision

不要重新搞“所有节点全局 Spearman”。

必须是scheduler decision-state 内的 ranking。

对同一 decision state 候选 i,j：

y
ij
	​

=sign(C
j
∗
	​

−C
i
∗
	​

)
L
pair
	​

=w
ij
	​

log[1+exp(−
τ
y
ij
	​

(
C
^
j
	​

−
C
^
i
	​

)
	​

)].

其中：

C
^
=current+
S
^
+load.

w
ij
	​

 用真实 cost gap 加权，近乎 tie 的 candidate 权重低。

这和之前被降级的 global Spearman 完全不同：

之前问“所有 runtime 排名怎样？”

这里问：

“scheduler 此时正在二选一/多选一，它有没有把更便宜的候选排前面？”

这才是 downstream 对齐。

数据纪律

这一步非常重要：

不能拿本次 300 smoke episode 去构造 decision loss。

因为你已经看到它的结果，并据此设计了 loss。

decision-state training 必须来自：

scheduler train/dev-training split；

或从 training templates 按冻结的 v03 workload generator 新生成一批 train-only episodes。

J test 必须继续封存。

Full-Unfreeze 的预注册成功门

我建议现在就冻结。

Prediction integrity

必须：

RuntimeQScore<845.0

即不能退回 J3 以下；

calibration error≤0.0501

沿用原 integrity tolerance；

quantile crossing = 0。

我不要求一定保持 R1b 的 667，因为 decision-aware training 可以允许少量 prediction-side trade-off。

Development scheduler gate

冻结 consumer：

sum_scheduler_runtime_score_v1

Primary 仍然：

mean_completion_ms

相对 A0：

Δ=New−A0.

沿用：

δ
NI
	​

=485ms.

Viability：

CI
high
	​

(Δ)<+485ms.

Strong：

CI
high
	​

(Δ)<0

并且：

point(Δ)≤−485ms.

也就是：

统计上更好，而且 effect size 至少达到此前已经冻结的 practical margin。

最终 test gate

这 300 集现在已经被你反复看过，并用于决定：

full-unfreeze；

decision-aware loss；

consumer mismatch hypothesis。

因此它们只能继续作为开发验证集。

真正论文级结论必须：

全部 hyperparameter / loss / consumer 锁死；

只保留一个 winner；

打开 J test split 一次；

复用完全相同的 primary metric / contrast / 485-ms rule；

不根据 test 再调模型。

Q5. 全解冻训练前必须修什么？
VERIFIED

阻塞项：

修 validate_post_pack() 的 missing/NaN canonical-view 漏洞。

overlay probabilities 加：

not bool
finite
0 <= p <= 1

overlay loader 现场重算：

artifact_sha256
base_pack_sha256

与 manifest 比较。

修 A1-A0 contrast overwrite。

runner 强制：

每臂 N=300
no duplicate episode_id
exact same episode set
finite primary metric

冻结新的：

scheduler_runtime_score_v1
decision-state training protocol

然后才开始 full-unfreeze。

我还建议把 provenance 一并修。

当前 manifest_r1b.json 写：

producer_commit_sha = 78f000a...

review_bundle_eb88a7b

但真正包含 float64 validator 等最终逻辑的是后来的工作树/eb88a7b。这是典型的：

uncommitted code 生成 artifact → commit 后 manifest 指向旧 HEAD。

下次最好 artifact 只允许在 clean git tree 上生成，并保存：

producer_commit_sha
packer_source_sha256
git_dirty = false

这个我也会在正式新实验前修。

Q6. “相对 O 还有 2339.6 ms headroom”还能写吗？
判定：VERIFIED，但只能叫 truth-informed joint-future headroom，不能叫 resource-head headroom，也不能叫严格 oracle bound。

观测值成立：

A1−O=+2339.6ms

95% paired-bootstrap：

[1860.2,2856.8].

review_bundle_eb88a7b

所以能写：

在 300 集开发 smoke 中，以同一 scheduler key 形状使用真实 H=5 后继执行信息时，平均完成时间相对 A1 降低 2339.6 ms，表明当前预测系统距离 truth-informed future-information reference 仍存在显著性能空间。

不能写：

“resource predictor 还有 2339.6 ms 提升空间。”

因为 O 同时消除了：

runtime 预测误差；

topology/suffix error；

future length error。

报告自己已经正确提醒这一点。

review_bundle_eb88a7b

更不能写：

“O 是理论 oracle lower bound / ceiling”。

从 simulator limited_future_truth_cost() 看，它仍然只是：

H=5；

greedy scheduler；

相同 key structure；

current cost 仍按原逻辑；

部分 memory/residency 判断仍调用 train_stats。

而且我直接从 300 paired episodes 重算：

O 在约 76% episode 优于 A1，但在约 24% episode 反而更差。

一个真正数学意义上的 lower-bound oracle 不应该出现这种情况。

所以我建议把所有：

headroom ceiling
oracle ceiling

统一改成：

joint-future-truth reference

或：

truth-informed H5 reference

最终我会把这轮实验的科学结论压成一句：

这次结果没有证明“越保守越好”，而是证明了：提高概率预测质量不足以保证下游收益，因为 predictor replacement 同时改变了 scheduler 实际消费的 future-cost statistic 的尺度、当前/未来相对权重以及可能的局部候选排序。下一轮 full-unfreeze 应保留 calibrated distribution objective，同时显式学习一个 decision-aligned scheduler score，并用真实 scheduler 结果作为最终 gate。

这比现在 REPORT 里的“scheduler consumes conservatism”严谨得多，也更有论文价值。
