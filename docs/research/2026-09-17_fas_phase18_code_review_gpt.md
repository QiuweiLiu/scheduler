# Phase 18 聚合乐观性诊断 — 代码审阅（GPT，2026-09-17）

会话：https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e
模型：GPT-5.6 Sol + High（basis: high_ui_mapping）
输入：`analysis_phase18_aggregate_optimism.py` + `test_phase18_aggregate_optimism.py`（附件）+ 8 问 brief
裁决：**设计方向可用；代码暂不通过正式运行** —— 6 个 P0 修完再跑；不需要改预测器，也不需要 1000 集调度重跑。
本文件是 GPT 原始回复存档，结论未经本地验证。

---

我已经基于这两个附件逐行审过，并独立跑了单测，12/12 通过。但我现在不建议直接正式跑 Phase18：设计方向对，但代码里有几处会直接影响结论的 P0，尤其是 video bootstrap、零步预测和 DAG 分支标记。

[INFERENCE] 三种真值都应保留，但我不同意 layers_h 做“聚合乐观性”的主口径。

如果问题是“p95 是否在纠正预测 runtime 的聚合低估”，唯一干净的主口径应是：

S
linear,K
true
	​

且限制在真实未来前 K 步为单链的 anchor。

因为预测消费者本身就是 K 个线性 future slots。layers_h 在分支 DAG 中会把一层里的多个真实节点全加进去，而预测器每个 slot 只有一个节点；此时

S
layers
	​

−
S
^

同时包含“runtime 低估 + 没表示 DAG width”，不能叫纯 aggregate calibration。代码目前三者的定义确实如此。

analysis_phase18_aggregate_opti…

建议定位为：

Primary：linear_k，strict width-1/path-compatible subset；

Secondary：layers_h，解释为 horizon workload coverage；

Context：all，解释为完整 remaining-work coverage。

如果论文只能放一个，我选第一个。

[VERIFIED] 线性预测和分支真值可以比较，但只能叫“workload coverage”，不能直接叫 quantile calibration。

而且当前分支诊断有两个代码问题。

第一，descendants() 实际是 BFS layer order，不是文档声称的“全局 sequence_index order”。

analysis_phase18_aggregate_opti…

 我构造一个合法 DAG 就能让两种顺序不同。因此 linear_k 需要改成全局按 sequence_index 排序，或者更干净：在 strict-chain subset 上逐 successor 走 K 步。

第二：

Python
运行
branching_in_linear_window =
    bool(len(layers) > 1 and layers[0] > 1)

是错的。

analysis_phase18_aggregate_opti…

a→{b,c} 且 b/c 都 terminal 时 layers=[2]，这里反而返回 False；后续层发生 fan-out 也检测不到。

建议至少定义两个字段：

branch_within_H=max(layers[:H])>1

和真正的 unique_chain_for_K：从 anchor 开始连续 K 次要求 exactly one successor。

因此你问“宽度错配会不会让 B
τ
	​

 解释失效？”——**会。**它不会让数字失效，但会让“预测 runtime 乐观”这个解释失效。

[VERIFIED] 真值应该用 runtime_ms，不是 compute-only。

这个选择正确。你现在明确用完整 wall-clock runtime_ms，而 Shat 只累计 runtime quantile；额外 load 单独报告，没有混进去。

analysis_phase18_aggregate_opti…

既然 J runtime head 的训练标签就是 total runtime，换成 compute-only 反而造成 target mismatch。

但论文要注明：

这里的 truth 是原始 trace 中观察到的 realized wall-clock runtime，用于预测校准诊断；它不是改变 GPU residency 后的 counterfactual execution time。

另外 runtime_ms() 遇到非 numeric 当前静默返回0，也建议 fail-closed，否则可能人为降低真值。

analysis_phase18_aggregate_opti…

[INFERENCE] 640 个 :run:1 应直接排除，不要映射到第一个节点。

映射会改变 anchor 所拥有的信息，并且容易和第一个真实 node anchor 重复计数。

正确措辞是：

640 workflow-root pseudo-events lack a corresponding schedulable template node and were excluded from node-level calibration.

但当前“其他未 join 必须 fail”的实现只在 --dry-run 路径触发；正式运行如果直接不 dry-run，unjoined_other 不会报错。

analysis_phase18_aggregate_opti…

这个 hard assertion 应移到 dry-run 分支之外。

[VERIFIED + INFERENCE] 候选选择确实是另一个 population mismatch。

当前脚本分析所有 joined anchors，而且记录了 anchor_lane，但没有过滤。

analysis_phase18_aggregate_opti…

所以它首先回答的是：

R7 node population 上预测 future 是否聚合乐观？

它还不能直接回答：

真正影响 r95 scheduler 决策的 candidate 是否聚合乐观？

最低成本处理不是猜“曾被考虑”，而是：

Primary 再报 anchor_lane=="gpu" 子集；all-anchor 作为 sensitivity。

不要按“冠军实际选中过的节点”过滤，那会产生 policy-dependent selection bias。

真正的“competitive candidate（同一决策点 ≥2 个候选）”以后在四臂调度实验中顺手记录即可，不必现在重跑历史实验。

[VERIFIED] 现在的报告项还不够，而且 bootstrap 有两个 P0。

最严重的是：

Python
运行
video_by_template = {
    template_id: template_id.split("_")[0]
}

analysis_phase18_aggregate_opti…

不要解析字符串。R7 template 本身有 video_id，应直接保存它，并硬断言最终 n_unique_video==160。否则 cluster bootstrap 可能完全聚错。

第二个 bug 是：

Python
运行
r_values = [row[r] for row in records if row[r] is not None]
bootstrap_ci(r_values, clusters)

r_values 被过滤，clusters 没同步过滤；一旦中间存在 S_true=0，后续 value 会配错 video。

analysis_phase18_aggregate_opti…

必须传 (value, video) 成对过滤。

还有一个非常重要的 selection bug：

Python
运行
if pred["n_steps"] == 0:
    return None

analysis_phase18_aggregate_opti…

**不能丢掉预测 future=0 的 anchor。**这些恰恰可能是最严重的 optimistic cases。应保留：

S
^
τ
	​

=0

再与真实 future 比较。terminal 且 S_true=0 时 ratio undefined，但 bias 仍可定义。

同时 predicted_sums() 对缺失 quantile 当前直接补0，单测甚至固定了该行为。

analysis_phase18_aggregate_opti…

 

test_phase18_aggregate_optimism

正式诊断应该 missing/nonfinite quantile fail-closed，否则你可能自己制造 aggregate optimism。

我还建议增加：

∑
v
	​

S
v
	​

∑
v
	​

S
^
v
	​

	​


即 ratio-of-sums。不要只看

mean(
S
^
v
	​

/S
v
	​

)

后者会被很小的 S
v
	​

 放大。

还应给 Bτ、underestimate rate 都做 video-cluster CI；目前代码只给 rτ CI。

analysis_phase18_aggregate_opti…

[VERIFIED / UNVERIFIED] “完全不相交”能证明严格 out-of-sample，但不能仅凭这个叫 OOD。

你们可以确定写：

The predictor was fitted on a disjoint 300-video corpus and applied unchanged to an unseen 160-video scheduler workload.

这说明没有 scheduler-video fitting leakage。

但：

“R7 是 distribution shift / OOD”

目前 UNVERIFIED。视频身份不同 ≠ 分布一定不同。

如果 Phase18 发现 aggregate optimism，应写：

“deployment-workload miscalibration/generalization error”

而不是：

“quantile prediction method is intrinsically biased.”

如果以后比较 J validation 与 R7 的 type/runtime/length/quantile coverage，证明分布明显变化，再升级为 domain shift。

[INFERENCE] 如果最终得到 r50≈0.4, r90≈0.8, r95≈1.0，我建议先实验4，再实验2。

但有一个前提：这个形态必须首先在strict-chain / shape-matched / GPU subset成立。

然后实验4直接测：

U
h
	​

=p95
h
	​

−p50
h
	​


是否预测

R
h
true
	​

−p50
h
	​


和

∣R
h
true
	​

−p50
h
	​

∣.

聚合层面再测：

h
∑
	​

(p95
h
	​

−p50
h
	​

)vsS
true
−
h
∑
	​

p50
h
	​

.

如果成立，p95 有真正的 uncertainty/difficulty 信息；如果不成立、但 p95 只是把整体 scale 调到1附近，那么更像 global bias correction。

之后再进四臂调度实验，才知道这种信息是否真的改善 decision ranking。

必须改的代码（跑正式 Phase18 前）

P0

video_id 直接取 template 字段，并 assert 160 unique videos。

unjoined_other 正式运行也 fail-closed。

预测0步 anchor 不得丢弃。

missing/nonfinite runtime quantile 不得补0。

bootstrap value 与 video 必须成对过滤。

修 linear_k 顺序和 branching/unique-chain 判定。

P1
7. 加 GPU-only、strict-chain、branching 三套分层。
8. 报 ratio-of-sums、B 的 CI、underestimate-rate CI。
9. 报 n_pred_steps vs true path/layer node count，及 zero-step 数量。
10. runtime_ms 缺失/非法 fail-closed。

你现有的 diamond 单测证明了三种 sum 在那个特例下计算正确，但没有覆盖上述 main-path bug。

test_phase18_aggregate_optimism

必须写进论文的措辞

不要笼统写：

“p50 underestimates the true future by X%.”

应区分：

“On chain-compatible scheduler anchors, the summed predicted runtime quantiles were compared with measured runtime over an equal-cardinality realized continuation.”

而对 layers_h/all 写：

“These comparisons measure future-work coverage and jointly reflect structural omission and resource-estimation error; they are not pure runtime calibration tests.”

所以我的裁决是：**实验设计可用，但当前代码暂不通过正式运行。**把上面6个 P0 修完后再跑；不需要改预测器，也不需要任何1000-episode调度重跑。
