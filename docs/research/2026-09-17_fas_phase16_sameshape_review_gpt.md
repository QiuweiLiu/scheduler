# Phase 16 truth-same-consumer 实现审阅（GPT，2026-09-17）

会话：https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e（标题「架构设计评估」）
模型：GPT-5.6 Sol + High（basis: high_ui_mapping，页面无限制/回退提示）
输入：提交 ae61c0a（parent 8ad865b）+ phase16_review.patch + 6 问 brief
本文件是 GPT 原始回复的存档；其结论未经本地验证，不得直接进入 canonical 决策。

---

结论先说：**这次 Phase16 修复了“key 形状不同”这个大混淆，但当前 sameshape_h5_truth 还不能严格回答“只把预测 runtime 换成真值后，为什么 truth 打不赢 p95”。**它现在更准确回答的是：

在完全相同的 dispatch key 和 current estimate 下，“部署中的预测 future package”与“真实后继链上的 truth-aware future cost”谁产生更好的调度。

这个实验有价值，但仍混合了 topology/length、runtime、load、residency 四类信息。建议保留现有三个臂作为 Phase16-A，再新增一个更干净的 runtime-only 分解 Phase16-B；不用改任何 legacy 臂。

1. Key 形状判断

VERIFIED：predopt_h5_q95 vs trueopt_h5 的旧比较确实 confounded，不能当 oracle policy bound。

trueopt_h5 的第一排序项直接是 limited_future_truth_cost，priority 第二，而且不包含 current；predopt_h5_q95 则是 priority 第一，再比较 current+future。所以即使把 future 数值做成完全一样，两者也可能选不同动作。

但这里有一个需要修正文档的事实：

VERIFIED：你们写的“普通 predopt_h5 priority 第三”在 ae61c0a 当前代码中不成立。

当前 generic predopt_h* 也是：

(priority, current+future, future, ready,…)

priority 第一。

所以 Phase16 文档和 commit message 中“三个臂 priority 分别第1/第2/第3”的说法要改。真正确定存在的 key-shape 混淆主要是 legacy trueopt_h5 vs prediction family。这不影响你们撤回旧“q95 beats oracle”的决定。

必须写的论文措辞：

Legacy trueopt_h5 exposes privileged future information but uses a different dispatch-key semantics; therefore it is an oracle-information heuristic, not an oracle policy or performance bound.

不要叫“oracle 上界/下界”。

2. Truth successor walk vs predicted synthetic chain

VERIFIED：这是当前 Phase16 最主要的剩余混淆，而且对你提出的机制问题来说不可忽略。

预测臂消费的是：

L
^
,(
R
^
1
	​

,…,
R
^
L
^
	​

)

对应 predictor synthetic chain。

truth 臂消费的是：

L
∗
,(R
1
∗
	​

,…,R
L
∗
∗
	​

)

对应真实 successor walk。

所以如果 Truth 胜/负，都无法知道来源是：

runtime 估计更准确；

长度更准确；

topology/content 更准确；

load/residency 模型不同。

不过我不同意 Phase16 文档里“因为 future step 没 node_id，因此这个混淆不可再压缩”的判断。

**INFERENCE，但很容易验证：v3.1 是线性 slot 预测，artifact 又有 anchor node_id 和 step_offset，因此如果 R7 template 在 H≤5 内保持 width=1，那么完全可以通过 ordinal successor 对齐。**你们设计文档明确把未来定义为 linear H=5 continuation slot，packer 也给每槽 step_offset=t+1。

最低成本方案不是改 frozen artifact，而是生成一个privileged oracle sidecar：

anchor node_id
  -> true successor 1 runtime
  -> true successor 2 runtime
  ...
  -> true successor 5 runtime

先对全部 evaluated anchors assert：

outdegree <= 1 within evaluated continuation
step_offset = 1..K

如果通过，就能严格做：

TruthRuntime∣Predicted Shape

即保持预测长度 
L
^
，但把第 h 个 predicted runtime 替成真实第 h 个 successor runtime。

这是我最推荐的对照。

3. limited_future_truth_cost 还有没有坑？

VERIFIED：有，而且因此当前 truth 臂不是“runtime-only truth”。

它同时做了：

真 successor walk；

真 compute_ms/runtime_ms；

cold-load 真值；

根据当前 GPU residency 模拟未来 residency；

内存不足时清空 resident；

H 按 DAG layer 推进。

而你们新 p50/p95 臂也有一个容易忽略的问题：

VERIFIED：sameshape_h5_p95 并不是“纯 runtime p95 求和”。

它调用 _q95_step_cost，GPU step 在 load_occurrence_probability>=0.5 时还会加 load_duration p95；p50 同理加 conditional load p50。

所以：

sameshape_h5_p95 ≡ predopt_h5_q95

这个语义等价是成立的；

但：

predopt_h5_q95 = Σ runtime p95

这个描述不成立。

真正已有的 runtime-only 实现是 predopt_h5_r95，它显式使用 _runtime_only_step_cost()、不加 future load。

因此，如果问题真的是：

“只改变每一步 runtime 的信息来源会怎样？”

建议新建三个严格正交臂：

Pred50
R
	​

,Pred95
R
	​

,Truth
R
	​


共同 score：

(priority, current
table
	​

+∑R
h
	​

,∑R
h
	​

,…)

其中 Truth_R 不做 future residency/load，只取真实 successor runtime。

还有一个 UNVERIFIED/P0 contract audit：必须确认 predictor 的 runtime_ms_quantiles 标签到底是“total step runtime”还是“compute-only runtime”。Simulator 中 compute_ms=runtime_ms-load_ms，而 train resource stats 的 runtime 来源是 node.runtime_ms。 如果 J-series runtime target 本身包含 load，那么 _q95_step_cost(runtime + load) 就存在重复计 load 的语义风险。跑实验前应把 label builder 查清。

4. dev700 三分岔判据够不够？

INFERENCE：方向是对的，但三个结论标签目前过强。

可以预注册，但建议改成：

Truth
R
	​

<Pred95
R
	​

<Pred50
R
	​


只能说明 privileged runtime information 最优、p95 优于 median forecast；要叫“校准补偿”，还必须同时观察：

Bias
50
	​

=∑p50−∑R
∗

是否系统性为负。

Truth
R
	​

≈Pred95
R
	​


不能用“差异不显著”定义。应预先定义一个 practical equivalence margin δ，做 equivalence CI：

∣Δ∣<δ.

而：

Pred95
R
	​

<Truth
R
	​


且 paired CI 上界 <0，可以写：

the p95-based index yields better system-level decisions than ranking by exact H5 remaining runtime under the same consumer shape.

但还不能直接写“uncertainty premium 有价值”，因为需要机制实验说明优势真的来自 uncertainty gap，而不是其他排序效应。

dev700 足够做这个诊断，不需要1000。confirm300 已经被看过，不应再作为新的 confirmatory set；若最终把这个机制升级成主贡献，最后只在仍封存的 T_final 验一次。

必须同时报告：

paired mean completion Δ + miss-rate Δ；

每 anchor 的 ∑p50,∑p95,∑truth signed bias / MAE / underestimation rate；

p50/p95/truth 对候选集的 top-1 agreement、pairwise disagreement、Kendall τ；

只在 p95 与 Truth 选不同动作时的 downstream regret；

score margin；

p50≤p90≤p95 violation rate；

tie rate，以及多少次最终由 gpu.index 决胜。

其中“common-state ranking audit”尤其重要，因为三种策略一旦开始执行，后续访问的状态分布就不同，直接比较最终 episode 指标会混入 trajectory divergence。

5. 如果最终真的得到 r95 < Truth

INFERENCE：这时最值得做的不是 ρ-strength，而是把“uncertainty premium 是否提供决策信息”直接钉死。

优先顺序我建议：

第一，利用你们已有的 scale-matched p50：

如果同尺度 p50 仍输，排除“p95 只是数值更大”。

第二，利用已有 tail-shuffle：

如果保持 tail-gap 总分布但打乱 step 对齐后变差，说明有价值的是：

u
h
	​

=p95
h
	​

−p50
h
	​


与特定 future step 的对应关系。

第三，新做最关键的一项：

u
h
	​

vsR
h
∗
	​

−p50
h
	​


以及

u
h
	​

vs∣R
h
∗
	​

−p50
h
	​

∣.

如果 tail gap 能预测“这一步容易被低估/难预测”，那 p95 就有一个真正的数据性质。

然后在 p95 和 Truth_R 排序发生冲突的决策 上测：

Regret(Truth action)−Regret(p95 action)

并按 tail gap、contention 分层。

如果 tail gap 越大、系统越拥塞，p95 相对 Truth_R 越有优势，你就有很强的故事：

exact remaining work ignores forecast uncertainty as a scheduling signal; the uncertainty premium identifies actions whose forecast errors have larger downstream externalities.

ρ-strength 我会降到 synthetic stress-test，因为你们真实 T1–T3 已经没有支持 positive execution dependence。

6. 还有哪些第三处混淆？

VERIFIED：current load gating、priority、ready_time、候选池在三个新 same-shape 臂之间是共享的，因此它们本身不是 differential confound。 gpu.index 也是统一 tie-break；只需报告 tie 发生频率。

但还有四个重要点：

VERIFIED：truth future 是 GPU-dependent，预测 future 基本不是。
limited_future_truth_cost 从 gpu.resident 出发模拟后续 cache/load，而 predicted future 的 conditional load 来自 artifact。于是 truth 臂实际上获得了额外 state-dependent future information。

VERIFIED：H 的语义不同。
truth 是 DAG layer；prediction 是 synthetic slot。在线性链上才近似相同。

VERIFIED：当前单元测试不足以验证这些边界。
它的 synthetic future load occurrence=0，而且测试 template 基本没有真正的多步 successor truth，因此能证明 p95 branch regression-equivalence，但没有验证 multi-step Truth 对齐、load、residency。

UNVERIFIED：runtime/load accounting contract 可能是最值得先查的隐藏问题。
需要确认 J predictor 的 runtime label 是否已经包含 load；否则 p95/q95 的解释可能整体需要调整。

必须修改的代码

我把它分成 P0/P1。

P0：实验前必须做

修 Phase16 文档中 predopt_h5 priority 第3 的错误；代码本身不用改。

审计并写死 J-series runtime_ms_quantiles 的 target semantics：total runtime 还是 compute-only。

新增 runtime-only same-shape family，至少：

sameshape_h5_pred_r50
sameshape_h5_pred_r95
sameshape_h5_truth_r_predshape
sameshape_h5_truth_r_trueshape

在构造 truth_r_predshape 前做 path-invariant assert；若 v3.1 evaluated subset 不是严格单链，则改用独立 oracle-sidecar 映射。

P1：强烈建议

给单测加入：

两个以上候选；

非零 conditional load；

≥3-step successor chain；

两块 GPU 不同 residency；

直接断言每个 policy 的 candidate key，而不只是最终 summary。

必须写的论文措辞

现在 Phase16-A 可以写：

We control the dispatch-key ordering and current-node estimate, but the truth arm still replaces the predicted synthetic continuation with the realized successor continuation; it therefore changes both future structure and resource information and is not an isolated runtime-oracle experiment.

做完我上面 Phase16-B 后，才可以写：

To isolate resource-estimation effects, we hold the predicted continuation shape and dispatch rule fixed and replace only each ordinal future-slot runtime with its measured counterpart.

如果最终 Pred95_R < Truth_R：

Exact H5 remaining runtime is not a policy optimum; under the same ranking rule, the p95-derived uncertainty premium induces decisions with lower observed system completion cost.

不要写：

prediction is better than truth / p95 beats oracle.

所以我的最终判定是：

Phase16 当前实现：可用作“same-key system sensitivity”，但还不是严格的 Truth-SameConsumer 机制实验。

最值得补的不是另一个复杂 optimizer，而是下面这个四臂分解：

Pred50
R
	​

→Pred95
R
	​

→TruthRuntime∣PredShape→TruthRuntime∣TrueShape.

它会非常干净地把：

风险溢价 → runtime 预测误差 → topology/length 预测误差

三件事拆开。这个实验一旦跑完，你们才真正能回答“p95 为什么会打赢/打不赢真值”。
