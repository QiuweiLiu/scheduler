---
research_backend: "ChatGPT Web"
requested_model: "GPT-5.6 Sol"
requested_reasoning: "High"
model_verification_basis: "high_ui_mapping"
conversation_id: "6a9822da-e278-83e9-9c1a-675923acda0e"
conversation_generation: "1"
conversation_url: null
verified_before_submit: true
verified_after_response: true
timestamp: "2026-09-11T11:25+08:00"
status: "valid"
---

# Research Report

---
export_backend: "ChatGPT Web DOM"
conversation_id: "6a9822da-e278-83e9-9c1a-675923acda0e"
conversation_url: "https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e"
title: "架构设计评估"
source: "visible_dom"
completeness: "visible_dom_only"
message_count: "8"
timestamp: "2026-09-11T11:09+08:00"
---

# Conversation Export

## 1. user
review_joint_training_plan.md
文件
p9d_future_role_family_multitask.py
Python
p9d_data_sample_redacted.jsonl
文件
Please review the 3 attached files for our joint topology × resource training plan:

1. review_joint_training_plan.md — the consolidated plan (J1/J2/J3 stages, network deltas, label contract, loss contract, metrics/gates, stop table, and 5 review questions in section 8).
2. p9d_future_role_family_multitask.py — current model code (shared causal GRU + structure/content/next-step heads); the plan proposes adding a resource-signature head and a resource head on top of this exact structure.
3. p9d_data_sample_redacted.jsonl — one feature record + one label record showing the data schema (identifiers redacted).

Focus your review on the 5 questions in section 8 of the plan, especially: (a) whether J1 already confounds representation learning with interface learning; (b) whether the proposed Z signature is minimal and sufficient for load/memory identifiability; (c) whether hard per-node expansion should be rejected given weak family F1; (d) any leakage or circularity in the Case-A label join; (e) whether code shapes/masks/matching in file 2 are consistent with the contract in file 1.

Return a structured review with VERIFIED / INFERENCE / UNVERIFIED labels and working links for every cited paper. Research review only; no code changes requested.

## 2. user
review_three_revisions.md
文件
raw_trace_samples_redacted.jsonl
文件
Please review the 2 attached files:

1. review_three_revisions.md — three proposed revisions to our joint-training plan (J0 frozen-encoder probe; input_scale recovered from raw-trace clip windows/query lengths; strict longest-path re-layering), each ending with focused questions.
2. raw_trace_samples_redacted.jsonl — 3 randomly drawn raw-trace excerpts (first 10 events each; identifiers/paths/query text redacted) showing step-level parent_step_ids, clip windows, and query lengths.

We are working through your previous verdict item by item (J0 accepted; input_scale evidence found; layer semantics under discussion). Focus on: (a) whether the proposed input_scale fields are adequate or resolution/frame-count is strictly required; (b) longest-path re-layering vs last-event-only step-parent expansion — which is the right fix; (c) anything in the raw samples that contradicts our reading.

Return a structured review with VERIFIED / INFERENCE / UNVERIFIED labels and working links for cited papers. Research review only; no code changes requested.

## 3. user
chatgpt_p9d_v3_contract_brief_zh.md
文件
抱歉，上一条消息的附件没有送达；现在《P9d 拓扑标签契约 v3 草案》已作为附件重新上传，请查收并评审。

评审要求不变：
1. 按草案文末的 8 个问题逐条裁决（接受 / 修改 / 拒绝 + 理由），标出任何新的 P0 风险与具体修改建议。
2. 请用中文回答；VERIFIED / INFERENCE / UNVERIFIED 标签保留英文原词。
3. 不要索取原始 trace、本地路径、模型权重或凭据；本评审只作为研究输入。

## 4. assistant
执行摘要

我对 v3 的总体裁决是：MODIFY 后接受，不建议按当前草案原样冻结；但修改幅度已经不大。

v3 相比 v2 的方向是对的，而且是一次重要纠偏。根据草案给出的采集器源码语义，当前三类 workflow 的顶层执行事实上是同步串行的：兼容循环中 planner 先产生单个工具调用决策，随后该工具执行；解析失败后 retry 紧随；post-loop generalist.generate 在循环结束后执行；st_fixed 也是固定串行流程。

chatgpt_p9d_v3_contract_brief_zh

 因此，把 v2 人工制造出来的 planner/tool“并行兄弟”恢复成顺序关系，比继续维护一个并不存在的宽 DAG 更忠实于数据。

但我认为草案还混在一起了两个概念：

trace 中相邻发生

=必然具有因果依赖

外部的 process-mining 文献也明确指出，仅按时间/全局顺序连接事件可能制造虚假依赖。
Springer
 你们这次之所以可以用链，不是因为“文件顺序天然等于因果关系”，而是因为采集器控制流 + 1,360 run 审计共同证明了当前 workload 是串行执行的。

因此我建议把 v3 的核心原则从：

“边 = trace 文件顺序中相邻节点”

改成：

“当前支持的 workflow 已验证为顶层串行控制流；在剔除/合并内部嵌套调用后，按该已验证执行序列构造直接执行依赖链。若未来出现不能满足串行语义的 run，则触发 contract failure，而不是强制链化。”

这个措辞差异非常重要。

我的八问裁决先汇总如下：

#	问题	裁决
1	执行顺序链作为唯一边	修改后接受
2	嵌套答案调用合并	修改后接受
3	post-loop generate / final answer	修改
4	retry 是否保留	接受
5	width 恒 1 后是否还叫 DAG topology	修改，而且是论文定位上的重要修改
6	8 项 gate 是否足够	修改，需要增加独立串行性/资源核算 gate
7	predictor 与 scheduler 是否同步 v3 化	接受分阶段处理
8	是否退化为序列预测、损失研究价值	接受这个风险判断；必须重写研究表述

此外我识别出 3 个正式冻结前的 P0，后面单列。

1. “执行顺序链”是否可以成为唯一事件级边？
裁决：修改后接受
VERIFIED

草案提供的源码/审计证据非常强：

planner 的 tool_name 与同一步实际工具 100% 对应；

tool call ID 的 step 编号与 step_id 对齐；

planner retry 与失败 planner 同步；

当前支持的兼容循环每轮是 planner → 一个 tool；

st_fixed 是明确固定序列；

原型对 1,360 run 重建后全部为单根单链。

chatgpt_p9d_v3_contract_brief_zh +1

所以对于当前这三个 workflow implementation，单链是有程序控制流依据的，而不仅是猜测。

INFERENCE

真正应该冻结的不是：

edge(v
i
	​

,v
i+1
	​

)⟺v
i
	​

,v
i+1
	​

 在文件中相邻

而是：

edge(v
i
	​

,v
i+1
	​

)⟺经验证的顶层串行执行语义使 v
i+1
	​

 在 v
i
	​

 后执行

文件顺序只是这个执行关系的具体编码。

这一点有外部方法学支持。Process Mining over Multiple Behavioral Dimensions with Event Knowledge Graphs 明确展示了：如果简单把时间上相邻但语义上无关的事件连接起来，会产生虚假行为依赖。
Springer

需要增加的 P0 gate

当前 Gate 1：

每条 run 重建后必须是一条链

其实部分是自证式的，因为 builder 本身就是“把节点全部链起来”。

chatgpt_p9d_v3_contract_brief_zh

因此应增加独立于 builder 输出的：

Serial-control gate

至少断言当前支持 workload 中：

每个 planner iteration 最多产生一个顶层工具调用；

不存在 async spawn / fork / multiple-tool fan-out；

非嵌套顶层调用不存在无法解释的执行窗口重叠（如果已有时间窗字段）；

所有同一步相邻组合都能由采集器控制流规则解释；

出现未来新模式时 fail closed，不能默认为相邻链。

关于你问的“并行分支/外部到达”

UNVERIFIED：这套规则不能代表未来可能加入的并行工具、repair fan-out、多 agent 协作等。

但这不是当前数据集的缺陷。

外部 job arrival 更不应该放入单个 agent run 的内部 DAG；那属于 scheduler workload arrival process。

2. 嵌套 generalist.generate 应合并还是保留为子节点？
裁决：修改后接受合并
VERIFIED

草案给出的关键事实是：

nested generalist.generate 是 summarization-tool 内部实现；

它发生于 Summarizer 时间窗口内部；

Summarizer action 的 runtime 已经包含这次 nested generate。

chatgpt_p9d_v3_contract_brief_zh

v3 计划把 nested call 放入 nested_calls，而不再作为拓扑节点。

chatgpt_p9d_v3_contract_brief_zh

如果调度器实际能调度的原子单位是：

summarization_tool

而不是内部：

generalist.generate

那么从调度资源语义看，合并是正确的。

否则你会产生：

runtime
Summarizer
	​

+runtime
nested
	​


的重复核算。

OpenTelemetry 的标准追踪模型也把这类内部调用表达为父操作的 child/sub-operation；父 span 的执行区间可以包含子操作，而子 span 用来提供内部可观测性。
GitHub
+1

但我不建议现在的 effective_model_id 定义

草案写：

effective_model_id 优先使用嵌套调用模型 ID。

chatgpt_p9d_v3_contract_brief_zh

这里我建议 MODIFY。

因为 Summarizer 现在是一个复合资源节点：

outer tool+nested model call.

如果只把节点的模型 ID 改成 nested model，资源模型可能误认为这个节点“纯粹就是一次 LLM inference”。

更好的资源签名概念应该保留：

outer_exec_class
outer_tool_family
nested_model_class
merged_nested_call=1.

nested runtime/resource 可以保留作 label-side diagnostic/decomposition，但节点的真实 resource target 应当是整个 Summarizer 的：

R
total
	​

.

不能再次加 nested target。

结论

INFERENCE：

合并 topology node 是正确的；资源表征应该是 composite signature，而不是简单用 nested model ID 覆盖主节点身份。

3. post-loop generalist.generate 与最终 answer 怎么处理？
裁决：修改

这里草案还有一个容易污染 R0 的地方。

post-loop generate

VERIFIED：应保留独立节点。

它是循环结束后独立执行的一次答案模型调用。

chatgpt_p9d_v3_contract_brief_zh

它有真实：

模型；

runtime；

load；

memory；

因此应该是独立 compute node。

final answer

我不建议把它与 post-loop generate 合并，因为二者语义不同：

generalist.generate=计算
answer=结果提交/终止事件.

但是，我同样不建议让 final answer 成为与 GPU compute node 等价的资源节点。

推荐修改

最清晰的是：

...→generalist.generate→[terminal answer marker]

其中 answer：

可以保留在 trace/topology label 中；

用于确认终止；

resource_applicable=false；

不进入 runtime/load/memory regression；

不建立 model_resource_class。

如果你们的 H=5 定义是未来可调度计算节点数，我甚至倾向于让 answer 不消耗 H 的一个 slot，而由 termination_status 单独表达。

为什么这是 P0

如果 R0 把：

answer event

也当成普通 resource node，那么 oracle-resource 上界会混入一个本质上没有同类 GPU resource semantics 的目标。

因此 R0 前必须明确 answer 是否属于 schedulable-resource node ontology。

4. retry 应保留还是折叠？
裁决：接受

这里我支持草案原方案。

VERIFIED

失败 planner 本身已经执行过模型计算；retry 也会再次执行。草案保存失败节点→retry 节点，并保留 error/retry metadata。

chatgpt_p9d_v3_contract_brief_zh

如果折叠成：

failed+retry→successful planner

就会删除真实发生的：

runtime；

model call；

latency；

future work length。

这对于 GPU scheduling 是错误的抽象。

INFERENCE

retry 恰恰是未来预测中值得留下的不确定性：

P(retry∣prefix).

它会影响：

future node count

以及：

future resource demand.

所以不要为了“拓扑更干净”删除它。

一个小修改

retry_of=<真实 event id> 只应是 audit-side metadata。

identity-free 学习目标建议只保留例如：

is_retry
status_class

而不要让真实 future ID 进入可预测内容字段。

这与草案现有“未来 ID 不作为输入”原则一致。

chatgpt_p9d_v3_contract_brief_zh

5. width 恒为 1 后，layer/width schema 怎么办？
裁决：修改，而且这是最重要的科学定位问题

草案现在明确：

W
l
	​

=1,∀l.

chatgpt_p9d_v3_contract_brief_zh

这意味着在当前真实 workload上：

DAG=path.

于是：

width prediction 没有信息量；

并行度预测不存在；

longest-path depth 等同 sequence position；

layer count 等同 bounded remaining sequence length。

所以我不建议继续把 width 当作正式学习目标，更不建议用 width accuracy 来证明“topology prediction”。

但 schema 可以保留

为了：

与 v1/v2 artifact 兼容；

未来支持 repair/fan-out/并行 tool；

可以继续保持：

future_layers[ℓ].width

但当前 v3 数据中应把它视为：

deterministic field, width=1

而不是一个需要训练、加 CE、报告 leaderboard 指标的任务。

建议重新表述当前 prediction target

从：

future DAG topology prediction

改成更准确的：

bounded future execution-structure / execution-trace prediction

核心未知量变成：

future length/termination
future node type/family
future resource signature
future workload scale.
这是不是降低了论文高度？

VERIFIED / INFERENCE：确实削弱了“预测未知 DAG 分支结构”这一具体贡献。

不能靠命名解决这个问题。

但它不等于项目没研究价值。

2026 年 Pythia 的核心论点之一就是 agent workflow 结构/阶段具有可预测性，暴露这种 workflow semantics 可以改善 serving；它并不要求“每条 trace 必须有宽 DAG”才能产生系统价值。
arXiv

Pythia: Exploiting Workflow Predictability for Efficient Agent-Native LLM Serving — arXiv 2604.25899

因此你现在真正有数据支撑的研究问题可以是：

Can causal-prefix forecasting predict future agent execution stages and resource demand?
	​


而不是强行声称：

we predict future DAG branching
	​


后者目前没有数据支持。

6. 现在的 8 个 gates 足够吗？
裁决：修改

草案已有 gate 对工程可复现性做得比较全面，包括路径性、步号、显式信号、cycle、数量守恒、泄漏、v2→v3 diff 和 manifest。

chatgpt_p9d_v3_contract_brief_zh

但还缺三个独立语义 gate。

P0-A：Seriality gate

前面已经提到。

不能只验证：

builder 输出是一条链。

必须验证：

原始控制流没有任何证据要求非链结构。

我建议把以下情况设为 hard failure：

一 iteration 多 tool；

async/fork signal；

非 nested 顶层执行区间重叠且无法解释；

planner decision 无法唯一关联后续 tool；

新 workflow type 未被 contract 明确支持。

P0-B：Nested accounting gate

对每个合并节点断言：

1 nested call↔1 Summarizer

且：

window
nested
	​

⊆window
Summarizer
	​

.

同时保证：

R
node
	​

=R
Summarizer,total
	​


而不是：

R
Summarizer
	​

+R
nested
	​

.
P0-C：Schedulable-node gate

明确所有 node type 是否：

resource_applicable∈{0,1}.

尤其是：

final answer；

error marker（如果某些 error event 不是 compute）；

run control；

nested event。

R0 必须只在 resource-applicable target 上评价资源上界。

敏感性分析

“nested call 保留为单独节点 vs 合并”我建议做一个 label-level sensitivity audit，但不是必须加成正式训练 variant。

例如比较：

node count；

remaining-H distribution；

total runtime/resource sum 是否发生重复；

node class histogram。

这是 P1，不是 R0 的 P0。

7. scheduler 侧同源错误是否必须同时修？
裁决：接受“predictor 先冻结、scheduler 后单独治理”

草案已经禁止 predictor 验收前接入 scheduler。

chatgpt_p9d_v3_contract_brief_zh

我赞成保持这个边界。

对于：

R0；

J0；

J1/J2/J3 predictor training；

它们只依赖 predictor pool 的 v3 标签，不要求 scheduler job_templates 已经同步迁移。

所以没必要为了 R0 去同时修改调度器。

但是

INFERENCE：在未来第一次 scheduler integration 之前，scheduler-side graph contract 必须成为独立 P0 gate。

因为如果：

Predictor outputs=v3 chain semantics

但：

Scheduler oracle/templates=v1/v2 all−events expansion,

那么：

oracle；

PredOpt；

future cost；

predictor-vs-true comparison

将使用两个不同“真值世界”。

那时所有 scheduler 结论都会难以解释。

所以正确顺序是：

Predictor v3→R0→J0/J1/J2/J3→predictor freeze

之后才：

scheduler topology contract audit/migration→integration.
8. 链式重建是否让 topology predictor 退化成普通序列预测？
裁决：接受这个判断，但修改研究表述，而不是修改真实数据
VERIFIED

按照 v3 草案，全部 1,360 run 都变成 width=1 单链。

chatgpt_p9d_v3_contract_brief_zh

因此在这个数据集里：

非平凡 DAG topology prediction 已经不存在。

这一点应该正面承认。

不应该怎么做

不要为了保持论文故事：

人工恢复 v2 假并行；

把执行相邻事件伪装成 DAG branch；

构造没有来源的 fan-out。

那会把“更有研究感”放在“标签正确”之前。

应该怎么表达

我建议分两层：

当前实证任务：

causal future execution-trace forecasting with resource uncertainty

模型根据 causal prefix 预测：

P(N
future
	​

)
P(role/family/resource signature)
i
	​

P(termination)
P(runtime/load/memory)
i
	​

.

架构扩展能力：

现有 layer-wise identity-free representation 仍可以容纳：

width>1

但当前 benchmark 并没有验证这项能力。

因此论文应该说：

The current observed workflows instantiate the path-structured special case of the general execution-graph interface.

而不能说：

We demonstrate accurate future DAG branching prediction.

这种表述更可信。

Pythia 提供了一个相关系统层证据：仅利用 agent workflow 的语义可预测性，就可以为 serving 提供有效优化信息，因此“不是宽 DAG”不意味着预测任务失去系统意义。
arXiv

新发现的 P0 风险

我建议在 v3 冻结前明确解决以下三项。

P0-1：必须把“链”建立在 serial-control invariant 上，而非文件顺序本身

草案第 1 节当前写：

边 = 相邻节点间的单边。

chatgpt_p9d_v3_contract_brief_zh

建议改成：

对经源码语义和独立审计确认的顶层串行 execution stream，直接依赖边连接相邻 schedulable/terminal operations；文件顺序仅作为该已验证执行序列的序列化表示。任何不满足 serial-control invariant 的 run 不得被强制转换为单链。

这是我认为最重要的文本修改。

P0-2：R0 前必须冻结 schedulable/resource node ontology

至少明确：

节点	Topology	Resource target
planner	✓	✓
tool	✓	✓
merged Summarizer	✓	✓，使用 total
nested generalist	合并	× 单独计数
post-loop generalist	✓	✓
retry planner	✓	✓
final answer	terminal marker	×
run_control	×	×

特别是 final answer 与 nested call。

否则 R0 的“oracle resource upper bound”定义仍然不稳定。

P0-3：holdout 状态需要重新表述

这里我看到一个方法学上的潜在冲突。

草案一方面规定：

holdout 只能在 calibration/selection 冻结后读取。

chatgpt_p9d_v3_contract_brief_zh

另一方面，“全量 1,360 run 审计”看起来覆盖了整个 predictor pool。

chatgpt_p9d_v3_contract_brief_zh

如果这 1,360 包含那 40 个 holdout 视频，并且 holdout 的 trace/标签统计参与了 v3 规则制定，那么：

INFERENCE：它已经不能再严格称为“完全未查看的 confirmatory holdout”。

这不等于模型泄漏：

没有用 holdout 拟合参数；

没有根据 holdout prediction performance 选模型。

但 contract design 已经接触过它。

因此建议在记录中明确区分：

P_holdout_diag: contract-integrity inspected, never used for fitting/model/hyperparameter/calibration selection.

而不要声称：

completely untouched unseen dataset.

从现在起不再根据 holdout-specific 分布去修改 v3 契约即可。

这不会阻止 R0，但会影响将来论文如何描述 holdout 证据强度。

v3 是否可以冻结？

我的答案是：

当前草案：MODIFY
	​


完成以下四项契约级修改后：

把“文件顺序链”改写为“经验证 serial control-flow 的直接执行链”，并加入独立 seriality gate；

nested Summarizer 使用 composite resource signature，不让 effective_model_id 覆盖整个节点语义；

明确 final answer 是 terminal/non-resource event，post-loop generation 才是实际 compute node；

width=1 标为 deterministic compatibility field，不再把 width prediction 当成当前 v3 的有效结构学习任务；

我会给：

v3 CONTRACT ACCEPT
	​


并认为它足以解锁 node-level R0 oracle-resource/signature experiment。

R0 此时应明确验证的是：

在正确未来 execution sequence + oracle resource signature 条件下， runtime/load/memory 能否被可靠识别
	​


而不是验证：

DAG branching prediction.

这次 v3 最有价值的地方其实是把研究问题变得更诚实了：v2 看起来更像 DAG，但一部分宽度是数据构造制造出来的；v3 看起来“简单”，却更接近真实 agent execution semantics。对于后续 resource prediction 和 scheduler，这种标签真实性比保留一个漂亮但虚假的多分支拓扑更重要。

参考依据方面，WfCommons 将 workflow dependency graph 与 task/resource characteristics 分开建模，支持你们先冻结执行结构语义再研究资源映射的做法。
科学直通车
 WfCommons: A framework for enabling scientific workflow research and development
 Process-mining 文献则直接说明，简单从事件时间顺序构造依赖可能引入虚假行为关系。
Springer
 OpenTelemetry 的官方 trace 规范支持把内部嵌套操作建模为父操作的 sub-operation，而不是在上层工作单元中重复计费。
GitHub
+1

## 5. user
chatgpt_p9d_r0_design_brief_zh.md
文件
请评审附件中的《P9d R0 oracle-signature 资源上界实验设计草案》（中文）。

要点：
1. R0 用 oracle（真实标签侧）资源等价签名，测量单节点 runtime/load/memory 的可辨识性上界，作为 J 系列联合训练的前置 go/no-go gate；前提是已冻结的 v3.1 串行控制流标签契约。
2. 请按草案文末的 7 个问题逐条裁决（接受 / 修改 / 拒绝 + 理由），特别关注：测量契约（cold/warm、stall、缺失、memory 覆盖）、签名与特征边界、预注册阈值 G1–G4 是否合理、以及防乐观/泄漏（同一 run 内节点相关性）。
3. 标出任何新的 P0 风险与对设计的具体修改建议。
4. 请用中文回答；VERIFIED / INFERENCE / UNVERIFIED 标签保留英文原词。不要索取原始 trace、本地路径、模型权重或凭据；本评审只作为研究输入，不自动改变我们的代码或实验。

## 6. assistant
执行摘要

总体裁决：R0 的研究定位成立，但当前草案应当 MODIFY 后再冻结，不建议原样执行。 我认为有 3 个新的 P0，其中两个是明确的方法学错误，一个是测量语义尚未被草案证明：

P0：G3 的覆盖率定义错误。 [P50,P95] 是 45% 的名义区间，不可能合理要求总体覆盖率 90–98%。草案当前同时预测 q50/q90/q95。

chatgpt_p9d_r0_design_brief_zh

 若调度侧关心“95% 不低估”，直接检查 q95 的单侧覆盖更合适；若需要 90% 双侧区间，应增加 q05，使用 [q05,q95]。

P0：load_ms 不能既是预测目标，又参与 cold/warm 输入标签。 当前 cold if load_ms>0 会把测量真值反灌进特征，并且可能给 runtime 暴露一个高度相关甚至包含于 runtime 的执行后信息。

chatgpt_p9d_r0_design_brief_zh

 cold/warm 必须只由执行前可见状态定义，例如 model_resident_before。

P0-UNVERIFIED：显存测量是否真的是“单节点显存需求”需要在已有 instrumentation contract 中明确。 如果 peak_reserved_mb/peak_allocated_mb 是 node-window 内重置/基线校正后的峰值，当前 target 可以继续；如果是进程级累积 allocator peak，那么它会混入前序节点/cache 状态，R0 测的就不是单节点资源等价性。

此外还有一个重要逻辑修改：不能规定“G1 或 G2 任一个失败 ⇒ 当前接口不可辨识”。 G1 和 G2 应当是机制诊断，而不是两个串联的必要条件。比如连续 workload_scale 极重要时，条件中位数 G1 可以很弱，但 LightGBM 仍然可能表现很好；反过来，条件中位数已经足够强时，G2 只有 3% 增益也不能说明接口不可辨识。

完成上述修正后，我认为 R0 可以作为 J0/J1/J2/J3 前的有效 go/no-go gate。

1. R0 的定位是否成立？
裁决：修改后接受
VERIFIED

草案把 R0 限定为：

R
0
	​

:(Z, workload_scale, c)→(runtime,load,memory)

其中 Z 使用标签侧 oracle 信息，且明确不预测 topology、不进入 scheduler、不触碰 scheduler pool/T_final，也不声称这是可部署模型。

chatgpt_p9d_r0_design_brief_zh

这个职责边界与 J 系列是清楚的：

R0：如果未来节点“是什么”已经知道，资源还能不能估准？

J1：不依赖 predicted signature 时，共享表示能做到多少？

J2：predicted signature 是否提供增量资源信息？

J3：resource gradient 是否值得反向塑造 signature/encoder？

INFERENCE：建议修改“上界”的措辞

R0 不是严格的信息论 upper bound，因为你仍限制了：

estimator family；

当前 Z；

当前 workload descriptors；

当前测量质量。

LightGBM 失败不能证明任何模型都失败。

建议正式名称使用：

oracle-signature empirical resource ceiling

中文可写：

冻结接口与估计器族下的 oracle-signature 经验上界

而不是“不加限定的资源可辨识性上界”。

Vidur 也采取“profiling + predictive estimator”来逼近 operator runtime，而不是把某个具体回归器结果当作理论可预测极限。
MLSys 会议录
+1

Vidur — MLSys 2024

2. 测量契约：cold/warm、stall、missing、memory、load
裁决：MODIFY，且这里包含两个 P0
2.1 cold/warm

当前定义：

cold⇐load_ms>0

或 resident_before=false。

chatgpt_p9d_r0_design_brief_zh

REJECT 当前版本。

VERIFIED：load_ms 同时是 R0 预测目标之一。

chatgpt_p9d_r0_design_brief_zh

因此如果：

X
regime
	​

=1[load_ms>0],

你实际上把 target-derived information 放进了 feature。

建议改为：

residency_state=
⎩
⎨
⎧
	​

nonresident,
resident,
unknown,
	​

model_resident_before=False
model_resident_before=True
otherwise.
	​


并最好叫 residency_state，不要叫 cold/warm，因为“nonresident”与“最终真的发生多少 load”不是完全同义。

VERIFIED：现代 LLM serving 系统中 checkpoint locality/residency 会显著影响 startup latency，因此把真正的 pre-execution residency state 作为上下文是合理的。ServerlessLLM 就明确依据 checkpoint locality 估算并优化启动时间。
USENIX
+1

ServerlessLLM — OSDI 2024

2.2 load_ms 应是 target 还是 context？
裁决：保留为 target，不作为 context

如果 scheduler 将来需要估计：

setup/model load latency,

那么单独预测 load_ms 很有价值。

尤其是：

runtime
total
	​


与：

load
model
	​


最好不要过早揉成一个数字，因为前者可能受执行本身影响，后者更多受模型 residency/locality 影响。

所以我的推荐是：

R=(runtime
inclusive
	​

,load,memory).

但是 load_ms 只能出现在 Y，不能又出现在 X。

2.3 stall

草案计划把：

runtime>10×median
train,exec
	​


或

runtime>300000ms

的节点从主表排除。

chatgpt_p9d_r0_design_brief_zh

裁决：REJECT“主表排除”，接受作为副表

你已经观察到 planner 最大值约 2,695,000 ms，并怀疑含首载/worker startup。

chatgpt_p9d_r0_design_brief_zh

这些巨大 tail 恰恰可能是 scheduler 最关心的东西。

如果主表删除它们，R0 会回答：

“正常条件下资源是否可辨识”

而不是：

“实际系统资源是否可辨识”。

建议明确两个 endpoint：

R0-Inclusive（primary）

所有合法 status=success、finite、positive measurement。

R0-Normal（secondary diagnostic）

使用仅由 train 冻结的 stall rule 排除 extreme runtime。

如果：

R0
normal
	​

≫R0
inclusive
	​

,

这是一个极有价值的结果：

semantic/resource signature 本身足够，但当前接口缺少能识别 startup/stall 的 pre-execution system state。

不要把它通过删数据隐藏掉。

Sarathi-Serve 和 Llumnix 等近期系统工作也说明 tail/stall 行为本身就是 serving 的关键性能问题，而不是应天然视作噪声删除。
USENIX
+1

2.4 missing
裁决：ACCEPT，但需要更强的描述

禁止把缺失 memory/load 填 0 是正确的。

chatgpt_p9d_r0_design_brief_zh

但是 memory 的约 66% coverage 明显是按 node type 非随机缺失：

某些类别大量有 memory，另一些类别没有。

chatgpt_p9d_r0_design_brief_zh

因此：

INFERENCE：不能把 observed-memory subset 的结果写成“所有未来节点的 memory identifiability”。

只能写：

memory identifiability on the instrumented/observed subset.

建议至少报告：

coverage(memory∣exec_class)

和

coverage(memory∣model_class).

主 memory 指标只在“预注册的充分覆盖类别”上做 gate；其余类别只报告 descriptive metrics。

2.5 peak_memory_inclusive=max(reserved,allocated)
裁决：条件接受

UNVERIFIED：仅凭草案，我无法确认底层计数器是否：

在每个节点开始前 reset；

做了 pre-node baseline subtraction；

限定正确 device；

不携带前序 allocator cache peak。

如果这些保证成立，这个 target 可以继承。

如果不成立，则：

peak_reserved
i
	​


可能主要是在预测“此前 GPU allocator 状态”，而不是节点 i 的需求。

这应成为 R0 Stage-0 的测量语义断言，而不是重新采数据。

3. Z、Tier A/Tier B 和消融
裁决：MODIFY

草案的 Z 当前包含：

exec_class, role, exact model_id, planner_mode,merged_nested, nested_model, regime.

chatgpt_p9d_r0_design_brief_zh

3.1 exact model_id 要不要删？
裁决：R0 ceiling 保留，但必须加 coarse-model 对照

对于 oracle ceiling，exact model_id 是合理的，因为：

model architecture/size 显然影响 runtime/memory；

它不是 node ID/video ID；

如果 deployment 本来就知道候选 model，这个信息并非非法。

但“exact model ID”不等于：

resource-equivalence class。

所以建议把实验分成：

R0
exact
	​

: exact model id

和

R0
coarse
	​

: model resource class.

其中 coarse class 必须来自静态属性/配置，或者 train-only/cross-fitted resource grouping；不能偷看每个节点自己的 runtime/memory 再给它贴 class。

Vidur 对 operator runtime 的建模同样显式依赖 model/workload configuration，而不是只使用语义角色。
MLSys 会议录
+1

3.2 Tier B 应避免哪些字段？

我建议定义一条非常硬的边：

c 只能包含“节点准备执行时已经可观察”的状态。

安全候选：

stack/baseline；

pre-execution model residency；

candidate hardware class（如果本实验硬件不固定）；

已发生的 prefix 状态。

需要重新归类的：

chain position：对于 R0 可以作为 oracle structural context，但不能伪装成部署时天然已知状态；

previous node class：若指这个真实执行节点开始时已经执行完成的 predecessor，则是合法；若是从当前预测 anchor 看未来第 4 节点的真实 predecessor 类，则属于 oracle topology 信息；

retry：更像 node signature，而不是 runtime context。

明确禁止：

运行结束后得到的 frame count；

实际 decode/output token count；

qwen_image_count 若由未来执行结果产生；

load/runtime/memory；

实际后验 batch/interference 状态。

3.3 要不要 role-only/model-only 消融？
裁决：ACCEPT，而且建议改成递增 contract ablation

比孤立的 role-only/model-only 更有解释力的是：

C
0
	​

=exec_class
C
1
	​

=C
0
	​

+role
C
2
	​

=C
1
	​

+model
C
3
	​

=C
2
	​

+workload_scale
C
4
	​

=C
3
	​

+preexecution residency/context.

看：

ΔMAE,Δpinball

逐步下降多少。

这样能直接回答：

哪个变量修复了 identifiability？

而不是只知道“full signature 有效”。

4. 基线和指标是否足够？
裁决：MODIFY，但总体框架足够

全局中位数、train-only 条件中位数和 LightGBM quantile 已足以回答 R0 的核心问题。

chatgpt_p9d_r0_design_brief_zh

我不认为 R0 需要再加入复杂神经网络。

条件中位数需要额外记录

完整签名键可能非常稀疏，因此“逐级回退”必须预注册：

fallback hierarchy；

每一级覆盖率；

validation/test/holdout 使用了第几级 fallback。

否则条件中位数实际上是一个不透明的手工模型。

XGBoost/GBM point regression

INFERENCE：可留可删，不影响核心结论。

它主要证明：

改善不是 quantile loss 特有的。

但 R0 的主问题不是 model leaderboard，所以不应让 estimator 数量膨胀。

暂时不加 conformal
裁决：不建议作为 R0 正式基线

Conformal 是有价值的后续 calibration tool，但 R0 当前首先需要判断：

条件资源分布到底能不能学出来。

加入 post-hoc conformal 会混淆：

estimator 本身变准；

还是 interval 被后处理扩大。

近年工作也明确指出，相关/非交换数据会改变 conformal coverage 行为，因此不能简单把 node-level 样本当 i.i.d. 使用。
Proceedings of Machine Learning Research
+1

Conformal Predictions under Markovian Data — ICML 2024

长尾

runtime top-10% 单列是正确的。

建议主报告使用：

T
tail
	​

=q90(runtime ∣ train)

冻结阈值。

然后 validation/test/holdout 都用同一个 tail 定义：

runtime>T
tail
	​

.

比“每个 split 自己选 top 10%”更可比。

5. G1–G4
裁决：当前版本 REJECT；保留思路，重写 gate

这是草案第二个最需要改的地方。

G1

条件中位数比全局中位数 MAE 改善 ≥25%。

裁决：降为 diagnostic

UNVERIFIED：25% 是工程阈值，不是文献常数。

更重要的是它不是 identifiability 的必要条件。

例如：

runtime=f(clip_len)

高度连续，按类别 median 可能几乎没有改善，而 LightGBM 能改善 40%。

不能因为 G1 失败就停止。

G2

LGBM quantile 比 conditional median pinball 改善 ≥10%。

裁决：降为 diagnostic 或 practical-improvement gate，不得单独 veto

反例也很明显：

如果完整 signature 的条件中位数已经接近最优：

LGBM improvement=4%,

接口仍然完全可能是高度可辨识的。

所以：

G1∨G2 failure⇒unidentifiable

这个逻辑应删除。

G3

[P50,P95] coverage ∈ [0.90,0.98]

裁决：REJECT — P0

校准正确时：

P(q
0.50
	​

≤Y≤q
0.95
	​

)≈0.95−0.50=0.45.

不是 0.90。

Quantile regression 的 prediction interval 本身就是通过两个对应的上下分位数定义目标覆盖率。
Proceedings of Machine Learning Research

因此有两个选择。

我更推荐方案 A，因为 scheduler 更在意 underprediction：

检查：

Coverage
q50
	​

=P(Y≤
q
^
	​

50
	​

)≈0.50
Coverage
q90
	​

≈0.90
Coverage
q95
	​

≈0.95.

定义：

QCalError=mean
τ∈{.5,.9,.95}
	​

∣
P
^
(Y≤q
τ
	​

)−τ∣.

例如：

overall ≤ 0.05；

major stratum ≤ 0.10。

这两个阈值仍是 UNVERIFIED engineering margins，但统计定义正确。

方案 B：

新增 q05，然后：

[q05,q95]

检验 90% empirical coverage。

G4

holdout pinball 不得比 validation 恶化 >15%。

裁决：MODIFY

15% 可以作为预注册 alarm，但不要把它描述成科学通用阈值。

更稳健的是在每个 split 都先算相对同一个 train baseline 的 improvement：

I
s
	​

=
Loss
baseline,s
	​

Loss
baseline,s
	​

−Loss
R0,s
	​

	​

.

然后要求：

holdout 上 I
holdout
	​

>0；

video-cluster bootstrap CI 不支持大规模反转；

validation→holdout degradation 15% 可作为 secondary stability margin。

这样避免 validation loss 很小时，简单 ratio 爆炸。

我建议的 go/no-go 逻辑

不要：

G1∧G2∧G3∧G4.

改成：

Core GO

最佳预注册 R0 模型在 validation 上：

相对 global/coarse baseline 有稳定且实际有意义的 improvement；

q95 underprediction calibration 可接受；

inclusive runtime tail 没有灾难性失败；

memory 在 instrumented major classes 上可解释。

G1/G2

作为 mechanism diagnostics：

improvement 来自 categorical signature，还是 workload/context？

Frozen holdout

作为一次性 generalization acceptance，不参与任何改选。

6. 最大的乐观偏差/泄漏风险
裁决：MODIFY
6.1 同一 video/run 内节点相关性

视频级 split 已经很好地解决了最严重的 train→validation leakage：同一视频的所有 run 都不会跨 split。

chatgpt_p9d_r0_design_brief_zh

所以：

VERIFIED：run 内相关节点不会出现在训练与验证两侧。

但还有另一种问题：

统计单位错误。

一个 30-node run 不能被当成 30 个独立实验单位。

否则：

长 run 权重过大；

CI 过窄；

高频 workflow/model dominating。

推荐 primary statistics
source-video cluster bootstrap
	​


即以 video 为 resampling unit，video 内所有 runs/nodes 整块保留。

同时报告：

node-pooled metric；

per-video macro metric；

bootstrap CI。

这是比“节点级 bootstrap”重要得多的防乐观控制。

6.2 per-run sensitivity
裁决：ACCEPT，但不是另一个 split

因为 run 嵌套于 video，video-level group 已经是更保守的 leakage boundary。

所以不要再建立“run-level train/test split”。

应该做：

Metric
run macro
	​

=mean
r
	​

Metric(nodes
r
	​

)

与：

Metric
video macro
	​


以及 pooled node metric 做敏感性比较。

如果结果只有 node-pooled 好、per-run/per-video 显著差：

模型很可能只在长 run 或高频 stack 上表现好。

6.3 exact model ID shortcut

INFERENCE：这是合法 shortcut，但要明确 scope。

如果同几个 model 在 train/val/holdout 都出现，那么 exact model ID 很可能解释大量 memory/load variance。

这不是 leakage——model identity 确实影响资源。

但它意味着 R0 结论是：

known-model regime 下可辨识。

而不是：

unseen model generalization 已解决。

用前面建议的 coarse/exact ablation即可暴露这一点。

6.4 stack/baseline shortcut

同理。

stack/baseline 如果在执行前已知，可以使用。

但需要分层报告，避免总体好结果只是因为：

stack→fixed resource pattern.
7. Stage 0 / smoke / formal 是否足够？
裁决：MODIFY 后接受

草案现有 Stage 0、smoke、manifest/hash/artifact 方案已经相当完整。

chatgpt_p9d_r0_design_brief_zh

我建议 Stage 0 新增 6 个 hard assertions：

P0/P1	Gate
P0	load_ms 不得参与任何 input/regime 派生
P0	memory counter 的 node-window/reset/baseline/device 语义已冻结
P0	nested Summarizer 不重复计算 internal-call resource
P0	feature whitelist 逐列标注 pre_execution / oracle_label / target_only
P1	conditional-median fallback hierarchy 与 support count 固定
P1	tail/stall threshold 仅从 train 构造并冻结

另外 formal statistics 固定为：

video-grouped split+video-clustered bootstrap

并增加：

per-run macro

敏感性报告。

不需要另做 run-level split。

RESULT / 论文应该怎样限定 R0

建议把核心 claim 限制成：

VERIFIED if R0 passes: 在冻结的 v3.1 execution-node contract、已观测 workload descriptors、pre-execution context 与 oracle node signature 条件下，所测节点类别的 runtime/load/memory 具有可利用的经验可辨识性。

绝对不要写：

resource prediction solved.

也不要写：

future resources are predictable.

更不能写：

topology predictor can predict the resource signature.

R0 没测这些。

一个好的层次关系是：

R0:oracle Z→R

通过后才值得问：

J2:
Z
^
→R.

如果 R0 都失败：

Z
^

再准确也救不了当前接口。

最终 7 问裁决
问题	裁决	核心原因
1. R0 定位	接受（修改命名）	是 empirical oracle ceiling，不是理论 upper bound
2. 测量契约	修改	load→regime 是 P0 泄漏；stall 应主表保留；memory 仅能对 instrumented subset 下结论
3. 签名/特征	修改	exact model 可用于 ceiling，但要 coarse ablation；严格区分 oracle label 与 pre-execution context
4. baseline/metric	接受并修改	median+LGBM 足够；暂不需 conformal；tail 用 train-frozen threshold
5. G1–G4	拒绝当前形式	G1/G2 不是必要条件；G3 数学错误；G4 应改成 relative generalization stability
6. 风险	修改	video split 防泄漏，但统计必须 cluster at video；exact model/stack scope 要限定
7. 执行	接受并增加 gates	加测量语义、feature-role whitelist、video-cluster bootstrap；run-level做敏感性而非新 split
冻结前 P0 清单

我只会阻塞 R0 的三件事：

P0-1：删除所有 load_ms → regime/context 路径。

P0-2：修正 G3。
要么用 q50/q90/q95 各自的单侧 calibration；要么增加 q05 后检查 [q05,q95] 的 90% coverage。

P0-3：明确 memory measurement semantics。
必须能够声明当前 peak_* 是合法的 node-level measurement；若它实际是未重置的 process-global peak，则 memory R0 要先从 go/no-go 中移出，而不是用一个不正确的 target 强行做实验。

这三项清掉后，我的总体结论是：

R0 DESIGN ACCEPT
	​


而且我建议先跑 R0，再决定是否继续 J 系列。特别值得关注的不是“LightGBM 能不能击败一个 baseline 10%”，而是结果最终属于哪一种：

oracle signature 足够，normal 与 inclusive 都好

→ 直接进入 J；

normal 好，inclusive 差

→ 缺的是 cold/startup/stall system context；

runtime 好，memory 差

→ 检查 memory measurement / signature；

full oracle signature 仍整体差

→ 此时才真正有证据暂停 J，扩展 token/frame/resolution/context contract。

## 7. user
附件上传控件故障，改为正文直接粘贴《P9d R0 oracle-signature 经验资源上界实验设计 v2》（评审修订版）。
请按文末「复核问题」逐条回答；若 3 个 P0 已关闭且无遗留，请明确确认 “R0 DESIGN v2 ACCEPT（可冻结）”。
请用中文回答，VERIFIED / INFERENCE / UNVERIFIED 标签保留英文原词。

---

---
research_task: P9d R0 oracle-signature 经验资源上界实验设计 v2（评审修订版）复核
project: VideoSeek 调度研究 · predictor block
status: 脱敏评审包（无凭据、无原始 trace、无本地绝对路径）
language: 中文（请用中文回答；VERIFIED / INFERENCE / UNVERIFIED 标签保留英文原词）
---

# 复核请求

上一轮你裁决 R0 设计“定位成立，但 MODIFY 后冻结”，并给出 3 个 P0 与若干修改。现按你的意见更新为 v2。请核查：(a) 3 个 P0 是否已正确关闭；(b) 其余修改是否落实且自洽；(c) 是否还有遗留问题；若无阻塞，请确认该版本可冻结。

# v1 → v2 变更记录

| 评审意见 | v2 处理 | 位置 |
|---|---|---|
| P0-1 删除 load_ms → regime/context 全部路径 | regime 改为 residency_state ∈ {resident, nonresident, unknown}，只由执行前可见的 model_resident_before 派生；load_ms 仅作为 target，并新增 Stage 0 断言“load_ms 不得出现在任何 input/regime 派生” | §4、§5、§9 |
| P0-2 修正 G3 覆盖率 | 删除 [P50,P95] ∈ [0.90,0.98]；改为逐分位单侧校准 QCalError = mean_τ |P̂(Y≤q̂_τ) − τ|（τ∈{.50,.90,.95}），总体 ≤0.05、主要分层 ≤0.10（标记为工程余量）；可选扩展：加预测 q05 后检查 [q05,q95] 的 90% 覆盖 | §7、§8 |
| P0-3 memory 测量语义 | Stage 0 硬断言：必须证明 peak_* 是 node-window / reset / 基线校正 / 设备正确；若不可证明，memory 从 go/no-go 移出，仅作 descriptive；所有 memory 结论限定在 instrumented subset | §5、§9、§10 |
| stall 不得从主表删除 | 主表 **R0-Inclusive**（全部合法正测量）；副表 **R0-Normal**（train 冻结 stall 规则）；tail 用 train 冻结 q90 阈值 | §5、§7、§10 |
| G1/G2 降级 | G1/G2 改为机制诊断，不参与 veto；go/no-go 重写为 Core GO + diagnostics + frozen holdout acceptance | §8 |
| 统计单位 | 主统计改为 **video-cluster bootstrap** + 报告 node-pooled / per-run macro / per-video macro；不建 run-level split，run 级仅作敏感性 | §3、§7 |
| 签名/特征边界 | 新增 feature role 白名单（pre_execution / oracle_label / target_only）；明确 context 允许/重分类/禁止项；exact model 保留但加 coarse ablation | §6 |
| 消融 | 新增 **C0–C4 递增 contract ablation** | §6.3、§8 |
| 命名 | 正式命名改为 **oracle-signature empirical resource ceiling**（冻结接口与估计器族下的经验上界），删除“theoretical upper bound”表述 | §1、§11 |
| conformal / XGB | R0 不加 conformal；XGB 点回归标记为可选对照 | §7 |
| Stage 0 gate | 新增 6 项硬断言 | §9 |
| 结论限定 | 新增 claims 边界与四象限决策表 | §10、§11 |

# 1. 定位与命名

- 名称：**R0：oracle-signature empirical resource ceiling（冻结接口与估计器族下的经验上界）**。
- 形式化：对每个 resource-applicable 计算节点，R₀: (Z_oracle, workload_scale, c_pre) → (runtime, load, memory)。
- 定位：只回答“当未来节点是什么已知（oracle 签名）时，当前接口能把单节点资源估到多准”；作为 J 系列的 go/no-go gate。
- 非目标：不预测结构/签名、不接调度器、不产出可部署模型、不计算聚合 future cost、不触碰 S_*/T_final。
- 明确不是理论界：结果受估计器族、当前 Z、当前 workload descriptors、当前测量质量限制；LightGBM 失败不能证明任何模型失败。

# 2. 数据与划分

- 数据：v3.1 标签（16,841 链节点中 15,481 个 resource-applicable；planner 7,211、post-loop 生成 1,192、工具 7,078）+ 原始 trace 资源字段的标签侧 join。
- 划分：P_dev/train 拟合 → P_dev/validation 选型 → P_dev/test 诊断 → P_holdout_diag 冻结后一次性验收；视频级划分（同一视频所有 run 同侧）。
- **统计单位**：以 video 为 resampling unit 的 cluster bootstrap；同时报告 node-pooled、per-run macro、per-video macro；不做 run-level split。

# 3. 目标与测量契约

- 目标：runtime_ms（主）、load_ms（次，仅 target）、peak_memory_inclusive_mb = max(peak_reserved_mb, peak_allocated_mb)（次，条件见 P0-3）。
- 有效行：status=success 且目标为正有限值；缺失不得填 0；逐分层报告覆盖率。
- **residency_state**（唯一 regime 定义，执行前可见）：model_resident_before=false → nonresident；true → resident；缺失 → unknown。禁止使用 load_ms 或任何执行后字段派生 regime。
- **stall 规则**（仅由 train 构造并冻结）：runtime_ms > 10 × median_train(exec_class) 或 > 300,000 ms（先触发者）。主表 R0-Inclusive 保留全部合法行；R0-Normal 副表排除 stall 行；stall 行数与分布必须单独报告。
- tail 阈值：T_tail = q90(runtime | train)，validation/test/holdout 统一使用，不按 split 重选。
- memory 语义：Stage 0 必须断言 peak_* 为 node-window/reset/baseline 校正/设备正确；否则 memory 仅 descriptive（见 §9、§10）。

# 4. 签名、workload_scale 与上下文（特征角色白名单）

- **Z（oracle 签名，标签侧）**：exec_class（node_type + raw_action）、role、model_class（exact model_id；另设 coarse 版本）、planner_mode、merged_nested_call 与 nested_model_class、residency_state。
- **workload_scale**：clip_len、query_char_len、nested_api_call_count + 缺失标志。
- **c_pre（执行前上下文）**允许：stack/baseline 标识；residency_state；已固定的硬件类别（若实验硬件固定）；当前 anchor 之前已完成 prefix 的统计。重分类：chain position 仅作 oracle structural context 标注；previous node class 仅当其指“节点开始时已完成的前驱”时才合法，否则标示为 oracle topology 信息；retry 归入节点签名而非上下文。**禁止**：执行后 frame count、实际 decode/output token 数、执行后 qwen_image_count、load/runtime/memory、后验 batch/interference 状态。
- **feature role 白名单**：Stage 0 逐列标注 pre_execution / oracle_label / target_only；特征只用前两类；target_only 永不进特征（P0-1 的具体检查项）。
- **exact vs coarse**：R0_exact 用 exact model_id；R0_coarse 用 model resource class（来自静态属性/配置或 train-only 交叉拟合分组，禁止查看节点自身 runtime/memory 生成 class）。

# 5. 消融（C0–C4 递增 contract ablation）

C0=exec_class → C1=+role → C2=+model → C3=+workload_scale → C4=+residency/context。报告每级 ΔMAE、Δpinball，用于回答“哪个变量修复了可辨识性”。（G1/G2 机制诊断即由此产生。）

# 6. 估计器与基线

1. 全局中位数（train）——粗基线。
2. 条件中位数：按完整签名键分层 + **预注册回退层级**（逐级 key 简化）；记录每级 support count 与 validation/test/holdout 实际使用的回退级。
3. LightGBM quantile（τ=0.50/0.90/0.95，log1p 目标，C4 特征）——主模型；可选加 q05 用于 [q05,q95] 扩展报告。
4. （可选对照）XGBoost 点回归（log1p）。
- 暂不加入 conformal（避免混淆“估计器变准”与“区间被后处理扩大”；相关/非交换数据的 coverage 行为本身不确定）。

# 7. 评估指标

- 逐目标：MAE、median AE、pinball（τ=0.50/0.90/0.95）、逐分位单侧校准 QCalError、q95 低估率、signed bias。
- 区间描述：[q50,q95] 经验覆盖率（作为描述，不做 90–98% 断言）、以及可选 [q05,q95] 的 90% 覆盖报告。
- 分层：exec_class × model_class、residency_state、workload_scale 分位桶、tail（runtime > T_tail）；每格报告 n 与覆盖。
- 聚合：node-pooled、per-run macro、per-video macro、video-cluster bootstrap CI。
- 泛化稳定性：I_s = (Loss_baseline,s − Loss_R0,s) / Loss_baseline,s，要求 I_holdout > 0 且 bootstrap CI 不支持大规模反转；validation→holdout 相对退化 ≤15% 仅作次要稳定性余量（非科学常数）。
- 结论限定：memory 指标只能在 instrumented/observed subset 上表述，并给出 coverage(memory | exec_class)、coverage(memory | model_class)。

# 8. Go/No-Go（重写）

- **Core GO**（全部满足）：验证集上最佳预注册 R0 模型相对 global/coarse baseline 有稳定且实际有意义的 pinball/MAE 改善（工程余量提案：中位 pinball 改善 ≥15% 且 bootstrap CI 下界 >0，标记为项目余量）；q50/q90/q95 的 QCalError 总体 ≤0.05、主要分层 ≤0.10；R0-Inclusive 的 tail 无灾难性失败（tail 子集误差单独报告）；memory 在 instrumented 主要类别上可解释（若 P0-3 未通过，改用 runtime + load 作为 Core GO，memory 仅 descriptive）。
- **机制诊断（不 veto）**：G1 条件中位数 vs 全局中位数 MAE 改善；G2 LightGBM vs 条件中位数 pinball 改善；C0–C4 逐级增益。
- **冻结 holdout**：一次性验收，不参与任何改选；用 I_holdout 与 bootstrap CI 判定泛化稳定性。
- **四象限决策表**（R0 之后）：
  1. Normal 与 Inclusive 都好 → 直接进入 J 系列；
  2. Normal 好、Inclusive 差 → 缺的是 cold/startup/stall 的执行前系统上下文；
  3. runtime 好、memory 差 → 检查 memory 测量语义与签名覆盖；
  4. 全 oracle 签名仍整体差 → 才有证据暂停 J，扩展 token/frame/resolution/context 契约。

# 9. Stage 0 硬断言与执行计划

Stage 0（P0，全部通过才进入 smoke）：
1. load_ms 不出现在任何 input/regime 派生路径（P0-1）；
2. peak_* 的 node-window/reset/baseline/device 语义已冻结，或 memory 已降级为 descriptive（P0-3）；
3. nested Summarizer 不重复计算内部调用资源（R_node = R_total）；
4. feature role 白名单逐列完成（pre_execution / oracle_label / target_only）；
5. 条件中位数回退层级与 support count 固定（P1，预注册）；
6. stall/tail 阈值仅由 train 构造并冻结（P1）。

Smoke：约 20 run / ≤300 节点的最小链路，检查有限值、分层计数、join 覆盖率、产物哈希。
正式：全量 + holdout 一次性评估；本地 CPU，预计 < 10 CPU-min；无 GPU、无下载。
正式前更新 .project/EXPERIMENT_GATE.json；实验 ID 建议 EXP-20260911_p9d_r0_oracle_signature_ceiling。

# 10. 交付物与可复现性

- 实验目录：config.json、run.sh、metrics.json、RESULT.md、逐节点预测/真值 artifacts（label 侧）、run_manifest.json（源哈希、seed、环境、命令、阈值）。
- 唯一控制面更新；v1–v3 数据集与既有实验只读。

# 11. 结论限定（claims）

- 允许：在上述冻结契约、已观测 workload descriptors、执行前上下文与 oracle 签名条件下，所测节点类别具有**经验可辨识性**（empirical ceiling）。
- 禁止：resource prediction solved；future resources are predictable；topology predictor can predict the resource signature；observed-memory subset 结果外推为全部节点。

# 复核问题

1. 3 个 P0 是否已正确关闭？是否还存在把 target 信息泄漏进特征或统计的路径？
2. residency_state 的三值定义与“不得使用 load_ms”的断言是否足以替代原 cold/warm？
3. QCalError 的定义与阈值（0.05/0.10）是否可作为预注册？是否需要同时预注册 [q05,q95] 扩展？
4. Core GO 的“实际有意义改善”工程余量（中位 pinball ≥15% 且 CI 下界 >0）是否可接受，还是应改为相对 baseline 的另一种形式？
5. C0–C4 消融与 exact/coarse 对照是否覆盖了你要的机制诊断？
6. 四象限决策表与 claims 边界是否准确？
7. 还有没有遗留的 P0/P1 或应删除的内容？若没有，请明确确认 “R0 DESIGN v2 ACCEPT（可冻结）”。

# 约束

- 不训练 topology/behavior 模型、不接入调度器、不读取 S_*/T_final；不改动 v1–v3 数据集与既有实验；R0 只新增实验目录与产物；测量资源不得作为特征；holdout 不参与选型/校准。

## 8. assistant
执行摘要

这版已经比 v1 严密很多。我的复核结论是：

原先的 3 个 P0 均已按要求正确关闭。
但我发现了 1 个新的 P0：load_ms=0 的目标语义尚未冻结。另外还有几项不构成数据泄漏、但应在正式运行前写死的 P1，包括 Core GO 的 primary pinball 定义、“主要分层”的确定规则、quantile crossing、LightGBM 选型/超参数规则，以及 holdout failure 后的处理规则。

因此我现在还不能给出“R0 DESIGN v2 ACCEPT（可冻结）”。只要把下面这个新的 load P0 处理清楚，并补齐几个 P1 的预注册歧义，我预计不需要再改实验框架。

总体裁决表如下：

复核项	裁决
原 P0-1：load→regime 泄漏	VERIFIED 已关闭
原 P0-2：错误的 [q50,q95] 90% coverage	VERIFIED 已关闭
原 P0-3：memory measurement semantics	VERIFIED 已条件关闭
新问题：load_ms=0 是否是真实“无需加载”	P0，需冻结语义
residency_state	接受
QCalError	修改后接受
Core GO 15% margin	修改后接受
C0–C4 + exact/coarse	接受，需补齐 feature-set 定义
四象限决策	基本接受，措辞需从“结论”降为“诊断指向”
video-cluster bootstrap	接受
1. 三个原 P0 是否已经关闭？还有没有 target 泄漏？
P0-1：load_ms → regime/context

裁决：ACCEPT，VERIFIED 已关闭。

现在 residency_state 仅由执行前的：

model_resident_before

得到，而 load_ms 被明确列为 target_only，Stage 0 还会检查它不出现在任何 input/regime 派生路径。

这已经解决了上一版最直接的 target leakage。

现在的边界是正确的：

X: residency_before
Y: load_ms.

而不是原来的：

Y→X.
P0-2：quantile coverage

裁决：ACCEPT，VERIFIED 已关闭。

你已经删除了：

[q50,q95]≈90%

这一错误要求，改为：

C
τ
	​

=
P
^
(Y≤
q
^
	​

τ
	​

)

并比较：

∣
C
τ
	​

−τ∣.

这是正确的 quantile calibration 思路。Quantile regression 本身就是通过估计条件分位数来表达不确定性；若构造双侧 nominal interval，则需要相应的上下分位点。
Proceedings of Machine Learning Research

P0-3：memory measurement semantics

裁决：ACCEPT，VERIFIED 已条件关闭。

这版设计的处理非常合理：

能证明 node-window/reset/baseline/device 语义 → memory 进入 go/no-go；
证明不了 → memory 自动降级为 descriptive。

这意味着 instrumentation 不确定性不会再把整个 R0 阻塞掉，同时也不会拿不可靠 memory target 做正式资源结论。

我赞成。

新 P0：load_ms=0 到底是什么？

这是我认为当前唯一真正需要阻塞 freeze 的地方。

裁决：MODIFY — P0

当前目标有效行定义仍是：

status=success∧target>0.

但是 load_ms 和 runtime/memory 不一样。

在你们前一版契约中曾经出现过：

load_ms=0⇒warm

的解释。

这意味着数据里至少可能存在这样一种语义：

load_ms=0=合法观测：本节点没有发生模型加载.

如果是这样，那么现在的：

load_ms>0

过滤会把所有“无需加载”的真实样本删除。

此时 R0 测到的不是：

p(load_ms∣X)

而是：

p(load_ms∣X, load_ms>0),

即：

已经发生 load 的条件下，load 要多久。

这是完全不同的问题。

必须在 Stage 0 冻结两种情况中的一个

情况 A：0 是合法 no-load measurement。

那么 load target 应允许：

load_ms≥0.

并且最好概念上拆成：

P(load event∣X)

以及：

P(load duration∣load event,X).

不一定必须真的实现 hurdle model；但至少 overall load metric 不能把真实 0 删除。

情况 B：0 是 instrumentation sentinel / 无效值。

那就明确写进 measurement contract：

load_ms=0 不代表 no-load，而代表 unavailable/invalid。

并且缺失状态必须与真实 0 可区分。

为什么这是 P0

因为 load 在当前 R0 是三个正式 resource target 之一；memory 降级后，它甚至可能成为 Core GO 的一部分。

在 0 的语义没定义之前：

“load 可辨识性”

没有稳定 target definition。

所以这里必须先冻结。

2. residency_state 三值定义是否足够？
裁决：ACCEPT

我认为：

{resident, nonresident, unknown}

已经足够作为第一版执行前 residency context。

VERIFIED

它满足最重要的 causal requirement：

节点开始之前就已知。

而不是通过执行后 load duration 反推。

INFERENCE

我建议继续用 residency_state，不要再改回 cold/warm。

因为：

nonresident

表示的是“执行前模型不驻留”。

它并不严格保证：

一定会产生某个固定的 load latency.

加载仍可能受：

checkpoint locality；

I/O；

worker state；

implementation；

等影响。

因此这个名字更精确。

近期系统工作也反复显示 model/configuration/workload context 对推理性能建模很重要；例如 Vidur 就把 profiling 和 configuration-aware predictive modeling 结合起来，而不是假设单一资源类别决定运行时间。
MLSys 会议录

3. QCalError 是否可以预注册？需不需要 q05？
裁决：MODIFY 后 ACCEPT

定义：

QCalError=
3
1
	​

τ∈{.5,.9,.95}
∑
	​

∣
P
^
(Y≤
q
^
	​

τ
	​

)−τ∣

本身是合理的 descriptive calibration score。

你提议：

overall ≤ 0.05；

major strata ≤ 0.10；

也可以作为 UNVERIFIED project engineering margins 预注册。

不过我建议做两个小修改。

第一：不要只 gate 平均值

平均值可能掩盖单个分位点失败。

例如：

e
50
	​

=0.12,e
90
	​

=0.01,e
95
	​

=0.01

则：

QCalError≈0.047

仍然通过。

因此最好同时要求：

∣C
95
	​

−.95∣≤0.05

至少把 q95 单独保护。

更干净甚至是：

τ
max
	​

∣C
τ
	​

−τ∣≤0.05

作为 overall gate。

对 scheduler 来说 q95 比 q50 更值得单独盯住。

第二：“主要分层”必须提前定义

不能等看到结果以后决定哪个 stratum 算“主要”。

建议用与结果无关的 support rule，例如：

仅对 validation 中达到预注册最小 node count 且覆盖若干独立 source videos 的 strata 应用 0.10 gate；其余只 descriptive。

具体 n 阈值是项目选择，属于 UNVERIFIED，但必须先写死。

q05 是否必须？

不必须。

R0 主要关心：

q50,q90,q95

尤其是上尾 underprediction。

所以单侧 quantile calibration 已足够。

如果你想报告真正的 90% prediction interval，那么：

[q05,q95]

当然是干净的 secondary output。Quantile regression 构造这类 interval 是标准做法。
Proceedings of Machine Learning Research

但这里建议二选一：

正式运行前就把 q05 固定成 secondary；

或者本轮彻底不做。

不要保留“跑完以后可选加”，否则 preregistration 不够干净。

这是 P1，不是 P0。

4. Core GO 的 ≥15% 改善是否合理？
裁决：MODIFY

15% 本身可以作为：

UNVERIFIED project practical-effect margin

没有问题。

问题在于当前“中位 pinball 改善 ≥15%”还存在三个歧义。

4.1 “中位 pinball”是什么意思？

如果你指：

Pinball
τ=.5
	​

,

那它过度偏向典型样本，对 q90/q95 不够敏感。

我更建议 Core GO 的 primary resource loss 定义成：

PB
primary
	​

=
3
PB
.50
	​

+PB
.90
	​

+PB
.95
	​

	​


对 runtime 做主 gate。

然后：

Improvement=1−
PB
baseline
	​

PB
R0
	​

	​

.

要求：

Improvement≥15%

且 video-cluster bootstrap 的改善 CI 下界：

>0.

这样“15%实际意义”与统计方向同时存在。

4.2 baseline 必须唯一

现在写的是：

global/coarse baseline

这还是有歧义。

正式 freeze 时应指定一个 Core baseline，例如：

B
core
	​

=exec_class-conditional train median

或者其他你们认为最合理的 coarse-only reference。

Global median 继续作为 very weak baseline 报告即可。

否则结果出来后选：

“比较起来更好看的那个 baseline”

会留下 researcher-degree-of-freedom。

4.3 pinball 在什么 scale 上算？

模型训练可以用：

log1p(Y).

但 Core GO 必须说明 pinball metric 是：

log-scale；

raw physical scale；

还是 train-normalized raw scale。

我建议：

训练：log1p。

正式资源评价：raw physical target 上计算 pinball；跨 resource 比较时再做 train-only normalization。

因为 scheduler 真正承担的是：

ms, MB

上的 underprediction，而不是 log error。

对于 runtime 单目标相对 baseline 的比例改善，直接 raw-scale pinball 就可以。

这是 P1，必须在 run 前写死。

5. C0–C4 + exact/coarse 是否足够？
裁决：ACCEPT，补一处定义

这套递增消融比单独 role-only/model-only 更好：

C0→C1→...→C4

可以直接看：

哪组信息消除了条件方差。

我赞成。

需要补充

完整 Z 里还有：

planner_mode

merged_nested_call

nested_model_class

retry

而当前 C0–C4 的描述没有明确这些字段在哪一级进入。

所以冻结时建议写成一个真正的 feature manifest，例如：

C0={exec_class}
C1=C0+{role,planner_mode,retry}
C2=C1+{model,nested_model,merged_nested}
C3=C2+workload
C4=C3+preexecution context.

具体怎么归类可以由你们决定；关键是不要结果出来后移动字段。

exact/coarse

也足够。

但 INFERENCE：R0_exact 代表：

known-model empirical ceiling

不能外推成 unseen-model generalization。

R0_coarse 更接近你后续想要的 resource-equivalence abstraction。

6. 四象限决策表与 claims 是否正确？
裁决：基本 ACCEPT，但四象限要改成“诊断假设”

claims 边界现在写得很好。

尤其是明确禁止：

resource prediction solved；

future resources are predictable；

topology predictor can predict signature；

memory observed subset → 全节点。

这些限制正确。

四象限需要稍微弱化

例如：

Normal 好、Inclusive 差 → 缺 cold/startup/stall context

这个方向很合理，但不是逻辑必然。

也可能是：

instrumentation glitch；

rare failure mode；

unmodeled contention；

estimator capacity；

true stochastic heavy tail。

所以最好改成：

优先检查 cold/startup/stall 的 pre-execution context 与测量质量。

同理：

full oracle 仍差

首先说明：

current interface+measurement+estimator family

不足。

然后才依次检查：

measurement quality；

estimator misspecification；

missing token/frame/resolution/context；

是否值得暂停 J。

因为 R0 自己已经明确不是 theoretical bound，所以 decision table 也应该遵守这个限制。

7. 还有哪些遗留 P0/P1？
新 P0

只有我上面指出的这一项：

load zero semantics 必须冻结。

如果 load_ms=0 是合法“无需加载”，就必须保留这些样本或拆成 occurrence + positive duration；如果 0 是 invalid sentinel，则明确记为 missing。

在此之前，我不能确认整个三目标 R0 contract 已经稳定。

其余都是 P1

我建议正式 run 前一并写死，但不要求再重新设计 R0：

P1	建议
Core pinball	明确 τ 聚合方式；建议 mean(.50,.90,.95)
baseline	指定唯一 Core baseline
metric scale	明确 raw/log/normalized；建议 gate 用 raw target
major strata	用结果无关的 n/video-count rule 定义
q95	加单独 calibration gate，不只看平均 QCalError
q05	正式前决定“有”或“无”，不要保留运行后可选
quantile crossing	预注册 crossing rate 报告及处理方式
C0–C4	固定每一级精确字段
LGBM selection	固定 hyperparameter/tuning grid、seed、selection metric
Inclusive/Normal	明确 Normal 是同一个 inclusive-trained model 的 subset diagnostic，还是另训 filtered model；我建议前者
holdout failure	明确失败后若改 contract/model，则当前 holdout 不再具备 confirmatory 身份
关于 video-cluster bootstrap

这一部分我认为设计是对的。

节点明显嵌套在 run/video 内，简单按 node 当独立样本会低估不确定性。Cluster bootstrap 的标准思想就是整簇重采样，保留簇内相关结构，而不是逐 observation 独立抽样。
OUP Academic

因此：

video

作为 primary resampling unit 是合理的。

同时报告：

node-pooled；

per-run macro；

per-video macro；

也能识别“性能只来自长 run”的问题。

不需要另建 run-level split。

最终裁决

原 3 个 P0：

全部正确关闭
	​


但出现一个新的 measurement-contract P0：

load_ms=0 的语义
	​


所以目前我的结论是：

R0 DESIGN v2：尚不能最终 ACCEPT 冻结，仅差这一项 P0 + 若干 P1 预注册细化。

如果你们确认并写入合同：

load_ms=0 到底是合法 no-load measurement 还是 invalid/missing sentinel，

并据此正确处理 0 样本；同时把上面的 Core baseline / pinball 聚合 / major-stratum / quantile crossing 等 P1 写进 config/contract，我不会要求再增加新的实验阶段。

届时我的预期最终裁决就是：

R0 DESIGN v2 ACCEPT（可冻结）。
