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
timestamp: "2026-09-11T15:56+08:00"
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
message_count: "9"
timestamp: "2026-09-11T15:56+08:00"
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

## 4. user
chatgpt_p9d_r0_design_brief_zh.md
文件
请评审附件中的《P9d R0 oracle-signature 资源上界实验设计草案》（中文）。

要点：
1. R0 用 oracle（真实标签侧）资源等价签名，测量单节点 runtime/load/memory 的可辨识性上界，作为 J 系列联合训练的前置 go/no-go gate；前提是已冻结的 v3.1 串行控制流标签契约。
2. 请按草案文末的 7 个问题逐条裁决（接受 / 修改 / 拒绝 + 理由），特别关注：测量契约（cold/warm、stall、缺失、memory 覆盖）、签名与特征边界、预注册阈值 G1–G4 是否合理、以及防乐观/泄漏（同一 run 内节点相关性）。
3. 标出任何新的 P0 风险与对设计的具体修改建议。
4. 请用中文回答；VERIFIED / INFERENCE / UNVERIFIED 标签保留英文原词。不要索取原始 trace、本地路径、模型权重或凭据；本评审只作为研究输入，不自动改变我们的代码或实验。

## 5. assistant
有遗留阻塞项，暂不能给出最终冻结裁决。

P0 — load 非劣性 CI 方向写反了。
VERIFIED：Brier 和 LoadDurationQ 都是“越低越好”的 loss。若定义

Δ
load
	​

=Metric
Jx
	​

−Metric
B1
	​

,

则非劣性应限制“J 不能高太多”，所以应为：

CI
upper
	​

(Δ
Brier
	​

)<+0.005

和类似

CI
upper
	​

(Δ
Duration
	​

)<+5%

而当前写的“CI 下界 > −0.005 / −5%”保护的是相反方向，不能阻止 J 显著变差。

P0 — checkpoint NI 对不同指标统一用了错误方向。
VERIFIED：future content accuracy、next-role accuracy、next-family accuracy 是越高越好，此时

CI
lower
	​

(Δ)>−δ

正确。
但 future length MAE、若 termination 使用 BCE/NLL，则是越低越好，必须改成

CI
upper
	​

(Δ)<+δ.

需要把 NI manifest 明确写成“每个 endpoint 的 metric + direction + δ”，不能对所有 endpoint 统一用 lower-bound 规则。

P0 — Core success 必须明确在冻结后的 P_dev/test 上判定。
VERIFIED：validation 已用于 NI-feasible checkpoint 筛选和 argmin RuntimeQScore。因此正式

CI
upper
	​

(Δ
rt
	​

)<0

不能再在同一个 validation 上作为成功证据，否则存在 selection optimism。应明确：

train：拟合；

validation：checkpoint/配置选择；

test：冻结后一次性 Core success + load NI 正式判定；
-旧 holdout：不参与正式结论。

除此之外，B1/J0/J1/J2/J3 的隔离、固定 L_total、T=1 soft conditioning、width 退出目标以及不在首轮引入 GradNorm/PCGrad，我没有发现新的阻塞。

## 6. user
正文提交《J 系列实现规划 v1（工程执行计划）》。科学设计 v3 已冻结；本轮请评审工程实现：数据管线、代码复用边界、structure 头实现细节、训练协议隔离、NI manifest 的 per-epoch 可实现性、环境/预算与里程碑。
请按文末 7 个问题逐条裁决；若无阻塞，请给出 “J IMPLEMENTATION PLAN v1 ACCEPT”。请用中文回答，VERIFIED / INFERENCE / UNVERIFIED 保留原词。

---

---
research_task: J 系列实现规划 v1 评审（工程执行计划）
project: VideoSeek 调度研究 · predictor block
status: 脱敏评审包（无凭据、无原始 trace、无本地绝对路径）
language: 中文（请用中文回答；VERIFIED / INFERENCE / UNVERIFIED 保留原词）
---

# 研究问题

J 系列科学设计（v3，两轮评审后冻结：五变体 J0/B1/J1/J2/J3 + OracleAttr 诊断、runtime 单一 primary、load hurdle 次要点、NI manifest、test 冻结判定）已就绪。现请评审**实现规划**：代码落点与复用、数据管线、模型改造、训练协议、评测实现、环境/算力、门禁与里程碑。请指出工程上的坑与遗漏。

# 1. 依据与范围

- 设计：docs/p9d_j_series_design.md（frozen v3）；标签：topology_predictor_p9d_v3（v3.1 契约）；资源真值来源：R0 已建的节点级实测表（15,481 个 resource-applicable 节点）。
- 范围：本地实现与运行；不改 v1–v3 数据集与现有实验；不接 scheduler；不读 S_*/T_final；holdout 冻结前不可见且不进正式决策。

# 2. 组件与代码落点

1. **数据管线** scripts/build_j_dataset.py（新）
   - 输入：v3 features（prefix 输入，逐字节沿用）、v3 labels（未来 H=5 节点属性）、R0 node table（按 node_id 关联实测 runtime/load/memory）。
   - 输出：results/processed/j_series_dataset_v1/{split}.jsonl.gz，每条 = prefix 引用 + future 节点列表（属性 + 资源目标 + flag）。
   - Stage 0 审计：join 覆盖（runtime 100%、load 期望 ~99.5%、memory ~63% 仅描述）、无泄漏（资源只进 target）、split 与 node 计数不变、N/A mask 正确。
2. **训练/评测** scripts/j_series_train_eval.py（改造 scripts/p9d_future_content_multitask.py，1505 行，复用 GRU encoder/heads/codec/loss/训练循环/stage0 模式）
   - 改造点：① structure 头改为 future length（分类）+ termination（BCE）；② 删除 width head/width loss；③ 内容头从 4 字段扩为 7：exec_class、role、action_family、model_class、merged_nested_call、is_retry、nested_model_class（+N/A mask）；④ 新增资源头（每槽 runtime 分位 τ=.50/.90/.95，log1p；load occurrence BCE；load duration 分位仅正样本）；⑤ 变体开关（encoder frozen/trainable；是否输入 sg(q(A))；属性梯度是否 stop）。
3. **实验目录** experiments/EXP-20260911_p9d_j_series_joint_resource/：config.json、run.sh、stage0_audit.json、artifacts/<variant>/seed_*/checkpoint.pt、predictions_*.jsonl.gz、metrics.json、RESULT.md、run_manifest.json。

# 3. 数据管线细节

- Prefix 输入完全沿用 v3 features（model_input 不含未来/资源；已知与 v2 逐字节一致）。
- 未来节点属性取自 v3 labels 的 future_layers[*].nodes[*]（含 is_retry、merged_nested_call、nested_calls 等）。
- 资源目标按 node_id 关联 R0 节点表：runtime_ms（primary）、load_ms（hurdle；0=合法 no-load，缺失=unavailable）、peak_memory_inclusive_mb（仅描述）。
- structure 目标派生：horizon_length = min(len(future_layers), 5)；termination = 1[termination_status == terminated 且在 H 内]；截断样本不得当作精确长度（沿用 v3 censored 语义）。
- 每 split 行数不变（13,754/2,029/1,520/1,380）；node 语义与 R0 相同（15,481）。

# 4. 训练协议

- 超参沿用旧配置：hidden 128、embeddings 16/12/16、dropout 0.1、batch 256、AdamW lr 3e-4、wd 1e-4、max_history 64、max_epochs 30、patience 6。
- 顺序：**backbone（attribute-only，无资源损失，3 seeds）→ 冻结生成 B1 预测器 → J0/B1/J1/J2/J3 各 3 seeds**；所有变体用**同一 backbone 初始化**（matched seeds）；J0/B1 冻结 encoder。
- Checkpoint 选择（每 epoch，validation）：先算 NI manifest（长度/终止/内容/next-role/next-family，逐 endpoint 的 metric+方向+δ），仅在 feasible 集合内按 argmin RuntimeQScore 选择；无 feasible → 变体 fail。
- 训练中记录 ||g_R||/||g_C||/||g_B|| 与 cosine（仅诊断，不改优化器）。
- 评测（test 冻结后一次性）：Δ_rt = RuntimeQScore_Jx − RuntimeQScore_B1，paired source-video bootstrap（1000）上界 <0；load NI：CI_upper(Δ_Brier)<+0.005、CI_upper(Δ_Duration)<+5%；次要：校准 max_τ≤0.05、q95 低估率、per-class 分层；memory/tail 仅诊断。

# 5. 环境与算力

- 本地安装 torch（当前缺失）：优先 CUDA wheel（匹配现有 driver），失败则 CPU 轮子；本机 RTX 3060 Laptop 6GB。
- 运行数：backbone 3 + J0 3 + B1 3 + J1 3 + J2 3 + J3 3 = 18 个 run；单 run 预计 3–10 分钟；总计 <3 GPU 小时。显存按 batch/序列可调（6GB 余量充足的概率高，但不作硬保证）。

# 6. 门禁与里程碑

- 门禁：EXPERIMENT_GATE 记录（配置/数据哈希/成本/停止条件）→ Stage 0（数据 join/泄漏/计数）→ smoke（约 40 run 子集全链路）→ 正式。
- 里程碑：M1 数据管线 + Stage 0；M2 backbone 训练与验收（结构/内容/行为不低于旧 P9e 参照）；M3 J0/B1（诊断）；M4 J1/J2/J3；M5 test 评测 + RESULT。
- 停止条件：Stage 0 任一失败即停；backbone 未达 NI 参照先修 backbone；J3 出现不稳定仅记录不优化；超出预算（>3 GPU 小时 ×1.5）暂停报告。

# 7. 风险与回退

- torch 安装失败/显存不足 → CPU 模式或缩减 batch（不换远端）。
- 数据 join 不一致（如 node_id 覆盖不足）→ fail-closed，不猜测补齐。
- 旧代码改造引入静默差异 → 用旧实验的 stage0/单元模式做回归（同接口 smoke）。
- 时间超预算 → 先交付 backbone + J0/B1 诊断，J1–J3 顺延。

# 8. 交付物

- 数据集（v1，新目录 + manifest）、训练/评测脚本、实验目录全部产物（checkpoint、预测、metrics、RESULT、run_manifest）、控制面更新（STATE/HANDOFF/DECISIONS/INDEX/EXPERIMENT_GATE）。

# 需要你回答的问题

1. 数据管线（join by node_id、长度/终止派生、N/A mask、censored 语义）是否与冻结设计一致？有无遗漏的泄漏/口径问题？
2. 代码复用边界（改造 1505 行旧脚本 vs 新建）是否合理，还是应拆分出独立的共享模块？
3. structure 头采用"长度分类 + termination BCE"是否是可接受的具体实现？长度分类的类别边界建议（1..5  vs 含 0）？
4. 训练协议（共享初始化、J0/B1 冻结、per-epoch NI-feasible 选择、patience 与 max_epochs）是否有隔离漏洞或过拟合风险？
5. NI manifest 的 per-epoch 实现（小样本 bootstrap、缺失掩码、censor 处理）会不会不可计算或不稳定？需要什么保护？
6. 环境/显存/回退与预算估计是否合理？
7. 里程碑、停止条件与交付物是否足够？若两项都已落地，请明确给出 **“J IMPLEMENTATION PLAN v1 ACCEPT”**；若有阻塞，请只列阻塞项。

# 约束

- 不改动 v1–v3 数据集、R0/R0.1 产物与既有实验；不接 scheduler；不读 S_*/T_final；不重采数据；holdout 保持不可见。
- 请标注 VERIFIED / INFERENCE / UNVERIFIED；不要索取原始 trace、本地路径、模型权重或凭据。

## 7. assistant
执行摘要

工程方向总体正确，但当前实现规划还有 4 个阻塞项，因此我暂时不能给出 J IMPLEMENTATION PLAN v1 ACCEPT。其中最重要的不是模型代码，而是两个会造成静默泄漏/错配的问题：

prefix_model_reuse 不能直接从未来节点真实 model_class 派生后喂给 J 模型，否则 J1/B1/J2/J3 会获得 future-label-derived context；

资源表不能用裸 node_id 做全局 join，必须使用 trace/run 身份 + node_id 的复合键，并区分“15,481 个唯一资源节点”和“anchor×future-slot 展开后的监督实例数”。

另外两个阻塞是：future-length 的 0..5/censor 定义必须落死，以及 patience=6 与 NI-feasible checkpoint 选择的交互目前没有定义。

逐问裁决如下。

1. 数据管线
裁决：MODIFY — 有两个 P0
P0-1：资源 join key 不能只是 node_id

INFERENCE：规划文本只写“按 node_id 关联 R0 节点表”，但没有声明 node_id 在 1,360 run 全局唯一。

因此实现必须用类似：

(source_trace_hash, node_id)

或：

(run_id, node_id)

作为 join key。

Stage 0 必须断言：

resource table 的复合键唯一；

v3 future node 的复合键唯一定位资源节点；

无一对多/多对一异常；

unmatched fail-closed。

另外必须分开记录两种数字：

N
unique−resource−node
	​

=15481

与

N
anchor×future−slot
	​

.

同一个真实 future node 可以从多个不同 prefix anchor 被预测，这是合法的训练实例重复，不能错误要求展开后的 supervision pair 也是 15,481。

P0-2：prefix_model_reuse 当前有潜在 future-label leakage

这是当前最重要的问题。

定义是：

“这个未来节点的 model_class 是否已经在当前可见 prefix 中出现过”。

从某个当前 anchor 预测未来第 k 个节点时，prefix 本身当然已知，但：

model_class
future,k
	​


并不知道。

如果 dataset builder 用 GT future model_class 计算：

prefix_model_reuse = GT_future_model in visible_prefix_models

再作为 c 输入 J1/B1/J2/J3，那么这是标签派生特征。

这与 R0 不同。R0 是 oracle-signature ceiling，未来 model_class 本来就是 oracle；J 的目标是 deployable prediction。

两种合法修复

最干净的是：

本轮 J 全部删除 prefix_model_reuse context。

只保留：

stack；

baseline；

纯 prefix-only statistics。

或者对于 B1/J2/J3，用预测模型分布构造：

p
reuse,k
	​

=
m
∑
	​

q
k
	​

(model=m)1[m∈prefix_model_set].

这样：

B1/J2 使用 sg(q(model))；

J3 可以保持可微；

J1 因没有 attribute interface，不能使用 slot-specific reuse。

后一个方案会让 J1→J2 的“interface information”同时包含 model reuse 信息，科学上仍可解释，但实现更复杂。

第一轮我更推荐直接移除。

N/A mask

ACCEPT。

但需要：

merged_nested_call=0 时 nested-model loss mask；

q(A) 输入资源头时给 nested-model 一个固定 zero/N/A representation；

不能把真实 nested ID 带入模型输入。

2. 复用 1505 行旧脚本还是拆模块？
裁决：ACCEPT with recommendation

不建议直接修改原 P9d/P9e 脚本本体。

更稳妥的是：

existing p9d script     # 保持原实验可复现
        ↓ reuse/tested pieces
j_series_common.py      # 少量稳定共享组件
        ↓
j_series_train_eval.py  # 新实验逻辑

适合抽出的主要是：

codec/vocab；

causal GRU encoder；

batching/masking；

video bootstrap；

common metric helpers。

J-specific 的：

hurdle resource loss；

B1/J1/J2/J3 gradient routing；

NI checkpoint selector；

应该留在新脚本。

INFERENCE：为了“减少重复”而先大规模重构 1505 行 legacy script，风险反而高于复制少量稳定实现。

所以这不阻塞，但原则是：

old experiment path immutable，J 使用新入口。

3. structure = length classification + termination BCE
裁决：MODIFY — P0

模型形式本身可以接受，但 target 必须定义成：

L
H
	​

∈{0,1,2,3,4,5}.

不是默认 1..5。

因为 final answer 已经是 non-resource terminal marker，不占 H slot，所以存在理论上：

L
H
	​

=0

的合法 anchor，例如当前节点已经是最后一个 compute node、后面只剩 terminal answer。

除非 Stage 0 VERIFIED 证明数据中永远没有这种 anchor，否则必须保留 class 0。

censored 的正确解释

建议直接把变量改名：

bounded_future_length

定义：

L
H
	​

=min(N
remaining compute
	​

,5).

那么：

terminated 且只剩 3 个 compute node → L
H
	​

=3, termination=1；

还有 >5 个 → L
H
	​

=5, termination=0/censored。

所以 censored example 的：

L
H
	​

=5

是精确的窗口内长度标签，只是它不是“最终总剩余长度=5”。

这样就不需要把 censored sample 从 length CE 中删除。

当前文本里的：

“截断样本不得当作精确长度”

容易与 CE class=5 冲突，必须改成上述表述。

CE 可以吗？

INFERENCE：对离散的 0..5 bounded count，6-class CE 是合理实现。

NI 仍然可以用：

MAE(argmaxL
H
	​

, L
H
GT
	​

)

作为 metric。

但如果冻结的 scientific v3 明确把 MAE 本身规定为 training loss，实现不能静默改成 CE。需要保证 scientific contract 冻结的是 endpoint/NI metric，而非具体 length loss。

4. 训练协议、共享初始化、patience
裁决：MODIFY — 有一个 P0

共享 pretrained backbone + matched seeds 的思路正确：

B1/J0 freeze；

J1/J2/J3 从同一 seed 对应 backbone 开始；

data order 匹配。

这满足隔离要求。

P0：patience=6 没有定义监控什么

现在 checkpoint 的正式规则是：

NI-feasible；

feasible checkpoints 中最小 RuntimeQScore。

但同时又保留：

patience=6。

如果 early stopping 直接监控 RuntimeQScore，那么可能出现：

前 6 epoch resource 变好；

但还没出现 NI-feasible checkpoint；

training 被提前终止。

或者不同 variant 的 NI 可行时间不同，导致训练预算不一致。

最简单的处理

因为只有 30 epochs、模型很小，我建议正式实验：

取消 early stopping，所有正式 variant 固定训练 30 epochs。

训练结束后从全部 epochs 中：

NI-feasible→argminRuntimeQScore.

这样最干净。

如果一定要保留 patience，则必须定义为：

从第一个 NI-feasible checkpoint 出现以后，才按 feasible RuntimeQScore 计算 patience。

但没有必要增加这个复杂度。

matched randomness

建议额外冻结：

dataloader shuffle seed；

backbone seed；

resource-head seed；

CUDA deterministic 配置能固定则固定。

不同输入维度导致 J1 与 J2 resource head 不可能完全同权重初始化并不是问题，只需要随机种子协议一致。

5. per-epoch NI manifest 是否可实现？
裁决：ACCEPT with protections

30 个 validation videos 做 video-cluster bootstrap，在计算上完全可行。

真正需要保护的是 mask 和 Monte-Carlo 抖动。

建议：

固定 bootstrap resample indices

正式实验开始前生成一次：

B=1000

组 validation video bootstrap indices。

所有：

epochs；

variants；

seeds；

复用同一 bootstrap index set。

这样 checkpoint feasibility 的变化来自模型，不来自每 epoch 重新随机 bootstrap。

endpoint-specific eligible set

例如 next-family 只在对应有效节点上计算。

对于每个 NI endpoint：

先得到有该 endpoint GT 的 video/node set；

variant/reference 使用完全相同 mask；

paired difference 后再做 video bootstrap。

不能：

variant 一套有效节点，reference 另一套有效节点。

empty bootstrap replicate

如果一个 bootstrap sample 恰好没有某 masked endpoint：

不得填 0；

标成 invalid replicate。

预注册一个最低有效 bootstrap 比例，例如：

≥95%

否则该 endpoint 的 NI 视作不可可靠计算，并 fail-closed。

具体 95% 是 UNVERIFIED engineering rule，可以换，但要事前固定。

censor

按上面的 bounded_future_length 后：

length metric：全部样本都可用；

termination：全部样本都可用；

不需要把 censored length 当 missing。

6. 环境、6GB 与预算
裁决：基本 ACCEPT

INFERENCE：hidden 128、H=5、小 GRU、batch 256 在 RTX 3060 Laptop 6GB 上成功的概率很高，但不能事前保证。

预算 <3 GPU-hours 也作为工程估算是合理的。

需要加两条正式运行规则。

规则一：不能各 variant 自己调 batch

如果 J3 OOM 而 J1 不 OOM，不能：

J1 batch=256
J3 batch=128

然后直接正式比较。

如果必须降 batch：

在 smoke 阶段统一选定 formal batch，之后所有正式 variant 一致。

或者用 gradient accumulation 保持相同 effective batch。

规则二：正式 run 不混 CPU/GPU backend

CPU fallback 可以用于 smoke/debug。

但一旦正式实验启动，应固定：

device；

torch version；

CUDA version；

precision；

batch/effective batch。

如果 CUDA 安装失败而决定正式使用 CPU，那就所有 18 runs 都用 CPU。

不要 B1 CPU、J2 GPU 混跑。

7. 门禁、里程碑与交付
裁决：MODIFY

总体够完整，但有两个顺序/治理问题。

J0 顺序有矛盾

前面定义 J0 为：

冻结 backbone 的 resource probe。

因此必须：

backbone training→J0/B1.

但 §8 写的是：

J0 probe → backbone training → ...

这不可能满足 J0 定义。

应统一为：

Stage0→Smoke→Backbone→J0/B1→J1/J2/J3→frozen Test.
holdout 最好彻底不构建进正式 J pipeline

既然科学设计已经规定：

holdout 不参与 J 正式决策，默认不报告，

工程上最安全的做法就是：

build_j_dataset.py 正式默认只构建 train/validation/test。

不要 Stage 0 顺手读取/统计 holdout。

如果以后真要 exploratory report，再使用一个显式：

--include-nonconfirmatory-holdout

之类的独立路径。

这样最不容易误用。

超预算停止

“先交付 backbone + J0/B1，J1–J3 顺延”可以作为工程 partial completion。

但必须明确：

在 J1/J2/J3 全部完成前，不生成最终 RESULT 的 joint-training 科学结论。

只能标：

partial / incomplete experiment.

最终 7 问裁决
问题	裁决
1. 数据管线	MODIFY / P0：复合 join key；禁止 GT-derived prefix_model_reuse
2. 代码复用	ACCEPT：新入口 + 小型共享模块，legacy 保持不动
3. structure head	MODIFY / P0：bounded length ∈ 0..5，明确 censor 语义
4. 训练协议	MODIFY / P0：patience 与 NI 规则冲突，建议固定 30 epochs
5. per-epoch NI	ACCEPT with guards：固定 bootstrap indices + mask/empty-replicate 规则
6. 环境/预算	ACCEPT：formal backend/batch 必须统一
7. 里程碑/交付	MODIFY：Backbone 必须先于 J0；holdout 最好不进入默认 pipeline

因此当前结论是：

J IMPLEMENTATION PLAN v1 — MODIFY BEFORE ACCEPT
	​


需要关闭的核心阻塞是 4 项：

resource join 改为 trace/run + node_id 复合键，并区分 unique node 与 expanded supervision pair；

从 deployable J 输入中删除 GT-derived prefix_model_reuse，或严格改造成由预测 q(model) 派生的 soft reuse；

structure target 固定为 bounded_future_length ∈ {0,...,5} + termination；

正式训练取消含糊的 patience，或明确只对 NI-feasible checkpoints 计 patience；我更推荐统一 30 epochs。

这四项关闭后，工程计划本身没有需要重新设计模型架构的地方。

## 8. user
最终确认轮（正文提交）。已按 v1 实现评审关闭全部 4 个 P0：
1) 资源 join 改为 (run_id, node_id) 复合键 + 唯一性断言 + 区分 15,481 唯一节点与 anchor×future-slot 监督实例；
2) 从 J 上下文删除 GT-derived prefix_model_reuse（c 仅 stack/baseline），R0 RESULT 加口径说明；
3) structure target 固定 bounded_future_length ∈ {0..5} + termination BCE，censored 语义落死；
4) 取消 early stopping，固定 30 epochs 后从全部 epoch 选 NI-feasible → argmin RuntimeQScore。
另采纳：固定 bootstrap indices、endpoint 成对 mask、空 replicate/95% 规则、正式 backend/batch 统一、J0 在 backbone 之后、holdout 默认不入管线、partial 标记、新入口+小共享模块（legacy 不动）。
请确认四项 P0 已关闭，并给出最终裁决 “J IMPLEMENTATION PLAN v2 ACCEPT（可开工）”；若有遗留请只列阻塞项。请用中文回答，VERIFIED / INFERENCE / UNVERIFIED 保留原词。

---

# J 系列实现规划 v2（评审修订版）

依据：docs/p9d_j_series_design.md（frozen v3，含 v3.1 修订）；实现评审记录 docs/research/2026-09-11_j_impl_plan_review.md。
状态：v2（关闭 v1 评审的 4 个 P0 + 采纳保护性建议），待最终确认后开工。

## v1 → v2 变更（P0 关闭）

| P0 | v2 处理 |
|---|---|
| 资源 join 键 | 改为复合键 **(run_id, node_id)**（可选附带 source_trace_sha256 校验）；Stage 0 断言：资源表复合键唯一、v3 future node 复合键唯一命中、无一/多对多异常、unmatched fail-closed；同时分别登记 N_unique_resource_nodes = 15,481 与 N_supervision_pairs = anchor × future-slot 展开数（同一直实节点可被多个 anchor 预测，属合法重复实例） |
| prefix_model_reuse 泄漏 | **从 J 的上下文 c 中删除**（本轮）；c 仅保留 stack/baseline；R0 RESULT 已加口径说明（该特征依赖 oracle model_class，非纯执行前可见）；N/A mask 规则：merged_nested_call=0 时 nested-model 损失 mask、资源头输入给固定零/N-A 表示、禁止真实 nested ID 进入输入 |
| structure target | 固定 bounded_future_length = min(剩余 compute 节点数, 5) ∈ {0,1,2,3,4,5}（含 0）+ termination BCE；censored 的 L_H=5 视为窗口内精确长度，全部样本进 CE；NI length metric = MAE(argmax L_H, L_H_GT) |
| patience 与选择冲突 | **取消 early stopping**；所有正式变体固定训练 30 epochs，结束后从全部 epoch 选择：先 NI-feasible，再 argmin RuntimeQScore；无 feasible → 变体 fail |

## 采纳的保护性建议

- **代码落点**：不修改旧 P9d/P9e 脚本本体（保持可复现）；新增 scripts/j_series_common.py（codec/vocab、causal GRU encoder、batching/masking、video bootstrap、指标工具）与 scripts/j_series_train_eval.py（hurdle 资源损失、B1/J1/J2/J3 梯度路由、NI checkpoint 选择器、评测）。
- **固定 bootstrap**：正式运行前一次性生成 B=1000 的 validation video-cluster bootstrap indices，全部 epoch/variant/seed 复用；逐 endpoint 使用成对相同 mask；空 replicate 标 invalid；有效比例 <95% 的 endpoint 判不可靠并 fail-closed。
- **正式一致性**：统一 device/torch/CUDA/precision/batch（必要时统一 gradient accumulation）；严禁跨变体单独调 batch、严禁 CPU/GPU 混跑；smoke 阶段先定死。
- **顺序修正**：Stage 0 → smoke → **backbone** → J0/B1 → J1/J2/J3 → 冻结 test 判定。
- **holdout**：默认数据管线只构建 train/validation/test；仅在显式独立路径下可作 exploratory，不进入正式决策。
- **部分完成**：只完成 backbone + J0/B1 时标记 partial / incomplete，不生成 joint-training 结论。

## 未变化部分

数据集（v3.1 标签 + 实测资源）、五变体与隔离、L_total 固定等权、runtime primary / load hurdle 次要点、NI manifest（metric+direction+δ）、test 冻结判定、预算 <2–3 GPU 小时、本地 RTX 3060 6GB 或统一 CPU。

## 9. assistant
四项 P0 都已经关闭；我没有发现新的实现阻塞项。

资源 join 键：已关闭。
VERIFIED：改为 (run_id, node_id) 复合键，并增加唯一性、唯一命中、无多对多、unmatched fail-closed，同时明确区分 15,481 个唯一资源节点与展开后的 anchor × future-slot 监督实例。这个口径已经足以避免此前的静默错配风险。

prefix_model_reuse 泄漏：已关闭。
VERIFIED：本轮 J 上下文仅保留 stack/baseline，不再使用由 GT future model_class 派生的 reuse 特征；nested-model 的 mask / N-A 表示以及真实 ID 禁入输入也已明确。

structure target：已关闭。
VERIFIED：bounded_future_length ∈ {0,…,5} 与 termination 分离；censored 样本的 L_H=5 被正确定义为窗口内精确长度，而不是“真实总剩余长度=5”；class 0 也已纳入。该实现与冻结的 H=5 串行链契约自洽。

patience / checkpoint selection：已关闭。
VERIFIED：取消 early stopping、固定 30 epochs，然后在全部 checkpoint 中执行
NI-feasible → argmin RuntimeQScore，彻底消除了 early stopping 与 NI 可行性出现时点之间的干扰。

其余保护措施也已落地：固定 video-bootstrap indices、endpoint 成对 mask、空 replicate fail-closed、正式 backend/effective batch 统一、backbone 先于 J0/B1、holdout 默认退出正式管线、partial 实验不产出联合训练结论，以及 legacy 脚本保持不动。INFERENCE：这些措施已经足以把后续风险主要收敛到实现 bug 和实际训练稳定性，而不是实验协议本身。

J IMPLEMENTATION PLAN v2 ACCEPT（可开工）。
