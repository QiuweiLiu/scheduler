---
research_backend: ChatGPT Web
requested_model: GPT-5.6 Sol
requested_reasoning: High
model_verification_basis: high_ui_mapping
conversation_id: "6a9822da-e278-83e9-9c1a-675923acda0e"
conversation_url: "https://chatgpt.com/c/6a9822da-e278-83e9-9c1a-675923acda0e"
conversation_generation: 1
parent_conversation_id: null
verified_before_submit: true
verified_after_response: true
timestamp: "2026-09-02 21:30 CST"
status: valid
scope: "结构 topology head、全未来节点 behavior/content decoder 与 identity-free H=5 DAG 输出契约"
---

# Future-DAG 结构与全未来节点行为解码架构评审

## 0. 结论先行

**VERIFIED（会话）**：已在 ChatGPT Web 新建会话“架构设计评估”，发送前页面可读，显示“高”推理控件；回答完成后 URL 稳定为本记录中的 `conversation_url`。本报告保存的是完整回答的结构化整理，不是把网页建议直接升格为项目决策。

**INFERENCE（推荐）**：可以把当前“只预测下一步行为”的设计扩展为：

```text
causal GRU encoder
    ├── structure head：未来 DAG 的层数、每层宽度、可选依赖边
    ├── future-node content decoder：每个未来节点的 type/family/action/model 等属性
    └── next-step behavior head：保留当前 next-role/execute-family 任务作为局部辅助头
```

但不建议做成“先输出硬 topology，再把 argmax topology 强制喂给 behavior decoder”的不可逆串联系统。更稳的首个候选是**结构引导的层内 set decoding**：结构预测提供 soft context、层级 embedding 和 mask，内容 decoder 同时保留从共享编码状态 `h_t` 到每个未来 slot 的直接路径。

**最重要的口径修正**：`layer_count + width_vector` 只是 leveled-DAG profile，不是完整 DAG skeleton。只有补充直接依赖边、父节点关系或等价的偏序表示后，才能称为完整拓扑；是否值得引入边，必须用下游 scheduler 收益验证。

## 1. 建模分解

**VERIFIED（来源支持的建模原则）**：对层内无序节点，目标不应依赖任意 slot 顺序。Deep Sets 对集合上的 permutation-invariant/equivariant 函数给出了基础形式；DETR 展示了固定输出 slots 加全局二分图匹配来预测无序集合的可行性。来源见文末。

**INFERENCE（应用到本项目）**：将未来图写成结构与节点属性两部分：

```text
X = 当前可见 causal prefix + current node/task state
G_future = (S_future, A_future)

p(G_future | X)
  = p(S_future | X) · p(A_future | S_future, X)
```

其中：

- `S_future`：相对当前可见边界的未来层数、层宽、可选的依赖关系；
- `A_future`：每个未来节点的 `node_type`、`action_family`、`raw_action`、`model_category` 等身份无关属性；
- `y_next`：原有的 next-role/execute-gated family，是局部行为辅助目标，不等同于完整未来节点序列。

因此最终有三个不同语义的输出：

1. **structure**：未来 H 层的形状和依赖结构；
2. **content**：结构中每个预测未来节点的具体属性；
3. **local behavior**：当前时刻之后最先发生的行为分布。

这种分解对本项目是合理的，但“这个精确分解已经在 video-agent future-DAG forecasting 上被论文验证”属于 **UNVERIFIED**；现有证据支持的是集合预测、图生成和多任务训练的组成方法，最终组合仍需本项目实验验证。

## 2. 推荐的 target 与输出契约

### 2.1 先固定 layer 的定义

**INFERENCE**：应从观察到的 prefix 边界开始，用可复现的 longest-path depth 定义未来层：

```text
d(v) = 1 + max d(parent),  parent 取未来子图中的父节点
max(empty) = 0
Layer_l = {v : d(v) = l}
```

需要在项目现有 `predecessor_node_ids` 重建口径上核对这一公式，而不能让普通拓扑排序的任意顺序决定标签。该定义通常带来：跨层边、无同层父子依赖、层内节点无序。

这里的 `width_l = |Layer_l|` 只是**结构层的节点数**。它说明同一层节点之间没有祖先关系，不表示它们会同时占用 GPU、同时开始、同时完成，也不表示物理并发；GPU 数量、显存、模型驻留、运行时和调度顺序仍可能把它们串行化。

### 2.2 首轮输出张量

| 输出 | 建议形状 | 语义 |
|---|---|---|
| `layer_logits` | `[B, Lmax+1]` | 未来非空层数 `L`，包含空未来 |
| `width_logits` | `[B, Lmax, Wmax+1]` | 每一层节点数 `W_l` |
| `future_slots` | `[B, Lmax, Kmax, D]` | 层内未来节点 slot 表示 |
| `exist_logits` | `[B, Lmax, Kmax]` | slot 是否对应真实预测节点 |
| `type_logits` | `[B, Lmax, Kmax, C_type]` | 节点类型/角色 |
| `family_logits` | `[B, Lmax, Kmax, C_family]` | execute-gated 行为族 |
| `action_logits` | `[B, Lmax, Kmax, C_action]` | 抽象/raw action 类别 |
| `model_logits` | `[B, Lmax, Kmax, C_model]` | 模型类别 |
| `edge_logits`（可选） | `[B, Lmax, Kmax, Lmax, Kmax]` | 合法跨层未来节点对的依赖概率 |

无效层、inactive slots、同层非法 edge pair 都要 mask。slot index 只是 decoder 坐标，不能赋予“slot 1 永远是 detector”的固定语义。

推理时应让结构 head 的 width 约束有效节点数，再从对应层的 `exist_logits` 选择 slot；不能只依赖“前几个 slot 默认有效”。

**UNVERIFIED 条件捷径**：如果未来窗口的 H=5 实际表示“最多只有 5 个未来节点”，可以把全部合法 `(L, W_1, ..., W_L)` 形状合成有限类别（包含空未来时是 32 个 composition）。但当前项目的 H=5 主要表示 DAG 层深窗口，不能未经确认就使用这个 32 类 shortcut；若 H=5 是最大深度，应单独保留 `Nmax` 和每层 width。

### 2.3 边的标签必须先定语义

**INFERENCE**：如果加 `edge_logits`，应明确预测的是：

- 原始直接依赖边；
- 传递闭包；还是
- 经过某种工作流转换后的邻接关系。

首选 canonical direct dependency；如果业务允许，应使用传递约简或等价的最小偏序表示。否则同一语义 DAG 可能因为多写了传递边而得到不同标签。

## 3. Decoder 方案比较

| 方案 | 判断 | 原因 |
|---|---|---|
| 固定并行 slots + permutation-invariant matching | **首轮推荐** | 适合 H=5，推理成本固定，没有人为节点顺序，也没有 autoregressive exposure bias |
| 层内 set decoder | **最贴合当前语义** | 层顺序有意义，层内节点无序；可用 slot interaction/self-attention 建模同层相关性 |
| 自回归 layer/node decoder | **作为消融** | 能表达生成条件依赖，但会引入任意节点顺序、误差累积和 teacher-forcing/inference mismatch |
| 完整 graph decoder | **最后引入** | 可直接建边，但参数、匹配、合法性约束和数据需求都显著增加 |

GraphRNN 展示了自回归节点/边生成的可行性；GRAN 用 block-wise 节点/边生成减轻顺序和长序列瓶颈；D-VAE 专门讨论 DAG 生成。这些是“可以做”的来源，不是“在本项目一定更好”的证据。

因此推荐的中间形态是：

```text
shared causal GRU
  → structural shape head
  → per-layer parallel set slots
  → optional pairwise edge head
  → next-step auxiliary head
```

不需要首轮同时替换为 Transformer、GNN 和自回归图生成器，否则无法判断收益究竟来自目标因子化、encoder 变化还是 decoder 变化。

## 4. topology 如何帮助内容预测

### 4.1 用 soft structure，不用 hard argmax

**INFERENCE**：从结构分布构造 soft context：

```text
c_S = f_L(P(L)) + Σ_l f_W(P(W_l))
q_lk = layer_embedding_l + slot_embedding_k
       + f_h(h_t) + f_s(c_S)
A_hat = Decoder(q_lk)
```

内容 decoder 必须保留 `h_t → every future slot` 的直接 residual/information path。这样结构预测错一个 width 时，不会把已经包含在 causal history 里的语义信息全部截断。

同时，`p(L)`、`p(W_l)` 比 argmax 更能传递不确定性，并可以参与训练；结构仍然通过 layer embedding、existence mask、cardinality 和合法 edge pair 约束内容输出。

### 4.2 训练时区分 oracle 与 deployable path

**INFERENCE**：做两个诊断路径：

- `oracle-structure`：内容 decoder 读取 ground-truth structure，用来回答“结构正确时内容上限是多少”；
- `predicted-structure`：内容 decoder 只读取模型自己的 soft structure，是部署口径。

oracle 结果不能当作端到端预测成绩。两者差距很大，说明 structure 是瓶颈；预测结构优于 oracle 结构不应被解释为模型真正看到了未来。

可将从 structure probabilities 到 content 的梯度 stop-gradient 作为 **UNVERIFIED 消融**，防止结构表示只为降低内容 loss 而偏离结构目标；后续再与完全联合反传比较。

### 4.3 不形成循环依赖

只要 content 不再反馈给 structure decoder，就不存在推理时的循环依赖。两个 loss 共同更新 shared encoder 是优化耦合，不是因果循环。避免把 content 的 hard top-1 再反馈给 topology，除非另设独立消融并能审计误差传播。

## 5. 是否保留当前 next-step behavior head

**INFERENCE：保留。** 它是一个有价值的 auxiliary/local anchor，因为它是局部、低视野、已有基线、相对容易学习，并能暴露 shared encoder 是否发生负迁移。

首轮联合损失可以抽象写成：

```text
L = λ_S L_structure
  + λ_A L_future_set
  + λ_E L_edge
  + λ_N L_next
```

不要把 `L_next` 当作无关紧要的小正则项；实验应把“next-role/family 不显著变差”列为明确验收约束。多任务损失权重和梯度冲突是已知问题，必要时再审计 GradNorm/PCGrad，不在首轮自动加入。

一致性 loss 需要谨慎：如果 layer 1 只有一个未来节点，可以比较 next-step 分布与该节点属性；如果 layer 1 有多个无序节点，不能随意规定某个 slot 等于“物理上下一步执行的节点”。最多比较 next-step 分布与 layer-1 合法节点的 aggregate/marginal，否则会把 DAG 层宽度偷偷改成执行顺序。

## 6. 损失和评估指标

### 6.1 结构

- `layer_count`、每层 `width`：categorical cross-entropy；
- layer exact accuracy、layer MAE；
- per-layer width accuracy/MAE；
- complete `(L, width_vector)` exact match；
- total future-node-count error；
- NLL、Brier、ECE 或 reliability curve。

### 6.2 节点属性

对每一层的预测 slots 与真实无序节点做 Hungarian assignment。预测 slot `i` 与真实节点 `j` 的匹配代价可以是：

```text
C_ij = α_type CE(type_i, type_j)
     + α_family CE(family_i, family_j)
     + α_action CE(action_i, action_j)
     + α_model CE(model_i, model_j)
```

未匹配 slot 训练为 no-node。`family` 继续 execute-gated，只在 node type 适用时计算，避免制造大量没有业务意义的 `family=None` 学习样本。

当同层节点属性完全相同、但父节点不同，属性-only matching 会让 edge target 任意翻转。可选：将邻域信息加入 matching cost、做 edge-aware second-stage assignment，或在 H 很小的情况下直接对合法 layer-preserving permutations 做联合 node+edge 最小化。H=5 且总节点数确实很小时，最多 5! 个排列可能足够小；这是 **UNVERIFIED 实验选项**，不是当前契约。

### 6.3 边与整图

- edge BCE 只在合法跨层 pair 上计算；
- 报 edge F1 和 AUPRC，不只报 accuracy，因为 non-edge 会占多数；
- matched node macro-F1；
- per-layer set exact match；
- whole-future set exact match；
- `(L, width)` exact match；
- 若加边，graph exact match、aligned edge F1/AUPRC、reachability/partial-order accuracy。

### 6.4 下游价值

必须同时跑：

```text
no-future scheduler
vs predicted-future scheduler
vs oracle-future diagnostic
```

结构/属性统计变好，不等于 scheduler 变好。最终必须报告 same scheduler、same workload、same information boundary 下的 completion/JCT、queue、deadline、planning latency、fallback 和 capacity/failed-job gate。

## 7. 推荐的分阶段实验

### Stage 0：标签与可识别性审计

先固定并统计：

- layerization 是否确定性；
- `(L, widths)` 的频数；
- 同层属性重复/碰撞比例；
- 每类 node attribute 的支持度；
- H 截断与真实终止是否被混为一类；
- `raw_action` 是否含任务/视频近似标识；
- 是否严格按 video split 隔离。

如果 `raw_action` 是高基数、含目标名或近似 ID 的字段，应先抽象化；否则即使它不是输入泄漏，也可能使指标主要反映记忆。空未来、真实终止和“只是达到 H”必须区分。

### Stage 1：最简单的 all-future baseline

保持现有 causal GRU，只加层内 fixed slots 和 permutation-invariant matching，预测：

```text
L + widths + node type + family + action + model category
```

先不加 edges，也不做 topology→content conditioning。目的是判断“从 next-step 改成 all-future-node 目标”本身是否可学。

### Stage 2：结构引导内容

比较：

```text
content = D(h)
content = D(h, soft(predicted_structure))
content = D(h, ground_truth_structure)  # 仅上限诊断
```

这是最关键的消融。如果 oracle 很好而 predicted 很差，说明结构误差正在传播；保留 topology head，但不能让它成为唯一信息通路。

### Stage 3：加入 next-step auxiliary

比较：

1. future-only；
2. future + next-step auxiliary；
3. future + auxiliary + 合法的 layer-1 marginal consistency。

next-step 质量的 non-inferiority 阈值应参考现有 baseline 的 seed/video bootstrap variability，而不是随意指定一个百分比。

### Stage 4：依赖边

比较 `profile-only` 与 `profile + dependency edges`。如果 edge 指标提高但 scheduler 没有收益，说明当前 scheduler 只需要层形状或节点数量，不应为无效分辨率承担 graph decoder 复杂度。

### Stage 5：decoder family

最后才比较 fixed set、autoregressive layer/node 和 richer graph decoder。当前先验是 layer-wise set prediction 更适合短 H、层内无序和中等规模数据，但这仍是 **INFERENCE**。

### 失败判据

- 无 conditioning 的 future-set decoder 已经不差或更好：topology 是辅助目标，不适合作为内容条件；
- oracle topology 显著好，predicted topology 反而降低内容：存在误差级联，保留直接 `h_t` 路径；
- 单节点 accuracy 好但 whole-set/whole-graph exact 很差：模型只生成了独立的“看起来合理”的碎片；
- edges 接近类别先验或不带来 scheduler 收益：删除 edge head；
- next-step 明显下降：先处理任务权重/梯度冲突，不把负迁移合理化；
- random/node split 好而 unseen-video split 崩溃：优先判断 trace/template 记忆，而不是扩大模型；
- 结构指标变好但 predicted-future scheduler 不变好：预测目标与 scheduler objective 尚未对齐。

## 8. 证据状态与来源

### VERIFIED（本次独立打开的原始/官方页面）

- [Deep Sets — arXiv](https://arxiv.org/abs/1703.06114)：集合任务的 permutation invariance/equivariance 基础。
- [Set Transformer — arXiv](https://arxiv.org/abs/1810.00825)：用 attention 建模无序集合成员交互。
- [Conditional Set Generation with Transformers — arXiv](https://arxiv.org/abs/2006.16841)：条件集合生成和集合大小建模。
- [GraphRNN — PMLR](https://proceedings.mlr.press/v80/you18a.html)：自回归节点/边生成和图分布建模。
- [GRAN — arXiv](https://arxiv.org/abs/1910.00760)：block-wise 图生成、顺序依赖缓解和边相关性。
- [D-VAE — NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/e205ee2a5de471a70c1fd1b46033a75f-Abstract.html)：面向 DAG 的生成模型。
- [Scheduled Sampling — NeurIPS 2015](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)：自回归训练/推理暴露偏差的相关工作。

### VERIFIED（官方实现/相关方法入口）

- [DETR official repository](https://github.com/facebookresearch/detr)：固定 slots 与 bipartite matching 的官方实现入口。
- [Kendall et al., Multi-Task Learning Using Uncertainty](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)：多任务 loss weighting。
- [PCGrad — NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)：梯度冲突处理。
- [On Calibration of Modern Neural Networks — PMLR](https://proceedings.mlr.press/v70/guo17a.html)：分类置信度 calibration。

### INFERENCE（针对本项目的综合建议）

- 首轮使用 shared causal GRU + structure head + layer-wise set decoder + 现有 next-step auxiliary head；
- topology 以 soft context、mask 和 cardinality 约束帮助内容，不以硬 argmax topology 作为唯一输入；
- 先不加完整 graph decoder，除非 edge 指标和 scheduler 结果都证明必要；
- 保留 oracle-structure 诊断，但不把它当部署成绩；
- 将下一步行为预测从“唯一输出”改成“局部锚点 + all-future-node content 输出”，二者不是同一个 head 的同一时间尺度。

### UNVERIFIED（必须由本项目实验决定）

- stop-gradient 是否优于完全联合梯度；
- H=5 是否足以使用精确 permutation search；
- 将结构压成单一 composition class 是否优于分头 layer/width；
- edge head 是否有下游调度价值；
- `raw_action` 是否在当前数据规模上可学；
- 同层 slot self-attention 是否优于独立 slot MLP；
- next-step auxiliary loss 的具体权重。

## 9. 对当前项目的处理

本轮只完成网页版讨论、原始来源核验和研究记录；**未修改代码、数据、预测产物或 scheduler 策略**。该报告不自动改变 R8-P9d 的验收状态，也不解封 `T_final`。下一步若要落地，应先由用户确认是否把“all-future-node content decoder + layer-wise set matching”纳入下一轮正式实验，再另建唯一实验目录并补充配置、seed、指标和验证门禁。
