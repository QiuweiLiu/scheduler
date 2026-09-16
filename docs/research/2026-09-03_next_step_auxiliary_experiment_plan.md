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
timestamp: "2026-09-03 01:08 CST"
status: valid
scope: "保留 next-step behavior auxiliary 后的 all-future-node predictor 首轮实验设计"
---

# Next-step Auxiliary 保留后的预测器实验规划评审

## 0. 结论先行

**VERIFIED（会话状态）**：已在绑定的 ChatGPT Web 会话“架构设计评估”中提交脱敏实验规划 brief。发送前页面 URL/title 可读，并显示“高”推理控件；回答已完整返回，之后“停止回答”控件消失。brief 未包含本地路径、凭据、原始 trace、私有日志或文件上传。

**INFERENCE（当前推荐）**：暂时保留现有 `next-step behavior head`，但将它严格定义为 local auxiliary/local anchor，不把它复制成 future-node content 的一部分。首轮正式比较收缩为四个模型：

| ID | 结构输出 | 全未来节点 content | next-step auxiliary | soft topology conditioning |
|---|---|---|---|---|
| `N0` | — | — | ✓ | — |
| `A` | ✓ | ✓ | ✗ | ✗ |
| `B` | ✓ | ✓ | ✓ | ✗ |
| `D` | ✓ | ✓ | ✓ | ✓ |

- `N0`：当前 next-step-only 行为模型，作为行为能力的 preservation reference。
- `A`：future-only，回答结构和所有未来节点内容能否独立学到。
- `B`：在 A 上加入 next-step auxiliary，回答辅助头是否改善未来表征，同时是否保住局部行为能力。
- `D`：在 B 上加入 soft topology conditioning，单独回答结构概率是否能帮助 content 解码。

`C`（layer-1 consistency）、`E`（显式依赖边）和 `F`（自回归/完整 graph decoder）后置。`oracle-structure` 不是第五个模型，而是同一 checkpoint 的诊断路径。

这是一份研究建议，不自动改变 `.project/PLAN.md`，不创建正式实验目录，不修改代码，也不开始训练。

## 1. 与当前项目边界的对应

**VERIFIED（项目状态）**：当前预测器开发边界为 `P_dev=300` 个视频，`P_holdout_diag=40` 个未见视频；scheduler 数据集 `S_train/S_val` 和封存的 `T_final` 不应参与 predictor fitting 或 model selection。当前已有 shared causal GRU 的 topology/behavior diagnostic，但尚未把本轮 all-future-node content 方案接入 scheduler。

**INFERENCE（本轮建议）**：新实验应继续使用与行为/资源预测器一致的视频级边界和 source-video 分组；未来 DAG 的 layer、width、节点属性是 topology/content 自己的标签。不能用 scheduler 的执行结果反向选择 predictor。

若采用网页 GPT 推荐的交叉验证，应在正式执行前明确它是否替换当前 R8-P9d 的固定 `train → validation → test → holdout` 方案：推荐方案为 `P_dev` 上 5-fold source-video grouped CV，每 fold 使用 3 个固定 seeds（11/22/33），每个模型 15 个 fit，四个模型共 60 个 fit；冻结后才打开 `P_holdout_diag`。这属于需要确认的路线调整，不在本报告中静默替换现有计划。

## 2. 三个 head 的职责和输出契约

### 2.1 Shared encoder

```text
visible causal prefix + current node/task state
                    ↓
          shared unidirectional causal GRU
                    ↓ h_t
      ┌─────────────┼─────────────┐
 structure head  content decoder  next-step head
```

输入只含当前可见 prefix、当前节点/任务状态以及预先批准的冻结当前预测特征；不含未来真实节点、真实 successor IDs、执行 truth 或资源 truth。

### 2.2 Structure head

建议预注册：

- `layer_logits: [B, Lmax+1]`，预测未来非空 DAG layer 数 `L`，包含空未来；
- `width_logits: [B, Lmax, Wmax]`，只对真实存在的 layer 计算 width CE，不额外用 `width=0` 监督 inactive layer；
- `node_mask` 由预测的 `L` 和 `W_l` 派生，不另设一个独立 occupancy/mask head；
- 首轮不预测显式 edge。

这里的 width 是 DAG layer 的节点数量，不是物理时间点并发数。`layer_count + width_vector` 仍只是 leveled-DAG profile，不应称为完整 DAG skeleton。

结构损失可以写成：

```text
L_S = 0.5 * CE(layer_hat, layer)
    + 0.5 * mean_{l <= L} CE(width_hat_l, width_l)
```

当 `L=0` 时 width loss 为 0。这样 layer existence 和 layer cardinality 不会被重复监督。

### 2.3 Future-node content decoder

每个未来 layer 使用固定数量的无序 slots：

- `type_logits: [B, Lmax, Kmax, C_type]`
- `family_logits: [B, Lmax, Kmax, C_family]`
- `action_logits: [B, Lmax, Kmax, C_action]`
- `model_logits: [B, Lmax, Kmax, C_model]`

slot index 没有真实身份含义。每层只做一次 joint Hungarian matching，匹配代价同时使用已定义的有效属性；同一 assignment 必须复用于 type/family/action/model 和后续 edge metric，不能每个属性重新匹配。

未匹配真实节点记为 FN，未匹配预测 slot 记为 FP。属性 loss 只在匹配节点和业务上有效的属性上计算；`family` 继续使用 execute-gated 规则，不能用大量无意义的 `family=None` 样本稀释目标。

### 2.4 Next-step behavior head

继续输出：

```text
p(next_role | h_t)
p(next_family | h_t, next_role=execute)
```

建议 loss：

```text
L_N = 0.5 * CE(next_role_hat, next_role)
    + 0.5 * CE(next_family_hat, next_family)   # 仅 execute 样本
```

它负责回答“当前之后的局部下一步行为是什么”，不负责为 future content 指定某个固定 slot，也不等同于未来 DAG 的第一个 layer。

### 2.5 总损失

```text
A:     L = L_S + L_C
B/D:   L = L_S + L_C + L_N
```

首轮先在每个 head 内归一化，再采用等权 `1:1:1` 作为中性起点。该权重不是理论最优值，而是可解释的 preregistered baseline；如果出现负迁移，先记录而不是立即加入 PCGrad/GradNorm。

`D` 的第一版用结构概率（soft probabilities）作为 content 的条件输入，并保留 `h_t → every future slot` 的直接路径。为保持 `B→D` 的归因清晰，可先隔离 content loss 对 topology conditioning representation 的反向耦合；stop-gradient 是待验证的实验选项，不是文献已证明的最优方案。

## 3. 第一轮实验执行顺序

### Stage 0：正式实验前的 hard gates

不训练，先核验：

1. layerization 是确定性的 canonical DAG layering；同一 DAG 不会因任意 topological sort 得到多套标签。
2. 真实终止与 H=5 截断明确区分。
3. `raw_action` ontology 不含 video-specific literal、query-specific identifier 或 target-specific value，避免把指标变成记忆测试。
4. 同一 source video 的 traces、prefixes、runs 和 derived variants 绝不跨 fold。
5. 同一层重复属性节点的碰撞比例可审计；如只靠属性无法唯一匹配，应提前决定是否加入邻域信息或联合 edge-aware matching。

任一 target、leakage 或 split gate 不通过，停止进入模型比较。

### Stage 1：P1 architecture factorization

按 `N0/A/B/D` 在同一数据边界、同一 encoder 容量、同一训练预算下比较。主要归因关系：

```text
A → B : next-step auxiliary 的增量与 negative transfer
B → D : soft topology conditioning 的增量
N0    : next-step 能力非劣性参考
```

同一 checkpoint 同时跑两条 future content 评估路径：

- `oracle-structure`：用真实 `L/W` 激活 slots，回答 content decoder 在结构完美时的上限；
- `predicted-structure`：用模型预测的 `L_hat/W_hat` 激活 slots，作为部署口径的主要 future 结果。

### Stage 2：只有 P1 通过后才做

- `C`：只有当 layer-1 marginal 与 next-step target 的语义确实一致时，才加入 distribution-level consistency；不能把任意 layer-1 slot 当作物理“下一步”。
- `E`：显式 direct dependency edge，先冻结 content assignment，再在合法跨层 pair 上计算 edge AUPRC/F1、graph exact、DAG-valid rate。
- `F`：fixed layer-wise set decoder 与 autoregressive/block-wise graph decoder 的复杂度消融，最后做，不能与 encoder 和 label contract 同时变化。

## 4. 指标和判定规则

### 4.1 主指标

- 结构：`StructureExact = 1[L_hat=L 且所有 active width_hat_l=width_l]`。
- 全未来节点 content：以 predicted-structure path 为主的 `E2E-AttrMacroF1`；结构错误通过 slot 激活、FP/FN 真实反映到结果。
- next preservation：`role Macro-F1` 与 execute-conditional `family Macro-F1` 分别报告。

### 4.2 次指标

- layer-count accuracy/MAE；per-layer width accuracy/MAE；total future-node-count MAE；
- `OracleStruct-AttrMacroF1` 和 `Gap_struct = OracleStruct - PredictedStruct`；
- type/family/action/model 分属性 Macro-F1；strict layer-set/whole-future exact；
- layer、width、next role、主要节点属性的 NLL/Brier/ECE。

oracle 分数只能回答“结构完美时能做到什么”，不能替代部署分数。

### 4.3 Next-step 非劣性

对 `B/D` 相对 `N0` 分别计算：

```text
Delta_role   = MacroF1_role(M)   - MacroF1_role(N0)
Delta_family = MacroF1_family(M) - MacroF1_family(N0)
```

使用 source-video-level paired cluster bootstrap，而不是 node-level bootstrap。网页 GPT 建议的工程性 margin 是 `δ_role=0.01`、`δ_family=0.02`，判定为单侧 95% CI 下界大于 `-δ`；这两个数是 **UNVERIFIED** 的项目容忍度，不是通用理论阈值，若项目已有历史门槛应优先复用。

推荐的层级 gate：

1. `A→B`：B 的 future 结果不实质性差于 A，B 相对 N0 通过 next-step NI，且 seed/fold 不明显不稳定。
2. `B→D`：D 相对 N0 通过 next-step NI；`StructureExact` 相对 B 非劣；predicted-structure `E2E-AttrMacroF1` 有稳定正方向；oracle/predicted gap 没有显示 topology error amplification。
3. 若只有 aggregate point estimate 变好但 fold 方向反复翻转，结论写成 promising but inconclusive，不写成 outperforms。

## 5. 失败模式和停止条件

- width/layer 很准但节点语义很差：先看 oracle content，不能直接上 graph decoder。
- oracle content 高、predicted content 低：structure 是瓶颈，重点审计结构误差放大。
- B 比 A 明显差且 next NI 不通过：记录 auxiliary negative transfer，不马上引入梯度修复算法。
- D 与 B 持平或更差而 structure 不差：topology 可能适合作为 auxiliary target，不适合作为 content condition。
- D 的 oracle 很高但 predicted 路径崩溃：soft condition 过强，需削弱依赖。
- layer target 不唯一、终止/截断混淆、raw-action 泄漏或 source-video 跨 fold：hard stop。
- future structure 接近 trivial baseline、oracle content 本身很差或 B 不稳定：停止 C/E/F。

## 6. Evidence status

### VERIFIED：独立来源直接支持的方法学前提

- Deep Sets 支持集合函数对元素置换不敏感的基本建模原则；
- DETR 展示了固定数量 parallel queries 与全局 bipartite matching 的 set prediction 范式；
- Set Transformer 针对无序集合内元素交互提供 attention-based 结构；
- PCGrad 的原始论文明确讨论多任务优化中的 gradient interference；
- Cawley & Talbot 说明反复优化有限样本上的 model-selection criterion 会引入选择过拟合/偏差，支持 holdout 在模型冻结后才打开；
- Guo et al. 说明现代神经网络的置信度可能未校准，因此 calibration 应作为概率输出的次指标。

### INFERENCE：针对本项目的推荐

- 首轮采用 `N0/A/B/D`；
- 保留 next-step auxiliary 作为局部能力锚点；
- predicted-structure content score 作为主要 future endpoint，oracle 仅作诊断；
- `node_mask` 从 `L/W` 派生，不再训练独立 mask head；
- 首轮使用固定 layer-wise unordered set slots 和一次/层 Hungarian matching；
- scheduler 数据集完全排除在 predictor fitting/selection 外；
- C/E/F 后置，避免一次改变多个机制。

### UNVERIFIED：必须由实验回答

- `1:1:1` 是否是最佳多任务权重；
- stop-gradient topology conditioning 是否优于 joint-gradient；
- soft topology conditioning 是否提升 content；
- next-step auxiliary 是否改善 future prediction；
- 显式 edge 是否对 scheduler 有用；
- fixed-set 是否最终优于 graph/autoregressive decoder；
- `δ_role=0.01`、`δ_family=0.02` 是否适合作为本项目的工程容忍度。

## 7. 研究来源

- Deep Sets — NeurIPS 2017: <https://papers.nips.cc/paper/6931-deep-sets>
- End-to-End Object Detection with Transformers (DETR) — arXiv: <https://arxiv.org/abs/2005.12872>
- Set Transformer — PMLR: <https://proceedings.mlr.press/v97/lee19d.html>
- Gradient Surgery for Multi-Task Learning — NeurIPS 2020: <https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html>
- On Over-fitting in Model Selection — JMLR: <https://www.jmlr.org/papers/v11/cawley10a.html>
- On Calibration of Modern Neural Networks — PMLR: <https://proceedings.mlr.press/v70/guo17a.html>

