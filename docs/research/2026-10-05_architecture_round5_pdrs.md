# 第五轮架构讨论（用户提案"前缀→未来预测→滚动调度"）— 正式主架构定为 PDRS

## 来源

网页版 GPT 2026-10-05：用户提案"从实例前缀预测未揭示短期 future graph（结构概率/运行时间/资源分布），
用于 node-level 异构 GPU 滚动调度"，并明确"不想让实测表进入架构"。GPT 裁定：**赞成该方向**，
并给出三个必须降级的措辞 + 正式架构 PDRS。

## 裁定摘要

- **主架构**：**PDRS — Prefix-Conditioned Distributional Receding-Horizon Scheduling**。
  一句话：仅凭每个运行中 agent 实例的执行前缀，预测其**未揭示后缀的概率信念**，
  并在每次新节点揭示后**滚动**重排 node–GPU 动作。
- **方法不读实测表**：表只存在于评估环境（执行器 + 准入规则）；方法只看到"合法动作集合"
  （GPU 可/不可派发），看不到 slowdown/interference 数值。
- **三个措辞降级**：①"联合 future graph"→ 不成立（只有边际；串行链无分支边）；
  ②"GPU 资源需求分布"→ 没有显存预测；③"异构感知预测"→ 预测器不按 device 条件。
  准确表述：**probabilistic future suffix / multi-horizon probabilistic future-chain prediction**。
- **关键好消息**：第一版**不需要联合分布**——加法期望线性性 ΣE[C_k] 只需边际；无需重训。
  joint 只在做 CVaR(Σ) 或序列依赖时才必要（留 future work）。
- **放置项 MVP 只看 H1**：无机器动力学时声称 H5 放置影响不可靠；H1 有效则故事更漂亮
  （"很少的部分未来信息就够"）。
- **heterogeneous 的严谨说法**：node-level rolling scheduling on a heterogeneous GPU substrate
  （状态异构：容量/驻留），不是 heterogeneity-aware prediction。

## 方法式（全部不读表）

q_k = P(L≥k)；W_i = Σ_k q_k [E(T_k) + p_load_k·p95(load_k)]；
A(i,g) = q_1·p_load_1·R(L_1)·[1−P_hit(i,g)]，P_hit = 下一步模型分布 ∩ GPU 预期驻留集合；
S(i,g) = C_now(i,g) + W_i + A(i,g)。

## 实验矩阵（主表不动）

Myopic / F0 / **PDRS-O**（存活加权边际，current-only）/ **PDRS-P**（+H1 模型概率放置）/
train-prior 对照 / shuffled-belief 对照 / Truth-Oracle 天花板；
负控制：单 GPU（放置贡献→0）、全驻留（A→0）。

## 执行顺序

1. **PDRS-O**（不改预测器）：q_k + W_i，回答"随机视野边际是否超过冻结 F0"；
2. **PDRS-P**（仍不改预测器）：+H1 放置亲和；
3. 三个信息对照（prior/shuffle/oracle）；
4. 只有前面成立才考虑：完整 role 分布（可能只需重跑推理）、robust coupling、H2/H5、runtime_probs 风险、VRAM 预测。

## 文献位置

- vs **HexAGenT**（最漂亮）：它只用**已揭示前沿**调度；我们加 P(未揭示后缀|prefix)——
  问题变成"**该不该预测还没出现的工作**"。
- vs SAGA：它是 workflow-type 图→缓存/亲和；我们是 per-instance 在线后验+滚动决策。
- vs Latency-Aware：它是完整物理执行优化器；我们刻意**不读机器模型**。
- vs Scepsy：aggregate-history vs per-instance short-horizon（→ instance vs train-prior 对照）。
- 理论 framing：learning-augmented online scheduling under gradually revealed workflows /
  learned partial clairvoyance（ESA 2026 + stochastic signal 线）。
- 安全表述（To our knowledge…）见原文。

## 核心研究问题（GPT 建议）

> Can learned partial clairvoyance about an unrevealed agent-workflow suffix improve online
> node-level GPU scheduling beyond using only the revealed frontier?

审核回复原文如下（未改动）。

---

炜，这一轮我会做一个比较明确的收敛：

**我赞成你把主线改成“前缀 → 未揭示短期未来预测 → 滚动调度”，并且赞成把实测机器表彻底移出方法架构。**  
但你现在这句话里有三个词需要降级，否则论文会被抓住：

> “联合 future graph” → 目前不成立  
> “GPU 资源需求分布” → 目前也没有  
> “heterogeneous-GPU-aware prediction” → 现有 predictor 也不是 device-conditioned

所以真正能够立刻落地、且不需要推翻现有资产的版本应当是：

> **从每个 Agent 实例已经执行的前缀预测尚未揭示的短期概率 suffix，并在每次新节点揭示后，用该 per-instance future belief 重新进行 node-level GPU 调度。**

我认为这比 BCSR 那条“预测 × 实测机器表”的路线更干净，也更适合作为论文真正的主架构。

---

## 1. 这个新 framing 成立吗？

### 结论：成立，而且我认为比 BCSR 更适合作为主线

但它不是简单“替代”前几轮，而是一次**向上收敛**：

\[
\text{BCSR}
\subset
\text{Prefix-conditioned rolling scheduling}
\]

前几轮实际上混在了一起三个东西：

1. 预测未揭示未来；
2. 如何消费未来信息；
3. 如何利用机器响应模型。

现在可以把它们拆干净：

\[
\boxed{
\text{Observed prefix}
\rightarrow
\text{future belief}
\rightarrow
\text{rolling scheduler}
}
\]

其中 scheduler 只允许读取：

\[
\mathcal I_{\rm method}
=
\{
\text{future prediction},
\text{current observable system state}
\}
\]

**不读取实测 slowdown/interference/batching table。**

实测表仅存在于：

\[
\mathcal E_{\rm evaluator}
\]

也就是被调度的“物理世界”。

这样之后，BCSR 原先那种：

\[
\Phi=
E[C_{\text{machine table}}]
\]

降级成一个 **oracle/model-based consumer variant**，不再是主架构。

Dual-Uncertainty 也不用做成主故事了；它变成 robustness extension。

而之前很好的 **Information × Actionability** 仍然保留，用于实验归因。

---

# 2. 但“联合 future graph”目前不能这么说

这是这一轮最重要的措辞修正。

你当前因果 projection 满足：

\[
task\_group\_size=1
\]

所有 workflow 都是串行链。

所以你的真实对象其实是：

\[
v_t
\rightarrow v_{t+1}
\rightarrow
v_{t+2}
\rightarrow\cdots
\]

并不存在需要预测的：

- 分支边；
- sibling；
- fan-out；
- fan-in；
- arbitrary DAG connectivity。

因此当前工作里：

> future graph prediction

容易被 reviewer 反问：

> “Where is the graph?”

你真正拥有的是：

\[
P(L)
\]

以及对每个 offset \(k\)：

\[
P(M_k),\;
P(R_k),\;
P(T_k),\;
P(L^{load}_k)
\]

所以最准确的名字是：

> **probabilistic future suffix prediction**

或者：

> **multi-horizon probabilistic future-chain prediction**

我甚至更推荐第一种，因为未来如果数据升级成 DAG，这个词不用改。

---

# 3. 另一个必须纠正的词：“联合分布”

你现在没有：

\[
P(Z_1,\dots,Z_H)
\]

只有：

\[
P(Z_1),P(Z_2),\dots,P(Z_H)
\]

即逐 horizon marginal。

所以不能说：

> “predict a joint distribution over the future graph.”

不过有一个非常重要的好消息：

### 你第一版 rolling scheduler 根本不需要联合分布

如果目标是 additive remaining-work value：

\[
V_i
=
E\left[
\sum_{k=1}^{L_i}C_{ik}
\right]
\]

利用期望线性性：

\[
E\left[\sum_k C_k\right]
=
\sum_kE[C_k]
\]

根本不要求：

\[
P(C_1,\ldots,C_H)
\]

存在。

于是用你现有 artifact 可以严格写：

\[
V_i
=
\sum_{k=1}^{H}
P(L_i\ge k)
\cdot
E[C_{ik}]
\]

这不是近似“联合 distribution”。

这是对 additive expectation **完全合法的 marginal computation**。

这个点很重要。

只有你以后想算：

\[
CVaR\left(\sum_k C_k\right)
\]

或者模拟：

> step1 是 8B planner → step2 大概率因此是 3B spatial

这种 sequence dependency 时，joint 才成为必要条件。

---

# 4. 所以第一版完全不应该补 joint predictor

我现在反而**不建议重训**。

原因是投入产出比不好。

你的 workload 本身是 chain。

即使重新训练 autoregressive H5 predictor：

\[
P(z_1,\dots,z_5\mid prefix)
=
\prod_kP(z_k\mid prefix,z_{<k})
\]

你得到的主要价值也是：

> 更好的 sequence correlation。

但论文最明显的创新并不是 correlation predictor，而是：

> **预测尚未揭示 future，然后滚动消费。**

先证明这个最重要。

而且 ESA 2026 已经把“processing operations gradually revealed during execution”正式定义成介于 clairvoyant 和 non-clairvoyant 之间的在线调度模型。citeturn439302search4  
更早 ICML 2023 也已经系统研究了 online precedence constraints 下，“额外预测信息究竟能怎样帮助调度”。citeturn548514search0

所以你的理论语境其实已经很清楚：

\[
\boxed{
\text{learning-augmented online scheduling under gradually revealed workflows}
}
\]

这比“我们预测了一个 graph”更稳。

---

# 5. “表不进架构”：我强烈推荐解释 (a)

也就是：

> **方法不读表；环境继续用表。**

这是我认为最合理的正式设计。

方法侧：

\[
I_{\rm machine}=M0
\]

只看到：

- 当前 ready nodes；
- GPU busy/idle；
- 当前 queue；
- 当前 resident model；
- 当前可用显存/容量；
- 当前已经揭示的 workflow prefix；
- predictor 给出的 per-instance future distributions。

不看到：

- 11-cell co-location slowdown；
- 12-cell load interference；
- batching latency-factor table；
- 任何这些 table 的具体数值。

环境侧仍然使用这些真实测量计算：

\[
T_{\rm actual}
\]

这完全合理。

可以把它想成：

> 汽车驾驶策略不需要知道发动机每一个物理方程，但真实车辆依然按照真实物理运动。

scheduler 不应该知道隐藏 plant 的 exact response model。

---

## 6. 准入怎么办？

这个需要稍微区分。

你现在：

> 是否允许 dispatch / batch / co-location

由 substrate 使用真实 profile 判断。

我认为可以继续保留。

方法看到的只是：

\[
\mathcal A_t
=
\{\text{currently legal actions}\}
\]

而不是：

> 为什么某动作合法、具体 slowdown 是多少。

这相当于 runtime 提供：

```text
GPU0: eligible
GPU1: not eligible
```

而不是：

```text
GPU0 slowdown = 1.1843
GPU1 slowdown = ...
```

所有 scheduler 都看到同一个合法动作集合。

所以公平。

论文里可以明确：

> Machine profiles are used only by the execution substrate to enforce feasibility and realize runtime effects; their numerical values are hidden from the scheduling policy.

这句话就够了。

---

# 7. 解读 (b)：整个系统连环境都不使用表

我**不推荐**作为主实验。

如果连 DES executor 也不用表，你只剩几个选择。

### 方案 B1：trace-only independent execution

每个节点：

\[
runtime = trace\ truth
\]

每个 load：

\[
load = trace\ truth
\]

不同 GPU 上任务之间：

\[
interference=0
\]

不做：

- co-location slowdown；
- load interference；
- measured batching speedup。

这个环境最干净，但也最不现实。

它相当于：

> independent-server replay simulator。

可以作为 **secondary sensitivity environment**。

不能替代你花很大成本校准好的 substrate。

### 方案 B2：解析模型

例如固定 slowdown coefficient、queueing model。

问题是：

> 你只是把真实测量表换成一个更假的模型。

没有任何科学优势。

### 方案 B3：外部 simulator

同理，你只是换了一个 machine model。

还会丢掉你已经独立审核的 substrate。

所以我的结论是：

\[
\boxed{
\text{表退出 scheduler，不退出 evaluator}
}
\]

这是最干净的线。

---

# 8. “GPU resource demand distribution”目前该怎么处理

目前你没有显存 distribution。

这一点不能模糊。

你拥有：

- runtime distribution；
- load-duration distribution；
- model identity distribution。

它们都不是：

\[
P(VRAM)
\]

所以第一版不要写：

> GPU resource demand distribution.

准确写：

> **runtime and loading-cost distributions, together with probabilistic future model identities.**

---

## 9. 有没有一种不重训的方法得到“部分资源分布”？

有一个小技巧可以做，但必须准确命名。

如果每个 model 的静态 memory footprint：

\[
w(m)
\]

是已知模型元数据，而不是实测 performance profile，那么：

\[
P(M=m)
\]

自然诱导出：

\[
P(W=w(m))
=
P(M=m)
\]

于是你可以得到：

> **future model-footprint distribution**

这不需要训练。

但注意它只表示：

- model weight residency footprint，

不包括：

- KV；
- activation；
- input-length-related peak；
- batching peak。

所以绝对不能把它叫：

> predicted total VRAM demand。

如果你的 scheduler 只用它做 capacity compatibility，这是合理的。

---

# 10. 真正预测显存是否值得？

我现在判断：

**不值得作为第一阶段。**

如果要认真预测：

\[
P(VRAM_{k})
\]

你至少需要每个 future node 的：

- model；
- input size；
- output/decode size；
- batch；
- execution phase；

以及对应 memory labels。

如果现有 trace 没有 peak VRAM label，就不是“加个 head”那么简单。

你实例已经关了，更不划算。

所以第一篇：

> runtime/load/model/length prediction

已经足够。

把 VRAM 预测放 future work。

---

# 11. Full role distribution 能不能补？

这倒值得检查一次代码。

你 artifact 现在只保存：

\[
(role^*,p^*)
\]

如果 frozen predictor 内部的 role head 实际是一个 softmax classifier，并且 checkpoint 还在，那么：

**可能只需要重新跑 inference serialization，完全不用重训。**

即把：

```text
top1_role
top1_probability
```

改成：

```text
role_probabilities{...}
```

但是：

> 必须先确认模型 head 本身输出完整 logits。

如果模型训练时就只有 binary/top1 confidence 或做了别的压缩，那么不成立。

即使能补，我也把它列第二阶段，而不是现在阻塞主架构。

---

# 12. Cross-step joint 呢？

这个不同。

如果当前预测器结构是：

\[
h(prefix)\rightarrow
head_{k=1},\ldots,head_{k=5}
\]

各 horizon 独立输出 marginal，那么：

**重新跑 inference 不会凭空产生 joint。**

需要：

- autoregressive suffix decoder；
- structured sequence model；
- 或 conditional rollout head；

基本意味着重训。

所以我会明确：

> **现在不补。**

如果最后 rolling scheduler 的收益很强，再考虑第二篇/增强版。

---

# 13. Rolling scheduling 应该具体长什么样？

这是最关键的一部分。

我建议正式方法叫：

# **Prefix-Conditioned Distributional Receding-Horizon Scheduling**
简称可以先用：

\[
\boxed{\text{PDRS}}
\]

---

## 第一步：每次 workflow 发生 reveal

假设已经观察到：

\[
x_{1:t}
\]

预测器输出当前 prefix 的 future belief：

\[
B_t
=
\{
P(L),
P(M_k),
P(T_k),
P(D^{load}_k),
P(load_k)
\}_{k=1}^{H}
\]

---

## 第二步：得到 survival probability

由 `length_probabilities`：

\[
q_k=P(L\ge k)
\]

这样不用固定：

> 一律看满 H=5。

如果 workflow 高概率只剩一步：

\[
q_1\approx1,\quad q_2\ll1
\]

后面的预测自然权重很低。

这就是之前“adaptive horizon”的更干净版本。

---

## 第三步：计算剩余工作 belief

例如第一版直接：

\[
W_i
=
\sum_{k=1}^{5}
q_{ik}
[
E(T_{ik})
+
p^{load}_{ik}R(L_{ik})
]
\]

这里：

- runtime 可以直接用 frozen `runtime_mean_ms`；
- load duration 如果没有 mean，可以先固定使用 p95，并明确叫 `probability-weighted p95 load cost`；
- 也可以做 p50/p95/CVaR ablation。

注意：

**这里没有任何机器表。**

---

# 14. 但这样还是和 F0 一样 GPU-independent

对。

所以还需要一个很轻量的、**不用 machine table** 的 action-conditioned placement term。

我们只利用：

- 当前 observable residency；
- future `model_probabilities`；
- predictor 自己给的 load-duration estimate。

定义当前选择：

\[
a=(i,g)
\]

之后 GPU \(g\) 可预期保留/拥有的模型集合：

\[
\mathcal R_g^+(a)
\]

这里不需要 slowdown table。

只是：

> GPU 上有什么模型。

然后下一步 future model 命中概率：

\[
P_{\rm hit}(i,g)
=
\sum_m
P(M_{i,1}=m)
\mathbf1[m\in\mathcal R_g^+(a)]
\]

因此 expected avoidable loading term：

\[
A(i,g)
=
q_{i1}
p^{load}_{i1}
R(L_{i1})
[1-P_{\rm hit}(i,g)]
\]

最终：

\[
\boxed{
S(i,g)
=
C_{\rm now}(i,g)
+
W_i
+
A(i,g)
}
\]

这里：

- \(C_{\rm now}\)：现有 F0 的 current consumer；
- \(W_i\)：预测 remaining suffix；
- \(A(i,g)\)：未来模型概率导致的 GPU-specific placement value。

**完全不读机器 profile。**

---

# 15. 为什么我建议 placement term MVP 只看下一步

非常重要。

你当然可以写：

\[
\sum_{k=1}^{5}
P(M_k=m)
\]

来影响 GPU g。

但当前 action 放到 GPU g 后，到 k=5：

- 中间可能发生很多其他调度；
- resident state 可能变了；
- eviction 可能发生；
- 其他 workflow 可能进来。

如果没有 machine dynamics model，声称：

> 当前 placement 对第 5 步 residency 有精确影响

并不可靠。

所以第一版：

\[
\text{H5 future workload for ordering}
\]

但是：

\[
\text{H1 future model belief for placement}
\]

非常合理。

后面再做：

- H1;
- H2;
- H5

placement horizon ablation。

如果 H1 已经有效，故事反而更漂亮：

> 很少的 partial future information 就够了。

这与 online scheduling with predictions 的“minimal useful information”路线很契合。citeturn548514search0

---

# 16. runtime_probs 怎么真正进入，而不是继续闲置？

第一阶段我甚至**不强求**。

因为你已经证明：

> runtime statistic axis 接近饱和。

不要为了“所有字段都用上”而污染主方法。

第一版真正的新信息就是：

- `length_probabilities`
- `model_probabilities`

这两个 F0 没用，而它们改变了：

- future horizon；
- placement。

`runtime_probs[16]` 留给第二个 variant：

\[
R(P_T)
=
E[T]+\lambda CVaR_\alpha(T)
\]

或者 deadline risk：

\[
P(T_{\rm remain}>slack)
\]

但它是 robustness / risk ablation。

不是核心。

---

# 17. 关于“异构 GPU”我要给一个重要警告

如果方法完全不读 machine table，而且你的 predictor：

\[
P(T_k)
\]

也**不以 GPU type 为条件**，那么当前 frozen 方法不能严格声称：

> **heterogeneity-aware runtime prediction**

例如它没有：

\[
P(T_k\mid GPU=4080)
\]

和：

\[
P(T_k\mid GPU=3090)
\]

两个分布。

因此论文当前最好说：

> **node-level rolling scheduling on a heterogeneous GPU substrate**

而不要过度写：

> **heterogeneity-aware predictive placement**

除非你还有 scheduler 可见的：

- GPU capacity；
- current queue state；
- static model compatibility/residency，

这些确实让不同 GPU action 不同。

但它利用的是**状态异构性**，不是预测 device speed。

这一点必须严谨。

---

# 18. 与 SAGA 的真正区别

SAGA 2026 已经相当危险，因为它明确用：

\[
G=(V,E,P,\phi)
\]

表示 Agent Execution Graph，并为边定义 transition probability；对 ReAct chain，其转移概率甚至直接和 termination probability 联系起来。它还支持从历史 traces 推断 AEG。citeturn439302search3

因此不能 claim：

> “first probabilistic agent workflow.”

你的区别应当是：

### SAGA

更接近：

\[
\text{workflow/agent-type graph}
\rightarrow
\text{cache reuse / affinity / batching}
\]

### 你

更接近：

\[
\boxed{
\text{this instance's current execution prefix}
\rightarrow
P(\text{still-unrevealed suffix}\mid prefix)
\rightarrow
\text{receding node-level decisions}
}
\]

而且每 reveal 一步就重新预测。

这就是 **per-instance online posterior**，而不是一个静态/历史模式 AEG。

SAGA 的主要调度贡献还是 program-level KV reuse、session-affinity batching 和 fairness。citeturn439302academia57

---

# 19. 与 HexAGenT 的区别更重要

HexAGenT 非常接近你的问题设置。

它明确研究：

> workflow DAG online reveal；

并根据**当前已经揭示的 graph**维护 workflow completion horizon，再联合选择 P/D placement 和 queue priority。citeturn932683search1turn932683search12

所以它几乎就是你的最重要对照：

### HexAGenT

\[
G^{revealed}_t
\rightarrow
scheduler
\]

### 你的核心问题

\[
G^{revealed}_t
+
P(G^{unrevealed}_{t:t+H}\mid prefix)
\rightarrow
scheduler
\]

即：

> **是否应该预测还没出现的工作？**

这其实是非常漂亮的区别。

---

# 20. 与 Latency-Aware Orchestration 的区别

这是另一个最大威胁。

它已经：

- workflow forecast；
- device-specific activation latency；
- peak memory；
- model loading；
- future model demand；
- physical execution graph；
- placement/order/lifecycle 联合优化。citeturn126512academia71

如果你继续走 BCSR：

> workflow forecast × machine cost model

会和它越来越近。

反而现在你要求：

> **不读 machine-response model**

使得你的问题变得不同：

你研究的不是：

> 如何建立一个完整 physical execution optimizer？

而是：

> **per-instance partial future belief 本身是否足以改善 rolling scheduling？**

这条线更干净。

---

# 21. 与 KVFlow / Halo / Scepsy

KVFlow 利用 Agent Step Graph 的 `steps-to-execution` 管 KV eviction/prefetch，本质上是已知/抽象 future activation 的 cache management。citeturn126512academia72

Halo 把结构化 workflow 当作 query-plan DAG，并通过 prefill/decode/cache/GPU-placement cost model 做 plan optimization，更偏“已知 plan 的物理优化”。citeturn126512search6

Scepsy恰好走另一个极端：它认为 workflow branching/fan-out/recur 很难精确预测，因此不预测具体 suffix，而是利用每个 LLM 的 aggregate execution share 来做资源配置。citeturn198522academia24

所以你和 Scepsy 可以形成一个很好的 conceptual contrast：

> **aggregate-history view vs per-instance short-horizon view**

你需要实验证明：

> per-instance prefix-conditioned future 比 unconditional aggregate/history prior 更有价值。

这正好对应一个很好做的 control：

\[
\text{instance prediction}
\quad vs\quad
\text{train prior}
\]

---

# 22. 理论位置也很好

ESA 2026：

> 后续 operation 只有前一 operation 完成后才逐渐揭示。citeturn439302search4

Online Scheduling with a Stochastic Signal 则说明非 clairvoyant 与完全 clairvoyant 之间存在一种“部分、随机获得未来信息”的有意义区间，并能改善调度保证。citeturn439302academia58

这给你一个比“AI agent 很动态”高级很多的理论 framing：

\[
\boxed{
\text{Agent execution is a gradually revealed scheduling problem with learned partial clairvoyance.}
}
\]

你的 predictor 就是：

> learned partial clairvoyance。

这个概念我很喜欢。

---

# 23. 安全的 “To our knowledge…” 表述

我建议论文暂时写成：

> **To our knowledge, existing agent-serving schedulers either rely on known or historically inferred workflow structure, optimize the currently revealed workflow frontier, or aggregate workflow-level demand. We are not aware of prior work that repeatedly derives an instance-specific probabilistic belief over the still-unrevealed short-horizon suffix from the execution prefix and uses that belief to reschedule node-level GPU actions as the workflow unfolds, without relying on an offline machine-response cost model.**

这个表述目前比较安全。

它没有声称：

- first probabilistic graph；
- first workflow-aware scheduler；
- first heterogeneous GPU scheduler；
- first predictive scheduler。

而是锁死在：

> **instance-specific + unrevealed suffix + re-predict/replan + no machine response model**

这个交叉点。

---

# 24. 我推荐的新正式架构

我会暂定名：

## **PDRS — Prefix-Conditioned Distributional Receding-Horizon Scheduling**

架构只有三块：

\[
\boxed{
\text{Execution Prefix}
\rightarrow
\text{Probabilistic Future Suffix}
\rightarrow
\text{Receding-Horizon Scheduler}
}
\]

不要画 machine profile block。

系统图里 machine substrate 只放在最下面：

```text
Agent instance
      │
      ▼
 observed prefix
      │
      ▼
Prefix-conditioned predictor
      │
      ├── remaining-length distribution
      ├── future-model probabilities
      ├── runtime distributions
      └── loading probability/duration
      │
      ▼
Receding-horizon scheduler
      │
      ▼
   (node, GPU)
      │
      ▼
Execution substrate
      │
      └── next observation ───────┐
                                 │
                                 └──── back to predictor
```

重点是最后那个闭环。

**rolling/receding horizon 才是这版架构真正应该突出的地方。**

---

# 25. 最小实验矩阵

现有主表：

**完全不动。**

新增 mechanistic table：

| Arm | Future info | Horizon | placement info |
|---|---|---|---|
| Myopic | none | 0 | current only |
| F0 | scalar p95 | fixed H5 | current only |
| **PDRS-O** | marginal future distributions | survival-weighted H≤5 | current only |
| **PDRS-P** | same | survival H≤5 | + next-model probabilistic residency |
| Prior-control | train unconditional prior | same | same consumer |
| Shuffle-control | wrong instance's future belief | same | same consumer |
| Truth/Oracle | realized future | same consumer | ceiling |

我尤其推荐：

### PDRS-O

只测试：

> fixed H5 scalar → per-instance stochastic horizon + marginals

### PDRS-P

在 PDRS-O 基础上再测试：

> model distribution 是否能改变 GPU placement。

这样归因非常清晰。

---

# 26. 两个负控制继续保留

### 单 GPU

只有一个 GPU：

\[
\text{placement contribution}=0
\]

如果 PDRS-P 相比 PDRS-O 仍变化很大，说明实现有串味。

### all-resident

所有 future model 均 resident：

\[
A(i,g)\approx0
\]

于是 placement belief 应退化。

非常适合证明机制。

---

# 27. 先做哪一步

我会按这个顺序：

**第一步，不改 predictor。**

实现：

\[
q_k=P(L\ge k)
\]

以及：

\[
W_i=
\sum_kq_k
[
runtime\_mean_k+
p^{load}_k\cdot p95(load_k)
]
\]

得到 **PDRS-O**。

先回答：

> 使用完整 length probability 做 stochastic-horizon marginal lookahead，会不会超过 frozen F0？

成本非常小。

---

**第二步，仍然不改 predictor。**

加入：

\[
P(M_{1}=m)
\]

做 **PDRS-P** 的 H1 placement affinity。

这一步才是真正把“概率分布”进入 node×GPU 动作。

---

**第三步。**

加入：

- train-prior；
- shuffled belief；
- oracle next-model；

三个 matched information controls。

如果 PDRS-P 过不了这些 control，立即停止继续复杂化。

---

**第四步，只有前面成立才考虑：**

- full role probabilities；
- robust coupling；
- H2/H5 action rollout；
- runtime_probs risk；
- VRAM/resource predictor。

---

# 最终裁定

我现在会把五轮讨论收敛成下面这个版本：

### 不再主推
**BCSR = future belief × exact measured machine table**

它太容易陷入 perfect-model 争议，而且和 Latency-Aware Orchestration 越来越近。

### 也不主推
**joint future-graph predictor**

因为你当前：

- 是串行链；
- 没有 joint；
- 没有 edges；
- 没有 VRAM distribution。

现在这么说会过度包装。

### 正式主推

\[
\boxed{
\textbf{Prefix-Conditioned Distributional Receding-Horizon Scheduling}
}
\]

一句话：

> **Given only the execution prefix of each running agent instance, predict a probabilistic short-horizon belief over its still-unrevealed suffix and repeatedly use that belief to reschedule current node–GPU actions as execution unfolds.**

方法**不读取机器行为实测表**。

实测表继续作为隐藏的、所有方法共享的 calibrated execution environment。

这既保留了你已经做完的 GPU 测量和 frozen substrate，又消除了“方法在解自己仿真器”这个最危险的问题。

而且从现在的文献格局看，我认为真正最值得守的创新问题已经可以非常明确地写成：

> **Can learned partial clairvoyance about an unrevealed agent-workflow suffix improve online node-level GPU scheduling beyond using only the revealed frontier?**

我认为这比前四轮任何一个单独的 “p95 / BCSR / machine-profile-aware placement” framing 都更像一篇完整论文的核心研究问题。memcite