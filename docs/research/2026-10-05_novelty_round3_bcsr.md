# 第三轮创新点讨论（闲置字段挖掘）— 结论：BCSR（Belief-Conditioned Substrate Rollout）

## 来源

网页版 GPT（GPT-5.6 Sol + High）2026-10-05 第三轮：用户指出"预测产物里大量字段没用上太可惜"，
要求把闲置字段逐一映射到调度动作、挖掘全部创新点并排序。

## 结论摘要

- **最可惜的闲置字段（三组）**：`model_probabilities`（下一步是什么模型）、
  `role/role_probability`（以什么执行形态运行——substrate 查表按 (model, role)）、
  `length_probabilities`/`termination_probability`（未来真的会发生吗）。
- **主方法升级为 BCSR**：Φ(i,g) = Σ_k q_k · E_{m,r}[C_k(s'=T(s,(i,g)), m, r)]，
  q_k = P(L≥k)（存活加权），C_k 全部来自**实测 substrate**（load/evict/coloc/interference/feasibility），
  S_BCSR = S_F0 + Φ。评分 9.3/10。
- **成长路线**：M0 模型分布 → M1 +存活加权 → M2 +role 敏感实测表 → M3 耦合稳健。
- **最小实验（6 臂，node 选择与 F0 完全一致，只改 GPU）**：
  F0 / P1 model-only / P2 +survival / P3 +role / train-prior 对照 / oracle-next。
- **新候选**：Probabilistic Action-Set Preservation（最小化"未来无可行动作"概率；
  lexicographic 第一键，无超参）——适配我们的 fail-closed 准入（容量/共置表/批处理/嵌套）。
- **只当 ablation**：统计量切换、自适应 H、软加载概率、置信度熵、retry、raw_action、lane 等。

## 本轮契约检查（本仓库实测）

1. **role 只有 top-1 + `role_probability`，没有完整分布**（0/1103 步有 `role_probabilities`）
   → E[C] 不能直接对 role 求和；用 nominal（残余质量→train 先验）或 robust（最坏情况）→ 正是
   Coupling-Robust 方法的天然动机。
2. **`length_probabilities` 与 `termination_probability` 不是同一量**（|P(L=0)−term| 中位 0.012、
   最大 0.9999；6 值分布求和恒为 1，质量集中在索引 5=视野上限）→ 存活权重用 length_probabilities
   算 q_k = P(L≥k)，**不要**再乘 termination（避免双折扣）。
3. 文献警告：不可声称"首个把未来用于 lifecycle/admission"（FATE/Latency-Aware/KVFlow/WarmServe/
   CONCUR 已占）；可用窄表述："To our knowledge, a scheduler that propagates probabilistic beliefs over
   unrevealed agent-workflow futures through measured, role-sensitive GPU interaction profiles to
   evaluate current node–GPU actions."

审核回复原文如下（未改动）。

---

炜，这一轮把字段逐个拆开后，我的结论反而更清楚了：

**真正“浪费”得最可惜的不是 `runtime_probs`，而是三组字段：`model_probabilities`、`role/role_probability`、`length/termination`。** 它们恰好分别回答：

- 下一步**可能是什么模型**；
- 它以什么**执行形态/role**运行；
- 这个未来**到底有多大概率真的会发生**。

而你的 substrate 恰好又有 `(model, role)` 敏感的共置/干扰表、model residency/load/evict、batch admission 和容量约束。

所以前两轮的 A* 可以进一步升级成一个更完整、也更有辨识度的东西：

\[
\boxed{\text{workflow future belief}
\rightarrow
\text{measured substrate consequences}
\rightarrow
Q(s,(node,GPU))}
\]

我暂时把它叫：

**Belief-Conditioned Substrate Rollout，BCSR**

它比“model probability → residency-aware placement”强一档。

文献上，FATE 已经明确对已知 DAG 的 `(stage,device)` assignment 做 future-state preservation，包括 residency、locality、future reachability；最新 Latency-Aware Orchestration 也已经用 workflow forecasts 预测 future model demand，并联合 placement/order/lifecycle。citeturn161654view4 citeturn453370search3  
但在这轮扫描里，我仍没有找到一个工作明确做了：

> **对尚未揭示 workflow suffix 的概率 belief，逐步通过实测的 model×role GPU interaction surface，来评价当前 `(node,GPU)` 动作。**

这个是目前最值得守的窄创新区。

---

# 1. 闲置字段 × 调度动作映射

| 闲置字段 | 最自然的调度动作 | 推荐的消费方式 | 文献覆盖 / 新颖性判断 | 成本 | 不重训 |
|---|---|---|---|---:|---|
| `model_probabilities` | **placement、residency、prewarm、evict** | 对不同 GPU 计算未来模型命中/加载/驱逐的期望后果 | 最近邻很多，但“未揭示概率 future → 当前 GPU action value”仍有空间。FATE/Latency-Aware 是最大威胁。citeturn161654view4 citeturn453370search3 | 小→中 | ✅ |
| `model_probability` | reliance gating、ambiguity | top-1 置信质量；决定是否值得相信 model identity | 单独很弱；full distribution 明显更有价值 | 小 | ✅ |
| `length_probabilities` | **lookahead depth、placement、lifecycle** | 用 \(P(L\ge k)\) 给第 k 步 future consequence 加 survival weight | stochastic horizon 本身不是新思想；online gradually revealed jobs 已成为正式调度模型。citeturn453370search2 | 小 | ✅ |
| `predicted_future_length` | horizon control | point-horizon baseline | 信息比 `length_probabilities` 弱；适合 ablation | 极小 | ✅ |
| `termination_probability` | prewarm/evict、lookahead | “还有没有下一步”的 continuation gate | termination-aware lifecycle 可用，但单独不够新；KVFlow/WarmServe 已经做 future-aware prefetch。citeturn894623search4 citeturn894623search0 | 小 | ✅ |
| `scenario_probability` / `synthetic_rollout` | scenario rollout | 多场景 expectation/CVaR/robust action value | **当前 prob=1 且单场景，所以目前没有新增信息量**；不能包装成创新 | — | — |
| `role + role_probability` | **co-location、interference、batch compatibility** | 把未来 role uncertainty 映射到 `(model,shape=role)` 实测表 | 这一条非常有价值；我没找到近邻明确传播 uncertain role belief 进实测 role-sensitive GPU interaction table | 中 | ✅ |
| `action_family + family_probability` | execution-class routing、tool placement | 只有 substrate/训练表存在 family→resource/action 映射时才有意义 | 当前没有直接 substrate hook，单独很弱 | 小/中 | ✅ |
| `is_retry_probability` | residency retention、eviction、prewarm | retry 高则提高保留当前/上一模型状态的 option value | agent recurrent workflow 已普遍存在，但 retry-probability-aware GPU residency 我未找到直接近邻；较窄 | 小 | ✅ |
| `merged_nested_call_probability` | **memory reservation、placement、admission** | 对潜在 nested GPU call 预留 headroom / 避免把未来内层调用卡死 | 比较新颖，但需要知道 nested demand/model identity；仅一个概率不够 | 中 | ✅* |
| `execution_lane` | CPU/GPU routing、future GPU demand masking | future CPU 步不应贡献 GPU residency/co-location cost | 主要是正确性，不是创新 | 极小 | ✅ |
| `raw_action` | tool-specific compatibility | 仅在 raw_action 与 batch/nested/resource contract 有明确关系时使用 | 易过拟合 workload ontology；不适合作主贡献 | 小 | ✅ |
| `step_offset` | prewarm timing、eviction priority、horizon discount | 越近的 future demand 更有价值；也决定预热 lead time | KVFlow 的 steps-to-execution 已非常接近。citeturn894623search4 | 小 | ✅ |
| runtime `p50/p90/p95/mean/CVaR` | ordering、deadline、risk score | 已有 | **基本榨干**；TIE 和你自己的 Phase20-B 都说明继续换 scalar 风险低回报。TIE 已直接做 distribution-aware length scheduling。citeturn936616search3 | 已有 | ✅ |
| `runtime_probs[16]` | **batch compatibility、deadline risk、preemption、robust action evaluation** | 不再压成 p95；用于 distribution overlap / probability of incompatibility | 比纯 scalar 更有空间，但“用 distribution 排队”已有 TIE，且 ICML 2026 甚至指出 prediction-driven size scheduling 可脆弱。citeturn936616search3 citeturn936616search0 | 中 | ✅ |
| `load_occurrence_probability` | placement、lifecycle | 把 hard `≥0.5` 改成连续 probability weighting | **正确性改进 > 创新** | 小 | ✅ |
| `load_duration` quantiles | placement、prewarm risk | probability-weighted tail load surcharge | 注意只有 quantiles，没有 conditional mean，不能无假设地声称“expected load duration” | 小 | ✅ |
| `bin_schema_id` | 无动作 | compatibility/fail-closed contract | **绝不能当策略 feature** | — | — |
| `resource_head_id` | 无动作 | provenance / head compatibility | **绝不能当策略 feature** | — | — |

这里最关键的一点是：

**不是所有 predictor output 都应该进 scheduler。**

好的 action-aware predictor output 必须满足：

\[
\exists a_1,a_2:
\quad
C(a_1\mid z)\neq C(a_2\mid z)
\]

也就是这个信息 \(z\) 真能改变动作之间的相对价值。

这正是为什么：

- `p95 runtime` 主要改变 **node ordering**；
- `model_probabilities` 能改变 **GPU placement**；
- `role_probability` 能改变 **co-location/interference feasibility/cost**；
- `length_probability` 能改变 **这些后果应当计到第几步**。

---

# 2. 有一个特别重要的技术坑：role 可能并没有“完整分布”

你的完整字段清单写的是：

> `role + role_probability`

而不是像 model 一样明确的 `role_probabilities{...}`。

如果实际上只有：

\[
(r^*,p^*)
\]

即 top-1 role 及它的概率，那么你**不能**直接写：

\[
E[C]
=
\sum_r p(r)C(r)
\]

因为其他 role 的概率没给出来。

这反而给 Coupling-Robust 一个非常自然的入口。

最简单可以写成：

\[
C_{\rm role}^{robust}
=
p^* C(r^*)
+
(1-p^*)\max_{r\neq r^*}C(r)
\]

或者残余质量按 train-only role prior 分配，作为 nominal 版本：

\[
C_{\rm role}^{nom}
=
p^* C(r^*)
+
(1-p^*)E_{r\sim p_{\rm train}}C(r)
\]

然后：

**nominal vs robust vs oracle-role**

正好是一套漂亮 ablation。

如果其实 artifact 中存在完整 role probability vector，那么再升级成 model×role coupling。

---

# 3. 我现在最看好的完整字段组合

这是这轮最大的收获。

对当前动作：

\[
a=(i,g)
\]

执行后得到状态：

\[
s_a=T(s,a)
\]

先从 `length_probabilities` 得到：

\[
q_k=P(L\ge k)
=
\sum_{\ell=k}^{5}P(L=\ell)
\]

于是第 4、5 步如果 workflow 很可能早就终止，就不会像现在 F0 一样被满额计入。

然后第 k 步根据未来 model belief：

\[
p^M_k(m)
\]

和 role belief/置信度：

\[
p^R_k(r)
\]

评价未来：

\[
C_k(s_a,m,r)
\]

这个 \(C\) **不是神经网络预测值**，而是你冻结 substrate 已经有的实测机制：

\[
C_k=
C_{\rm load}
+
C_{\rm evict}
+
C_{\rm coloc}
+
C_{\rm interference}
+
C_{\rm feasibility}
\]

最终：

\[
\boxed{
\Phi(a)=
\sum_{k=1}^{5}
q_k\,
\mathbb E_{m,r}
[
C_k(s_a,m,r)
]
}
\]

然后：

\[
S_{\rm BCSR}(i,g)
=
S_{\rm F0}(i,g)+\Phi(i,g)
\]

这就已经远远不是：

> “把 p95 换成 distribution”

而是：

> **预测哪种 future state 会出现，以及当前 action 会如何改变这些 future states 的系统代价。**

FATE 已经证明“当前 assignment 产生的 future state”是有价值的，但它的 formulation 是显式 DAG stage 和 state-dependent cost；你的潜在区别是未揭示 future 的 probability belief。citeturn161654view4

最新 Latency-Aware Orchestration 则已经把 workflow forecast、future model demand、model lifecycle 和 placement/order 放到一起，因此你不能泛泛地声称“首个 predictive lifecycle scheduler”。citeturn453370search3

你真正应守的是更窄的：

> **belief over unrevealed future × action-conditioned measured interaction surface**

---

# 4. 你提出的六个具体方向逐一裁定

## (a) 自适应视野：有用，但不要单独做创新

我的评价：**中价值，低主创新性，强烈建议作为主方法组成部分。**

而且最好别写成：

> termination 高 → H=1，否则 H=5。

更漂亮的不是离散“选 H”，而是：

\[
\Phi=\sum_k P(L\ge k)C_k
\]

也就是 **stochastic horizon / survival-weighted lookahead**。

这样：

- 不需要阈值；
- 不需要调 H；
- 自动退化；
- 直接消费完整 `length_probabilities`。

`predicted_future_length` 只作为 point-horizon ablation。

注意不要同时再乘 `1-termination_probability`，除非 artifact contract 证明两者独立。如果 `length_probabilities[0]` 已经表示“零个未来步骤”，再乘 termination 会**重复折扣**。

建议先检查：

\[
P(L=0)\stackrel{?}{\approx}P(\mathrm{terminate})
\]

若本质重复，就确定一个 canonical source。

自适应 horizon/MPC 在调度和控制中本身很成熟，2026 的 resource-allocation 工作也直接使用 finite receding horizon，所以这不能单独做 headline。citeturn987586search1

**定位：主方法组件。**

---

## (b) role 分布 × substrate 查表：非常有价值

我的评价：**高价值，这轮新增字段里最值得升级成方法的一条。**

因为你的实测表不是：

\[
C(model_A,model_B)
\]

而是：

\[
C(model_A,role_A,model_B,role_B)
\]

或者干扰：

\[
C(infer\_model,infer\_role,load\_model)
\]

这意味着：

**role prediction 是真正 action-relevant 的。**

当前 F0 完全没消费这个信息。

例如两个 GPU：

- GPU0 正在跑 `8B/planner`
- GPU1 正在跑 `8B/videotool_spatial`

未来模型都可能是 3B。

即使：

\[
P(model=3B)
\]

相同，未来 interference 可能完全不同。

所以 placement 应该依赖：

\[
P(model,role)
\]

而不只是：

\[
P(model)
\]

WarmServe 已显式考虑 prewarming interference，FATE/Latency-Aware 已做 state-aware placement，但我本轮没有找到一个工作明确把**未来 role uncertainty**传播进 empirical role-sensitive co-location/interference matrix。citeturn894623search0 citeturn161654view4

**这是你当前最值得开发的“新字段”。**

不过有一个 joint 问题：

你有 full model distribution，但 role 可能只是 top-1 confidence。

因此第一版千万不要默认：

\[
P(m,r)=P(m)P(r)
\]

这正适合 coupling-robust extension。

---

## (c) “期望加载”替代 ≥0.5：应该做，但只是修正消费者

我的评价：**应该，创新性低。**

现在：

\[
I(p_{load}\ge0.5)\cdot p95(L)
\]

在 0.499 和 0.501 之间存在不自然跳变。

更合理可以做：

\[
p_{load}\cdot C_{\rm measured-load}(m)
\]

如果你的 `C_measured-load` 是 substrate 已实测的确定性 load cost，那么这个确实可以解释为预期 load transition cost。

但如果你写：

\[
p_{load}\cdot p95(L)
\]

它不是数学意义上的：

\[
E[L]
\]

应该称：

**probability-weighted tail-load cost**

而不是 expected load。

因为你只有 load duration 的 p50/p90/p95，没有 conditional mean/full distribution。

它很适合做：

F0-hard-threshold vs F0-soft-load

结果无论涨不涨都可以解释，但**不能做主贡献**。

---

## (d) 分布感知 batch admission：有价值，但我不把它放前三

我的评价：**中等潜力，中等风险。**

当前 homogeneity：

\[
\frac{\max(\hat T_1,\hat T_2)}
{\min(\hat T_1,\hat T_2)}
\le \tau
\]

可以变成：

\[
D(P_{T_1},P_{T_2})\le\tau_D
\]

例如：

- Wasserstein-1；
- quantile distance；
- overlap probability；
- \(P(T_1/T_2\in[1/\tau,\tau])\)。

这种思想是合理的，因为 homogeneous B=2 profile 的有效性本来就是一个 distribution compatibility 问题。

但它有两个问题。

第一，TIE 已经证明 distribution-aware runtime/length scheduling 是热点，而 2026 年 9 月 vLLM 社区甚至已经在讨论 length-aware batch composition；所以“distribution → batching”并不是完全没人想到。citeturn936616search3 citeturn453370search0

第二，更关键：

**你这些 `runtime_probs` 是未来 step 的 runtime distribution。**

如果正式 batch admission 需要的是“当前两个 ready request”的 runtime distribution，那么必须确认 decision time 确实能拿到**当前候选节点自己的 distribution**。

如果当前节点只有 point estimate，而 `runtime_probs` 只是它对 successor 的预测，那么不能偷偷拿 predecessor 过去的 forecast 当成当前新鲜预测。

所以这条先做 contract audit。

如果 contract 成立，再做：

- point-homogeneity；
- distribution-homogeneity；
- oracle-runtime homogeneity。

否则它更适合做**future batchability value**，而非当前 admission。

**定位：secondary mechanism。**

---

## (e) 终止感知 prewarm / eviction：有用但文献覆盖较多

我的评价：**中低新颖性，高可实现性。**

比如：

\[
V_{\rm prewarm}(m)=
P(\text{future uses }m)
\times
\text{load saving}
-
\text{interference}
-
\text{eviction opportunity cost}
\]

而：

\[
P(\text{future uses }m)
\]

自然应含 survival probability。

这比 Hermes 当前的 point downstream demand / threshold 更完整。

但：

- KVFlow 已根据 future activation 做 prefetch/eviction；citeturn894623search4
- WarmServe 已 forecast future workload 做 interference-aware GPU prewarming；citeturn894623search0
- FATE 也将 lifecycle/locality 纳入 future state。citeturn161654view4

所以不能单独当贡献。

最合适的角色是：

> BCSR 的 lifecycle instantiation。

---

## (f) merged_nested / retry probability：有意思，但先别做主线

### Retry

可以定义：

\[
V_{\rm keep}(m)
=
P(retry)\times C_{\rm reload}(m)
\]

retry 高：

- 不急着 evict；
- 倾向保持 residency；
- 同一 job 后续节点 placement 更 sticky。

这个很自然，而且我本轮没有找到一个 agent GPU scheduler 明确把 `retry_probability` 当 residency value。

但 reviewer 很可能问：

> retry prediction 真的准吗？
> retry 后是否一定使用同一个模型？

所以它需要额外语义证明。

**定位：机制 ablation / appendix。**

### merged_nested_call

这个反而可能挺新。

如果当前 action 把 GPU 塞得太满：

\[
s_a
\]

未来 CPU composite 节点内部突然触发 nested GPU call，就可能：

- 等待；
- 驱逐；
- OOM/fail-closed。

所以可以计算：

\[
P(nested)\times C_{\rm latent-capacity}
\]

甚至：

\[
P(\text{future nested call has no feasible GPU}\mid a)
\]

这本质上是：

**latent resource-demand-aware placement**。

但有一个硬条件：

`merged_nested_call_probability` 必须能和：

- nested model identity；
- memory demand；
- runtime

建立 train-only/frozen mapping。

如果只有：

> 有 0.3 概率发生 nested call

但不知道是哪种 nested demand，那么它不能直接支持精确 reservation。

可以做 conservative headroom，但会容易被质疑是 heuristic。

**新颖性：中高。**
**可行性：中。**
**风险：高。**

我会先放第 4–5 顺位。

---

# 5. 这轮还挖出了一个前两轮没单独讲清楚的创新：Probabilistic Action-Set Preservation

你的 substrate 有大量 fail-closed admission。

所以一个 current action 不仅改变 future cost，还可能改变：

\[
\mathcal A_{t+k}
\]

即未来还能选哪些 GPU。

定义：

\[
N_{\rm feasible}(a,z)
=
|\mathcal A(T(s,a),z)|
\]

那么可以评价：

\[
R(a)
=
P_{z\sim belief}
\left[
N_{\rm feasible}(a,z)=0
\right]
\]

或者：

\[
E[N_{\rm feasible}]
\]

也就是说：

> 不只问未来贵不贵，而问当前 action 是否会把未来逼进狭窄甚至不可执行的 action set。

这非常适合你的：

- capacity fail-closed；
- sparse measured co-location table；
- batching compatibility；
- nested calls。

FATE 已经明确提出 **future device reachability**，所以“保存未来可达设备”本身不能 claim 新。citeturn161654view5

你的可区分点必须是：

> **unrevealed future 的概率 feasible-set risk**

而不是 known DAG reachability。

我认为这是第一名方法里非常好的一个 penalty：

\[
\Phi(a)
=
E[C_{\rm future}]
+
\lambda
P(\mathcal A_{\rm future}=\varnothing)
\]

但先不要引入 λ。

MVP 可以 lexicographic：

1. 最小 `future infeasible probability`
2. 再最小 expected future state cost。

这样甚至无超参。

---

# 6. 另一个可用新点：Decision-Relevant Uncertainty，而不是“预测置信度”

这一轮我也更喜欢这个版本。

不要问：

> predictor uncertainty 大不大？

而问：

> **预测的不确定性会不会改变 scheduler 选什么？**

比如未来可能是：

\[
z_1,z_2,z_3
\]

如果无论哪个未来：

\[
\arg\min_a C(a,z_j)=a^*
\]

那这个 uncertainty 对当前 decision 根本不重要。

只有不同 plausible futures 导致不同 best action：

\[
a^*(z_1)\neq a^*(z_2)
\]

才需要复杂 belief reasoning。

这和 2026 predict-then-optimize 文献中的“switching threshold”思想非常接近：不确定性只有跨过使最优动作发生变化的阈值时，才真正产生 decision cost。citeturn936616search7

可以定义：

\[
A_{\rm disagreement}
=
1-
\max_a P(a=a^*(Z))
\]

高 disagreement：

- 用 robust / belief rollout；
- 低 disagreement：
- 直接 F0，省计算。

这会比：

\[
p95-p50
\]

作为 confidence 高级得多。

**但它仍然更适合作为 robustness/efficiency contribution，不是第一贡献。**

---

# 7. 真正的创新候选清单

| 候选 | 一句话主张 | 关键字段 | 最近邻差异 | 最小实验 | 最大风险 |
|---|---|---|---|---|---|
| **C1 BCSR** | 对未揭示 workflow belief 计算每个 `(node,GPU)` 动作诱导的实测 substrate future cost | model probs + length + role + load prob | FATE/Latency-Aware 有 future state，但这里是 probabilistic unrevealed suffix + measured role-sensitive state | F0 / placement-only / BCSR / oracle-next | “FATE + probability” |
| **C2 Coupling-Robust BCSR** | 不伪造 model×role joint，直接在所有合法 coupling 上评价 worst-case future action cost | model marginals + role confidence/full marginal | DRO 技巧已知，但问题结构很贴合你的 artifact contract | independence/prior / robust / oracle joint | 被说成标准 DRO |
| **C3 Information×Actionability** | 同一预测信息在 ordering、placement、lifecycle、admission 中价值不同 | 全字段 + trace | 不是新 scheduler，而是系统性测“预测何时 actionable” | Myopic/F0/New + matched controls | 被说成只是 ablation |
| C4 Probabilistic action-set preservation | 最小化未来无可行 GPU / action-space collapse 风险 | model+role+length | FATE 有 reachability；你的差异是 uncertain future | cost-only / +reachability / oracle | 和 FATE 太近 |
| C5 Distribution-aware batch compatibility | 用 runtime distribution 而非点值判断同质 batching | runtime_probs | TIE 管 ordering；当前没找到同样的 agent-future batch gate | point gate / dist gate / oracle | incremental；contract 问题 |
| C6 Termination-aware lifecycle | 只有高概率真正到达的 future 才值得 prewarm/保留 | length/termination+model | KVFlow/WarmServe 很近 | current prewarm / survival-aware | 新颖性弱 |
| C7 Retry-aware residency | retry 概率成为 model keep-value | retry+model | 较少直接近邻 | with/without retry value | 窄、语义依赖 |
| C8 Latent nested-demand reservation | 对未揭示 nested GPU call 保留未来容量 | nested prob+model/mem map | 本轮没找到直接近邻 | no reserve / expected / oracle nested | workload-specific |
| C9 Decision-relevant uncertainty gate | 只有预测 uncertainty 会改变 best action 时才启动复杂 rollout | full belief | 与 predict-then-optimize switching threshold 接近 | always-on / gate / F0 | reliability 定义 |
| C10 Soft load probability | 去掉 0.5 hard threshold | load occurrence | 数学修正 | threshold / soft | **只能 ablation** |

---

# 8. 最终排名

这轮字段扫描以后，我的前三名实际上不是完全换了，而是**第一名变得更强、更具体了**。

## 第一名：BCSR — model × role × stochastic horizon 的 action-conditioned substrate rollout

综合评分我给 **9.3/10**。

最小版本：

\[
\Phi(i,g)
=
\sum_m
P(m_{t+1}=m)
C_{\rm load/evict}(m\mid T(s,i,g))
\]

然后分阶段加字段：

\[
\text{M0: model}
\]

\[
\text{M1: model + length survival}
\]

\[
\text{M2: model + length + role-sensitive coloc/interference}
\]

最后才：

\[
\text{M3: robust coupling}
\]

这是最自然的研究成长路径。

---

## 第二名：Information × Actionability decomposition

评分 **8.8/10**。

它不是一个算法，但它非常可能让整篇论文“从一个 heuristic scheduler”升级成一篇有明确 scientific question 的系统论文：

> 哪类 partial future information 对哪类 scheduling action 有价值？

你的已有 F0/Myopic/Truth/Oracle 和 decision trace 恰好已经完成了一半。

新方法完成另一半。

而且“逐渐揭示信息”现在已有正式 online scheduling 理论基础。citeturn453370search2

---

## 第三名：Coupling-Robust model×role action value

评分 **8.1/10**。

它完美解决你的真实数据契约：

> marginals available, joint unavailable。

尤其 role 如果实际上只有 top1 probability，这个动机更强。

它比“直接假设 model/role independent”严谨很多。

但建议作为 BCSR 的 robust variant，而不是独立做整篇。

---

# 9. 哪些只能当 ablation

我会明确降级以下内容：

- `p50 / p90 / p95 / mean / CVaR` 切换；
- 自适应 H 本身；
- `load_prob≥0.5 → soft probability`；
- `model_probability` entropy/confidence；
- termination-aware prewarm 单独一项；
- retry-aware residency 单独一项；
- raw_action / family probability；
- execution lane；
- predicted_future_length；
- `bin_schema_id / resource_head_id`。

这些都可以让方法更完整，但拿出来独立 claim 会显得碎。

---

# 10. 我建议现在真正实现的最小实验

先**完全不要碰 batch/preemption/nested**。

只做 6 个臂，而且全部在冻结 DES/substrate 上：

| Arm | node ordering | GPU placement future info |
|---|---|---|
| F0 | F0 | 无 |
| **P1 Model-only** | 与 F0 完全相同 | `model_probabilities` |
| **P2 Model+survival** | 与 F0 相同 | + `length_probabilities` |
| **P3 Model+role** | 与 F0 相同 | + role-sensitive measured tables |
| Prior control | 与 F0 相同 | train unconditional model prior |
| Oracle-next | 与 F0 相同 | realized next model/role |

第一轮最好继续坚持：

> **node selection 与 F0 一模一样，只允许 GPU placement 改变。**

这样结果极其干净。

如果：

\[
P1 < F0
\]

你证明 model belief 对 placement 有价值。

如果：

\[
P2 < P1
\]

你证明 stochastic horizon 有价值。

如果：

\[
P3 < P2
\]

你证明 **role-aware measured interaction belief** 有价值。

然后再放开：

\[
(node,GPU)
\]

joint optimization。

同时看机制链：

\[
load\_ms\downarrow
\]

\[
evictions\downarrow
\]

\[
future\ residency\ hit\uparrow
\]

\[
colocation/interference\ cost\downarrow
\]

最终：

\[
JCT\downarrow
\]

这就会形成非常漂亮的因果证据。

---

最后，关于你说的：

> “能不能做第一个把 workflow future belief 用于 admission/lifecycle 的每一类动作？”

**我现在不建议这样 claim。**

因为：

- FATE 已经把 placement、state preservation、reachability 放进来；citeturn161654view4
- Latency-Aware 已 joint placement/order/lifecycle；citeturn453370search3
- KVFlow 已 future-aware eviction/prefetch；citeturn894623search4
- WarmServe 已 forecast-driven interference-aware prewarming；citeturn894623search0
- CONCUR 已经把 agent-level admission control 当第一等机制。citeturn936616academia27

但我认为有一个更窄、目前扫描后仍然很有希望的“first-like”表述：

> **A scheduler that propagates probabilistic beliefs over unrevealed agent-workflow futures through measured, role-sensitive GPU interaction profiles to evaluate current node–GPU actions.**

这个我会继续作为你最值得验证的创新核心。

不过论文里在彻底做完 final related-work audit 前，最好仍写：

> “To our knowledge…”

而不要直接写 absolute “the first”。

**所以最终执行结论很明确：下一步不要继续研究 runtime statistic；先实现 `model_probs → placement`，然后立即加 `length survival`，第三步才加 `role-sensitive co-location/interference`。如果这三层能逐级给出增益，你的创新故事就基本成型了。**