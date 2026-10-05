# 方法创新性讨论（网页版 GPT，含 2024–2026 文献扫描）— 结论：方向 A* 而非"分位数调度"

## 背景

用户判断"当前 F0 创新性不足"，提出围绕分位数/分布预测做文章。本轮把现有资产（F0 消费者骨架、
预测产物字段、已探索的功能家族、substrate 机制）与候选方向 A–E 交给网页版 GPT 讨论，
并要求给出新颖性评估 + 推荐 + 风险 + 归因干净的实验设计。

**核心结论**：不建议升级为"分位数/分布预测调度"（2026 已饱和：TIE 已做 output-length 分布调度，
MAPS 做 uncertainty-calibrated bound；"p95 vs mean vs CVaR" 难成主贡献）。
建议升级为 **A\*：Probabilistic Action-Conditioned Future-State Scheduling**——
把"预测未来会发生什么"变成"**当前 action (i,g) 会怎样改变未来 GPU 状态分布**"。

## 关键事实（本仓库已核实）

- F0 的未来项 `sameshape_future_cost(node_id)` **与显卡无关** → 未来信息只影响 ordering、
  不影响 placement（这是缺口）。
- 预测产物已含 `model_probabilities`（下一步模型分布）、`runtime_probs`（16 桶）、
  load 分位数与发生概率、链终止/长度分布；**model_probabilities 目前完全未被消费**。
- 功能家族（p50/p95/condmean/stepcvar95/risk）已探索且 p95 为冻结冠军；Phase 20-B：
  oracle 只比 p95 好 ~1.3s → 换统计量已榨干。
- 逐步边际分布**不可识别联合 CVaR**（代码已明示）→ 方向 D 危险。

## 推荐的最小可实现版（H=1，无需重训预测器、无需联合分布）

Q(i,g) = C_now(i,g) + F_F0(i) + β·Φ(i,g)，
Φ(i,g) = Σ_m p(m|prefix) · min_h C_next(m, h | s')，s' = T(s,i,g)（放置后的 GPU 状态），
C_next 直接用已实测 substrate（驻留/加载/驱逐/加性干扰/容量）。

## 实验设计（信息 × 消费者 2×2 + 对照）

|  | GPU-agnostic consumer | Action-conditioned consumer |
|---|---|---|
| 预测未来 | 现有 F0 | **新方法** |
| 真值未来 | 现有 truth arm | **Oracle-AC 天花板** |

外加 Myopic=无未来。分解：F0−Myopic=信息价值；New−F0=同信息下消费者价值；
Oracle-AC−Pred-AC=predictor 还是 scheduler 是瓶颈。
负控制：①单 GPU 时应退化为 F0；②全驻留/加载=0 时 placement 价值→0。
记录：load 次数/驱逐/总加载 ms/未来驻留命中/浪费预取/干扰 ms/JCT/P95/deadline/决策开销。

## 风险

- **最近邻工作**：2026-09 的 *Latency-Aware Orchestration for Multi-Agent LLM Workflows on
  Heterogeneous GPUs*（workflow forecast → future model demand → placement/lifecycle/order 联合优化）
  必须作为 Related Work 第一近邻；"普通版 A"已不够。
- B/C/D/E 降级为 ablation/稳健性机制（B 调参感、C 拥挤、D 数学不可识别、E 概念错误：
  p95−p50 是 aleatoric 变异不是 predictor 置信）。
- 定位必须强调：**不是已知 DAG 上的 future-state planning，而是未揭示未来的 belief-state planning**。

## 对现有资产的意义

F0 恰好成为**极好的 ablation**（已证明"未来信息仅用于 ordering"值 ~1.49s）；
substrate（加载/驱逐/干扰实测）正是 C_next 所需；预测产物已带 model_probabilities。

审核回复原文如下（未改动）。

---

炜，我查完 2024–2026 这条线后，结论比“选 A 还是 B”更明确：

**不建议把论文升级成“分位数 / 分布预测调度”。** 这个表述到 2026 年已经不够新。更值得升级的是：

> **部分揭示 Agent workflow 的概率未来 → 当前 action-conditioned GPU state → 前瞻 placement。**

也就是把“预测未来会发生什么”进一步变成“**当前把这个节点放到 GPU g，会怎样改变未来的 residency / eviction / loading / interference 状态**”。

这可以继续使用你已经有的 p95 / `runtime_probs`，但它们应当是方法的输入，不再是创新点本身。

### 1. A–E 的新颖性判断

| 方向 | 我的判断 | 原因 |
|---|---|---|
| A 简单 GPU-aware future load | **中等，但单独不够强** | 方向正确，但 2026 已有 FATE、WarmServe、Hermes，以及 9 月刚出的 Latency-Aware Orchestration，都在做 future demand / residency / lifecycle / placement |
| **A\* action-conditioned probabilistic future state** | **最强** | 不只是“预测未来需求”，而是把**未揭示未来的概率 belief**与“当前 action 会造成的 GPU 状态转移”结合 |
| B 状态自适应 p50/p95/CVaR | **弱** | 很容易被看成风险系数/调参；adaptive risk appetite 本身不是新思想 |
| C chance constraint / deadline probability | **中弱** | SLO、deadline、stochastic scheduling 已非常拥挤；QLM、HexAGenT、JITServe 等已经覆盖很多 |
| D runtime_probs + CVaR/DRO | **弱～中** | TIE 已直接把 output-length distribution 用于调度；MAPS 也做 uncertainty-calibrated upper bound。你还缺 joint distribution，无法严谨定义总未来 CVaR |
| E p95-p50 gating | **弱** | 最大问题是 `p95-p50` 是**条件运行时离散度**，不是“预测器可信度”；拿它当 confidence 容易被审稿人抓 |

2026 年的文献压力其实已经很大。TIE 已明确提出“不要预测一个长度，要预测分布并基于尾部调度”，并在 ICML 2026 发表；MAPS 也用 uncertainty-aware calibration 得到有目标覆盖率的 output-length upper bound。单纯“p95 比 mean 好”“CVaR 比 p95 好”现在很难成为主贡献。citeturn426026search6turn426026search13

而且 Microsoft 的 *Beyond Prediction* 甚至反过来论证：prediction-driven scheduling 在 distribution shift、bursty arrival 和 GPU memory pressure 下可能脆弱，并提出 prediction-free 的 tail-aware scheduling。换句话说，审稿人现在会要求你解释“为什么这个 distribution prediction 真正改变了系统决策”，而不是看到一个新的 risk statistic 就满意。citeturn426026search1

A 的问题则更微妙。Hermes 已经有 **PDGraph → probabilistic backend demand → Gittins scheduling + backend prewarm**；WarmServe 根据未来 workload forecast 做 GPU prewarming/model placement；FATE 直接把 model residency、prefix reuse、parent locality 和 future device reachability 定义成 future state。citeturn189125academia24turn425580search0turn823940academia28

尤其需要注意 **2026-09-03 的 Latency-Aware Orchestration**：它已经明确写到预测 activation latency、memory、model-loading cost，传播 workflow dependency 来预测 future model demand，再联合优化 placement、ordering 和 model lifecycle。这个工作和“普通版 A”已经非常近。citeturn944297academia32

所以：

**“future model demand → residency-aware placement”本身已经不能作为你论文最核心的新意。**

但是我没有在这些我能核实的工作中看到一个非常明确的组合：

> **online partially revealed workflow + calibrated probabilistic unrevealed futures + action-conditioned GPU-state rollout。**

这才是你现在真正可以抢的位置。

---

## 2. 我推荐的主方法：A\*，不是原始 A

我会把方法重新定义成：

**Belief-State Action-Conditioned Lookahead Scheduling**

当前 F0 实际是在做：

\[
S_{F0}(i,g)
=
C_{\mathrm{now}}(i,g)
+
F_{\mathrm{H5}}(i)
\]

关键问题正是你发现的：

\[
F_{\mathrm{H5}}(i)
\quad\text{与 }g\text{ 无关}
\]

所以它预测了未来，却没有计算：

> “执行 action \((i,g)\) 以后，未来执行环境发生了什么变化？”

真正应该加入的是：

\[
Q(i,g)=C_{\mathrm{now}}(i,g)
+\lambda F_{\mathrm{order}}(i)
+\beta
\underbrace{
\mathbb E_{\omega\sim q_\theta(\cdot\mid prefix)}
[
V_H(T(s,i,g),\omega)
]
}_{\text{action-conditioned future-state value}}
\]

其中：

- \(s\)：当前 GPU residency / busy / cache / interference 状态；
- \(T(s,i,g)\)：把节点 \(i\) 放到 GPU \(g\) 后形成的新状态；
- \(\omega\)：尚未揭示的 workflow future；
- \(q_\theta\)：你的 predictor 对未来的概率 belief；
- \(V_H\)：在这个新 GPU 状态下执行未来 H 步的代价。

这和 F0 的本质区别非常大：

**F0：哪个 job 以后比较贵？**

变成：

**A\*：如果现在做这个 assignment，未来会不会因此变贵？**

后一句才是真正的 scheduling contribution。

### 最小可实现版，甚至不用重新训练 predictor

先只做 **H=1 probabilistic placement**。

你的每个节点已经有下一步：

\[
p(m_{t+1}=m \mid prefix)
\]

即 `model_probabilities`。

当前 action \((i,g)\) 执行后，得到 GPU 状态：

\[
s' = T(s,i,g)
\]

然后：

\[
\Phi(i,g)
=
\sum_m p(m\mid prefix)
\min_h
C_{\mathrm{next}}(m,h\mid s')
\]

这里 `C_next` 可以直接用现有实测 substrate：

- 是否 resident → load=0/非0；
- model load cost；
- 必需 eviction cost；
- measured additive load interference；
- capacity feasibility；
- 后续可以加 batching/co-location。

于是：

\[
Q(i,g)=C_{\mathrm{now}}(i,g)+F_{F0}(i)+\beta\Phi(i,g)
\]

最重要的是：

**H=1 不需要联合分布。**

因为你只对“下一模型”做期望，现有 `model_probabilities` 已足够。

这比直接上 runtime_probs × CVaR 安全得多。

---

## 3. 更强的完全体：联合未来 scenario，而不是继续换统计量

如果 H=1 有明显收益，我建议最终论文方法升级成：

\[
\omega^{(k)}
=
(m_1,r_1,m_2,r_2,\dots,m_H,r_H,L)
\]

也就是让 predictor 输出/采样 **joint future scenarios**。

然后：

\[
Q(i,g)
=
C_{\mathrm{now}}
+
\frac1K\sum_{k=1}^{K}
V_H(T(s,i,g),\omega^{(k)})
\]

这时你真正解决的是：

> 动态 Agent 的未来 DAG 尚未完全展开时，如何用概率未来预测驱动当前 GPU placement。

这和 FATE 的区别就非常好讲了。FATE 的核心是“当前 assignment 如何保存未来 state”；HexAGenT 的核心是 online-revealed DAG + heterogeneous placement；Latency-Aware Orchestration 已经进一步引入 workflow forecasting。citeturn823940academia28turn823940search4turn944297academia32

你的差异必须强调：

**不是已知 workflow 上的 future-state planning，而是 unrevealed future 的 belief-state planning。**

LLMSched 和 Hermes 虽然显式建模结构/需求 uncertainty，但 LLMSched主要用 uncertainty reduction + JCT scheduling，Hermes主要通过 Gittins ordering 和 backend prewarm 消费概率需求，并没有把每个当前 GPU assignment 当作一个 state-transition action 来评价未来 placement regret。citeturn369105search0turn189125academia24

这个交叉点，我认为明显比“p95 scheduler”强。

---

## 4. B/C/D/E 怎么处理

我都不会删除，但会**降级成 ablation / robustness mechanism**。

B 不要作为贡献。写成：

“Given the same predictive belief, we additionally test mean/p50/p95/CVaR consumers.”

你的 Phase 20-B 已经很好地说明这条路为什么不是主线：p90≈p95、oracle只剩约 1.3s、tail shuffle 后收益消失，说明 statistic-selection 已经接近饱和。

C 也不建议成为核心。QLM 本身就是 Bayesian uncertainty + stochastic optimization + SLO attainment；HexAGenT 已直接按 workflow-level SLO risk 排 ready calls；JITServe 面对 imprecise request information 做 SLO-aware serving。citeturn357195search0turn823940search4turn426026search10

可以把 deadline violation probability 作为第二目标：

\[
Q=C+\lambda P(T_{\rm finish}>D)
\]

但它是“目标函数增强”，不是主贡献。

D 最危险。你目前只有 per-step marginal `runtime_probs`。

所以：

\[
\mathrm{CVaR}\left(\sum_k T_k\right)
\]

**不能从这些 marginal 唯一确定。**

你现有代码里那句 “joint not identifiable” 是数学上正确的，不要为了论文强行绕过去。否则 reviewer 很容易问“你假设独立还是 comonotonic？为什么？”

如果未来真的建立 joint scenario predictor，那 D 才重新有意义。

E 则需要改名。

不要写：

> p95−p50 大 → predictor confidence 低。

这是不成立的。

例如一个预测器可以非常准确地知道某节点 runtime 天生重尾，那么 p95−p50 很大，但模型其实非常“有信心”。

更合理的是：

- runtime spread = **aleatoric variability**；
- model probability entropy = **future ambiguity**；
- held-out calibration residual / conformal error = **predictor reliability**。

如果真的做 gating，应写：

\[
\beta(x)=f(\text{calibrated prediction reliability})
\]

而不是直接拿 `p95-p50` 当 confidence。

---

# 5. 实验怎么把“信息”和“机制”彻底拆干净

这是我认为你下一阶段最重要的实验设计。

做一个非常干净的 **2×2 information × consumer**：

| | GPU-agnostic consumer | Action-conditioned consumer |
|---|---|---|
| **Predicted future** | 现有 F0 | **新方法** |
| **Oracle future** | 现有/对应 oracle arm | **Oracle action-conditioned ceiling** |

再加 Myopic=no future。

这张表可以回答三个不同的问题：

\[
\text{F0}-\text{Myopic}
\]

= “future information 有没有价值？”

\[
\text{New}-\text{F0}
\]

= “**同样预测信息**，action-conditioned consumer 有没有价值？”

\[
\text{Oracle-AC}-\text{Pred-AC}
\]

= “剩余瓶颈到底在 predictor 还是 scheduler？”

这比再做十个 p90/CVaR 参数非常有论文价值。

而且还有两个特别漂亮的 negative control：

**单 GPU。**  
如果你的升级只贡献 placement，那么在只有一个 GPU 时应该退化到 F0。若单 GPU 仍有大幅提升，就说明机制混进了 ordering。

**所有模型都 resident / load cost=0。**  
这时未来 placement value 应趋近于 0，新方法再次退化为 F0。

这两条是非常漂亮的机制验证。

另外建议记录：

- model load 次数；
- evictions；
- total load ms；
- future-model residency hit；
- wasted residency/prewarm；
- additive interference ms；
- JCT / P95；
- deadline miss；
- decision overhead。

如果 JCT 提升同时 `load_ms↓ / eviction↓ / future hit↑`，因果链就非常完整。

---

# 6. 最终推荐

如果只允许我选一个：

> **选 A，但必须升级成 A\*：Probabilistic Action-Conditioned Future-State Scheduling。**

不是：

> “预测未来哪个模型会来，所以尽量放到有这个模型的 GPU。”

而是：

> **“维护未揭示 Agent workflow 的 future belief，并评估每一个当前 placement action 对未来 GPU execution state 分布的影响。”**

第二个配套贡献，我也**不选 B/C/D/E 原版**。

我会选：

> **calibrated robustness of the belief-state scheduler**

即研究预测错误时 action-conditioned lookahead 如何 graceful degradation；必要时才对 prediction weight 做 calibration / ambiguity-set robustification。

这样论文故事可以变成三层：

**C1 — Prediction：**  
从当前 Agent prefix 对未揭示 H-step future 建模，而不是假定完整 DAG 已知。

**C2 — Consumption：**  
提出 action-conditioned future-state value，把概率未来真正映射成 GPU placement / lifecycle consequences。

**C3 — Robustness：**  
在 prediction error / distribution shift 下校准或限制 lookahead，使其安全退化到 myopic/F0。

这比：

> “我们预测 p95，然后 p95 调度最好”

要强很多。

而且从你现在的已有实验看，**F0 恰好是一个极好的 ablation，而不是需要推倒重做的失败版本**：它已经证明“未来信息仅用于 ordering”有约 1.49s 的价值；现在你要验证的自然下一问就是：

> **如果未来信息不仅决定 who goes first，还决定 where it should go，会不会进一步释放价值？**

这条研究逻辑非常顺。

还有一点需要特别警惕：9 月刚出的 *Latency-Aware Orchestration for Multi-Agent LLM Workflows on Heterogeneous GPUs* 已经是你目前最接近的工作，不再是 Parrot/Pythia。它已经做到 workflow forecast → future model demand → placement/lifecycle/order 联合优化。你后续 proposal 和 related work 必须把它作为第一近邻；仅做原始 A，我认为已经不够。citeturn944297academia32

所以我最终的选题判断是：

**“围绕分位数/分布预测”——不建议。**

**“围绕概率预测的未揭示未来如何转化为 action-conditioned GPU placement”——建议，而且这是目前我认为最有希望把你现有工程和实验积累升成真正方法贡献的一条线。** memcite