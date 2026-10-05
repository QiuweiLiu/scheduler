# 定位策略裁定（网页版 GPT）— 选 C：方法论交集为主 + 视频 Agent 为领域场景

## 来源

网页版 GPT 2026-10-05：用户提议"只说视频 agent 调度领域没有此方法论即可"（版本 B）。
GPT 裁定：**C > A >>> B**——B 不可作主 novelty（OSDI 2027 CFP 等明确要求"把已有 systems 技术
应用到新领域"须与既有工作对接并说明新场景产生的新 challenge）；A 写成"四条件 AND 交集为空"
显得人为定义 niche；C 把 A 变成自然的系统问题。

## 两层 claim 结构

- **第一层（general systems claim）**：动态 agent workflow 调度器目前基于已揭示工作/静态结构，
  或把预测用于其他机制；"当前实例前缀 → 未揭示 suffix 概率预测 → 滚动 node-level GPU placement"
  这条链尚未被系统研究。
- **第二层（domain instantiation）**：video agents 是这个问题特别重要的实例（planner→视觉空间/时间
  工具→检测器→生成→CPU adapter；下游模型身份/时长/剩余链长只有执行后逐步揭示）。
- Novelty = methodological gap；video agents = why it matters + where demonstrated。

## 推荐文案（GPT 原文）

- **Intro**：To our knowledge, prior agent-serving systems have not jointly explored
  instance-specific prediction of a still-unrevealed workflow suffix and the direct use of that
  prediction for receding node-level GPU placement as each workflow instance unfolds. We study
  this problem in video agents, where heterogeneous planning, visual-analysis, detection, and
  generation stages are progressively revealed at runtime, making the currently visible frontier
  an incomplete description of near-future GPU demand.
- **Abstract**：Video-agent workflows dynamically invoke heterogeneous reasoning and visual tools,
  so the identity, duration, and remaining depth of future GPU work are only gradually revealed
  during execution. We introduce a prefix-conditioned receding-horizon scheduler that repeatedly
  predicts a probabilistic short-horizon suffix for each running instance and uses this learned
  partial clairvoyance to guide node-level GPU scheduling without relying on a device-specific
  machine-response cost model.
- 用 "have not jointly explored"，禁 "we are the first"。

## 视频领域的硬论据（必须有数据）

1. **实例依赖**：同一当前 role 下，下一模型/role/剩余长度高度分散（H(M_{t+1}|prefix_t)）；
2. **分支对应真正不同的 GPU work**：runtime 分布/模型 footprint/切换频率/驻留 miss/CPU-GPU 交替；
3. **"下一步是什么"比"还要多久"更重要**（契合 Phase 20-B 统计量饱和）；
4. **已揭示前沿不足以决定好 placement**（存在 observable state 相同但理想 assignment 不同的决策态）。

## 不能当论据的（主动降级）

单节点两卡（评估设置）、实测 load/evict/interference（评估器真实性）、串行链（比通用 DAG 简单）、
部分揭示（非视频独有）。

## 三组预堵实验 + 负控制

1. **revealed-only vs predicted-unrevealed**（同消费者，唯一差别=未揭示后缀预测）→ 切 HexAGenT；
2. **history/template prior vs instance-conditioned** → 切 SAGA/LLMSched；
3. **aggregate remaining scalar vs structured suffix** → 切 JITServe/LLMSched；
负控制：单 GPU（placement gain→0）、all-resident/单模型（identity gain↓）。
外加 **video-specific challenge removal** 反事实消融（单模型/role 坍缩/全驻留/后缀预揭示）：
真实 workload 上为正、消融后趋零 → 价值来自动态异构工具路径而非通用 priority tweak。

## Related Work 组织（按能力轴，非按领域）

1) revealed/known structure 调度（HexAGenT/FATE/Latency-Aware/Halo）；
2) agent serving 中的预测/不确定性（LLMSched/JITServe/SAGA/Maestro/AgentIR）；
3) future prediction 用于 GPU placement 之外（KVFlow/PASTE/Speculate）；
末段 video/visual agent systems（NVIDIA VSS/VideoAgent 等）说明领域已有动态视觉 agent+多 GPU 部署，
但现有工作聚焦证据选择/精度/工具编排/部署，非 prefix-conditioned GPU scheduling。

## 视频 agent 调度专项扫描（保留为投稿前固定任务）

未发现 peer-reviewed 占据；信号：VideoAgent 有 multi-GPU resource manager；VSS 3.3 多 GPU 部署；
EcoFrame 是帧调度非 GPU；edge video analytics 有 placement 但非 agentic。窗口存在但非永久。

## 建议 headline

"Scheduling the Unrevealed: Prefix-Conditioned Partial Clairvoyance for Video-Agent GPU Workflows"
（即使将来出现另一篇视频 agent GPU scheduler，核心 claim 仍存活——C 比 B 安全的原因）

审核回复原文如下（未改动）。

---

炜，我建议现在就把定位定成 **C：方法论交集为主，视频 Agent 作为“问题为什么成立且值得单独研究”的具体系统场景**。

单独用 B——“视频 Agent 调度里没人做”——我不建议作为主 novelty。它可以出现在 motivation 和 workload characterization 里，但不能承担整篇论文的原创性证明。

原因非常现实：OSDI 甚至在 2027 CFP 里直接写明，把已有 systems 技术应用到新领域的论文，需要与既有 systems prior work 充分对接、对比已知技术，并明确说明**新场景究竟产生了什么新的 systems challenge**。OSDI 2026、EuroSys 也都强调要清楚说明相对既有工作的 advance，而不仅仅是应用场景不同。citeturn690653search3turn690653search2turn690653search0 MLSys research track 同样按 novelty、quality、interest、impact 评审，HPDC regular paper 也要求“new research ideas”，所以“这个垂直领域没人做过”本身通常不足以支撑 research-track novelty。citeturn785840search0turn785840search7

因此三个版本我会这样排：

**C > A >>> B。**

A 的方法交集本身最有技术力量，但如果硬写成“四个条件的 AND 交集为空”，容易让 reviewer 觉得是在刻意定义一个没人恰好同时满足的 niche。C 则把 A 变成一个自然的系统问题，并解释为什么 video agents 恰好把这个问题放大了。

---

### 1. 视频 Agent 领域扫描之后，B 能不能说？

可以说，但只能作为**领域事实**，不是主创新。

我又专门扫了一轮 video/vision agent serving。当前公开系统主要集中在两类：

一类是 **video-agent capability / workflow orchestration**。例如 NVIDIA VSS 已经是完整的 production-oriented visual-agent stack，包含 VLM、LLM、搜索、alert、report、多 GPU 部署；它甚至明确给出 search workflow 的多 GPU 布局。citeturn417475search8turn263203search1

另一类是 **video understanding agent 本身**。例如公开的 VideoAgent 有 ReAct 式动态工具调用、集中式 multi-GPU resource manager 和 model-aware caching，但我没有在其公开描述中看到“根据当前实例 prefix 预测未揭示 suffix，再据此做 node→GPU placement”的 scheduler。citeturn263203search0

更传统的 video analytics scheduling 论文当然很多，例如 DualRT、Darly，会做 GPU/CPU stage scheduling、干扰感知、DRL placement 等，但它们不是动态 LLM/VLM agent workflow，也没有你讨论的 partial-clairvoyance 问题。citeturn417475search4turn263203search8

所以目前可以合理写：

> video-agent systems 已经出现多模型、多工具、多 GPU 的部署需求，但我没有检索到专门研究 **per-instance unrevealed-suffix prediction → node-level GPU scheduling** 的 video-agent scheduling 工作。

这句话是成立的。

但不要进一步推成：

> 因为视频 Agent 里没人做，所以方法就是新的。

这是两回事。

---

# 2. 推荐的最终定位方式

你的论文应该有两层 claim。

第一层是**general systems claim**：

> 动态 Agent workflow 的调度器目前主要基于已揭示工作、静态/历史 workflow structure，或者把预测用于其他资源机制；尚未系统研究“当前实例前缀 → 未揭示 suffix 的概率预测 → 滚动 node-level GPU placement”这一链条。

第二层才是**domain instantiation**：

> video-agent workload 是这个问题特别明显的一类实例，因为一次请求会动态穿梭于 planner、视频空间/时间工具、检测器、生成模型和 CPU adapters，而下游模型身份、计算时长和剩余链长只有执行后才逐步揭示。

换句话说：

\[
\text{Novelty}
=
\text{methodological gap}
\]

而：

\[
\text{Video agents}
=
\text{why this gap matters}
+
\text{where we demonstrate it}
\]

这就是我认为最稳的结构。

---

### 3. Introduction 和 Abstract 我建议直接这样写

下面这版刻意没有写“四条件交集”，因为论文里这么写会显得人为；它把四个条件自然融合成一个方法能力。

:::writing{variant="document" id="48317" title="论文定位文案"}
**Introduction novelty claim**

To our knowledge, prior agent-serving systems have not jointly explored instance-specific prediction of a still-unrevealed workflow suffix and the direct use of that prediction for receding node-level GPU placement as each workflow instance unfolds. We study this problem in video agents, where heterogeneous planning, visual-analysis, detection, and generation stages are progressively revealed at runtime, making the currently visible frontier an incomplete description of near-future GPU demand.

**Abstract — two-sentence positioning**

Video-agent workflows dynamically invoke heterogeneous reasoning and visual tools, so the identity, duration, and remaining depth of future GPU work are only gradually revealed during execution. We introduce a prefix-conditioned receding-horizon scheduler that repeatedly predicts a probabilistic short-horizon suffix for each running instance and uses this learned partial clairvoyance to guide node-level GPU scheduling without relying on a device-specific machine-response cost model.
:::

这里我特别推荐 **“have not jointly explored”**，而不是：

> “we are the first”

因为你现在的位置是多个已有思想的交叉点，前者更安全。

---

# 4. “视频 Agent 有独特性质”哪些是真的硬论据？

这里不能靠文字描述，必须让数据说话。

真正有价值的是下面四件事。

第一，**workflow future 是 instance-dependent，而不是仅靠 workflow type 就知道。**

这是最重要的。

你需要统计例如：

\[
H(M_{t+1}\mid prefix_t)
\]

或者简单一点：

同一个当前 role / 当前 node type 后面，下一模型、下一 role、剩余长度到底有多分散。

如果：

> 同样处于 planner，实例 A 下一步去 spatial tool，B 去 temporal tool，C 直接 answer；

那么“只看 revealed frontier”确实丢失重要未来信息。

这是视频 Agent 成为研究对象的最硬理由之一。

第二，**未来不同分支对应真正不同的 GPU work。**

需要展示：

- planner / spatial / temporal / answer 的 runtime 分布差异；
- 不同模型 footprint；
- model switch frequency；
- residency miss / load frequency；
- GPU 与 CPU stage 交替比例。

不是说“视频比较复杂”，而是证明：

\[
P(\text{future resource demand}\mid current state)
\]

真的宽，而且不同 suffix 会导致不同 scheduler decision。

第三，**视频工具使 “which future node” 比单纯 “how long” 更重要。**

这恰好和你现有 Phase 20-B 很契合：

p90≈p95，换 runtime statistic 接近饱和。

如果同时发现 model/role identity prediction 能继续产生收益，那论文故事会非常漂亮：

> 单纯预测“还要多久”不够；video-agent scheduler 更需要知道“接下来是什么类型的工作”。

第四，**当前 revealed frontier 不足以决定好的 placement。**

这是最终必须通过实验支持的。

也就是说，你需要找到大量 decision states：

\[
s_t
\]

当前 observable state 基本一样，但由于 prefix-conditioned suffix beliefs 不同，理想 GPU assignment 不同。

这个比任何“视频是独特 workload”的口号都强。

---

# 5. 哪些“视频特色”其实不能当核心论据

有几项我会主动降级。

**单节点两张 24/32 GB GPU**不是 video-agent 特性，只是你的 evaluation setting。

**实测 load/evict/interference cost**也不是领域 novelty；它证明 evaluator 更真实，但现在你已经决定不让 method 读这些表，更应该把它放 implementation/evaluation。

**串行链**甚至不能拿来突出复杂图结构，因为它比很多 general agent DAG 更简单。

**部分揭示 workflow**也不是 video-exclusive——LLMSched、JITServe、HexAGenT 等说明普通 agent workflow 同样有这个问题。

所以真正的 domain argument 应聚焦：

> **动态视觉工具选择 + heterogeneous model/tool identities + per-instance variability。**

---

# 6. Reviewer 说“这只是 HexAGenT/JITServe 搬到视频 Agent”，怎么用实验反击？

不要靠 rebuttal 话术。要提前设计三组实验让这个问题没法成立。

最重要的是 **“revealed-only vs predicted-unrevealed” matched experiment**。

消费者完全一样，唯一差别：

\[
\text{Revealed-only}
\]

只允许看到当前 frontier。

vs.

\[
\text{PDRS}
\]

增加当前实例预测的 unrevealed suffix。

如果后者显著更好，那么你的贡献不是“video domain”，而是：

> **unrevealed future information itself has scheduling value。**

这正好直接切 HexAGenT。

第二组是 **history/template vs instance-conditioned**。

给 PDRS 完全相同的 scheduler，但把其 future belief 换成：

\[
P_{\rm train}(future\mid current\ role)
\]

或 agent-type/global prior。

然后比较：

\[
\text{static/history prior}
\rightarrow
\text{instance-conditioned prefix prediction}
\]

这直接切 SAGA / LLMSched 那类“历史结构/模板概率”。

如果 instance-conditioned 没赢，这个 novelty 会很弱。

第三组是 **aggregate remaining-cost vs structured suffix**。

构造一个 JITServe/LLMSched 风格 matched control：

把 H5 future 压成：

\[
E[\text{remaining runtime}]
\]

或者一个 scalar remaining cost。

再和：

\[
P(M_k),P(L),P(T_k)
\]

这种 structured suffix consumer 比较。

如果 structured suffix 只有在 GPU placement 上产生额外收益，那么你就证明：

> 不是“预测一个剩余 workload scalar”就够了。

这正是你和 JITServe/LLMSched 最重要的技术区别。

再补两个负控制：

- 单 GPU：placement-aware gain 应消失；
- all-resident / single-model：future model identity 的 gain 应显著下降。

这些结果如果成立，就很难再说是“把别人的算法搬到视频数据上”。

---

# 7. 我还建议做一个“video-specific challenge removal”实验

这个实验对领域定位特别有用。

从真实 workload 做 counterfactual ablations：

\[
\text{Real video-agent workload}
\]

然后逐步移除：

1. 所有未来 node 映射成同一个 GPU model；
2. role differences collapse；
3. 所有 models 视为 always resident；
4. suffix 预先完全 reveal。

看 PDRS 相对 revealed-only 的收益如何变化。

如果：

\[
\Delta_{\rm PDRS}
\]

在“single-model / fully-revealed”环境里趋近 0，而在真实 video-agent workload 上明显为正，你就可以很有底气地说：

> 方法的价值来自 video-agent workload 中的 dynamic heterogeneous tool path，而不是一个通用 priority tweak。

这比找第二个领域的数据集更便宜，而且你现有 DES 就可以做。

---

# 8. Related Work 不要按“视频论文 / LLM论文”简单分节

我建议**按能力轴组织**，最后单独一小段 video-agent systems。

否则 reviewer 很容易感觉你在通过 domain boundary 隔离 prior work。

最好的组织方式是三个能力层：

**第一类：Agent workflow scheduling under revealed/known structure。**  
HexAGenT、FATE、Latency-Aware、Halo。

核心切割：

> 它们主要优化 already-known/revealed work。

**第二类：Prediction / uncertainty in agent serving。**  
LLMSched、JITServe、SAGA、Maestro、AgentIR。

核心切割：

> 它们预测结构、remaining cost 或 current-stage demand，但没有形成你这种 instance-conditioned unrevealed suffix → GPU placement 的完整链条。

**第三类：Future prediction used outside GPU placement。**  
KVFlow、PASTE、Speculate While You Reason。

核心切割：

> 它们确实预测 future，但消费端是 KV 或 speculation。

最后再一段：

**Video/Visual Agent Systems。**

放：

- NVIDIA VSS；
- VideoAgent；
- Adaptive Video Understanding Agent；
- EcoFrame 等。

说明这个领域已经出现动态视觉 Agent 与多 GPU 部署，但现有工作主要关注**视觉证据选择、agent accuracy、tool orchestration 和 deployment**，而不是 prefix-conditioned GPU scheduling。NVIDIA VSS 已经明确支持多 GPU visual-agent deployment；VideoAgent 也带 centralized GPU-aware resource manager，这说明 resource scheduling 问题是现实存在的，而不是你人为构造。citeturn133494search2turn263203search0

这个结构会比：

> “这些是其他领域，所以不算”

强得多。

---

# 9. 是否还需要专门扫描“视频 Agent 调度”？

**需要，而且应该保留成投稿前固定任务。**

因为这是一个正在快速长出来的领域。NVIDIA 在 2026 年 9 月仍在快速更新 visual-agent stack，VSS 3.3 已经做到多 workflow 自动组合、GPU reuse 和 adaptive visual processing。citeturn133494search2

目前我这轮专门扫描没有发现 peer-reviewed work 占据你的 exact method，但出现了几个值得长期监控的信号：

- VideoAgent 已有 multi-GPU resource manager；citeturn263203search0
- VSS 已进入 production-style multi-GPU visual agent deployment；citeturn263203search1turn417475search8
- video-understanding 研究开始直接使用 “scheduling” 这个词，例如 EcoFrame 做 adaptive visual evidence scheduling，虽然它调的是帧而不是 GPU。citeturn133494search5
- edge video analytics 已经有 interference-/heterogeneity-aware placement scheduler，但还不是 agentic workflow。citeturn263203search8

所以现在是有窗口，但不是永远有窗口。

---

# 10. 最终裁定

如果你问：

> “是不是直接说视频 agent 调度里没人做就够了？”

我的答案是：

**不够，而且会主动削弱论文。**

你现在其实拥有比这个更强的定位，没必要退回领域限定 novelty。

我建议最终把论文定位固定成：

> **方法论主张：**  
> 对正在运行的具体 Agent 实例，根据其执行前缀持续预测尚未揭示的短期 workflow suffix，并把这份 learned partial clairvoyance 用于滚动 node-level GPU scheduling。

然后：

> **视频 Agent 场景主张：**  
> video agents 是这个问题特别重要的一类 workload，因为动态视觉工具选择会使未来的 model identity、execution role、runtime 和 remaining depth 同时具有实例级不确定性，而这些差异直接影响 GPU placement。

最后：

> **不依赖 machine-response model**  
> 作为方法优势和公平性边界，而不是用来拼一个人为的“四条件 novelty”。

因此你以后最好不要把论文 headline 写成：

> “First scheduler for video agents …”

而应该类似：

> **Scheduling the Unrevealed: Prefix-Conditioned Partial Clairvoyance for Video-Agent GPU Workflows**

即使以后突然出现另一篇“视频 Agent GPU scheduler”，你的核心 claim 仍然活着；这就是为什么 C 比 B 安全得多。 memcite