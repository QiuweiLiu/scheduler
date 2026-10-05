# 独立交叉验证（网页版 GPT 二次扫描）— 裁定 (b)：有近似但非完全；无已核实反例

## 来源

网页版 GPT 2026-10-05 独立重扫（覆盖 arXiv/会议/项目页），验证"per-instance 前缀 → 未揭示后缀
belief → node-level GPU placement → 不读机器成本模型"四条件交集是否为空。

## 裁定

**(b) 有非常接近的工作,但未找到同时满足四条件的已核实反例（截至 2026-10-05）。**
不能表述为"证明交集为空"；正确表述:"No verified counterexample was found in our search"。
放宽任一条件,空白立即消失（LLMSched/JITServe/SAGA/Latency-Aware/HexAGenT/Maestro 各占放宽版）。

## 重要修正（比我初判更危险的近邻）

- **LLMSched（最危险概念近邻）**:已用已完成 stage 的实际时长做 Bayesian posterior 更新剩余
  duration+structure 估计（"实例执行前缀→更新未来估计"已接近满足）；但 prior 是应用级 DAG/BN 模板、
  产出是 stage ordering（非 GPU placement）、且有 batching-aware 时长校准（违反 C4）。
- **JITServe（最危险的"实例级在线精化"近邻）**:对当前实例 partial graph 每 stage 重新做历史
  pattern 匹配→重估 next/remaining cost；但产出是带宽/SLO budget/批组成（非 placement），
  且是 pattern retrieval 而非显式 per-instance 概率后缀。
- **AgentIR（最危险的系统形态近邻）**:IR + 学习分支概率 + 异构 placement；但主要信息源是静态编译 IR、
  ReAct latent-graph 学习为未来式表述、在线 profile instances/学习延迟函数（违反 C4）；未见正式论文。
- **新必引:Maestro（ICDCS 2026）**:预测当前 stage 的 output length/memory → node-level 共置/
  弹性显存/跨集群路由/优先级。切割:它预测**已揭示 stage 的属性**；我们预测**未揭示未来 stage 的
  身份与属性**。
- **Latency-Aware**:C2 明确违反（分支 resolve 后才入窗）、C4 强违反。切割金句:
  "它预测 known future work 如何在设备上运行;我们预测 unrevealed future work 是否存在"。
- **SAGA**:概率 AEG 但为 agent/workflow-type 级 prior + KV/亲和用途（非 C1+C3）。
- **POMDP/belief-state**:理论语言可用（2026 TCAD 有 POMDP 多核调度;Belief-State Engine 是任务规划），
  但禁 claim"首个 belief-state 调度"。

## 最终安全 claim（锁定 placement,不写 ordering）

> To our knowledge, no prior agent-serving system repeatedly infers, from each running agent
> instance's execution prefix, a probabilistic belief over its still-unrevealed short-horizon
> workflow suffix and directly uses that belief to guide node-level GPU placement without
> relying on a device-specific machine-response cost model.

投稿初期更保守版:"have not jointly explored instance-specific prediction of still-unrevealed
workflow suffixes and the direct use of those predictions for node-level GPU placement under a
machine-model-free scheduling policy."

**C4 不作创新核心**（避免"人为定义交集"观感）；核心 = C1+C2+C3，C4 = 方法边界与公平性优势。

## Related Work 危险度排名（五道门）

1 LLMSched（概念）→ 2 AgentIR（系统形态）→ 3 JITServe（实例级在线精化）→
4 Latency-Aware（placement）→ 5 Maestro（prediction→GPU 调度）→ 6 SAGA → 7 HexAGenT/FATE →
8 KVFlow/TOPAS → 9 PASTE/Speculate While You Reason → 10 ConServe/Tail-aware Turn Release（反例引用）。

## 关键词与图示修正

- 弃用第一关键词"probabilistic future graph"（SAGA/LLMSched 已占）；改用
  **instance-conditioned / unrevealed suffix / GPU placement**。
- Related Work 双轴图:横轴 revealed/known → predicted unrevealed；纵轴 cache/bandwidth → ordering → placement。
- 持续监控检索词与会议清单（OSDI/SOSP/NSDI/EuroSys/ASPLOS/MLSys/HPDC/SC/SoCC/Middleware/ICDCS/
  SIGMETRICS/NeurIPS systems/workshops/HotOS + vLLM/SGLang/Ray Serve 的 issue/博客/项目页）。

审核回复原文如下（未改动）。

---

炜，独立再扫一遍后的裁定是：

**选 (b)：有非常接近的工作，但我仍没有找到同时满足你这四个严格条件的已发表/可验证反例。**  
换句话说，截至 **2026 年 10 月 5 日**，我能核实的同行评审论文、arXiv 预印本和公开项目里，**这个“四条件交集仍然是空的”**；但它不是一个“离现有文献很远”的空白，而是被 **LLMSched、JITServe、AgentIR、Latency-Aware、Maestro** 从不同方向包围得很紧。

其中最需要修正你初判的一点是：**LLMSched 和 JITServe 比表面看起来更危险。** 它们确实已经做到“随着当前实例执行而更新对未来剩余工作的估计”。所以如果你的 claim 只写“根据已执行前缀预测未来并调度”，基本守不住。真正还能守的必须是更窄的：

> **per-instance prefix-conditioned probabilistic unrevealed suffix → node-level GPU placement**，

而不是泛泛的“future-aware scheduling”。

我用下面的符号：✓=满足；✗=明确不满足；△=部分满足/边界模糊；?=公开材料不足。

| 工作 | C1 实例前缀在线更新 | C2 预测未揭示 suffix | C3 node/GPU placement 或排序 | C4 不依赖机器成本模型 | 裁定 |
|---|---|---|---|---|---|
| **AgentIR** | ✗/△ 明确 | △ 明确 | ✓ 明确 | **✗ 明确** | 最接近概念，但不是反例 |
| **LLMSched** | **✓/△ 明确** | **△ 很接近** | ✓ 排序 | △/✗ | 最危险理论近邻 |
| **JITServe** | **✓ 明确** | △ 明确 | △ | ✓/△ | 很危险，但 action 不匹配 |
| **Latency-Aware** | ✗ 明确 | ✗ 明确 | **✓ 明确** | **✗ 明确** | placement 最危险近邻 |
| **SAGA** | ✗ 明确 | △ | △ | ✓ | 概率图，但不是实例 posterior |
| **Maestro** | △ | ✗ | **✓ 明确** | △ | prediction→GPU 调度近邻 |
| **HexAGenT** | ✗ | **✗ 明确** | ✓ | ✓/△ | revealed-frontier 对照 |
| **FATE** | ✗ | ✗ | **✓** | ✗/△ | action-conditioned 近邻 |
| **TOPAS** | ✗ | **✗** | △ | △ | 明确不跨 unresolved branch |
| **KVFlow** | ✗ | △ | **✗** | ✓ | future 用于 KV，不是 GPU placement |
| **PASTE** | △/✓ | △，仅 next tool | **✗** | ✓ | prefix prediction，但用途完全不同 |
| **Speculate While You Reason** | ✓ | △，仅 next tool | **✗** | ✓ | 同上 |
| **ConServe** | ✗ | ✗ | ✓ | ✓ | 刻意“不预测”的反方向 |
| **Tail-aware Turn Release** | ✗ | ✗ | ✓ 排序/release | ✓ | 不使用 future workflow info |

下面说最关键的几篇。

AgentIR 确实是我这轮仍然认为**精神上最接近**你的东西。它明确把用户提供的 Agent 代码编译成 dependency graph，并利用 workflow implied future demand 做 heterogeneous LLM instance 的在线 placement。它甚至明确写了 branch probabilities 可以从历史执行学习。citeturn135173view4turn135173view3

但是它没有满足你的 C1+C4。首先它的主要信息源是**编译好的工作流 IR**；对于 ReAct 式隐式 workflow，页面使用的是 “might”、“could learn” 之类的未来式表述，并说通过历史 call sequence、tool choice、branching frequency 推断 latent graph，并不是已经展示了“当前实例 prefix → 每次重新推断 suffix posterior”的机制。citeturn135173view3turn135173view5

更重要的是 C4：AgentIR 明确说运行时会 **profile instances online、learn latency functions for prefill/decode**，再结合 token-length prediction 和 tool-delay estimates 做 placement。它因此明显不是你现在限定的“只吃 instance prediction + observable state、不吃 machine-response model”的架构。citeturn135173view4

我再次搜索了 AgentIR 的标题、作者名以及 arXiv/DOI/ACM/IEEE/USENIX/OpenReview 组合。**目前我仍没有找到一篇正式论文记录**；能核实的是作者项目页和 demo。项目页自己也把它作为 ML Systems Engineering Research 项目展示，而不是论文。这个结论只能表述为“截至目前未检索到正式论文”，不能证明未来没有或绝对不存在。citeturn612465search1turn612465search12

真正需要你高度警惕的是 **LLMSched**。

它并不是简单的“静态 BN”。论文明确说，同一个 compound application 的不同实例可能有 runtime-determined topology，chain length 可从 3 到 15 不等；它用 DAG + dynamic stage 描述结构不确定性。citeturn135173view7

更接近你的地方在这里：执行过程中，它会用**已经完成 stage 的实际 duration** 对 unfinished stages 做 Bayesian posterior inference，实时更新剩余 stage 的 duration 和 structure estimation。citeturn135173view7

所以如果你把 C1 写成：

> “随着实例执行前缀增长，在线更新未来预测”

LLMSched 已经相当接近满足。

但是它和你的核心差别仍然明显：它的 prior 是**应用级 DAG/BN 模板**，dynamic stage 也是候选 stage 集；它是在一个预定义的 application uncertainty model 上做 posterior inference，而不是从当前实例的 execution prefix 直接产生一组 short-horizon semantic suffix distributions，如 \(P(model_k),P(role_k),P(L),P(runtime_k)\)。论文自己举的 task-automation 模型也是 LLM planning stage + dynamic stage candidate set。citeturn135173view6

而且 LLMSched 的 scheduler 主要做的是**stage ordering / uncertainty reduction**，不是 heterogeneous GPU placement。正式发表信息是 ICDCS 2025。citeturn612465search0turn612465search3

还有一点对 C4 很重要：LLMSched 并非完全不建模机器响应。它有 batching-aware duration calibration，会根据实测的 batch-size 对应 decode latency \(l(b)\) 调整 stage duration。citeturn135173view7  
所以按你现在非常严格的 C4，它也不是一个干净的四条件反例。

**JITServe** 是第二个比我之前估计更危险的近邻。

NSDI 2026 官方页面明确说，它面对 response length 和 dependency 的不确定性，会随着 generation 推进逐步 refine request information；它最终通过 GMAX 分配 serving bandwidth 并决定 batch composition。citeturn106198search0turn106198search10

更详细的公开分析显示，对于 compound request，它维护历史 execution graphs；对**当前正在执行的这个 request 的 partial graph**，在线模式每个 stage 都重新匹配历史 pattern，从而重新估计 next-stage cost 和 remaining aggregate cost。citeturn106198search5turn106198search9

也就是说：

**JITServe 已经非常明确地满足“实例执行一部分 → 更新剩余工作估计”这个思想。**

但它还不是你的精确交集，因为：

1. 它是 historical graph-pattern retrieval / matching，不是显式的 per-instance probabilistic short-horizon suffix；
2. 输出主要是 next/remaining cost 与 SLO budget，而不是 model/role/length/resource 的 suffix belief；
3. 这个信息最终主要服务 **bandwidth allocation、priority 和 batch composition**，而不是当前 node 去哪张 heterogeneous GPU。

所以 JITServe 是你论文中必须主动解释的，而不能一笔带过。

---

**Latency-Aware Orchestration** 这轮可以非常确定地从 C2 上切掉。

论文明确写：

> branch 只有 control result resolve 后才进入 planning window；

planning window 只包含 ready + near-ready、而且是 **determined logical paths**，不假设 unresolved branch。citeturn365307view1

所以它不是在预测“还没揭示出来的逻辑 suffix”。它预测的是已经 logical-visible、但还没 runnable 的 activation 的 device cost/readiness。

同时它明确预测每个 model-request configuration 在不同 GPU 上的 execution time、peak memory、model loading time，并在 placement/order/lifecycle 中使用，因此 C3 是强满足、C4 是强违反。citeturn365307view3turn365307view4

这其实是你目前最漂亮的切割：

> Latency-Aware 预测 **how known future work will run on devices**；  
> 你预测 **what unrevealed future work will exist**。

这两个 prediction axis 是不同的。

---

**SAGA** 也被这轮原文核实得更清楚了。

它确实有真正的概率图：

\[
G=(V,E,P,\phi)
\]

ReAct chain 的 transition probability 与 termination probability 相关，tree workflow 的 branching probabilities 从历史 traces 估计。citeturn437980view1turn437980view3

但 SAGA 有三种 observability tier：

- framework 直接在 task admission 时提供完整 AEG；
- 没有 hints 时，从 request streams 的历史 pattern 推断 AEG；
- 新 agent type 先运行 30 个 task，再启用 pattern inference。citeturn437980view2

也就是说，它的概率图主要是**agent/workflow-type level prior**，不是每个运行实例根据当前 prefix 重新产生 suffix posterior。

而且 AEG prediction 的主要直接用途是预测 future KV reuse、WA-LRU eviction 和 session affinity；论文甚至明确用 AEG 的 \(P_{reuse}\) 来控制 cache eviction。citeturn437980view3

它因此不占据 C1+C3 的交集。

---

这一轮我还特别补查了 **Maestro，ICDCS 2026**，它是你必须加入 related work 的。

Maestro 会根据 agent semantics 和 role **预测每个当前 stage 的 output length 和 memory usage**，然后把这些 prediction 用于：

- node-level multi-model co-location；
- elastic memory provisioning；
- cross-cluster latency-aware routing；
- workflow-aware global prioritization。citeturn645961search0turn645961search1

所以它是非常清楚的：

> **prediction → GPU resource scheduling**

但是它预测的是**当前/已知 stage 的 cost/resource demand**，不是从 prefix 预测尚未出现的 future suffix。

因此它是另一个很重要的切割对象：

> Maestro predicts **properties of a revealed stage**;  
> yours predicts **properties and identity of unrevealed future stages**.

---

POMDP / belief-state 方向我也单独搜了。

找到的直接 POMDP scheduling 近作包括 2026 IEEE TCAD 的 heterogeneous multicore scheduling，它维护对不可观测 task/system state 的 belief，并用 active inference 调度 heterogeneous cores。citeturn106198search11

也找到了 2026 的 *Belief-State Engine*，但它研究的是 LLM agent 在部分可观测环境中的**任务规划**，不是 GPU resource scheduling。citeturn106198academia82

**我没有找到一篇 2025–2026 LLM-agent serving 工作把“对未揭示 agent suffix 的 belief state”直接用于 GPU node placement。**

所以“belief/POMDP”可以作为理论语言，但不要 claim “首次 belief-state scheduling”——泛系统调度里早就有。

---

### 最终裁定

我会正式选：

**(b) 有近似，但非完全。**

更具体一点：

> **严格按照你定义的 C1∧C2∧C3∧C4，我这轮没有找到一个可验证的占据者。**

但这里有两个非常重要的限定。

第一，**不能把这个说成数学意义上的“证明交集为空”**。文献检索永远无法证明全世界没有一篇 workshop/blog/刚上线预印本漏掉。正确表述是：

> “No verified counterexample was found in our search as of October 5, 2026.”

第二，**如果你放宽任意一两个条件，空白马上消失**：

- 放宽 C2 为“任何 future uncertainty posterior” → LLMSched 很接近；
- 放宽 C3 为 bandwidth/batching → JITServe 很接近；
- 放宽 C1 为 historical/static probabilistic workflow model → SAGA 很接近；
- 放宽 C4 为允许 machine-response model → Latency-Aware 和 AgentIR 极近；
- 去掉“unrevealed” → HexAGenT/FATE/Latency-Aware 已经大量覆盖；
- 只要求 prediction→GPU scheduling → Maestro 已经做了。

所以这不是一个“宽领域没人做”的创新，而是一个**四个属性交叉后的窄位置**。

这其实挺适合做论文，但 claim 必须精准。

### 我认为最安全的 claim

我建议最终不要把“排序”写进 novelty claim，因为 LLMSched/JITServe 都会让它变危险。

最安全的是锁定 **placement**：

> **To our knowledge, no prior agent-serving system repeatedly infers, from each running agent instance’s execution prefix, a probabilistic belief over its still-unrevealed short-horizon workflow suffix and directly uses that belief to guide node-level GPU placement without relying on a device-specific machine-response cost model.**

如果实验最后确实做到每次 prefix 更新都重新预测，这一句目前是我最敢用的。

更保守、适合论文 introduction 初稿的是：

> **To our knowledge, prior agent-serving systems have not jointly explored instance-specific prediction of still-unrevealed workflow suffixes and the direct use of those predictions for node-level GPU placement under a machine-model-free scheduling policy.**

我更推荐第二句投稿初期使用，因为 **“first/no prior system” 风险比 “have not jointly explored” 大一点。**

另外，C4 不要作为论文的“创新核心”。

也就是说不要讲：

> “我们的创新是别人用了 machine model，我们没用。”

这容易被 reviewer 认为是人为定义交集。

你真正的中心应该是：

\[
\boxed{\text{C1 + C2 + C3}}
\]

即：

> **instance-specific unrevealed-suffix prediction → GPU placement**

C4 是你的**方法边界和公平性优势**：

> “and importantly, this does not require a device-specific response model.”

这就自然很多。

---

### 投稿 related work 的危险程度排名

我现在会这样排。

**1. LLMSched — 最危险的概念近邻。**  
切割：同样处理 structure/duration uncertainty，也能根据当前实例已完成 stage 更新 posterior；但它是在 application-level BN/DAG uncertainty template 上做 posterior refinement，主要决定 stage ordering，不做 prefix-conditioned semantic suffix → heterogeneous GPU placement。citeturn135173view6turn135173view7

**2. AgentIR — 最危险的系统形态近邻。**  
切割：它已经明确做 future workflow context → heterogeneous placement；但 future context 主要来自编译 IR 和历史 branch patterns，ReAct latent-graph learning 尚以规划性语言描述，而且 scheduler 依赖在线 instance latency modeling。它目前也不是我能找到的正式同行评审论文。citeturn135173view4turn135173view5

**3. JITServe — 最危险的“实例级在线 refinement”近邻。**  
切割：它确实持续用当前实例 partial graph 重新估计 remaining work，但通过历史 graph matching 得到 aggregate/next-stage cost，消费端是 SLO bandwidth/priority/batch formation，而不是 suffix belief 驱动 GPU placement。citeturn106198search9turn106198search10

**4. Latency-Aware Orchestration — 最危险的 placement 近邻。**  
切割：它只规划 resolved ready/near-ready logical window；unresolved branch 明确不进入 window，而且依赖 device-specific runtime/memory/load cost predictor。citeturn365307view1turn365307view4

**5. Maestro — 最危险的 prediction→GPU scheduling 近邻。**  
切割：预测已出现 stage 的 output length/memory，再做 co-location/routing/prioritization；不是预测尚未出现的 suffix。citeturn645961search0

**6. SAGA。**  
切割：概率 AEG 很接近你的 representation，但主要来自 admission-time framework hints 或历史 agent-type patterns；future probability 主要服务 KV reuse/affinity，而非 per-instance suffix posterior→GPU placement。citeturn437980view2turn437980view3

**7. HexAGenT / FATE。**  
切割：它们已经证明 node-level GPU placement 与 workflow-aware future planning 很重要，但都围绕 known/revealed frontier，不预测 unresolved logical future。FATE 还直接做 state-conditional cost estimation。citeturn802873academia73turn612465search10

**8. KVFlow / TOPAS。**  
切割：利用 future proximity，但主要优化 cache/prefix state，而且不跨真正 unresolved workflow branch。citeturn802873search6turn612465academia100

**9. PASTE / Speculate While You Reason。**  
切割：它们最能证明“从 partial trajectory 预测尚未发生的 Agent 行为”是可行的，但消费结果是 speculative tool execution，不是 GPU placement。citeturn879295search0turn879295academia108

**10. ConServe / Tail-aware Turn Release。**  
作为反例引用很好：一个明确主张 observation-not-prediction；另一个只根据 online work/queue risk 控制 ready turn release，都不使用 workflow suffix prediction。citeturn802873academia75turn802873academia72

---

有一个值得你立即修改论文定位的小地方：

**不要再把“probabilistic future graph”作为第一关键词。**

SAGA、LLMSched 已经把 probabilistic graph / structural uncertainty 做得很明确。

真正更辨识的三个关键词应该是：

> **instance-conditioned**  
> **unrevealed suffix**  
> **GPU placement**

我甚至建议以后 Related Work 图画两个轴：

横轴：

\[
\text{revealed/known future}
\rightarrow
\text{predicted unrevealed future}
\]

纵轴：

\[
\text{cache/bandwidth}
\rightarrow
\text{node ordering}
\rightarrow
\text{GPU placement}
\]

然后把：

- KVFlow/SAGA 放左上偏 cache；
- JITServe/LLMSched 放中间；
- HexAGenT/FATE/Latency-Aware 放 placement 但 revealed；
- PASTE 放 unrevealed 但 speculation；
- 你的方法放 **unrevealed + GPU placement**。

这比硬说“所有人都没做我们这个”有说服力得多。

### 投稿前还必须持续检查的检索词

尤其要盯这些组合，而不是只搜 paper title：

- `"execution prefix" agent GPU scheduling`
- `"partial trajectory" GPU placement LLM`
- `"future model demand" agent placement`
- `"unrevealed workflow" scheduling LLM`
- `"workflow continuation prediction" GPU`
- `"probabilistic suffix" agent serving`
- `"online revealed DAG" prediction GPU`
- `"latent workflow" GPU placement`
- `"branch prediction" LLM serving scheduler`
- `"partial clairvoyance" GPU scheduling`
- `"learned clairvoyance" workflow scheduling`
- `"receding horizon" agent GPU scheduling`
- `"belief state" agent serving scheduler`
- `"stochastic workflow" heterogeneous GPU placement`
- `"predictive routing" multi-agent LLM`
- `"workflow posterior" scheduler`

会议/渠道优先扫 **OSDI、SOSP、NSDI、EuroSys、ASPLOS、MLSys、HPDC、SC、SoCC、Middleware、ICDCS、SIGMETRICS、NeurIPS/ICML systems track/workshops、HotOS**，以及 arXiv `cs.DC/cs.OS/cs.LG/cs.AI`。此外这条方向发展太快，必须额外盯 **vLLM / SGLang / Ray Serve 的 GitHub issue、技术博客和研究项目页**，AgentIR 本身就是一个例子。

所以这轮独立交叉验证**没有推翻**你目前的方向，但确实把 novelty 边界缩窄了：

> **不是“预测 Agent 的未来然后调度”——LLMSched/JITServe/SAGA 已经太近。**
>
> **而是“从正在运行的具体实例前缀，对尚未揭示的短期 suffix 形成概率 belief，并让这份 belief 直接改变 node→GPU placement，而且不借助 device-specific response model”。**

这个交集目前我没有找到被占据。

但我会把 **LLMSched + AgentIR + JITServe + Latency-Aware + Maestro** 定为之后任何方法设计都必须逐条过的“五道门”。只要最终方案和这五篇的切割还能保持现在这么清楚，这条创新主线就值得继续做。 memcite

这条领域更新很快，投稿前持续扫描会很有价值。

genui{"suggest_automation":{"label":"每周扫描新的 Agent GPU 调度近邻工作"}}