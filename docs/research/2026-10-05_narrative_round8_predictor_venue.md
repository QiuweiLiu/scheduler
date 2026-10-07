# 第 8 轮:预测器定位 + 创新点 + 图 + 发表评估(GPT 回复全文)

> 来源:网页版 GPT(GPT-5.6 Sol + High;generation 11 会话 `6ac1f777-6728-83ec-9281-18a3cb787f23`),
> 2026-10-05 第 8 轮(含 venue CFP 联网核查)。原始回复全文照录。

炜，这轮我会给一个更“收口”的版本：**预测器不要再当第二主角，核心论文也不要围绕“预测器很复杂”来讲。** 现在最强的故事是：一个很克制的 prefix-conditioned lookahead，配上真正能作用于未来状态的 residency control；预测器的大量额外输出则变成“我们系统探索过、但发现没有必要消费”的证据。

## Q1 预测器怎么定位

汇报里预测器讲到“三件事”就够了，不需要展开网络结构细节。

第一，输入是什么：**当前实例已执行前缀**。第二，预测什么：**未来最多 5 步的 runtime/load/model/remaining-length**。第三，如何训练和隔离：**train-only、冻结产物、SHA 固定，confirm300 不参与训练**。

主报告里甚至可以只放一个小框：

\[
\text{prefix}
\rightarrow
\{T_{1:H},\;P(M_{1:H}),\;P(L)\}
\]

然后立刻进入 scheduler。

架构层数、hidden size、loss、训练曲线、每个 head 的详细精度都放 appendix。除非老师专门问“预测器准不准”，否则不要在 15–20 分钟里花 3 分钟讲模型结构。你现在的研究问题已经不是“我们做了一个多头 predictor”，而是“有限未来信息怎样进入调度”。

预测指标也不要铺满。主文最多保留和调度真正相关的 2–3 个摘要指标；如果目前没有一套非常有解释力的 predictor metric，就干脆把 accuracy table 放附录。因为你已经有更强的 downstream oracle 证据：**即使未来身份/信息变真值，原动作空间也没有明显 headroom。** 这比单纯 predictor F1/AUROC 更能回答系统问题。

“很多输出没用上”也完全不尴尬。建议直接定义成：

> **The predictor exposes a superset interface; the final scheduler consumes only the prediction dimensions shown to be actionable.**

中文就是：

> “最初预测器提供了一个较完整的未来状态接口，但后续消费实验发现，不是所有预测维度都会转化为调度价值，因此最终方法只保留实际有消费者的输出。”

甚至可以把它讲成一个正结果：

- runtime p95 → ordering；
- model probability / length → residency；
- CVaR、16 桶、retry、nested、role 等 → **当前系统没有证据证明需要消费**。

这不是“模型白做了”，而是**设计空间裁剪**。

我现在明确不建议重训瘦 predictor。重训会带来三个问题：它改变共享 representation 和各 head 的误差，导致你原来所有冻结 downstream 实验不能再严格对应；需要重新调参、重新冻结；即使性能变好，也会把论文重新拖回 predictor engineering。除非你最后发现 predictor latency/显存本身已经成为真实部署瓶颈，否则收益非常有限。

现在只做“表述剪枝”和“接口剪枝”：论文图只展示被方法消费的 heads，artifact 里保留完整 frozen outputs；如果以后想展示部署轻量化，可以**不重训**，只序列化/读取所需字段。真正删 head 重训放 future optimization。

---

# Q2 创新点怎么说

我建议一句话创新声明不要写成“提出了复杂概率调度器”。

中文：

> **我们研究部分揭示的视频 Agent 工作流中如何利用实例执行前缀获得有限前瞻，并发现只有当调度器能够控制模型驻留状态时，这类未来信息才能进一步转化为系统收益；据此设计了有限前瞻排序与低破坏性驻留控制相结合的 GPU 调度方法。**

英文：

> **We study how learned partial clairvoyance from each running agent instance can guide GPU scheduling, and combine a simple prefix-conditioned finite lookahead with low-disruption residency control to make near-future workload information actionable.**

这两句都没有把 mean 的 1.21% 错归因给完整概率 belief。

三个贡献我建议有明显主次：

**第一贡献，方法/系统贡献。**  
Prefix-conditioned finite lookahead + runtime residency control：排序解决当前选择，驱逐/预取处理跨步骤模型状态，而且 scheduler 不读取 device-specific response table。

**第二贡献，实验归因贡献。**  
系统性分离“预测信息丰富度”和“动作域”：在原动作空间，distributional PDRS 甚至 oracle 都没有额外收益；加入 residency 后才出现稳定 mean/p95 提升。这个 negative-to-positive 证据非常有价值。

**第三贡献，机制洞见。**  
发现 mean 与 tail 的来源不同：residency mechanism 解释主要 mean gain，而实例特异 belief 的独立收益目前只在 tail；同时 SRPT-like ordering/preemption 与 starvation 共同解释 tail degradation。

如果老师问“方法怎么这么简单”，不要防御。应该直接说：

> **简单是实验筛出来的，不是最开始就拍脑袋决定的。**

你已经实测过：

- richer distribution；
- survival weighting；
- oracle；
- risk variants；
- explicit preemption。

这些都没形成更强结果。最后留下 F0 + residency 反而意味着：

> **the final design is the minimum mechanism supported by the evidence.**

在 systems 论文里，这其实是优点。EuroSys 明确要求 solution 要 compelling、evaluation 要清楚展示 benefits 和 limitations；OSDI 也强调 significant problem、compelling solution、practical benefit 和超越既有工作的明确 advances。citeturn137211search2turn137211search3

五道门的切割只需要最短版本，不要再讲一页 related work：

- **LLMSched**：处理 compound-LLM 的结构/时长不确定性，用 DAG+BN 做 JCT-oriented scheduling；你这里关注 **per-instance unrevealed suffix → node/residency actions**。citeturn465748academia60turn465748search0
- **AgentIR**：从 agent code 编译 workflow IR，并做 heterogeneous online placement；与你最接近，但它依赖显式/历史 workflow structure，并做 online instance latency profiling；你这里从运行实例 prefix 预测未揭示 suffix，而且 policy 不读 machine-response model。citeturn465748search14
- **JITServe**：随执行逐步 refine imprecise request information，但消费端主要是 SLO bandwidth 和 batch composition，不是 future-model belief 驱动 residency/placement。citeturn651678search0turn651678search1
- **Latency-Aware**：已经 joint placement/order/model-lifecycle，但 predictor 明确估 device-specific latency、memory 和 loading cost；你的边界是 machine-model-free、instance-prefix-conditioned short-horizon future。citeturn465748academia61
- **Maestro**：预测当前 stage 的 output length/memory，驱动 co-location/routing/caching；你预测的是**尚未揭示 downstream stages 的 identity/work**。citeturn465748search3turn465748search4

这五句就够。

---

# Q3 最小图集

我建议正文/汇报 **6 张核心图**，最多再加 2 张。

| 图 | 内容 | 放哪 | 一句话目的 |
|---|---|---|---|
| **Fig.1 Problem** | 一个 video-agent prefix，后面节点灰色未知；旁边两张 GPU 有不同 resident models | 第1–2页 | **调度决定发生在未来揭示之前** |
| **Fig.2 System** | Prefix → frozen predictor → finite-lookahead ordering + residency controller → GPU，machine profile 只画在 evaluator 下层 | 方法页 | **方法只吃预测+observable state，不读机器响应表** |
| **Fig.3 Main-table Mean** | F0=0 的 forest plot，五 baseline Δmean+CI | 结果1 | **简单前瞻已经显著改善 mean** |
| **Fig.4 Design-space pruning** | F0 / PDRS / oracle / preempt 的 mean+p95 小 heatmap | 结果2 | **更复杂预测与破坏性动作没有进一步价值** |
| **Fig.5 Residency result + mechanism** | 左：mean −1.21%、p95 −1.95% CI；右：evict−18%、cold-load−22%、reload−23% | 核心结果 | **驻留动作产生可解释的正收益** |
| **Fig.6 Attribution** | point / distribution / shuffle / oracle，分别画 mean、p95 | 机制归因 | **mean主要来自动作，instance belief额外价值目前在 tail** |

两个可选图：

**Fig.7 Tail story**：F0 / Parrot / preemption / residency 的 p95 对比，说明 starvation/fairness。

**Fig.8 Related-work positioning**：横轴“revealed→unrevealed future”，纵轴“queue/batch→GPU residency/placement”。把五道门放进去。这个很适合论文，不一定要放两天后的汇报。

如果只能做五张，删 Fig.4，把 negative-result 作为 Fig.6 的一个 inset。

---

# Q4 发表量级

先给最重要的判断：

> **以“现在这一刻”的证据，我认为还没有达到 OSDI/EuroSys 级别的稳态证据强度；但已经明显不是一个只能发小 workshop 的项目。补完几个关键实验后，MLSys / HPDC / ICDCS / Middleware 这一档是现实目标。**

而且 venue fit 其实很好。MLSys 2027 CFP 明确把 autonomous/agentic AI、multimodal AI systems、LLM inference，以及 ML/job scheduling 列为 topics。citeturn137211search1

我会这样分档：

| 档位 | Venue | 当前判断 | 真正缺什么 |
|---|---|---|---|
| **冲刺** | **MLSys research / EuroSys / OSDI** | MLSys 最贴；EuroSys/OSDI 明显更难 | same-interface 2×2、Stress-v2、至少一组真实 GPU end-to-end validation、第二类 workload/trace 或更强 generalization；最好把 tail/fairness 机制闭环 |
| **稳妥** | **HPDC / ICDCS / ACM Middleware / SIGMETRICS（偏 measurement framing）** | 我认为这是当前最合理主目标区间 | same-interface 2×2 基本必须；Stress-v2 强烈建议；真实机器 sanity check 会明显增色，但未必像 OSDI 那样接近硬要求 |
| **保底** | **CCGrid / IEEE Cluster 一类系统/HPC venue** | 题目和 scheduling/agentic-AI fit 很直接 | 把公平对照和 reproducibility 做扎实即可；最好补一个 stress/generalization |

CCGrid 2027 甚至专门设了 **Systems for LLM Applications and Agentic AI** track，并明确列 scheduling、placement、serving、resource management；同时 AI/ML systems track 也覆盖 heterogeneous inference 和 model placement，所以作为保底/稳妥下沿非常匹配。citeturn338848search0turn338848search2

Middleware 也很契合：官方 scope 包括 AI/ML middleware、resource management 和 scheduling，而且它还接受 Experimentation/Deployment 类型论文，更看重完整系统、广泛实验和 lessons learned。citeturn810981view2

SIGMETRICS 如果走“性能规律/信息—动作解耦/尾部 tradeoff”路线也能匹配，它明确覆盖 workload characterization、resource allocation/scheduling、performance modeling；但如果你不做更系统的 characterization 或分析模型，我会把它放在 Middleware/HPDC 后面。citeturn467278search7

### OSDI/EuroSys 还差什么

OSDI 2027 明确要求 significant problem、compelling solution、practical benefit，并要求新 domain 的 systems work 要充分对接已有 systems prior work 和 baseline，说明新场景产生的新 systems challenge。citeturn137211search3

你现在最容易被 OSDI reviewer 打的不是 novelty，而是：

1. **最终正方法主要在 DES 上。**
2. gain 只有约 1–2%。
3. 新 action package 与五条 baselines 不是 same-interface。
4. workload 只有当前这一个 video-agent substrate/chain family。
5. shuffle 说明 mean gain 主要不是 predictor-specific。

所以如果真冲 OSDI，我会把下面四件事视为近乎必需：

**same-interface 2×2**：no-future/point-future × no-residency/residency，直接把 information 和 action mechanism 拆开。

**Stress-v2**：系统性扫 residency pressure / arrival contention / capacity topology，证明 1.2% 不是偶然小效应，而是在“什么时候有用”上有规律。

**真实 GPU validation**：不用重跑整个 confirm300，哪怕挑几十 episode，证明 DES 对 policy ranking 和关键 mechanism counters 的方向是对的。

**第二 workload family / generalization**：不一定要另一个大数据集，但至少要一个不同 agent workflow regime，让结论不是只对当前 chain traces 成立。

EuroSys 同样强调 solution 的 significance、rigorous evaluation、benefits 和 limitations。citeturn137211search2

### MLSys 是最值得冲的吗？

**从 topic fit 看，是。**

MLSys 2027 现在明确接受 agentic AI、multimodal AI systems、LLM inference、ML for scheduling，而且重视 artifact/reproducibility。citeturn137211search1

但有一个现实问题：**MLSys 2027 截稿是 2026 年 10 月 30 日**，今天是 10 月 7 日。citeturn137211search1

只有三周多。

以你现在还缺：

- same-interface 2×2；
- Stress-v2；
- 真机验证；

我不建议为了赶这个 deadline 把论文仓促锁死。除非这些实验一周内能全部出结果，否则更合理的是按 MLSys 质量线完善，再投后续 venue。

OSDI 2027 full paper deadline是 **12 月 8 日**，时间更充裕，但门槛也高得多。citeturn137211search3turn137211search6

NSDI 我反而建议直接排除：NSDI 2027 CFP 明确把 **GPU resource scheduling** 列为 out-of-scope。citeturn137211search70

---

## 如果改成 systems / measurement 论文，标题和 contribution 要不要变

**要。**

如果走纯“新算法”路线，标题类似：

> **Scheduling the Unrevealed: Prefix-Conditioned Lookahead and Residency Control for Video-Agent Workflows**

但以你现在的结果，我更推荐 systems/measurement hybrid：

> **When Does Lookahead Help? Information and Actionability in Video-Agent GPU Scheduling**

或者：

> **From Prediction to Action: Scheduling Partially Revealed Video-Agent Workflows**

这种标题更适合你现在真正最强的资产：

- 完整 substrate；
- 五条 baseline；
- frozen/reproducible protocol；
- 很多负结果；
- information/action decomposition；
- mean/tail 机制差异。

Contribution 也要从：

> “we propose algorithm X”

改成：

> “we formulate / build / characterize / design”。

我甚至会让贡献顺序变成：

1. **Problem + platform**：建立 partially revealed video-agent GPU scheduling 的可复现实验 substrate 与严格信息接口。
2. **Design**：提出 simple finite-lookahead + residency-control scheduler。
3. **Finding**：通过 point/distribution/oracle/shuffle/preemption 控制，揭示 future information 的收益受 actionability 限制，并展示 mean/tail 的不同因果来源。

这会比强行把 PDRS 包装成“新算法”更符合你当前证据。

---

## 最后给你一个非常具体的优先级

如果两天汇报之后继续做论文，我只建议按这个顺序：

**P0：same-interface 2×2。**  
这是现在最大的 causal hole。

**P1：Stress-v2。**  
回答“1.21% 为什么值得研究，以及在什么 regime 变大”。

**P2：真实 GPU 小规模 validation。**  
不用重开大规模 profiling；目标只是验证 simulator 的 policy ordering 与机制方向。

**P3：aging/fairness control。**  
把你现在很漂亮的 starvation/tail 解释坐实。

**P4：第二 workload/regime。**  
冲 MLSys/OSDI 才特别重要。

预测器瘦身、joint distribution、更多 heads、RL、重新训练 predictor——**现在全部排后面。**

因为当前最有价值的问题已经不是“能不能预测得更多”，而是：

\[
\boxed{\text{哪些未来信息，在什么动作空间和系统压力下，真正改变调度结果？}}
\]

这才是你这个项目现在最有机会变成一篇成熟 systems paper 的核心。