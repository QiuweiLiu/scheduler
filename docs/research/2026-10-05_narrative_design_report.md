> **更正注(2026-10-05,后加)**:文中机制数字 "eviction −18%" 系口径混用——−17~18% 是"仅驱逐"臂
> (pdrs_evict)的口径;**主线 `pdrs_resident` 的精确值为:驱逐 −10.3%、按需冷加载 −21.5%、驱逐后重载 −22.3%**
> (预取 1432/600/174,命中 42%)。详见 `experiments/EXP-20261005_residency_comparison_v1/RESULT.md` 机制表。

# 第 7 轮:汇报/论文叙事设计(GPT 回复全文)

> 来源:网页版 GPT(GPT-5.6 Sol + High;generation 11 会话 `6ac1f777-6728-83ec-9281-18a3cb787f23`),
> 2026-10-05 第 7 轮。原始回复全文照录。

炜，我建议这次汇报不要讲成“我们终于找到一个更好的调度器”，而讲成一个更完整、也更可信的研究发现：

> **在部分揭示的视频 Agent 工作流中，有限前瞻的瓶颈不是预测是否更精细，而是预测信息有没有可作用的系统动作。**  
> 在只允许排序/普通放置时，更丰富的分布信息、甚至 oracle 都不能超过 F0；开放模型驻留/容量管理后，系统均值和尾部才显著改善。进一步看，动作机制本身解释了大部分均值收益，而实例特异性的未来信念目前只在尾部产生可测价值。

这个故事比“PDRS 赢了几个 baseline”更有研究味道，因为它包含一个被实验推翻的原假设，以及由此得到的新机制发现。

---

# 一、推荐的完整叙事弧

我会按“成功 → 瓶颈 → 反证 → 新假设 → 机制验证 → 边界”来讲。

### 第一幕：有限前瞻确实有价值，但 F0 已经很强

起点不是 PDRS，而是最简单的 F0：

\[
\text{current cost}+\text{H5 p95 future cost}
\]

它只把未来信息压成一个标量，甚至不消费大量概率字段。

但在冻结主表的**均值 JCT**上：

- F0 显著优于 Parrot、QLM、LLMSched、Hermes、Torpor；
- 也显著优于 Myopic/FCFS。

因此第一个事实是：

> **有限前瞻不是没用，简单前瞻已经足以改善平均完成时间。**

但是这里立即主动承认：

> **F0 不是全指标统治者。**

Parrot 的 p95 比 F0 好 3.15s，而且显著。

这提前埋下后面的“均值—尾部冲突”。

---

### 第二幕：自然问题——是不是因为 F0 把丰富预测压得太狠？

你的 predictor 明明还有：

- 链长/termination；
- model probabilities；
- runtime distributions；
- role 等。

于是提出 PDRS：

> 不再固定 H5，不再只吃 scalar p95，而是根据实例 prefix 消费 survival-weighted / distributional future。

这是一个非常自然的研究推进。

---

### 第三幕：结果反而把这个假设否了

这是汇报里最应该强调的一页，而不是略过。

PDRS v2：

- mean：全部不赢；
- p95：排序型 PDRS 甚至显著恶化；
- deadline：不赢；
- makespan：不赢；
- **连 oracle 都基本打平。**

因此不能再解释为：

> “预测器还不够准。”

因为：

\[
Oracle \approx F0
\]

也不能解释为：

> “我们只是优化错了 mean。”

因为你把 p95、deadline、makespan 都测了。

于是得到一个很强的诊断：

\[
\boxed{\text{information quality is no longer the main bottleneck}}
\]

在当前动作空间里，

\[
\boxed{\text{future information is largely saturated}}
\]

这是故事第一次真正转折。

---

### 第四幕：为什么 oracle 都没有用？

因为原来的 scheduler 真正能做的事情主要还是：

\[
\text{which ready node first?}
\]

以及一个很弱的：

\[
\text{which GPU now?}
\]

即便知道下一步是什么：

> scheduler 也无法提前保留它、提前加载它、避免把它驱逐掉。

于是你提出的新假设不是：

> 再提高预测精度。

而是：

> **预测必须有 actionability。**

即：

\[
\text{Future information}
\xrightarrow[]{?}
\text{Action that can change future state}
\]

所以开放：

- 驻留管理；
- 策略驱逐；
- 保守预取。

这个转折非常自然。

---

### 第五幕：动作域扩充第一次产生稳定正收益

这是核心正结果：

相对 F0：

\[
\Delta mean=-0.84s=-1.21\%
\]

\[
\Delta p95=-2.55s=-1.95\%
\]

且都显著。

与此同时：

- eviction −18%；
- demand cold load −22%；
- evict→reload −23%；
- prefetch 每集 4–8 次，命中约 40%。

这形成了非常完整的机制链：

\[
\text{Residency control}
\rightarrow
\text{fewer destructive state transitions}
\rightarrow
\text{fewer cold loads}
\rightarrow
\text{lower completion time}
\]

所以：

> **动作域确实是一个此前缺失的瓶颈。**

---

### 第六幕：但是不要把这个结果错误归因给“概率预测”

这里反而是这项工作最成熟的地方。

因为：

\[
F0point \approx PDRS
\]

mean/p95 口径也打平，

\[
Oracle \approx best
\]

而且 shuffle 后：

\[
mean\仍然显著改善
\]

所以不能说：

> “distributional belief 让 mean 好了 1.21%。”

这不成立。

目前证据更准确地说明：

> **驻留动作本身解释了主要的 mean 收益。**

但是尾部不同：

真实 belief 相比 shuffle：

\[
p95=-1.615s^*
\]

因此：

> **实例特异性未来信念的独立价值，目前只在尾部被检测出来。**

这个结果我建议你主动讲。

因为它把故事从：

> “我设计了一个很牛的概率 scheduler”

变成：

> “我们分离出了 action mechanism 与 predictive information 的真实贡献。”

学术可信度高很多。

---

### 第七幕：尾部结果又和整个系统现象形成闭环

这里有一个特别漂亮的三角证据。

F0 近似 cost/SRPT 风格：

> mean 好，但容易让长任务饥饿。

Parrot App-FIFO：

> mean 差，但 p95 好。

显式 SRPT 风格抢占：

> p95 +1.86～2.56s，显著恶化；
> victim waiting ≈90s/episode。

而 residency actions：

> 没有打乱执行进度，却同时改善 mean 和 p95。

三处独立实验都指向：

\[
\boxed{\text{tail latency is dominated more by starvation/fairness than by prediction error}}
\]

这可以成为你汇报里一个非常好的“额外系统洞见”。

---

# 二、我建议最终定义的三个贡献

不要硬凑成“三个算法模块”。

### Contribution 1 — Partial-clairvoyance scheduling benchmark / evidence

> 我们在部分揭示的视频 Agent workflow 上系统隔离了“未来信息量”和“scheduler 可执行动作”两个维度，并构建了冻结、可复现的比较协议。

证据：

- Myopic → F0；
- F0 → PDRS；
- PDRS → Oracle；
- original actions → residency actions。

它给后面的因果分析打基础。

---

### Contribution 2 — Actionability bottleneck

> **更丰富的 future distribution 在传统 ordering/placement action space 中并不会自动转化成更好的调度。**

证据特别强：

- PDRS 全四指标无胜；
- survival-weighted ordering p95 显著更差；
- oracle 仍无显著提升。

所以：

> **提高预测精度不是充分条件。**

这是你目前最有研究价值的负结果。

---

### Contribution 3 — Residency actions unlock system gains, while belief value is tail-specific

> 开放驻留/容量控制后，均值和尾部均显著改善，同时显著减少 eviction 和 cold loading；但 matched/shuffle/oracle 对照显示，均值收益主要来自动作机制本身，实例特异性 belief 的独立贡献目前只在尾部可见。

这一句非常重要，因为它没有过度 claim。

---

# 三、可以主张什么 / 不能主张什么

| 可以主张 | 推荐措辞 |
|---|---|
| F0 在冻结主表的 **mean completion time** 上显著优于五条 adapted baselines | “F0 achieves significantly lower mean completion time than all five evaluated adapted baselines.” |
| richer predictive distributions 在原动作空间没有增加价值 | “Richer distributional lookahead provides no measurable benefit under the original action space.” |
| oracle 也没有改善，说明 predictor accuracy 不是主要瓶颈 | “The flat oracle ceiling indicates that forecast accuracy alone is not the dominant bottleneck in this regime.” |
| residency action package 显著改善 F0 的 mean/p95 | “Adding residency-control actions reduces mean and p95 completion time by 1.21% and 1.95% relative to F0.” |
| 机制改善来自 eviction/load reduction | “The improvement coincides with fewer evictions, cold loads, and eviction–reload cycles.” |
| real belief 对 tail 有独立价值 | “Instance-specific belief yields an additional significant p95 benefit over shuffled belief.” |
| SRPT-like behavior 与 starvation/p95 有关 | “Across three independent interventions, aggressive short-job prioritization consistently trades mean latency for tail starvation.” |

不能主张的更重要：

| 不能主张 | 为什么 |
|---|---|
| “PDRS 显著优于 F0” | 不成立 |
| “概率分布带来 1.21% mean 提升” | shuffle 仍赢；F0point≈PDRS |
| “Oracle 证明模型已经完美” | 只能证明**当前动作/消费者下**更准 future 没有剩余价值 |
| “我们的系统所有指标都优于 Parrot” | Parrot 原动作空间 p95 显著优于 F0；新系统与它 p95 只是打平 |
| “我们的系统全面优于 Torpor” | Torpor makespan 显著更好 |
| “新方法公平击败全部 baseline” | 新方法有 residency actions，而 frozen baselines 没有同接口动作 |
| “residency actions 的收益由 per-instance prediction 导致” | mean 上被 shuffle 否定 |
| “抢占没有价值” | 只能说**当前 SRPT-style preemption policy / workload 下**净有害 |
| “未来信息只在 residency actions 下有用” | 目前只是实验性假设，且 mean 上还不能这么说 |
| “视频 Agent 调度里首次……” | 禁止 first；继续使用 “to our knowledge / have not jointly explored” |

尤其是跨运行系统对照，我建议始终标一句：

> **The end-to-end comparison uses the frozen baselines under their original action interfaces; it should not be interpreted as a same-interface attribution of the new residency mechanism.**

---

# 四、15–20 分钟汇报结构

我建议控制在 **9 张主 slide，约 17–18 分钟**。

| 时间 | 内容 | 核心问题 |
|---:|---|---|
| 0:00–1:30 | 1. 问题与一句话发现 | 为什么预测未来还不够？ |
| 1:30–3:00 | 2. 系统与实验协议 | partial reveal、prefix predictor、frozen DES |
| 3:00–5:30 | 3. F0 主表 | 简单 H5 为什么已经很强？ |
| 5:30–8:30 | 4. PDRS 失败实验 | richer distribution / oracle 为什么都没用？ |
| 8:30–10:00 | 5. 假设转变 | information bottleneck → actionability bottleneck |
| 10:00–13:00 | 6. Residency actions | mean/p95 正收益 + mechanism counters |
| 13:00–14:30 | 7. 信念因果拆解 | F0point≈PDRS、shuffle、oracle |
| 14:30–16:00 | 8. 尾部统一解释 | F0 / Parrot / preemption / residency |
| 16:00–17:30 | 9. 当前结论 + 下一步 | same-interface 2×2 / Stress-v2 |

如果老师只给 15 分钟，压缩第 2 页和 literature，把第 8 页并进第 7 页。

---

# 五、最值得做的图表

### 图 1：整篇故事的总图

不要首先画复杂 architecture。

画：

```text
Prefix prediction
      │
      ▼
 Future information
      │
      ├──── Original actions
      │     ordering / ordinary placement
      │             │
      │             └── richer belief ≈ no gain
      │
      └──── Expanded actions
            residency / eviction / prefetch
                      │
                      └── significant system gain
```

最下方一句：

> Prediction becomes useful only through actionable control channels — but its independent value is metric-dependent.

注意最后半句避免说所有收益都是 belief。

---

### 图 2：主表只画 mean

不要把主表四指标全塞一页。

横轴：

\[
\Delta mean\ vs\ F0
\]

用 forest plot：

- Hermes +350
- LLMSched +939
- Parrot +2048
- QLM +3379
- Torpor +3522

所有 CI。

角落一个小框：

> p95: Parrot beats F0 by 3.15s; others mostly tie/worse.

这样非常诚实。

---

### 图 3：PDRS “反结果”热图

行：

- place
- O
- P
- oracle
- myopic

列：

- mean
- p95
- deadline
- makespan

格子显示 ΔF0。

最应该圈出来：

\[
PDRS_O:\;p95+2260^*
\]

以及：

\[
Oracle:\;all\ n.s.
\]

标题：

> **More predictive information does not improve the original action space.**

这是汇报转折页。

---

### 图 4：Information × Actionability 2×2

我认为这是整个汇报最重要的一张 conceptual figure。

| | 弱动作域：ordering/placement | 强动作域：residency control |
|---|---|---|
| scalar / point future | **F0：强 mean baseline** | **动作包：mean/p95 均提升** |
| distributional belief | **PDRS：无增益/伤尾部** | **mean ≈ point；tail > shuffled** |

这张表把七轮讨论压成一张图。

标题：

> **Information is not equivalent to actionability.**

---

### 图 5：Residency positive result

左边两根 forest：

- mean −0.84s*
- p95 −2.55s*

右边 mechanism bars：

- eviction −18%
- demand cold load −22%
- eviction→reload −23%
- prefetch hit ~40%

这页负责证明不是随机 fluctuation，而是有机制链。

---

### 图 6：Prediction causality decomposition

建议只放 4 个 arm：

- F0
- F0point+actions
- PDRS+actions
- Shuffle+actions
- Oracle+actions

可以做两个 panel：

mean：

> point ≈ distribution ≈ oracle，shuffle 仍改善。

p95：

> real belief 显著好于 shuffle。

最下方一句：

\[
\boxed{\text{Action mechanism drives mean; instance-specific belief appears in the tail.}}
\]

这是最成熟的结论。

---

### 图 7：尾部机制统一图

三条证据并排：

- F0 cost/SRPT-like → mean好 / tail饥饿；
- Parrot FIFO → mean差 / tail好；
- explicit SRPT-preempt → tail显著更差；
- residency → 不制造 victim starvation，同时 p95改善。

可以画一个简单二维：

横轴 fairness ↑  
纵轴 work-conservation / mean efficiency ↑

展示策略位置。

---

### 图 8：系统级结果

两 panel forest：

- mean vs pdrs_resident；
- p95 vs pdrs_resident。

特别标：

- mean：全部基线劣；
- p95：只有 Parrot 打平；
- makespan：Torpor 更优。

页脚必须写：

> **Different action interfaces; end-to-end comparison, not mechanism attribution.**

---

# 六、中文汇报标题和摘要

:::writing{variant="document" id="63148" title="中文汇报标题与摘要"}
**标题**

《预测什么时候才真正有用？——部分揭示视频 Agent 工作流中的有限前瞻与动作可执行性》

**摘要**

本项目研究部分揭示的视频 Agent 工作流如何利用尚未出现的短期未来信息进行 GPU 调度。已有结果表明，一个仅使用未来五步 p95 代价的简单前瞻调度器 F0，已经在冻结测试集的平均完成时间上显著优于五条适配基线。然而，当我们进一步引入链长分布、存活加权和实例级概率预测时，PDRS 在平均、p95、违约率和 makespan 上均没有获得额外收益，甚至部分排序策略显著恶化尾部；即使使用未来真值，性能也基本不再提高。这说明在原有排序和放置动作空间中，未来信息已经接近饱和，预测精度并非主要瓶颈。

基于这一发现，我们进一步扩展了模型驻留、策略驱逐和保守预取动作。新的驻留动作包相对 F0 将平均完成时间降低 1.21%、p95 降低 1.95%，并分别减少 18% 的驱逐、22% 的按需冷加载和 23% 的驱逐后重载。进一步的 point/distribution/oracle/shuffle 对照表明，平均性能收益主要来自新增的驻留控制能力，而非更复杂的概率消费；实例特异性的未来信念目前只在尾部延迟上表现出独立价值。结果提示，在部分揭示 Agent 调度中，关键问题不仅是“能否预测未来”，更是“预测的信息是否存在能够作用于系统未来状态的调度动作”。
:::

---

# 七、英文论文标题与摘要

我会避免标题直接把 PDRS 写成主英雄，因为现有证据已经告诉我们“distributional PDRS 本身不是赢家”。

:::writing{variant="document" id="82416" title="English Paper Title and Abstract"}
**Title**

**When Lookahead Becomes Actionable: Scheduling Partially Revealed Video-Agent Workflows with Residency Control**

**Abstract**

Video-agent workflows progressively reveal heterogeneous planning, visual-analysis, detection, and generation stages, making near-future GPU demand only partially observable at scheduling time. We study whether learned short-horizon information about an unrevealed workflow suffix can improve node-level GPU scheduling. On a frozen trace-driven evaluation, a simple five-step lookahead scheduler already significantly reduces mean completion time relative to five adapted agent-serving baselines. Surprisingly, replacing this scalar lookahead with richer instance-level distributional predictions provides no additional benefit under the original ordering and placement actions: mean, tail latency, deadline misses, and makespan do not improve, and even an oracle future remains essentially flat.

This result points to an actionability rather than a prediction-accuracy bottleneck. We therefore extend the scheduler with model-residency controls, including policy-guided eviction and conservative prefetching, while keeping machine-response profiles hidden from the scheduling policy. The resulting action package reduces mean completion time by 1.21% and p95 latency by 1.95% relative to the original lookahead scheduler, while reducing evictions by 18%, demand cold loads by 22%, and eviction–reload cycles by 23%. Matched point, distributional, shuffled, and oracle controls further show that the mean improvement is primarily attributable to the expanded residency action space, whereas instance-specific future belief provides an additional measurable benefit only in tail latency. These results suggest that, for partially revealed agent workflows, richer prediction is useful only when the scheduler exposes actions capable of changing the future execution state.
:::

这里最后一句投稿前可能还需要稍微收紧，因为“only when”目前略强。最终可以改成：

> “These results suggest that the value of richer future prediction is strongly constrained by the actions exposed to the scheduler.”

这句更保险。

---

# 八、老师最可能问的 5 个尖锐问题

### Q1：“所以你那个概率预测到底有没有用？”

不要绕。

回答：

> **目前对 mean 没有证据证明完整概率信念有额外价值。**  
> F0point、PDRS 和 oracle 在驻留动作下基本打平，而且 shuffle 仍然能改善 mean，所以 mean gain 主要来自新增 residency actions。  
> 但是在 p95 上，真实实例 belief 显著优于 shuffled belief 1.615s，因此我们目前能确认的实例预测独立价值主要体现在 tail。

这个回答非常有力量，因为不逃避负结果。

---

### Q2：“那你论文为什么还需要 predictor？直接做 eviction/prefetch 不就好了？”

回答要承认这个问题**还没有完全关闭**：

> 这是目前最关键的开放问题。现有结果表明简单 point future 已经足以获得大部分 mean gain，因此不能声称复杂 predictor 对 mean 必不可少。我们现在把 predictor 的价值拆成两个问题：第一，简单 partial future 是否足以指导 residency；第二，instance-specific distribution 是否在高不确定、低驻留或更强 contention 场景下产生额外价值。目前 confirm300 已经观察到 tail benefit，但还需要 same-interface 2×2 和 Stress-v2 完成最终归因。

如果两天后的汇报是进度汇报，这个回答完全可以接受。

---

### Q3：“你新系统比 Parrot/LLMSched 好，是不是因为你多了 prefetch/eviction 动作？”

回答：

> **系统级比较确实存在动作接口不一致，所以不能拿它证明我们的机制本身比这些 baseline 好。**

然后说：

> 冻结主表只支持原动作空间下的 mean 比较；新的 residency experiment 是独立的 mechanism study。现在跨运行结果只能说明最终系统表现，而不能做 causal attribution。论文定稿前需要补 same-interface controls，至少对核心 F0/PDRS 体系完成信息×动作 2×2；若要对 baseline 做机制归因，还需要允许它们使用等价动作接口。

不要狡辩。

---

### Q4：“为什么 oracle 都没有提升？是不是整个未来预测课题没意义？”

回答：

> Oracle 平只说明在**原来的 ordering/placement action space，以及当前 confirm300 workload** 中，继续提高 future accuracy 没有剩余价值。实际上它帮助我们排除了 predictor accuracy 这个解释，并直接引导出了 actionability hypothesis。开放 residency actions 后，系统第一次出现稳定改善。因此 oracle-flat 不是实验失败，而是定位瓶颈的关键证据。

但别继续说 oracle 证明 distribution useful——它没有。

---

### Q5：“为什么抢占反而更差？”

这是你现在很好的机制解释：

> 当前策略是 SRPT-like，它本来就偏向短工作。显式抢占进一步把长请求变成 victim，虽然能让短请求更早运行，却累计了大约 90 秒/episode 的 victim waiting，结果 p95 和 makespan 都显著恶化。这个结果与 F0/Parrot 的 mean-tail tradeoff 一致，因此更像 starvation/fairness 问题，而不是实现异常。

然后强调：

> 所以我们现在没有把 preemption 放进最终正方法。

---

# 九、两天后汇报最后一页我建议只放三句话

第一句：

> **简单有限前瞻已经足以改善平均调度性能，但更丰富的预测在原动作空间中迅速饱和。**

第二句：

> **新增模型驻留控制后，系统首次同时改善 mean 和 p95，说明“能做什么”与“知道什么”同样重要。**

第三句：

> **当前证据同时限制了我们的 claim：动作机制解释 mean 收益，而实例级 future belief 的独立价值目前只在 tail latency 中得到确认。**

然后“Next”只留三件事：

1. **same-interface information × action 2×2**；
2. **Stress-v2：制造更高 residency pressure / contention，验证 belief value 是否扩大**；
3. **公平性/尾部机制：加入 aging 或 fairness control，确认 starvation 解释。**

这样老师会看到的不是“做了很多实验但是结果乱”，而是一条非常清楚的科研路线：

\[
\text{Future helps}
\rightarrow
\text{richer future does not}
\rightarrow
\text{why?}
\rightarrow
\text{action bottleneck}
\rightarrow
\text{residency actions help}
\rightarrow
\text{prediction and action contributions disentangled}
\]

这其实已经比最开始“预测未来 H5 然后调度”成熟了很多。