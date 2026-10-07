> **更正注(2026-10-05,后加)**:文中机制数字 "eviction −18%" 系口径混用——−17~18% 是"仅驱逐"臂
> (pdrs_evict)的口径;**主线 `pdrs_resident` 的精确值为:驱逐 −10.3%、按需冷加载 −21.5%、驱逐后重载 −22.3%**
> (预取 1432/600/174,命中 42%)。详见 `experiments/EXP-20261005_residency_comparison_v1/RESULT.md` 机制表。

# 第 7 轮追问:经典故事骨架(问题→方法→结果)的调和方案(GPT 回复全文)

> 来源:网页版 GPT(GPT-5.6 Sol + High;generation 11 会话 `6ac1f777-6728-83ec-9281-18a3cb787f23`),
> 2026-10-05 第 7 轮聚焦追问。原始回复全文照录。

炜，这个经典骨架是成立的，而且我建议你两天后的汇报就按这个讲。上一版“发现型”叙事适合论文讨论部分，但进度汇报最重要的是让老师在前三分钟内听懂三件事：

> **出现了什么新问题 → 你做了什么 → 最后提升了多少。**

你现在完全可以拼成这条主线：

> **视频 Agent 的工作流在运行中逐步揭示，GPU 调度器在当前时刻并不知道后面会调用什么模型、还剩多少工作；但模型加载、驻留和驱逐又具有跨步骤后果。我们因此设计了一个“有限前瞻 + 驻留控制”的调度器：利用执行前缀预测短期未来，用 F0 做当前任务排序，并增加策略驱逐与保守预取来管理 GPU 驻留状态。冻结 confirm300 上，F0 在平均完成时间上显著优于五条适配基线；进一步加入驻留动作后，相比 F0，mean 再降低 1.21%、p95 降低 1.95%。**

这就是你要的经典故事。

---

## 1. 主线方法到底是 F0 还是 F0+动作包？

**最终方法应该定义成 F0 + Residency Actions，而不是单独 F0，也不是 PDRS。**

最容易讲清楚的结构是：

\[
\boxed{
\text{Prefix-conditioned finite lookahead}
+
\text{Residency control}
}
\]

其中两部分各司其职。

F0 是**决策骨架**：

\[
\text{current cost}
+
\text{predicted H5 future cost}
\]

它解决：

> 当前 ready nodes 里，谁应该先执行？

驻留动作包解决：

> 当前与未来模型之间，GPU 上应该留下谁、赶走谁、提前加载谁？

所以最终方法逻辑是：

\[
\text{Prefix}
\rightarrow
\text{short-horizon prediction}
\rightarrow
\begin{cases}
\text{F0 ordering}\\
\text{residency control}
\end{cases}
\]

这样非常自然。

你甚至不需要再把 “PDRS” 放进方法名。

PDRS 已经从“候选主方法”变成：

> **为了验证更复杂地消费概率未来是否必要而做的增强实验。**

它失败了，但这不影响最终方法。

更重要的是，你必须注意一个边界：

**不能说“概率信念驱动驻留动作，从而使 mean 提升 1.21%”。**

因为 shuffle 仍然赢，F0point≈PDRS。

所以方法描述最好写成：

> “利用有限前瞻信息进行排序，并开放未来感知的驻留控制动作。”

而结果解释写成：

> “驻留动作解释了主要的平均收益；实例级未来信念的额外价值目前主要体现在尾部。”

这样就不会过度归因。

---

## 2. 负结果怎么放？只放一页，而且定位成“为什么最终方法这么简单”

不要按照实验时间顺序讲：

> F0 → PDRS → oracle → placement → residency → preemption……

这正是上一轮让你觉得乱的原因。

经典故事应该先把**最终正方法和结果讲完**。

然后用一页回答：

> **为什么没有设计得更复杂？**

标题甚至可以直接叫：

### “为什么没有继续增加预测与调度复杂度？”

这一页只需要三个结论：

\[
\text{Richer PDRS prediction}
\not\Rightarrow
\text{better performance}
\]

因为四指标都没有改善，而且排序显著伤 p95。

\[
\text{Oracle future}
\approx
F0
\]

说明继续提高 future accuracy 不是当前主要瓶颈。

\[
\text{Explicit SRPT preemption}
\Rightarrow
\text{worse tail}
\]

说明“开放更多动作”也不能无脑做；破坏执行进度的动作会制造 starvation。

这页的作用不是贡献，而是支持最终设计选择：

> **所以最终保留简单 F0，并只加入低破坏性的 residency actions。**

这样负结果就从“故事里的岔路”变成了：

> **设计空间裁剪依据。**

非常清楚。

---

# 3. 开场一句话怎么写

中文我建议直接说：

> **视频 Agent 的 GPU 调度有一个传统推理服务里不明显的问题：工作流是边执行边揭示的，但模型驻留、加载和驱逐的影响会跨越多个未来步骤，因此调度器必须在不知道完整未来的情况下提前做资源决策。**

然后第二句马上接方法：

> **我们用实例执行前缀预测短期未来，并将有限前瞻排序与模型驻留控制结合起来。**

英文对应可以是：

> **Video-agent GPU scheduling faces a mismatch between information and decision timing: future workflow stages are revealed only during execution, while model residency, loading, and eviction decisions have consequences that extend across those unrevealed stages.**

接：

> **We address this mismatch with prefix-conditioned finite lookahead combined with runtime residency control.**

这比说：

> “Existing systems cannot…”

安全很多。

也没有 first claim。

---

# 4. 经典故事和发现型故事冲突吗？

不冲突。

最好的处理方式是：

**经典骨架做主线，发现型逻辑做内部证据。**

主线：

\[
\boxed{
Problem
\rightarrow
Method
\rightarrow
Improvement
}
\]

也就是：

\[
\text{逐步揭示 workflow}
\rightarrow
\text{有限前瞻 + residency actions}
\rightarrow
\text{mean/p95 改善}
\]

而“发现型”内容只用来回答：

> 为什么你最后选了这个方法？

即：

\[
\text{PDRS doesn't help}
\]

\[
\text{Oracle doesn't help}
\]

\[
\text{Preemption hurts}
\]

因此：

\[
\text{simple lookahead + low-disruption residency control}
\]

这就是最终方案。

**两天后的汇报，我明确推荐经典版。**

论文未来写作则可以在 Results/Discussion 里重新展开 actionability discovery，因为那是更深入的 scientific insight。

换句话说：

> 汇报讲“我解决了什么问题”。

> 论文讨论讲“实验过程中我们还发现了什么规律”。

两者不是同一个叙事层级。

---

# 5. 8–9 页汇报，我建议这样重排

| 页 | 标题 | 这一页只让老师记住一句话 | 图/表 |
|---|---|---|---|
| **1** | **问题：视频 Agent 的工作流是边跑边揭示的** | 调度器现在做决定，但后续模型/工作量尚未出现，而 residency 决策有未来后果 | 一张简单 workflow：Planner → ? → ?，下面两张 GPU |
| **2** | **现有调度为什么不够？** | 只看 ready frontier 会忽略短期未来需求；单纯知道当前 cost 不能管理未来 residency | 对比 Myopic/F0 的概念图，不放复杂公式 |
| **3** | **方法：有限前瞻 + 驻留控制** | Prefix predictor 给短期未来；F0 排序；驱逐/预取管理驻留 | 最重要的系统框图 |
| **4** | **第一步：有限前瞻已经显著改善平均完成时间** | 冻结原动作空间下，F0 mean 显著优于五条 adapted baselines | F0 为 0 的 mean forest plot |
| **5** | **第二步：驻留动作进一步同时改善 mean 和 p95** | 相比 F0：mean −1.21%，p95 −1.95%，均显著 | 两个 forest/bar + CI |
| **6** | **为什么会改善？——少驱逐、少冷加载** | 驱逐 −18%、cold load −22%、evict→reload −23%，形成机制闭环 | 三个机制柱 + prefetch hit≈40% |
| **7** | **为什么最终方法没有更复杂？** | 更丰富 PDRS、oracle 都没带来额外收益；显式抢占反而伤尾部 | 一个小型对照图：PDRS/oracle/preempt |
| **8** | **最终系统表现与边界** | mean 对六个系统均占优；p95 与 Parrot 打平；Torpor makespan 更好；跨接口只作 end-to-end 对照 | mean/p95 forest + 角标注明不同 action interface |
| **9** | **结论与下一步** | 问题是“未来未知但资源动作必须提前”；当前解法是简单前瞻 + residency；下一步补 same-interface 2×2 和 stress | 三条 takeaway + next work |

如果只有 15 分钟：

- 第 2 页压 1 分钟；
- 第 7 页压 1 分钟；
- 第 8 页只说边界，不逐条念数字。

真正重点放 1、3、4、5、6。

---

# 6. 这个经典版里最关键的一张方法图怎么画

我甚至建议你整场汇报围绕这一张图。

左边：

```text
Observed prefix
Planner → Spatial Tool
```

中间：

```text
Short-horizon predictor
        ↓
possible future:
70% Model A
30% Model B
remaining work
```

右边：

```text
Scheduler

1. Ordering:
current + finite future cost

2. Residency:
keep / evict / prefetch
```

最右：

```text
GPU 0      GPU 1
Model A    Model C
```

底下一句：

> **Predict only what is needed to make near-future resource decisions.**

但是注意不要在图里放：

- slowdown table；
- interference table。

它们只属于 evaluator，不属于方法。

---

# 7. “好多少”应该怎么讲才不会乱

不要一次把所有数字都念出来。

只讲三级结果。

第一级：

> **在相同原动作空间里，F0 的平均完成时间相比五条适配基线低约 0.5%–5.0%，且全部统计显著。**

第二级：

> **在 F0 基础上加入 residency actions 后，mean 再降低 1.21%，p95 降低 1.95%，两者都显著。**

第三级：

> **最终系统在 end-to-end 对照中 mean 优于六个参照系统 1.73%–7.49%；p95 除 Parrot 外均更好或持平，与 Parrot 打平。**

然后主动补一句：

> “这里最后一组是不同动作接口的系统级比较，因此我只把它作为 end-to-end performance，不用它做机制归因。”

老师会觉得你很清楚自己实验能证明什么。

---

# 8. 三个最可能追问的问题

### Q1：“你的创新到底是预测器，还是 eviction/prefetch？”

这是老师最可能问的。

你要回答：

> **目前最终系统是两层：有限前瞻负责排序，驻留动作负责跨步骤管理 GPU 状态。现有结果表明，复杂概率消费并没有给 mean 带来额外收益，因此我不会把“更复杂 predictor”作为主要算法贡献。核心问题是如何在部分揭示的 workflow 中把有限未来信息与可执行的资源动作结合起来。驻留动作贡献了主要 mean 增益，而实例级预测的额外价值目前主要出现在 p95。**

这很诚实。

---

### Q2：“既然 shuffle 后 mean 还是好，那干脆不要预测，直接做 residency policy 不行吗？”

这是目前最危险的问题。

回答：

> **这是现在还没有完全关闭的问题。现有证据说明 residency action 本身确实解释了大部分 mean 收益，所以不能声称预测是 mean 改善的必要条件。另一方面，真实 belief 在 p95 上显著优于 shuffled belief，说明实例级预测至少在尾部有额外价值。下一步必须补 same-interface 2×2，把 no-future residency、point-future residency 和 instance-specific future residency 放在完全相同动作接口下比较。**

千万不要试图硬解释 shuffle。

这个实验就是下一步。

---

### Q3：“那为什么不直接用 Parrot？它 p95 本来就比 F0 好。”

回答：

> **Parrot 的确在原动作空间 p95 上显著优于 F0，所以不能说 F0 全面优于 Parrot。它的 App-FIFO 更公平，因此牺牲了一部分 mean 换取更好的 tail。加入 residency actions 后，我们把 F0 相对 Parrot 的 p95 劣势从显著 −2.40% 消解到统计打平，同时保持更好的 mean；这说明 tail 不完全是预测问题，也受 starvation/fairness 和 residency disruption 共同影响。**

如果老师继续问：

> “那是不是应该给 F0 加 aging/fairness？”

答案就是：

> 对，这正是下一阶段尾部方向之一，而且比继续堆预测精度更合理。

---

# 最后我帮你把整个汇报压成一句最容易记的中文

如果老师最后只记住一句，我希望是：

> **视频 Agent 的难点不是单纯“未来预测不准”，而是工作流在逐步揭示时，调度器必须提前决定哪些模型值得执行、保留、驱逐和预取；我们的方案用简单有限前瞻做排序，并通过驻留控制把这种前瞻转化成实际 GPU 动作，在冻结测试集上相对 F0 将平均和 p95 完成时间分别进一步降低 1.21% 和 1.95%。**

这就是现在最像“问题—方法—结果”的版本。

而那些很有价值的 PDRS/oracle/shuffle/preemption 结果，全部放在后面作为：

> **为什么最终设计是这个，而不是更复杂版本。**

这样故事就不会散了。