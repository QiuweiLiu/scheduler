# 第 6 轮架构裁定:动作域扩充——驻留动作(预取+驱逐)优先,抢占推迟

> 来源:网页版 GPT(GPT-5.6 Sol + High;generation 11 会话 `6ac1f777-6728-83ec-9281-18a3cb787f23`),
> 2026-10-05 第 6 轮架构讨论。原始回复全文照录(公式为 LaTeX 源)。

炜，这轮我给一个比较明确的裁定：

**你的新证据基本足以停止继续优化“排序/普通 placement 消费者”。下一步应该扩动作域，但第一阶段只扩“驻留动作”——预取 + 策略化驱逐；抢占先作为第二阶段。**

原因很直接：PDRS、placement-only、甚至 oracle 在现有动作空间都没有显著收益，而改变排序还显著伤害 p95。这已经非常像“信息有，但当前 action set 无法兑现”的症状。与此同时，抢占会同时改变运行进度、stall、完成顺序，是三个动作里最容易把因果归因搞乱的，所以不应该和 residency 动作第一次一起塞进去。

更重要的一点：**核心实验里 F0 与 PDRS 必须使用完全相同的 F0 node ordering。** 不要再让 PDRS 的 survival-weighted ordering 参与。否则你刚发现的 +2.26s p95 劣化会掩盖新动作究竟有没有价值。

---

## 一、先定义一个统一“信念接口”，再让三个动作共用

这样才能满足你要求的“唯一差异是如何消费信念，而不是不同启发式”。

对实例 \(j\)、未来第 \(k\) 步、模型 \(m\)，定义一个模型需求质量：

\[
D_{jkm}^{B}
=
q_{jk}^{B}
\cdot
p_{jkm}^{B}
\cdot
\ell_{jk}^{B}
\]

其中 \(\ell\) 用统一的 `p95 load duration` 表示价值尺度。

两种消费方式只在 \(B\) 上不同。

**F0-point 模式：**

\[
q_{jk}^{F}=1,
\]

\[
p_{jkm}^{F}
=
\mathbf1
[m=\arg\max p_{jkm}],
\]

\[
\ell_{jk}^{F}
=
\mathbf1[p^{load}_{jk}\ge0.5]\,
L^{95}_{jk}.
\]

也就是说，把同一份冻结预测**压成确定性 top-1 + hard load threshold + 固定 H5**。

这不是原始 F0 新增了训练信息，而是定义一个“F0 风格的点消费器”。我建议明确把 arm 名叫：

`F0-point-actions`

避免声称它和 frozen F0 完全相同。

**PDRS-distribution 模式：**

\[
q_{jk}^{P}=P(L_j\ge k),
\]

\[
p_{jkm}^{P}=P(M_{jk}=m),
\]

\[
\ell_{jk}^{P}
=
p^{load}_{jk}L^{95}_{jk}.
\]

因此它使用 survival + 全 model probability + soft load occurrence。

接下来所有 action policy 完全相同，只是把 \(D^F\) 换成 \(D^P\)。

这就是最干净的信息对等。

---

# 二、动作 1：预取——我建议做，但先做“保守预取”

不要第一版就让它为了预取主动驱逐模型。

那会把 prefetch 和 eviction 两个机制绑死，出了结果无法解释。

定义下一步需求：

\[
U_B(m)
=
\sum_jD^B_{j,1,m}.
\]

为什么只看 \(k=1\)？

因为 prefetch 是非常短视的执行动作。当前 action 对第五步是否仍然有价值依赖中间大量状态转移，而方法又明确不读 machine dynamics model。H1 最符合你的方法边界。

### 触发条件

每个调度 decision epoch：

1. 先按照 frozen F0 ordering 尽可能 dispatch 真实 ready nodes；
2. 真实工作优先级永远高于 speculative prefetch；
3. dispatch 完后，若存在：
   - \(U_B(m)>0\)；
   - \(m\) 当前不 resident、也不在 loading；
   - 至少一张 GPU 有足够**现有空闲显存**直接容纳；
4. 才允许一个 prefetch。

第一版：

**prefetch 不允许触发 eviction。**

所以 speculative action 永远不会主动赶走真实 resident state。

### 预取哪个模型

\[
m^*=\arg\max_m U_B(m).
\]

F0-point：

> 下一步 top-1、高于 0.5 load threshold 的模型。

PDRS：

> 所有实例下一步 model probability × survival × load probability 的概率质量。

没有另外调阈值。

### 放哪张 GPU

方法不读 slowdown/interference table。

在所有能够**无驱逐容纳** \(m^*\) 的 GPU 中：

1. free memory 最大；
2. 平局 GPU id 固定排序。

如果已经在任意 GPU resident，第一版不做 duplicate prefetch。

### 什么时候预取

立即。

不做“等多久再 preload”的第二个决策变量。

否则马上又需要 machine-response model 来估 timing。

环境自己结算：

- load time；
- load interference；
- 真实 delay。

policy 完全不知道这些表。

这意味着若 prefetch 太激进，它应该自然受到惩罚——这是好事。

---

# 三、动作 2：驱逐——这是我认为最值得先做的动作

相比 prefetch，我实际上更看好 **belief-guided eviction**。

因为它不需要回答“现在 preload 是否值得”这种 machine-cost 问题，只回答：

> **必须腾显存了，留下谁？**

这是 future model belief 最天然的 consumer。

定义模型的全 horizon 驻留价值：

\[
V_B(m)
=
\sum_j\sum_{k=1}^{5}
D^B_{jkm}.
\]

F0-point 和 PDRS-distribution 只通过前面的统一 belief encoder 不同。

### 触发条件

**只在真实容量压力已经发生时触发。**

即某个真正 ready node 所需模型不 resident，而且加载它后显存不足。

不要为了“未来可能来”提前驱逐。

### 驱逐对象怎么选

不是简单：

> evict 最低概率那个。

应该考虑一次可能需要腾多个模型。

对于候选 GPU \(g\)，在所有可驱逐 resident model 集合中寻找：

\[
E_g^*
=
\arg\min_E
\sum_{m\in E}V_B(m)
\]

满足：

\[
\text{freed\_mem}(E)\ge
\text{required\_mem}.
\]

因为模型数通常很少，DES 里直接枚举 subset 就行，不需要 greedy heuristic。

平局规则：

1. 驱逐总 future value 最小；
2. excess freed memory 最小；
3. eviction model-id tuple 字典序；
4. GPU id。

不能驱逐：

- 当前执行中的 pinned model；
- in-flight load；
- substrate 明确标记不可驱逐状态。

这样 F0 和 PDRS 的**动作规则一字不差**。

只变化：

\[
V_F(m)
\quad vs\quad
V_P(m).
\]

这非常漂亮。

### GPU 本身怎么选

如果目标模型可以在多张 GPU 上加载，则枚举：

\[
(g,E_g^*)
\]

选择总 eviction value 最小的那个。

若相同，再回退 frozen F0 的当前 placement tie-break。

所以未来信息这次真正进入：

\[
(node,GPU)
\]

动作域，但完全不读 machine table。

---

# 四、动作 3：抢占——能做，但我不建议放进第一次主验证

这里我和你的原提案有一点不同。

**model probability 对 prefetch/eviction 是天然信号，对 preemption 并没有那么天然。**

抢占解决的是：

> 哪个已经运行的 request 应该为更紧急的 request 让位？

它最自然消费的是**剩余工作量 belief**，而不是下一模型概率。

而且你刚证明 survival-weighted remaining-work 对 ordering 甚至会伤 p95。

所以第一次就把它加入，会显著增加解释难度。

如果仍然做，我建议用一个 parameter-free、SRPT 风格的规则。

对 job \(j\) 定义：

F0：

\[
R_F(j)
=
R_{\rm visible}(j)
+
\sum_{k=1}^{5}
[
T^{95}_{jk}
+
\mathbf1(p^{load}_{jk}\ge0.5)L^{95}_{jk}
].
\]

PDRS：

\[
R_P(j)
=
R_{\rm visible}(j)
+
\sum_{k=1}^{5}
q_{jk}
[
T^{95}_{jk}
+
p^{load}_{jk}L^{95}_{jk}
].
\]

仍然用 p95，而不是重新换 runtime statistic。

### 触发

只有同时满足：

- frozen F0 选出的最高优先级 ready node \(i\) 当前无法 dispatch；
- 阻塞原因是 GPU 被占；
- 至少存在一个 **decode-phase** victim；
- 暂停该 victim 后会让 \(i\) 立即成为合法 action。

不做 speculative preemption。

### victim

只有：

\[
R_B(i)<R_B(v)
\]

才允许 preempt。

然后：

\[
v^*=\arg\max_vR_B(v).
\]

直觉就是经典 SRPT：

> 让更长的任务给更短、更高优先级的任务让路。

注意这里我不建议“抢占最短剩余”。

那在 mean JCT 目标下方向基本反了——短任务更应该让它尽快完成。

### 恢复

必须沿用现有 request-preemption contract：

- prefill 原子，不抢；
- 只在合法 decode/token boundary；
- 保存已生成 token/progress；
- resume 不得 `progress=0`；
- 环境计入 \(R_m\)；
- 调度器不读取 \(R_m\)；
- deadline clock 不暂停；
- 不允许同一个 request 同时出现 running+queued 两份；
- resume 后至少完成一个有效 decode unit 才允许再次被抢占，防止零进展 ping-pong。

这里最后一条属于 liveness invariant，不是性能调参。

---

# 五、为什么我建议第一次只测试“预取 + 驱逐”

你现在真正的科学假设是：

> **预测的未来模型 belief 在模型驻留/容量动作出现以后才具备 actionability。**

最直接的动作就是：

\[
\boxed{\text{prefetch + eviction}}
\]

而不是 preemption。

抢占更像：

> future-work belief 是否能帮助 execution priority control。

这是第二个问题。

如果一次加入三个：

- New 赢了：不知道是哪个动作；
- New 输了：也不知道是谁拖后腿。

你刚用很多实验才把排序/placement 轴理干净，没必要再重新制造混乱。

---

# 六、我建议的 8 臂，不需要 Myopic+actions

正好用你说的一次 8-arm ≈40 min。

| arm | Ordering | Residency actions | belief |
|---|---|---|---|
| A0 | frozen F0 | 无 | 原 F0 |
| A1 | frozen F0 | 无 | PDRS（只作为已有 matched reference） |
| A2 | frozen F0 | prefetch+eviction | **F0-point** |
| A3 | frozen F0 | prefetch+eviction | **PDRS-distribution** |
| A4 | frozen F0 | prefetch+eviction | shuffled PDRS belief |
| A5 | frozen F0 | prefetch+eviction | oracle future belief |
| A6 | frozen F0 | prefetch+eviction+preemption | F0-point |
| A7 | frozen F0 | prefetch+eviction+preemption | PDRS-distribution |

其中 A0/A1 可以重跑，以保证：

- 同一代码版本；
- 同样 instrumentation；
- paired episode ID；
- 同一 bootstrap。

若 bit-identical，可注明与 frozen numbers 一致。

### Oracle 怎么定义

不要使用整个 workflow 全真值。

否则 ceiling 太夸张。

我建议：

**Oracle 使用与该 action 完全相同 horizon 的 realized future。**

预取：

\[
M_{t+1}
\]

one-hot。

驱逐：

真实未来最多 H=5 的 model sequence + 是否实际终止。

load duration 仍然使用冻结 predictor p95，不给 execution duration truth。

这样 oracle 只回答：

> 如果 future identity belief 完美，会不会有价值？

不混入 runtime truth。

### Shuffle

shuffle **整个 per-instance future belief package**：

- model distribution；
- length/survival；
- load probability；

在 episode 内实例之间 permutation。

保持：

- marginal statistics；
- action mechanism；
- action count opportunity；

破坏：

- prefix-specific informativeness。

这比随机打乱概率向量内部强很多。

---

# 七、Myopic+actions 要不要？

**第一轮不需要。**

原因是你的主问题不是：

> actions 本身有没有价值？

而是：

> **同一个 action space 中，distributional future belief 是否比 point consumption 更有价值？**

A2 vs A3 已经是最关键 matched pair。

A0 vs A2 则告诉：

> residency actions + 简单 future point information 本身有没有价值。

A3 vs A4：

> instance-specific belief 有没有价值。

A3 vs A5：

> predictor ceiling 还有多少。

已经足够。

Myopic+actions 最多作为第二轮 control，用来回答：

> completely no-future 的 residency policy 能做到多少。

但不是当前 blocker。

---

# 八、主指标别再四个一起“找胜利”

你刚这轮最有价值的一点，就是四个指标全部没有救活 PDRS。

下一轮反而应该更严格。

继续预注册：

\[
\boxed{\text{primary = mean completion time}}
\]

因为这是原论文主指标。

secondary：

- p95 completion；
- deadline miss；
- makespan。

机制指标必须新增：

- `prefetch_started`
- `prefetch_useful`
- `prefetch_wasted`
- `prefetch_hit_rate`
- `cold_loads_on_demand`
- `load_ms`
- `eviction_count`
- `evicted_then_reloaded`
- `resident_future_hit`
- `preemption_count`
- `resume_count`
- `preemption_stall_ms`
- `zero_progress_preemptions`（必须 0）
- `action_flip_vs_F0`

尤其是：

\[
\frac{\text{useful prefetch}}
{\text{all prefetch}}
\]

和：

\[
\text{evicted-then-soon-reloaded}
\]

会非常关键。

---

# 九、这次真正应该检验的是“交互效应”

别只看：

\[
A3-A2.
\]

你的假设其实是：

> **动作域扩充释放了 distributional belief 的价值。**

因此最漂亮的统计量是 difference-in-differences：

\[
\Delta_{\rm interaction}
=
(PDRS_{actions}-F0_{actions})
-
(PDRS_{base}-F0_{base}).
\]

现有 base 大概已经是：

\[
PDRS_{base}-F0_{base}
\approx +61\text{ ms}
\]

即无收益。

如果加 actions 后变成：

\[
PDRS_{actions}-F0_{actions}<0,
\]

且 paired-bootstrap 的：

\[
CI_{95\%}(\Delta_{\rm interaction})<0,
\]

你就获得了非常直接的证据：

> **不是预测突然更准了，而是 action space 改变后同样的信息才开始产生价值。**

这比单独说 New beats F0 强很多。

---

# 十、五道门重新切一次

**LLMSched：仍然切得很清楚。**  
它用 Bayesian structural/duration uncertainty 来决定 uncertainty-reducing stage 和 JCT-efficient ordering，本质仍然是 workflow scheduling / ordering，没有把未揭示 future model belief 映射为 model residency/prefetch/eviction policy。citeturn552872academia79turn552872search0

**AgentIR：危险度进一步升高。**  
它已经明确写了用 implied future demand 做 online placement，并结合 workflow structure、prefix locality、online instance profiling 和 heterogeneous vLLM routing；它也维护 conditional-edge probabilities。citeturn828074search0  
你现在必须切得更窄：

> AgentIR 利用编译 IR / 历史 workflow structure + online machine latency functions 来优化 placement；你的实验研究的是 **per-instance unrevealed future belief 在不读取 machine-response model 的条件下，能否直接改善 runtime residency actions**。

这个区别还在，但 AgentIR 已经绝对是最危险的一条。

**JITServe：有 preemption，但消费对象不同。**  
它会随着 generation 逐渐 refine imprecise request information，然后用 GMAX 做 SLO bandwidth allocation、batch composition，同时其 artifact 包含 scheduling/preemption integration。citeturn828074search4turn828074search3  
它没有做：

> future model identity belief → model prefetch / eviction / residency retention。

所以 residency-action 实验依然能切开。

**Latency-Aware Orchestration：这会是你 residency 方向最危险的正式论文近邻。**  
它已经明确预测 future model demand，并把 model-lifecycle alternatives、placement、ordering 一起优化，而且 predictor 会估 device-specific activation latency、peak memory、model-loading cost。citeturn552872search1turn552872academia80

所以你绝对不能 claim：

> “first future-aware model-lifecycle scheduler。”

你能守的是：

> **prefix-conditioned unrevealed belief → runtime residency actions，且不依赖 device-specific response/cost model。**

**Maestro：也很近，但信息对象不同。**  
它预测每个 stage 的 output length 和 memory usage，并用于 multi-model co-location、hierarchical weight caching、elastic memory provisioning、routing 和 workflow prioritization。citeturn828074academia36turn828074search1  
它不是从当前实例 prefix 预测**尚未揭示的 downstream model identity distribution**，所以未来模型 belief 驱动的驻留价值仍有切割空间。

五道门之后，我的判断是：

**加入 residency management 不会破坏你前面的 novelty 位置，但会让 Latency-Aware 和 AgentIR 的危险程度明显上升。**

所以一定要保住两个限定：

1. per-instance unrevealed suffix belief；
2. policy 不读取 machine-response cost model。

---

# 十一、仿真正确性上的 P0 级约束

这些必须在正式跑之前 gate。

**Prefetch：**

- 同一个 model 不得重复 load；
- resident / loading / demand-load 三种状态互斥正确；
- demand 到达时若 prefetch 尚未完成，必须等待同一个 in-flight load，不能再发第二次；
- speculative load 的 interference / duration 只计一次；
- prefetch 完成但从未被使用必须标记 wasted；
- 不允许 speculative prefetch 偷偷绕过 capacity gate。

**Eviction：**

- active/pinned model 不可驱逐；
- eviction 必须先发生，之后 cold load 才开始；
- model state 只有一个权威来源；
- eviction→reload 费用不能漏计；
- fail-closed 对不存在 profile cell 的行为保持原冻结 semantics。

**Preemption：**

- `batch_enabled=False` 始终保持；
- prefill 原子；
- decode token boundary；
- progress 单调不减；
- resume 不重置；
- \(R_m\) executor 计费但 policy 不读；
- deadline clock 连续；
- preempt/resume event 必须一一配对；
- completed/failed/preempted state 不得重复。

这里任何一条失败，都不能正式测量。

---

# 十二、与 frozen 主表的关系

**主表继续不动。**

但新 experiment 不再能说：

> “New 与五条 baseline 同动作空间公平比较。”

因为你的 action space 已经变了：

- prefetch；
- policy eviction；
- preemption。

所以这张实验应该明确叫：

> **Mechanism study under an expanded action space**

而不是主表替代。

最终论文证据链可以是：

1. Frozen main table：F0 在原动作空间胜过五条 adapted systems；
2. PDRS v2：更多 future distribution 在原动作空间没有额外价值；
3. Expanded-action experiment：测试是不是 actionability bottleneck；
4. Mechanism counters：解释为什么。

这其实是一条很好的科研故事。

---

# 十三、预注册证伪条件

我建议你正式把假设写成：

> **H-actionability:** Distributional future belief provides measurable scheduling value only after the scheduler is given residency/capacity-control actions that can directly act on predicted future model demand.

不要写绝对的“only”作为理论定理；这是实验假设。

### 强确认

必须同时满足：

1. 在无新动作时：
   \[
   PDRS_{base}-F0_{base}
   \]
   不优于 0——你已经基本确认。

2. 加 residency actions 后：
   \[
   PDRS_R-F0_R<0
   \]
   且 paired 95% CI 上界 < 0。

3. 交互效应：
   \[
   \Delta_{\rm interaction}<0
   \]
   且 CI 上界 < 0。

4. 机制方向一致，至少出现：
   - useful prefetch ↑ 或
   - demand cold-load ↓ 或
   - bad eviction/reload ↓。

这才是完整 confirmation。

### “动作有用，但 distribution 没用”

若：

\[
F0_R<F0_{base}
\]

但：

\[
PDRS_R\approx F0_R,
\]

则结论是：

> residency actions 有价值，但完整 probability distribution 没有额外价值。

这会直接否定你“distributional belief 是关键”的主张。

### “预测器质量不足”

若：

\[
Oracle_R\ll F0_R
\]

但：

\[
PDRS_R\approx F0_R,
\]

则：

> action space 有 headroom，但当前 predictor 无法释放。

这时再讨论重训 predictor 才合理。

### “动作空间仍然没价值”

若：

\[
Oracle_R\approx F0_R
\]

且预取/驱逐有足够 action opportunities，

那就很强：

> 即使未来身份完全已知，residency action 也没带来性能收益。

此时应停止这条线，不再堆 action。

### “工作负载没有机会”

如果 oracle 不赢，但：

- eviction trigger 极少；
- prefetch candidate 极少；
- resident hit 已经很高；

则不能说机制无效，只能说：

> confirm300 里 residency actionability 不足。

需要做 stress regime，而不是改算法。

### “belief 非因果”

如果：

\[
Shuffle_R\approx PDRS_R
\]

则即使两者都赢 F0，也不能把收益归因于 instance-specific prediction。

这点非常重要。

---

# 最后的执行顺序

我不会直接实现“三动作完全体”。

**第一优先：只实现统一 belief encoder + policy eviction。**

因为它是：

- 最自然；
- 零 speculative timing；
- 不依赖 machine model；
- 不改变 request execution progress；
- 最容易验证。

先跑：

\[
F0\text{-point eviction}
\quad vs\quad
PDRS\text{-distribution eviction}.
\]

如果 eviction 事件太少，再加 conservative prefetch。

**第二优先：prefetch + eviction residency package。**

形成正式 A2–A5。

**第三优先：只有前两项出现正信号，才打开 preemption A6/A7。**

所以我的最终裁定不是简单的“动作域扩充对”。

而是更具体：

\[
\boxed{
\textbf{先把未来信念接到 residency value，尤其是 eviction；不要先接 preemption。}
}
\]

你目前最值得验证的新问题已经从：

> “预测未来能不能让排序更好？”

非常自然地变成：

> **“既然未来预测无法改善选择哪个 ready node，那么它是否能够通过决定什么模型值得提前保留、加载或牺牲，改变未来可执行状态？”**

如果这一步连 oracle 都不赢，我会认为“有限前瞻预测”在你当前 workload/substrate 上基本已经被系统性证伪；反过来，如果 distributional belief 只在 residency actions 打开后开始显著胜出，那会比前面任何一次 +1–2% 的排序改进都更像真正的机制发现。