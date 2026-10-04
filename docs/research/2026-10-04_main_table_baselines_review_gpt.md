# 主表基线适配 manifest 审核（网页版独立审核，generation 10 会话）

## 来源与范围

网页版 GPT（GPT-5.6 Sol + High；generation 10 会话 `6ac131eb-5ee0-83ec-91e2-6d284f965c0a`，
上一会话 generation 9 达长度上限后换代）对 `2026-10-03_main_table_baselines_manifest.md`
（送审稿）做静态审核，并对四篇原论文正式 PDF/官方材料复核。无附件。

**结论：当前稿不批准冻结；按四项最小修正后批准进入实现阶段。**

修正要点：
- Parrot：去掉 critical-path，改为 App-FIFO projection（`Parrot-adapted (App-FIFO projection)`）；
- QLM：放弃 A/B，改为 SLO-oriented stochastic queue assignment（SAA，Δ=0.90）；
- Hermes：改用论文原式 Gittins + 在线 analytical prewarm trigger（K=0.5）；
- Torpor：改名 `Torpor-lifecycle-projection`，去掉加权分，改 lexicographic locality/swap 策略。

审核回复原文如下（未改动）。

---

炜，我把四篇原论文的正式 PDF/官方实现重新对过一遍后，结论比较明确：

**这版 manifest 还不能原样冻结。**  
Hermes 基本正确，只需要把公式和 prewarm 时机改回论文原式；QLM 需要中等幅度修正；Parrot 和 Torpor 当前定义都加入了原文没有的调度逻辑，若直接实现，确实存在 reviewer 指责 strawman / name-laundering 的风险。

尤其值得注意的是，我们上一代内部审查其实已经预警过：Parrot 在当前 serial-chain substrate 上很容易失去其主要 DAG/并行优化空间，而 Torpor 的有效机制主要依赖 model swapping / locality / interference。这次查完原文后，这个判断反而更确定了。

我建议冻结成下面这一版。

| Baseline | 裁定 | 推荐冻结形式 | 风险 |
|---|---|---|---|
| Parrot | **需修正** | App-FIFO / application-centric scheduling，不做 critical-path-first | 高 |
| QLM | **需修正** | SLO-oriented stochastic queue assignment，选"C"，不是 A/B | 中 |
| Hermes | **小修后冻结** | 原论文 Gittins 公式 + K=0.5 analytical prewarm | 低 |
| Torpor | **需修正** | lifecycle projection，lexicographic locality/swap policy，不做加权 runtime score | 中高 |

下面逐条说。

### 1. Parrot：不要用"剩余 critical path 大者优先"

这一条是当前 manifest 最大的问题之一。

Parrot 原文真正做的是：Semantic Variable 暴露真实 application DAG，然后进行 performance-objective deduction；实际调度算法先按 DAG **topological order** 排列，再倾向于把同一 application、同一 task group 连续调度。官方实现文档甚至直接把串行分析 workload 下的行为称为 **Application-level FIFO / depth-first order**。它不是一个"估计 remaining critical path，然后长作业优先"的算法。

论文 Algorithm 1 也很清楚：先 `Q.sort()` 拓扑序，再找 task group、shared prefix/context，然后选 engine；没有 `remaining_work` 或 critical-path-length priority。

所以我不批准当前：

> canonical remaining chain workload → critical-path-first

这实际上是我们自己造了一个 workflow-aware LJF，然后挂上 Parrot 名字。

建议冻结为：

`Parrot-AppFIFO-adapted`

在你们当前 serial-chain substrate 下：

\[
priority(j)=
(\text{service priority},
\text{application arrival order},
\text{ready time},
\text{job id})
\]

也就是同 service class 内：

**oldest application first，并尽量连续推进该 application 的 ready node。**

如果 A1、B1 都 ready，A 比 B 早到，那么执行 A1；A1 完成后 A2 ready，此时即使 B1 等得更久一些，仍继续 A2，形成 Parrot 文档描述的 application-level depth-first behavior。

这里**不需要 canonical future skeleton**。你们 readiness 本身已经保证拓扑合法；serial chain 又没有 task group parallelism。强行从 train 集学习未来骨架，反而给 Parrot 注入了原论文中来自"真实程序 DAG"的信息替代物，而且会跟 Ours/Hermes 的 future modeling 边界纠缠。

测试改成：

- App-FIFO 与 request-FIFO 至少发生一次 action flip；
- 一个 application 连续 stage 能获得 continuation preference；
- 修改不可见 suffix 的 runtime/role 不改变当前决定；
- serial chain 下 `task_group_size == 1` 应显式统计并报告。

名字最好也别叫 `Parrot-DAG-adapted`，推荐：

**`Parrot-adapted (App-FIFO projection)`**

表脚注：

> Semantic Variable API, prefix/context sharing, batching, and parallel task-group scheduling are unavailable; the serial-chain projection retains Parrot's application-centric topological/App-FIFO scheduling.

这个写法最安全。Parrot 原论文确实把 application-level scheduling、task grouping 和 context sharing作为关键机制。

换句话说，这条可以进主表，但你们应该**主动承认它在当前 workload 上退化明显**，而不是人为给它补一个 critical-path heuristic。

---

### 2. QLM：A 和 B 我都不建议，应该增加一个 C

这一条查完原文之后变化最大。

QLM 的 duration uncertainty 并不是你 manifest 现在描述的：

> completed same-key observations → Normal-Normal Bayesian posterior online update

原论文 RCT estimator 是从 offline request history 得到 output-token distribution，例如直接拟合

\[
O_q\sim \mathcal N(\mu_o,\sigma_o),
\]

然后把这个随机变量传播进 waiting time 和 completion time。论文明确写的是 workload profiling + hardware profiling；我没有看到"每完成一个同键 request 就做 conjugate posterior update"这个机制。

所以**不要自行增加在线 Bayesian posterior update**。这很容易变成"一个比原 QLM 更聪明的 QLM"。

更重要的是，QLM stochastic program 的目标不是平均 JCT。它优化的是 **SLO attainment**：

\[
ct_{g,j}
=
waiting + execution + model\ swap,
\]

\[
p_{g,j}=ct_{g,j}-SLO_{g,j},
\]

并要求：

\[
P(p_{g,j}\le0)>\Delta,
\]

最后最小化 SLO penalty。

所以你提出的 A：

> candidate a first，其余 SPT → expected total completion time

已经把 QLM 的核心 objective 换掉了。

B 的 mean+CVaR 更不是原 QLM。

我的裁定是新增：

**(C) QLM-SAA stochastic queue assignment**

冻结规则：

duration distribution：

\[
D_{m,r}
=
\text{train-only empirical duration distribution}
\]

key 建议就用：

`(model_id, role)`

样本不足时 fallback：

`model_id → global`

不要放 service priority 到 duration distribution 里，也暂时不要再细分 seq position，避免过拟合。

每次调度时拿当前 waiting/ready set，在固定的 **S=64 common-random-number scenarios** 下生成 execution time。

然后优化：

\[
C_i^{(s)}
=
waiting_i^{(s)}
+
runtime_i^{(s)}
+
load/swap_i
\]

并保持 QLM 的核心方向：

1. 优先使尽可能多的 request 满足 chance-SLO；
2. 在该条件下最小化 expected SLO penalty；
3. model transition/load 直接进入 completion time。

我建议 adaptation 固定：

\[
\Delta=0.90
\]

并且**明确标成 adaptation hyperparameter，不冒充论文原参数**。论文只写 "with a high probability"，公开版本里没有给我找到一个默认 Δ。正式主结果固定 0.90，不调 validation；appendix 做：

`Δ ∈ {0.8, 0.9, 0.95}`

敏感性。

如果当前 capacity 下 chance constraints 全部不可满足，则不要像原 QLM 一样 scale-up——因为你们 substrate 没 autoscaling。固定 fallback：

\[
\min
\left(
\#\text{chance violations},
\sum_i E[(C_i-d_i)_+],
\sum_i E[C_i]
\right)
\]

lexicographic optimization。

这样既保证任何状态都有合法 action，也没有偷偷变成 SJF。

warm-start 也不要单独再加一个 heuristic 权重。原 QLM 本身就是把模型 switching/warm-start 的后果通过 virtual queue ordering 映射到 LSOs。你们直接让 F3 load time 进入 \(C_i\) 即可。原论文明确说 virtual queue ordering 会驱动 model warm start / swapping。

因此最终：

**`QLM-queue-adapted` 可以保留这个名字。**

脚注建议：

> Retains QLM's uncertainty-aware stochastic queue reordering and model-transition cost; token-level batching, request eviction, KV state swapping, and autoscaling are unavailable.

测试增加一个很重要的：

> **同一组 mean runtime 不变，仅改变 variance，QLM 必须至少在构造 case 中产生不同 queue solution。**

否则 stochastic QLM 实际退化成 deterministic EDF/SPT。

---

### 3. Hermes：最接近可以直接冻结，但 Gittins 公式要改

你当前写的是：

\[
g=
\sup_\tau
\frac{P(\text{finish})}
{E[\text{time}]}
\]

概念上是 reward/time，但 Hermes 原文给了明确公式，所以没必要自己重新定义。

论文原式是：

\[
G(D,a)
=
\inf_{\Delta>0}
\frac{
E[\min(X_D-a,\Delta)\mid X_D>a]
}{
P(X_D-a\le\Delta\mid X_D>a)
}
\]

并且：

**G 越小，优先级越高。**

所以冻结这个原式。

也不需要自己造一个 `τ_max`。

Hermes 本身用 histogram/bucket representation，论文默认：

- distribution buckets = **10**
- Pearson correlation threshold = **0.5**
- prewarming effectiveness \(K=0.5\)

这些都有论文明确默认值。

你们直接：

\[
\Delta\in
\{\text{10-bucket support boundaries}\}
\]

枚举即可。

对于 non-preemptive node substrate，我建议：

- job 完成一个 node 后更新 PDGraph conditional distribution；
- 已观察服务量 \(a_j\) 用该 job 已执行 node 的 `observed_intrinsic_ms` 累加；
- 或直接构造 conditional remaining-demand distribution \(D_j^{remain}\)，令 \(a=0\)。

我更推荐后一种，实现不容易错：

\[
G(D_j^{remain},0)
\]

两种数学意义一致得多，而且天然适合 node boundary scheduling。

最大的修正反而是 prewarm。

你 manifest 现在写：

> 无 ready work 时才 prewarm

这**不是 Hermes**。

Hermes 的核心恰好是：**当前 functional unit 还在执行时，就提前启动下游 backend prewarm，把 warmup 从 critical path 隐藏掉。**

论文定义：

\[
p_e
=
p_s
P(t_c>t_s+t_p)
\]

其中：

- \(p_s\)：目标 downstream unit 被选择概率
- \(t_c\)：当前 unit completion time
- \(t_s\)：prewarm start
- \(t_p\)：prewarm duration

如果：

\[
p_s<K
\]

不预热；

否则选择 \(t_s\) 使：

\[
p_e=K.
\]

默认：

\[
\boxed{K=0.5}
\]

这刚好和你们已经实现的 F3/F4 非常匹配：

current node running  
→ PDGraph 推 downstream model probability  
→ 算 prewarm trigger event  
→ F3 model load  
→ F4 concurrent-load interference  
→ 若 F4 combination uncovered，按你们现有 fail-closed / serial fallback。

所以 Hermes 最终可以非常干净地冻结成：

**`Hermes-PDGraph-adapted`**

而 policy id `hermes_gittins` 没问题。

我甚至建议把测试加到：

- `gittins != mean-SPT` 至少一个 constructed state；
- prefix observation 后 conditional PDGraph 确实变化；
- \(p_s<0.5\) 不 prewarm；
- \(p_s≥0.5\) 时按 analytical trigger prewarm；
- prewarm 发生在 active-node execution window 内，而不是只在 idle；
- F4 uncovered fallback 可计数。

Hermes 是这四条里**最不担心 name laundering 的一条**。原论文确实就是 PDGraph → online refinement → Gittins → prewarm。

还有一个小细节：论文对 deadline 场景其实另有 `Hermes-DDL`，采用 worst-case LSTF：

\[
S(D,a)
=
t_{ddl}-t_{now}-(\sup X_D-a)
\]

升序排。主表如果主要优化 mean completion，就用 Gittins 没问题；如果未来专门做 deadline-miss 主结果，可以在 appendix 增加 `Hermes-DDL-adapted`，不要偷偷把 deadline term 混进 Gittins。

---

### 4. Torpor：不要做 `runtime + load + λ interference` 加权分

这一条也是需要明显修正的。

Torpor 原文根本不是一个 weighted cost scheduler。

它有三个相对清晰、分开的机制。

第一，queue priority 用的是 **RRC（required request count）**：

\[
RRC=
\frac{pn-m}{1-p}
\]

其中 \(p\) 是 tail-SLO percentile，例如论文默认实验用了 P98；RRC 越小意味着越可能通过后续 request 满足 SLO，因此优先。

第二，GPU/model placement 是明显的 lexicographic policy：

- target model 已在 available GPU → 直接执行；
- 否则如果模型在 busy GPU，优先通过 fast NVLink 做 GPU→GPU swap；
- 否则 host→GPU；
- host→GPU 时尽量选择邻居 idle / light-model 的 GPU，避免 PCIe interference。

第三，eviction：

- **light models first**
- 只剩 heavy models 后才 LRU。

所以：

> runtime + cold load + interference-risk weighted score

并不是 Torpor 的算法。

但你们这里还有一个现实问题：原版 RRC 很难干净映射。

Torpor 的 SLO 是"某个 inference function 的 tail latency percentile"，而你们的是 **workflow/job deadline**。如果硬把 job deadline 变成 function P98 SLO，会引入很多人工定义。

因此我建议不要强行伪造 RRC。

把这条明确降格成：

**`Torpor-lifecycle-projection`**

而不是含糊的 full `Torpor-adapted`。

冻结规则不使用任何 λ：

先保持 benchmark 的硬 service priority。

同 priority 内 job selection 用：

\[
FCFS
\]

不要加入 runtime-SJF，因为 Torpor 没这个机制。

选 GPU 则 lexicographic：

\[
\text{resident+available}
\succ
\text{covered low-interference load}
\succ
\text{covered higher-interference load}
\succ
\text{uncovered serial fallback}.
\]

也就是：

1. 当前 model 在 dispatchable GPU resident：优先；
2. 否则根据 F3 measured load cost + F4 measured interference 选 load path；
3. F4 未覆盖，不估一个假 interference penalty，直接执行已有 fail-closed policy；
4. eviction 时优先 evict **measured swap/interference cost 最低**的 resident model；
5. 同成本时 LRU。

这里我甚至不建议人为定义 `heavy if slowdown >10%`。

直接把原来的 binary heavy/light 思想推广成连续的：

\[
swap\_burden(m)
=
\widehat{load\_time}(m)
+
\widehat{interference\_cost}(m)
\]

然后 eviction 从 burden 最低开始。

这属于非常清楚的 adaptation，而且不需要拍脑袋的 threshold。

但是脚注一定要写：

> Retains Torpor's late binding, residency-aware model swapping, interference-aware loading, and swap-cost-aware eviction. RRC tail-SLO queueing and NVLink GPU-to-GPU swapping are unavailable in our substrate.

这样 reviewer 一看就知道你移植的是哪一部分。

如果你坚持表中只写 `Torpor-adapted`，我会认为 **name-laundering 风险中高**。写成：

**Torpor-lifecycle projection**

反而非常安全。

---

### 因此我建议最终 freeze 的四条正式定义

可以直接让 Codex 按这个版本改 manifest：

**Parrot**
> `Parrot-adapted (App-FIFO projection)`  
> serial-chain 下保留 application-centric topological/App-FIFO scheduling；service priority 第一键，随后 application arrival order。禁止 canonical remaining-work / critical-path scoring。

**QLM**
> `QLM-queue-adapted`  
> train-only `(model_id, role)` duration distributions；S=64 fixed common-random scenarios；stochastic queue assignment，以 SLO chance satisfaction 和 expected SLO penalty 为目标；Δ=0.90 固定，0.8/0.95 仅 sensitivity；load/swap 直接进入 completion time；禁止在线 same-key posterior invention。

**Hermes**
> `Hermes-PDGraph-adapted`  
> conditional PDGraph + exact Gittins rank；10 buckets；lower-G first；Pearson threshold=0.5；prewarm \(K=0.5\)，按 \(p_e=p_sP(t_c>t_s+t_p)\) analytical trigger；允许 active-node execution 中 prewarm。

**Torpor**
> `Torpor-lifecycle-projection`  
> service priority → FCFS job queue；resident-first late binding；F3/F4 measured swap/interference-aware placement；lowest swap-burden eviction first、tie LRU；uncovered interference fail-closed。明确不保留 RRC/NVLink。

---

### 关于顺序：我同意，而且再加一道 gate

你的：

> 实现 → 单测 → 机制激活审计 → 正式跑

完全正确。

我建议正式冻结成：

**implementation → unit/fidelity tests → leakage gate → mechanism-activation audit → independent code review → formal run**

其中正式跑之前必须生成一份 baseline activation report，例如：

- Parrot `app_fifo_action_flip_count`
- QLM `stochastic_reorder_count / swap_cost_affected_count`
- Hermes `gittins_vs_mean_flip / pdgraph_update / prewarm_trigger / prewarm_effective`
- Torpor `resident_hit / cold_swap / interference_aware_choice / eviction_choice / uncovered_fallback`

**不设 activation ≥10% 这种人为性能门槛。**

只要求：

\[
\boxed{\text{mechanism genuinely entered the decision path}}
\]

然后把真实 activation rate 原样报告。

这一点和此前内部方案是一致的：之前就明确提出"核心机制必须实际触发，但不要为了达到某个 action-flip 比例去调 workload"。

所以我的最终送审意见是：

**当前稿：不批准冻结。**  
**按上面四项最小修正后：批准进入实现阶段。**

最值得立刻改的是三个地方：**Parrot 去掉 critical-path；QLM 放弃 A/B 改成 SLO stochastic program；Hermes prewarm 从"idle 才触发"改成论文的在线 analytical trigger。** Torpor 则主要是改命名和删除人为 weighted score。

来源上，这次判定主要依据四篇原论文正式 PDF/作者官方材料，而不是二手综述；Parrot 还交叉核对了官方 ParrotServe scheduler 文档。QLM 和 Torpor 这次原文细节尤其改变了 manifest 结论，所以我会以这一版为准。
