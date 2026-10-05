# 架构重审（用户质疑"真值同源=透题"）— 结论：Dual-Uncertainty Action Evaluation + 信息契约

## 来源

网页版 GPT 2026-10-05：用户质疑"仿真器真值来源于实测表;调度模型拿到表=提前透题,架构不行,
要求重新提案"。GPT 裁定:**质疑对,但架构不废**。

## 裁定

> **现有 DES/substrate 合法;"BCSR 直接读取 DES 用作真值的同一张机器表"不应作为主方法的最终证据。**
> 保留为 **Oracle-Profile / perfect-system-model 上界**;主方法改为
> **Dual-Uncertainty Belief-Conditioned Action Evaluation**（工作流未来不确定 × 机器响应模型不确定）。

## 定性

- 离线 GPU profiling **不是特权信息**（LeMix/Festina/PackServe 等真实系统都用 offline profile /
  white-box latency model）；与"偷看逐任务真值"不同；workload 泄漏四硬保证仍然成立。
- 但当前 BCSR 越过"证据强度"线：ε_machine = 0（在完全已知转移模型上规划）→
  **model-in-the-loop optimism**；审稿会问"收益来自更好的 belief 消费,还是多拿了一个 oracle 机器模型"。

## 修复方案（GPT 推荐 (d)+(a)+(c),不做 (b)）

- **(d) Scheduler Information Contract（必做）**：信息分三类——I_state（所有臂共有）/
  I_workload（none→Myopic、scalar→F0、belief→New、truth→Oracle）/ I_machine（M0 无 / M1 估计 /
  M★ 精确仅 oracle）；所有实验标注 (FutureInfo, MachineInfo)。
- **(a) Machine-profile holdout（最推荐,无需新 GPU）**：原始校准有**逐轮测量**→
  measurement holdout（调度器用轮 {1,2}、执行器用轮 {3}，轮换）；或 cell holdout（显示 8/11 格、
  藏 3 格，藏格 → 方法自动退化为 F0）；画 **profile coverage → ΔJCT** 曲线。
- **(c) Calibrated perturbation（必做鲁棒性）**：用真实逐轮相对误差形成经验分布去扰动执行器真值
  （不是随便加高斯）；无重复数据的机制用 ±5/10/20% 敏感性分析并明确标注。

## 关键公平对照与新实验矩阵（6 臂）

**F0+M**（同 M̂,只用于当前动作）vs **New = F0+M + Φ_future** → New−(F0+M) 才是
"同机器信息下 action-conditioned future belief 的独立价值"。

A0 原 F0 / A1 F0+M / A2 Placement-only New（node 序同 A1,只改 GPU）/ A3 Full dual-uncertainty BCSR /
A4 Shuffled future（同机器模型、同算法、同边际,belief 跨 job 打乱）/ A5 Oracle machine（上界,保留）。

**双误差轴 2×2**：预测未来 × 机器模型（估计/oracle）→ 回答"瓶颈在 workflow predictor 还是 machine model"。

## 主表与基线

- **现有冻结主表不重跑、不改数字**：保持 Table 1（calibrated substrate 下的全系统对比）；
  新增 Table 2（信息匹配的机制/因果对照，F0 系列内）与 Table 3（机器模型鲁棒性）。
- **不要给五条基线硬加机器模型**（会破坏冻结保真度）；因果归因只在 F0 系列内做。

## 最终回答用户质疑

不是 workload leakage、不是作弊；属于 **perfect dynamics-model assumption**——
只能支持"若机器模型精确,方法有这些潜力",不能支持"真实部署也如此"。
修复不需要新 GPU 测量、不推翻主表、不重训预测器。

审核回复原文如下（未改动）。

---

炜，这个质疑是对的，但我不会判定“整个架构无效”。更准确的裁定是：

> **现有 DES/substrate 合法；但“BCSR 直接读取 DES 用作真值的同一张机器表”不应该作为论文主方法的最终证据。**
>
> 它可以保留为 **Oracle-Profile / perfect-system-model 上界**。真正的主方法应改成：**工作流未来有不确定性，机器响应模型也有不确定性。**

这反而可能把论文做得更强。

## 1. 先把“透题”问题定性清楚

### (a) 离线测 GPU 物理行为，本身不是特权信息

真实系统完全可以在部署前 profiling。现在很多系统就是这么做的：LeMix 将 offline profiling、execution prediction 和 runtime scheduling 联合起来；Festina 的 scheduler 直接查 offline profiles；2026 年 9 月的 PackServe 也用 white-box latency model 来预测 prefill/decode interference。citeturn995417academia12turn995417academia13turn995417academia14

因此：

> “scheduler 知道一张离线测出来的 GPU performance profile”

本身非常正常。

它与“scheduler 偷看 confirm300 某个 job 的真实 runtime”不是一回事。

你已有的 workload 泄漏四硬保证仍然成立。

### (b) 策略利用模拟器成本模型，也不是不允许

FATE 本身就在做 state-conditional cost estimation，然后评价当前 assignment 对 downstream state 的影响；Latency-Aware Orchestration 也预测 device-specific activation latency、memory、loading cost，再做 placement/order/lifecycle 优化。citeturn758449academia48turn758449academia49

所以“model-based planning”本身没有问题。

### (c) 但你当前 BCSR 确实越过了一条“证据强度”的线

问题不是 data leakage，而是：

\[
\hat M_{\text{scheduler}}=M_{\text{DES truth}}
\]

也就是说，在机器动态这一部分：

\[
\epsilon_{\text{machine-model}}=0
\]

于是 BCSR 实际是在一个**完全已知转移模型**上规划。

这会产生两个 reviewer 问题：

1. 现实里 co-location slowdown / loading interference 不可能永远精确等于离线 median；
2. 其他 baseline 不使用这个 exact transition model，因此 BCSR 的收益到底来自“更好的 future-belief 消费”，还是“多拿到一个 oracle machine model”？

这不是“作弊”，但属于典型的 **model-in-the-loop optimism**。

2026 的 model-based control / planning 文献本身就在强调这种 model mismatch 问题：即使规划器形式正确，不准确 dynamics 也可能导致明显的 closed-loop degradation，因此开始显式做 robust/OOD model uncertainty。citeturn753900search0turn753900search8

所以我会作如下裁定：

> **Exact-table BCSR 可以作为 Oracle-Machine 上界。**
>
> **不能作为最终主方法唯一证据。**

---

# 2. “大家都拿同一张表，新方法才使用”公平吗？

要分两个问题。

对于**系统级比较**：

> 可以。

如果论文明确声明：

> hardware profile 是公共 deployment interface，所有 scheduler 都允许访问；某 baseline 的原算法不使用该信息。

那么并不存在秘密特权。

现实系统也不会因为 FCFS 不使用 GPU profile，就禁止新的 scheduler 使用 profiling。

但是对于你要证明的：

> “我们的 future-belief action evaluation 本身更好”

仅仅说“所有人理论上都能拿表”就**不够**。

因为 reviewer 可以说：

> F0 根本没有这个消费接口，你只是给 New 多了一种信息。

因此必须增加一个**信息匹配 control**。

这和你之前的 F0/Myopic 信息纪律完全一样。

---

# 3. 我建议直接建立一个正式的 Scheduler Information Contract

以后不要再模糊说“scheduler 能看到什么”。

明确分成三类：

\[
\mathcal I =
(\mathcal I_{\rm state},
 \mathcal I_{\rm workload},
 \mathcal I_{\rm machine})
\]

其中：

**\(\mathcal I_{\rm state}\)**：所有人都有

- GPU busy/idle
- residency
- free memory
- ready queue
- 当前 token phase
- 当前已经发生的执行历史

**\(\mathcal I_{\rm workload}\)**：

- none：Myopic
- scalar H5：F0
- probabilistic future belief：New
- truth：Oracle

**\(\mathcal I_{\rm machine}\)**：

- M0：无 profile / 仅静态物理信息
- M1：**estimated profile**
- M★：完整精确 DES profile，仅作 oracle

这样你的实验以后全部可以标：

\[
(\text{FutureInfo},\text{MachineInfo})
\]

这个非常重要。

最近的 DynaSchedBench 甚至专门指出 dynamic scheduling benchmark 中 observability 本身会明显改变评价结果，因此把可见信息固定下来是必要的实验纪律。citeturn713500search7

---

# 4. 新提案：Dual-Uncertainty Action Evaluation

这是我现在最推荐的新架构。

把原来的：

\[
\Phi(a)
=
E_{Z\sim b}
[C_{M^\star}(a,Z)]
\]

改成：

\[
\boxed{
\Phi(a)
=
E_{Z\sim b_{\rm workflow}}
[
C_{\hat M}(a,Z)
]
}
\]

这里出现两个完全不同的不确定性：

### Workflow uncertainty

你已经有：

\[
b_{\rm workflow}
=
P(\text{unrevealed future}\mid prefix)
\]

包括 model / role / length。

### Machine uncertainty

真实环境是：

\[
M^\star
\]

但 scheduler 只拥有：

\[
\hat M\neq M^\star
\]

所以真正的问题变成：

> **Can probabilistic future information still improve scheduling when both the future workflow and the machine response model are imperfect?**

这个故事比原来的 BCSR 更强。

甚至后期可以加 robust version：

\[
\Phi_{\rm robust}(a)
=
\sup_{M\in\mathcal U(\hat M)}
E_Z[C_M(a,Z)]
\]

其中 \(\mathcal U\) 是机器 profile uncertainty set。

2026 的 learning-augmented algorithms 越来越强调这种“预测有用时利用，错时保持 robustness”的设计，而不是假设 prediction/model 永远正确。citeturn713500search0turn713500search2

但我建议第一版**先别实现 robust optimization**。

先解决实验公平性。

---

# 5. 四种修复方案怎么选

| 方案 | 我的评价 | 成本 | 主表要重跑？ | 能支持什么主张 | 最大问题 |
|---|---|---:|---|---|---|
| **(a) Machine-profile holdout** | **最推荐** | 中 | 否 | 对未见/不完整 machine profile 仍有效 | 23 个 interaction cells 比较少 |
| **(b) Online estimation** | 有潜力但暂缓 | 大 | 否 | unknown machine dynamics 下在线学习 | 会把论文扩大成 bandit/online learning |
| **(c) Machine-model perturbation** | 必做 robustness | 小-中 | 否 | 对 profile error 不敏感 | 随便加 Gaussian noise 会很假 |
| **(d) Information contract** | **必须做** | 极小 | 否 | 公平性和声明边界清楚 | 本身不是算法创新 |

我的方案是：

\[
\boxed{(d)+(a)+(c)}
\]

先不做 (b)。

---

# 6. Machine holdout 应该怎么做

这里甚至不需要重新开 GPU。

你的 profile 不是凭空得到的，而是有真实测量原始记录。

最理想情况，如果现有 raw calibration 保存了每个 cell 的多轮测量，那么可以做两种实验。

### 第一种：measurement holdout

例如某个 interaction cell 有三轮：

\[
x_1,x_2,x_3
\]

scheduler machine model 只使用：

\[
\hat M=\mathrm{aggregate}(x_1,x_2)
\]

DES evaluation 使用：

\[
M^\star=x_3
\]

轮换三次：

- train 12 → truth 3
- train 13 → truth 2
- train 23 → truth 1

这样非常漂亮：

**方法从未看到执行器使用的那个 measurement realization。**

而且机器误差不是人为编的。

如果某些 mechanism 没有逐轮 raw measurements，就不要假装有。只对具备重复测量的 co-location/interference 部分做这个实验。

### 第二种：cell holdout

例如 11 个 co-location cells：

- scheduler profile 只显示其中 8 个；
- 3 个隐藏；
- executor 仍使用全部 11 个真实值。

隐藏 cell 不能偷偷补真值。

只能：

- 用同 model-pair 的 train-only fallback；
- global median；
- conservative unknown；
- 或直接“不利用该 future-cost 项”。

我个人甚至更喜欢最后一个：

> **unknown cell → BCSR 不声称知道，退化为 F0。**

非常干净。

这样你可以画：

\[
\text{profile coverage}
=
0,25,50,75,100\%
\]

对：

\[
\Delta JCT
\]

的曲线。

这比只给 perfect-profile 一个点可信很多。

---

# 7. 不要直接“随机加噪声”，要做 calibrated perturbation

(c) 也值得做，但必须有依据。

最好从真实重复 measurement 得到 relative error：

\[
e_c=
\frac{x_c-\operatorname{median}(x_c)}
{\operatorname{median}(x_c)}
\]

形成 empirical error distribution。

然后 DES truth 采样：

\[
M^\star_c
=
\hat M_c(1+\epsilon_c)
\]

其中：

\[
\epsilon_c\sim\hat P_{\rm empirical}
\]

这叫：

> **profile uncertainty stress test**

比随便说：

> ±10%、±20%

强得多。

如果某些机制确实没有重复数据，再补 deterministic sensitivity：

\[
\pm5\%,\pm10\%,\pm20\%
\]

但明确写成 sensitivity analysis，而不是现实噪声模型。

---

# 8. 在线估计为什么现在不推荐

这个想法本身很好：

scheduler 初始不知道：

\[
C(m_i,r_i,m_j,r_j)
\]

执行过以后才观察结果：

\[
y_t=C+\epsilon
\]

不断更新：

\[
\hat C_{t+1}
\]

这会把项目变成：

> **learning-augmented scheduling with online system identification**

理论上很漂亮。

但它会马上带来：

- exploration vs exploitation；
- counterfactual cell 没观察到怎么办；
- online estimator 怎么初始化；
- 是否主动牺牲当前性能去 probe；
- adaptation speed；
- machine drift。

这几乎是另一篇论文。

而且 2026 已经有不少 work 开始研究 unknown system dynamics 与 model mismatch 下的 policy optimization。citeturn753900search8turn753900search9

所以：

**不要为了修一个公平性问题，把第一篇论文扩成 online system identification。**

可以放 Future Work。

---

# 9. 最关键的公平性对照：加入 Machine-Aware F0

这是新提案里最重要的实验。

不要比较：

\[
F0
\quad vs\quad
BCSR+\hat M
\]

直接这样比还是有问题。

必须加入：

\[
\boxed{F0+M}
\]

它拥有和 New **完全一样的 machine model \(\hat M\)**。

但它只用于当前动作：

\[
S_{F0+M}(i,g)
=
C_{\rm current}^{\hat M}(i,g)
+
F_{p95}(i)
\]

没有：

\[
E[\text{future substrate consequence}]
\]

然后 New：

\[
S_{\rm New}(i,g)
=
S_{F0+M}(i,g)
+
\Phi_{\rm future}^{\hat M}(i,g)
\]

这样：

\[
New-(F0+M)
\]

才真正回答：

> **在 machine information 完全一样时，把 unrevealed future belief 用来评价未来 action consequences 有没有额外价值？**

这条证据非常关键。

---

# 10. 我建议新的最小实验矩阵

不要一次跑十几个复杂 scheduler。

先做 6 个：

### A0 — 原始 F0

当前冻结版本。

机器 profile 只通过所有人共享的 substrate admission 使用，不作为 score input。

### A1 — F0 + Estimated Machine

与 New 拿完全相同的：

\[
\hat M
\]

但只改善**当前** `(node,GPU)` action cost。

无 future substrate rollout。

### A2 — Placement-only New

node order 与 A1 **完全相同**。

只允许：

\[
g
\]

由 future belief 改变。

这是最干净的 action-value 证据。

### A3 — Full Dual-Uncertainty BCSR

联合：

\[
(node,GPU)
\]

评价。

### A4 — Shuffled Future

和 A3：

- 相同 machine model；
- 相同算法；
- 相同 probability marginals；

但未来 belief 在 job 之间 shuffle。

验证收益真来自 prefix-specific future。

### A5 — Oracle Machine

\[
\hat M=M^\star
\]

就是你原先想做的 BCSR。

**明确标 Upper Bound。**

这样 perfect-table 版本不需要删掉，反而变得非常有价值：

\[
A5-A3
\]

直接告诉你：

> 还有多少性能损失来自 machine model error。

---

# 11. 更漂亮的是你会得到两个独立的 Oracle gap

你原来已经有：

\[
PredictedFuture\rightarrow TruthFuture
\]

现在再增加：

\[
EstimatedMachine\rightarrow OracleMachine
\]

于是整个研究有两个误差轴：

\[
\text{Workflow prediction error}
\]

和：

\[
\text{Machine model error}
\]

可以形成一个 2×2：

| | Estimated machine | Oracle machine |
|---|---:|---:|
| Predicted future | **实际方法** | machine ceiling |
| Oracle future | predictor ceiling | full oracle |

于是你可以真正回答：

> 当前系统瓶颈究竟在 workflow predictor，还是 machine model？

这个实验非常有研究价值。

甚至比原来“p95 vs oracle”更有解释力。

---

# 12. 现有主表要不要重跑？

我的结论是：

## **不需要。**

不要动已经冻结的主表。

因为主表回答的问题仍然有效：

> 在同一个 calibrated DES substrate 下，现有六个完整系统谁表现怎样？

它并没有使用未来 BCSR，也没有因为现在的新设计突然产生数据泄漏。

保持：

> **Table 1 — Frozen full-system comparison under calibrated substrate**

即可。

然后新增：

> **Table 2 — Information-matched mechanistic evaluation**

这里跑：

- F0
- F0+M
- Placement-New
- Full-New
- shuffled
- Oracle-M

再新增：

> **Figure / Table 3 — machine-model robustness**

例如：

- profile coverage；
- measurement holdout；
- profile perturbation。

这样完全不推翻以前工作。

---

# 13. 是否还要把五条主表 baseline 全部重新给 machine model？

**我不建议。**

原因很简单：

Parrot、QLM、LLMSched、Hermes、Torpor 有自己的冻结 adaptation。

为了所谓“公平”硬给它们改一个 machine-model consumer：

> 已经不再是原来的 faithful baseline。

反而会破坏你刚刚非常辛苦 freeze 下来的 baseline fidelity。

正确做法是区分两种证据：

**Table 1：complete-system external comparison**

保持五 baseline 不动。

**Table 2：matched-information causal comparison**

围绕 F0 系列进行：

\[
F0
\rightarrow F0+M
\rightarrow New
\]

这样才能严格归因。

如果论文中写：

> “New 优于 Hermes，因为我们的 action evaluation 更好”

不够严谨。

但可以写：

> “New achieves the best end-to-end result among evaluated systems; the matched F0 controls further isolate the gain attributable to belief-conditioned action evaluation.”

前半句 full-system。

后半句 causal attribution。

不要混。

---

# 14. 对你原始质疑的最终回答

> “仿真器里面的真值来自我们测出来的表；调度模型拿表，相当于透题。”

我会改成一个更精确的判断：

**不是 workload leakage，也不等于作弊。**

它属于：

> **perfect dynamics-model assumption。**

这在 simulation/model-based planning 中可以接受，但只能支持：

> “如果 machine response model 精确，方法有这些潜力。”

不能直接支持：

> “真实部署中也会得到这些收益。”

而你的论文完全没必要承担这个弱点，因为你已经有足够资产把它修掉，而且**不用重新测 GPU，不用推翻主表，不用重训 predictor。**

---

# 最终执行建议

我建议正式采用：

\[
\boxed{\textbf{Dual-Uncertainty Belief-Conditioned Action Evaluation}}
\]

架构层面彻底分离：

\[
\boxed{
M^\star_{\rm executor}
\neq
\hat M_{\rm scheduler}
}
\]

并实施：

**第一优先：** 信息接口 contract。  
**第二优先：** `F0 → F0+M → Placement-New → Full-New` matched controls。  
**第三优先：** machine-profile holdout。  
**第四优先：** empirically calibrated machine-error stress test。  
**Oracle exact-table BCSR：** 保留，但降级为 ceiling。

**现有冻结主表：不重跑、不修改数字。**

如果这套结果最后呈现为：

\[
F0+M < F0,
\]

说明 machine model 自己有价值；

\[
New < F0+M,
\]

说明 **future belief 的 action-conditioned 消费有独立价值**；

并且：

\[
New_{\rm estimated}
\approx New_{\rm oracle-machine},
\]

那会是最漂亮的结果——说明你的方法不是“解模拟器”，而是真正在利用可泛化的 future information。

即使最后发现 Oracle-M 很强、Estimated-M 收益大量消失，这个实验也极其重要：它会及时告诉我们 **BCSR 不适合作为主创新**，避免把后续论文建立在一个 perfect-model artifact 上。