# Forecast-aware Scheduling：形式化问题描述 v2（2026-09-16，吸收 GPT 审稿意见）

> v1 的 4 个必修问题已按 GPT 意见修正：①随机展开图 vs 固定模板；②符号冲突（$L$）；③$J^\star$ 不是系统级 action value；④命题 A/C 未处理随机长度。
> 本版同时给出命题 A–D 的严格陈述、三级主张结构、命题 C 的检验设计，以及所有消费者的统一形式。

## 1. 记号（修正版）

| 符号 | 含义 |
|---|---|
| $j\in\mathcal{J}$ | 一个 job（一次视频 Agent 请求），到达时间 $a_j$，可选截止期 $d_j$，优先级类 $w_j$ |
| $G_j=(V_j,E_j)$ | job $j$ 的**随机展开图**：节点与依赖在运行期由 Agent 决策逐步生成 |
| $G_j^{t}$ | 时刻 $t$ 已实际展开的前缀（调度器只能观察到 $G_j^t$，看不到 $G_j\setminus G_j^t$） |
| $v\in V_j$ | 节点；lane $\ell(v)\in\{\mathrm{gpu},\mathrm{cpu}\}$；模型 $m(v)$ |
| $R_v$ | 节点 runtime（随机变量） |
| $D^{\mathrm{load}}_v$ | 冷启动加载时长（$m(v)\notin\mathrm{res}(g)$ 时发生） |
| $N_v$ | 从 $v$ **之后**起算的剩余节点数（随机变量），预测记为 $\hat N_v$ |
| $g\in\mathcal{G}$ | GPU；容量 $C_g$；驻留集合 $\mathrm{res}(g)$ |
| $s_t=(R_t,G_t,\mathrm{res}_t,\mathrm{mem}_t,Q_t,t)$ | 状态：就绪集合、GPU 忙闲、驻留、可用显存、队列、时间 |
| $\mathrm{CT}_j$ | 完成时间（$C_j-a_j$） |

**约定**：
- 预测接口给出的是"$v$ **之后**"的链（不含 $v$ 自身）；
- **CPU lane 由独立执行器自动派发，不进入 GPU 动作优化**（否则动作空间不闭合）；
- 若存在外生优先级 $w_j$，声明**所有策略首先服从同一 lexicographic 约束**（保证 FCFS/SJF 等基线不被改造）。

## 2. 真实系统目标与理想动作价值

策略 $\pi$ 在每个决策点从动作空间
$$
\mathcal{A}(s)=\{(v,g): v\ \text{ready},\ g\ \text{free},\ \text{显存可行}\}
$$
中选 $a_t$；目标是 episode 级
$$
\min_{\pi}\ \mathbb{E}\Big[\tfrac1{|\mathcal{J}|}\textstyle\sum_j \mathrm{CT}_j\Big].
$$

**系统级动作价值**（理想目标，不可直接计算）：
$$
Q^{\pi}(s,a)=\mathbb{E}\Big[\textstyle\sum_j \mathrm{CT}_j \,\Big|\, s_t=s,\ a_t=a,\ \pi\Big],\qquad
Q^{*}(s,a)=\inf_{\pi} Q^{\pi}(s,a).
$$

**澄清**：v1 里写的 $J^{\star}(a)$（当前节点 + 后续节点真实时长累加 + 简化驻留）只是
**remaining-work oracle surrogate**，用于提供"信息上界"参照；它 $\neq Q^{*}$。
论文必须区分：`OracleInfo/OracleTopo` 近似的是**信息**，不是最优策略价值。

## 3. 预测接口与消费者（统一形式）

预测器对每个就绪节点 $v$ 给出
$$
\mathcal{F}(v)=\Big(\hat N_v,\ p_v(N),\ \big\{(\hat m_h,\hat\ell_h),\ Q_\tau(R_h),\ \mathrm{occ}_h,\ Q_\tau(D^{\mathrm{load}}_h)\big\}_{h\ge1}\Big),
$$
只有**边际**分位数与长度分布，**跨步依赖结构未知**。

调度器使用代理分数 $\mathrm{Score}(a)=\hat c_{\text{current}}(v,g)+\Phi(\mathcal{F}(v),s)$。
把**所有**消费函数统一写成
$$
\boxed{\ \Phi_{\theta,\rho}(\mathcal F,s)=\rho_s\Big(\textstyle\sum_{h} I_h R_h;\ C_\theta(s)\Big),\quad I_h=\mathbf 1\{N\ge h\}\ }
$$
其中 $C_\theta$ 是假设/估计的依赖结构（copula），$\rho_s$ 是风险泛函。于是：

| 消费者 | $C_\theta$ | $\rho_s$ | 备注 |
|---|---|---|---|
| independent-CVaR | 独立 copula $\Pi$ | $\mathrm{CVaR}_{0.9}$ | 场景采样臂 |
| **r95（冠军）** | **共单调 copula $M$** | $\mathrm{VaR}_{0.95}$ | 且固定 $N=\hat N_v$ |
| p90 / p50 | $M$ | $\mathrm{VaR}_{0.9},\mathrm{VaR}_{0.5}$ | 同族不同 $\tau$ |
| dependence-adaptive | 由数据估的 $t$-copula / 上尾依赖 | $\mathrm{VaR}_{0.95}$ | Q3 的实现 |
| upper-tail hybrid | 不声称对应某 copula | $\sum Q_{0.5}+\beta(s)\sum[Q_{0.95}-Q_{0.5}]$ | 低维参数化代理 |

## 4. 命题 A–D（严格版）

**命题 A（固定长度下的精确可加性）.** 给定候选 $a$ 与**确定**长度 $n$，若边际 CDF $F_{ah}$ 连续且存在 $U\sim\mathrm U(0,1)$ 使
$R_{ah}=F_{ah}^{-1}(U),\ h=1..n$，则对任意 $\alpha\in(0,1)$
$$
\mathrm{VaR}_\alpha\Big(\textstyle\sum_{h=1}^n R_{ah}\Big)=\sum_{h=1}^n \mathrm{VaR}_\alpha(R_{ah}).
$$
*证明*：直接引用共单调 quantile representation / VaR additivity（Dhaene et al. 2002；Kaas et al. 2002）。

> ⚠️ **随机长度**：我们的 r95 用 $N=\hat N_v=\arg\max p_v(N)$，所以 A 只能证明"**条件于所消费的确定长度**"。
> 若要覆盖真实随机 $N$，令 $Y_h=\mathbf 1\{N\ge h\}R_h$，必须假设**整个 $(Y_1,\dots,Y_H)$ 共单调**；
> 仅 $R_h$ 共单调**不够**。

**命题 B（dependence-robust ES 上界；不能换成 VaR）.** 给定边际、可积随机变量，对任意联合分布
$$
\textstyle\sum_h R_h\ \le_{cx}\ \sum_h R_h^{\mathrm{com}}
\quad\Longrightarrow\quad
\mathrm{ES}_\alpha\big(\textstyle\sum_h R_h\big)\le \mathrm{ES}_\alpha\big(\textstyle\sum_h R_h^{\mathrm{com}}\big)=\sum_h \mathrm{ES}_\alpha(R_h).
$$
*证明*：Kaas et al.（共单调和是 convex-largest）+ ES 对 convex order 单调且在共单调下可加。

> **必须紧接**：该结论**不能**替换成 VaR。任意依赖下的 VaR 保守界由 union bound 给出：
> $\mathrm{VaR}_\alpha(\sum R_h)\le\sum_h \mathrm{VaR}_{\alpha_h}(R_h)$，只要 $\sum_h(1-\alpha_h)\le 1-\alpha$；等分时 $\alpha_h=1-(1-\alpha)/n$。
> （$H=5$、总体 95% ⇒ 单步约 99%。）故 **r95 是有结构假设的保守代理，不是无条件鲁棒界。**

**命题 C（上尾共单调下的高分位精确性）.** 若 $(R_1,\dots,R_n)$ 满足 Cheung 的 **upper-comonotonicity**，则存在 $\alpha_0<1$，
使所有 $\alpha>\alpha_0$ 下 VaR（以及 TVaR/ES）可加。
若仅满足 Hua–Joe 的 **tail-comonotonicity** 及相应尾域条件，结论**降级为渐近可加**：
$$
\rho(q):=\frac{Q_q(\sum_h R_h)}{\sum_h Q_q(R_h)}\ \longrightarrow\ 1,\qquad q\uparrow 1 .
$$
**不能**写成"有限 0.95 下必然严格相等"。

**命题 D（局部排序引理，非全局最优）.** 单机、两个**同时就绪**的 job $i,j$，非抢占、无 sequence-dependent setup、
总时长 $P_i,P_j$ 有限期望且不因执行顺序改变，则
$$
\mathbb E[C_i+C_j\mid i\to j]-\mathbb E[C_i+C_j\mid j\to i]=\mathbb E[P_i]-\mathbb E[P_j],
$$
故按 $\mathbb E[P]$ 升序局部最优。若存在**严格递增** $g$ 使 $\mathbb E[P_a\mid x_a]=g(\Phi(x_a))$，则按 $\Phi$ 排序与局部最优交换排序一致。
> **不要声称 r95 天然等价于 $\mathbb E[P]$**——该关系必须由经验 calibration/ranking 验证。

## 5. Q2 能证到什么程度（三级主张）

多 GPU + release times + precedence + residency/setup + eviction 下，动作 $a=(v,g)$ 会改变其他 job 的可行性与服务时间，
**不存在简单 pairwise interchange**，因此**不能**从 D 推出全局最优。论文分三级陈述：

| 层级 | 主张 |
|---|---|
| **理论** | 单机、同时就绪的 **local ranking lemma**（命题 D） |
| **机制** | 若 $\Phi$ 与真实 downstream marginal cost **保持排序**，则贪心列表调度获得更好的局部决策 |
| **系统** | 多 GPU 下的整体效果由 paired simulation（+ 未来 real replay）验证，**不声称 optimality** |

## 6. 命题 C 的检验设计

$$
\rho(q)=\frac{Q_q\big(\sum_h R_h\big)}{\sum_h Q_q(R_h)}\quad (\text{越大越接近共单调}),
$$
- $q\in\{.5,.75,.9,.925,.95,.975\}$（样本不足不要硬做 .99）；
- **cluster bootstrap 整条 workflow/video**（必须保留跨步依赖，不能逐节点 bootstrap），$B\ge 2000$；
- 报告 pointwise 95% percentile/BCa CI；可加做跨 $q$ 的 sup-norm 同时置信带；
- **不做 "$\rho=1$" 的 NHST**；用**等价检验**：预设实质等价带 $\delta$（如 5%），检验 $H_1:|\rho(q)-1|<\delta$；
  仅当高 $q$ 的 $\rho(q)$ 趋近 1 **且进入等价带**时，才支持 tail-comonotonic 机制解释。

## 7. 实证协议（不变）

1,000 episodes（700 dev / 300 frozen confirm）；配对 bootstrap B=2000；预测器冻结；接口一致；
对照：E0 / FCFS / SJF / state-aware / E2 / OracleInfo / OracleTopo / 消费者族 / CP-SAT / MPC。

## 8. 措辞纪律

- 第 5 节的 "$\succ$" 称为 **empirical dominance on the evaluated workload**，不是数学 dominance；
- 不写"r95 是任意依赖下的上界"（见命题 B）；
- 不写"r95 = 最优"；写 *local ranking lemma + empirical dominance*；
- $J^\star$ 与 $Q^\star$ 严格区分（信息 vs 策略价值）。
