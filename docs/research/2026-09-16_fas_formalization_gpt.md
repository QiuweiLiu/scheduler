# 事件链：深挖文献 + 形式化证明路线（GPT，2026-09-16）

输入：我们的 r95 结果 + 三个对照（scale-matched / tail-shuffle / load 拆分）+ 上一轮的共单调理论定位。
输出：更深的文献（含依赖不确定性的 DRO 与 2026 新理论）、四个可证明命题（含假设与可检验性）、三个"比 r95 更好"的候选方案。

---

① 深挖文献后的结论

我没有找到一篇已发表系统/调度论文与“逐步边际 VaR 相加作为在线调度分数，并与独立/CVaR 联合采样系统比较”完全重合。最接近的调度文献反而说明你们的区别在哪里：

Liu & Urgo，A branch-and-bound approach to minimise the value-at-risk of the makespan in a stochastic two-machine flow shop，IJPR 2024，正式发表；Meloni et al.，Evaluation of VaR and CVaR for the makespan in interval valued blocking job shops，IJPE 2022，正式发表；以及 Evaluation of the quantiles and superquantiles of the makespan in interval valued activity networks，Computers & OR 2023，正式发表。它们都是先得到/逼近整个 makespan 分布再计算 VaR/CVaR，而不是把各 stage 的边际 quantile 直接相加作为决策 surrogate。
Taylor & Francis Online
+2
诊断大学
+2

依赖不确定性方面，Branda，Distributionally robust fixed interval scheduling on parallel identical machines under uncertain finishing times，Computers & OR 2018，正式发表，已经让 ambiguity set 显式包含 copula dependence；Novak et al.，Distributionally robust scheduling algorithms for total flow time minimization on parallel machines using norm regularizations，EJOR 2022，正式发表，使用均值/协方差 ambiguity，但不是未知 copula 本身。Gao & Kleywegt，Distributionally Robust Stochastic Optimization with Dependence Structure，2017，仅 arXiv/Optimization Online，未查到正式期刊版本，则直接研究 Spearman rank、comonotonicity、copula-Wasserstein 等 dependence ambiguity。
科学直接
+2
科学直接
+2

理论上最重要的新参照是 De Vecchi、Nendel、Streicher，Upper Comonotonicity and Risk Aggregation Under Dependence Uncertainty，Mathematical Finance 2026，正式发表：即便只掌握很弱的正依赖信息，允许的极端尾部仍可能达到完全相关的风险水平。这非常适合为你们“为何保守尾部耦合有意义”提供现代依据。
IDEAS/RePEc
+1

Agent/LLM 方向，LLMSched，ICDCS 2025，正式发表，确实同时建模 compound workflow 的结构和 duration uncertainty，但用 Bayesian network/entropy 去优先执行“降低未来不确定性”的 stage，没有研究链内 runtime dependence aggregation。TIE，ICML 2026，正式录用，arXiv版本可查，只针对单请求输出长度设计 tail-inflated score。PBKV、FATE、SAGA 目前分别是 2026 arXiv preprint，关注未来 invocation/cache、future execution state、workflow-atomic scheduling，也没有看到你们这种相关随机和比较。
arXiv
+4
IEEE Xplore
+4
ccchengff.github.io
+4

因此目前最安全的新颖性表述仍是：

To our knowledge, we provide the first systematic study of dependence-aware aggregation of per-stage predictive uncertainty for scheduling dynamically unfolding agent workflows.

不是“首次使用 comonotonicity”。

② 建议的形式化与证明路线

设候选请求 a 的活动步骤为 R
a1
	​

,…,R
aL
	​

，总未来执行时间

S
a
	​

=
h
∑
	​

R
ah
	​

,C
α
	​

(a)=
h
∑
	​

VaR
α
	​

(R
ah
	​

).
命题 A：共单调/公共慢化因子下的精确性

假设存在公共 U∼U(0,1)，使

R
ah
	​

=F
ah
−1
	​

(U),

即同一视频/会话的“快慢状态”同时作用于所有未来步骤。那么步骤共单调，因而

VaR
α
	​

(S
a
	​

)=
h
∑
	​

VaR
α
	​

(R
ah
	​

)=C
α
	​

(a).

这不是近似，而是经典 comonotonic VaR additivity。可直接沿 Dhaene et al. 与 Cheung 的 quantile representation 证明。Kaas et al. 还证明给定边际时，共单调和在 convex order 下最大。
剑桥大学出版社
+1

**可检验：**对同一 workflow 的不同 step，测 Spearman/Kendall、90/95% co-exceedance、upper-tail dependence。若整体相关一般但尾部显著同步，A 可放宽到 C。

命题 B：真正的“保守上界”应使用 ES，而不是声称 r95 普遍上界

对任意具有同样边际分布的联合结构，

S≤
cx
	​

S
com
.

因此对 convex-order-consistent risk measure（特别是 ES/CVaR）：

ES
α
	​

(S)≤ES
α
	​

(S
com
)=
h
∑
	​

ES
α
	​

(R
h
	​

).

这是严格的 dependence-robust upper bound。
剑桥大学出版社
+1

但非常重要：

VaR
α
	​

(S)≤∑VaR
α
	​

(R
h
	​

)

一般不成立。 VaR 不具普遍次可加性，也不保持 convex order。因此论文不能把 r95 写成“所有依赖结构下的 upper bound”。

若需要任意依赖下的 VaR 上界，可用 union bound：选

α
h
	​

=1−
H
1−α
	​

,

则

VaR
α
	​

(S)≤
h
∑
	​

VaR
α
h
	​

	​

(R
h
	​

).

H=5、总体95%时需要约单步99%而非95%。这反而说明你们 r95 是有结构假设的 conservative surrogate，而非无条件鲁棒界。

命题 C：只需“上尾共单调”

Cheung 的 upper-comonotonicity 给出：若各步骤超过某些阈值后共同单调，则当 α 高于相应阈值时，VaR/TVaR/ES重新具有可加性：

VaR
α
	​

(S)=
h
∑
	​

VaR
α
	​

(R
h
	​

).

Hua & Joe 进一步证明 tail-comonotonic 情况下，α→1 时 VaR/CTE 渐近可加。
IDEAS/RePEc
+1

这可能比“整个分布共单调”更符合你们：正常运行时各节点未必相关，但同一视频特别难、GPU共同受压、同一上下文特别大时，尾部共同变慢。

**可检验：**画

∑VaR
q
	​

(R
h
	​

)
VaR
q
	​

(∑R
h
	​

)
	​


随 q=.5,.75,.9,.95,.99 的曲线。若随 q 上升趋近1，这是极强机制证据。

命题 D：调度只要求排序正确，条件更弱

你们并不需要 C
α
	​

(a) 精确等于真实延迟，只需存在所有候选共享的严格递增函数 g：

E[S
a
	​

∣x]=g(C
α
	​

(a)).

则

C
α
	​

(a)<C
α
	​

(b)⟺E[S
a
	​

]<E[S
b
	​

].

在同时 ready 的单机两-job 情况，标准 interchange argument 给出

E[C
a→b
	​

]−E[C
b→a
	​

]=E[S
a
	​

]−E[S
b
	​

],

所以这种排序足以得到最小 expected completion 的局部 SPT 决策。

你们复杂多GPU系统不满足全局定理条件，所以应把它写成 local ranking lemma / intuition，而不是声称 q95 全局最优。

你们现有 scale-match 和 tail-shuffle 已经很适合 D：它们说明不是绝对 scale，而是candidate-specific tail ordering产生价值。

③ 比 r95 更进一步的办法

我最推荐不是立刻放弃 r95，而是构造 dependence-adaptive tail aggregator。

在 dev trace 上估计步骤间 upper-tail dependence λ
ij
U
	​

，然后用一因子/t-copula产生联合 runtime，计算

VaR
.95
	​

(
h
∑
	​

R
h
	​

)

作为 score。计算量很低，H≤5 时 CPU 足够。它位于 independence 与 comonotonic 两个极端之间。

更稳健的版本是 dependence-ambiguity DRO：边际分位数固定，只约束 Kendall/Spearman/tail-dependence 落在经验置信区间内，求该 ambiguity set 中 worst-case ES。理论更漂亮，但实现明显更重；Gao–Kleywegt 和近年的 upper-comonotonic dependence-uncertainty 工作可作为理论路线。
arXiv
+1

最便宜则是做一个“upper-tail hybrid”：

C=
h
∑
	​

p50
h
	​

+β
h
∑
	​

(p95
h
	​

−p50
h
	​

),

其中 β 由实测 tail-coexceedance 决定，而不是手调。若 β≈1 且性能仍接近 r95，你们就获得一个很漂亮的结论：

r95 的有效性并非偶然 heuristic，而是数据中的 common-tail dependence 推出了接近共单调的最优消费。

这比硬证明“r95 在你们完整在线多GPU调度器上全局最优”更现实，也更容易经得住审稿。
