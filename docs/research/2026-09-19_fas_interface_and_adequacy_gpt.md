# 接口兼容 / 分布怎么用 / 5-6秒够不够 + 近三年文献 — 对话 6aae601b

## 回答

结�?先�?在前�?��**R1b 已经�?���?��为�?�明显更好的 runtime distribution predictor”，但还不能仅凭�?0 秒→5�? 秒�?�判�?���?��已经“解决了调度�??”�??*近三年的系统�?decision-focused literature 没有给出类似“相对�?�?���?40% 就足够调度�?�的通用阈�?�；更一致的做法�?�� prediction improvement �?终关�?��多少 downstream decision gap。MLSys 2023 �?Azure lifetime scheduling 甚至专门把调度算法�?计成对不完美预测鲁�?；OSDI 2024 �?PCS 也直接研�?predictability、�?�能和公平�?�的 Pareto 权衡，�?�不�??�??测�?�?��值�?�\nMLSys Proceedings
+1

因�?，我建�? Phase R 现在先停止继�??练资源头，把 R1b 接入 scheduler，保留旧 Σq95 �?��，同时�?�?ΣE[T] 风险�??�臂和一�?���?��名的 tail-risk 臂�??00 �?paired v03 episodes 足以�?smoke；真正决定�?��?不�?”的，是它关�?��多少 J3→oracle 的调度�?�能缺口。\n
Q1. “真�?10 秒，现在�?��预测 5�? 秒，够用吗？”\n判定：UNVERIFIED。�?测上已经显著改善；调度上够不够，�?��没有证据。\n
不能直接从：

10.4s�?.6s

推出“不能用”�?�\n
也不能从 RuntimeQScore �?1% 推出“已经�?用了”�?�\n
原因�??测�?�?�� decision value 的映射一�?��非线性�?�任务相关的。Decision-Focused Learning 的近年综述和理�?工作都强调，�???prediction loss �?downstream decision regret 并不�?�?对应；Perdomo 2024 更直接研究了 prediction improvement 到最�?welfare/value 的关系�?�\nML Anthology
+1

在系统�?域，这一点更加明显�?�Barbalho et al. �?Azure VM placement �?���?���?runtime/lifetime �?��高度�?��，因此调度算法需要�? prediction errors 鲁�?，�?�不�?�� predictor“足够精�??�后才使用�?�LAVA 进一步改�?lifetime distribution + reprediction，在 Google hyperscale cloud �?��际部署并获得资源效率收益。\nMLSys Proceedings
+1

�?以，�?VideoSeek，�?��?不�?”的正确量不�?��

�?nT
^
−T�?n
�?��，�?�应首先看冻结调度目标：

J(scheduler;
D
^
T
	​\n
)

其中 J 就是你们 v03 原来已经使用�?makespan / flow-time 类目标�?�\n
我建�??外报告一�?��常直观�?�但�?��为解释�?�诊�?��量：

OracleGapClosed=
J
J3
	​\n
−J
oracle
	​\n
J
J3
	​\n
−J
new
	​\n
	​\n
	​\n

对于越小越好的目标�?�\n
比�? J3 flow time=100，R1b=94，perfect-runtime oracle=88：\n
100�?8
100�?4
	​\n
=50%

这比说�??.6 秒还�?���?46%”更直接回答负责人：

新资源�?测器已经解决了原资源�?���?造成调度损失的�?少？

这不�?���?��的优化目标，�?��把现�?scheduler objective �?oracle gap 归一化�?�\n
怎样判断 accuracy 已经进入“收益饱和区”？

你们�?��的实验最�?��：\n
固定同一�?scheduler consumer（�?先固�?Σq95），画：

predictor quality→scheduler objective

现有就有四个�?��点：

J3 �?R1b �?R3a-U �?perfect-runtime oracle

x 轴可以放 RuntimeQScore，y 轴放�?v03 flow time/makespan。\n
如果：\n
J3→R1b prediction 大幅改善，scheduler 也明显改善；

R1b→更准模型几乎不再改�?scheduler；\n
oracle 相�? R1b 也只剩很小空间；

才有资格说：

对于当前 scheduler，runtime prediction 已基�??用�?�\n
反过来，如果 perfect-runtime oracle 仍显著优�?R1b：\n
调度�??没有完全解决，resource information 仍有价�?��?�\n
这一判断和近年来 PtO/DFL 文献的�?�想�?��致的：最终应�?decision regret/value，�?�不�??找一�?��任务通用�?prediction-error threshold。\nNeurIPS 会�?论文集\n+2
Proceedings of Machine Learning Research
+2

Q2. 3 �?��位数变成 16-bin distribution，接口�?�么兼�?？\n判定：INFERENCE，文�?��持�?�保�?richer uncertainty representation、下游�?�择风险泛函”的原则，但没有论文规定你们该用�?�?JSON/artifact schema。\n
因�?这里应�?区分：\n
文献�?��的架构原则，和\n你们�?��的工程接口决策�?�\n
NeurIPS 2023 �?Predict-then-Calibrate 明确提出�?prediction �?downstream risk/robustness 解�?�：先生成�?�?不确定�?�表示，再由下游根据风险�?求使�?��LAVA 则是很直接的 systems 例子—�?�scheduler 利用 learned lifetime distributions，�?�不�?��依赖�?�?point prediction。\nNeurIPS 会�?论文集\n+1

我的建�?不是“full distribution �?functionals 二�?�一”�?�\n
而是：\n
完整分布作为 source of truth+固定函数值作为 canonical views
	​\n

也就�?? artifact 同时存两层�?�\n
字�?	建�?
artifact_schema_version	必须
producer_checkpoint_sha	0ee8... / �?head ID
resource_head_version	J3, R1b-16bin 等\nbin_schema_id	比�? mass16_v1
bin_edges_ms	16-bin 的全部边界\nbin_representatives_ms	计算 mean 等使用的代表值\nruntime_probs[H,K]	完整概率分布，核�?source of truth
q50_ms[H]	canonical derived view
q90_ms[H]	同上
q95_ms[H]	必须保留
mean_ms[H]	E[T]
cvar95_ms[H]	�?��定义�?��后才存\nquantile_rule	CDF inversion / bin interpolation 规则
overflow_rule	�?后一格�?何�?理\ncalibration_version	有无后校准及版本
runtime_semantics	unconditional / conditional-on-active
consumer_compat	�?��容哪�?scheduler adapter

为什么不能只存：

q50/q90/q95/E[T]

？\n
因为那会把这次好不�?易获得的 full distribution 又在接口边界上扔掉�?�以后想测：

CVaR；\n
tail probability；\n
deadline exceedance；\n
scenario sampling；\n
就必须重新跑 predictor。\n
反过来，如果�?�� 16 �??率，也会�?reproducibility 风险：\n
不同版本 scheduler �?��采用不同：\n
quantile interpolation；\n
overflow value；\n
bin representative；\n
CVaR 实现；\n
于是同一�?artifact 得到不同 q95。\n
�?�?canonical derived values 也一起固化�?�\n
Fluxion �?NSDI 2023 从系统�?度强调了 modular component + consistent interface，使模型部件�?���?��更新和组合；虽然它没有�?定�?率分�?schema，但与你这里“producer �?scheduler adapter 解�?��?�的设�?原则高度�?致�?�\nUSENIX

q95 �?��必须保留吗？
判定：VERIFIED：必须保留，至少作为 compatibility baseline。\n
建�?明确形成 scheduler adapter：\n
legacy_sum_q95_v1

定义永远固定为：

C
j
q95
	​\n
=
h
∑\n	​\n
q
0.95,j,h
	​\n

这样就可以做到真�?apples-to-apples：\n
J3 + legacy_sum_q95_v1

vs.

R1b + legacy_sum_q95_v1

变化�?�� predictor。\n
否则如果同时：\n
J3:q95 �?R1b:mean

然后 scheduler 改善了，你根�?��法区分到底是：\n
predictor 变好了；

还是 risk functional 变了。\n
�?有调度实验结果里再�?录：

predictor_artifact_id

consumer_policy_id

例�?：\n
R1b_mass16_seed11 / legacy_sum_q95_v1

以后接口升级也不会污染历史结果�?�\n
Q3. 16-bin distribution 到底应�?怎么�?��

这里先指出一�?��常重要的限制：\n
你现在�?测的�?���?步的 marginal distribution，不�?���?H �?runtime �?joint distribution。\n
因�?你已经能精确得到：\n
E[
h
∑\n	​\n
T
h
	​\n
]=
h
∑\n	​\n
E[T
h
	​\n
]

不需要任�?independence assumption。\n
但不能仅�?marginal distributions 得到：\n
Q
.95
	​\n
(
h
∑\n	​\n
T
h
	​\n
)

或�?�：

CVaR
.95
	​\n
(
h
∑\n	​\n
T
h
	​\n
)

除非再指�?future-slot dependence。\n
这决定了下面的优先级。\n
当前 VideoSeek，我会这样排序\n优先�?t消费方式	我的判断	适用条件 / 失效模式
1	(a) ΣE[T]	当前�?��新基线\t�?expected flow/makespan �?�?��；严格可加；但忽�?tail/deadline risk
2	(c) system-level CVaR	�?值得发展的�?险版本\t适合尾部失败成本高；但需�?joint/scenarios
3	(b) Σq95	必须保留�?legacy risk heuristic	�?单�?�历史兼容；但不�?total-runtime q95
4	(d) scenarios + stochastic/DRO planning	full distribution �?充分的长期用法\t�??理�?�?尾部；但计算贵且必须处理�?slot 相关�?n5	(f) decision-focused learning	后续�?��方向，不�?���?consumer API	预测准确度与决策价�?�明显脱钩时值得用\n6	(e) distribution distance 直接�?cost	�?��不推荐\t除非 scheduler objective �?��就定义为某�? distributional discrepancy/DRO

下面分别说原因�?�\n
(a) ΣE[T]

VERIFIED：从数�?接口角度，它�?��现在�?干净的新 consumer。\n
因为：\n
E[T
1
	​\n
+�?T
H
	​\n
]=
h
∑\n	​\n
E[T
h
	​\n
]

不�?求独立�?�\n
这�?好利用了你现�?��

ΣE[T]/ΣT=0.996

这个很强�?aggregate calibration 结果。\n
如果 scheduler �?��主�?�?expected mean flow time / expected completion cost，它�?��值得跑的 arm。\n
但它�?failure mode 很明�?��

heavy-tail �?expected cost 很准，不代表 deadline/tail-risk 很安全�?�\n
因�?它不应�?直接替代 q95，�?�应该与 q95 对照。\n
DOTE 的结果是�?�?��强的邻域证据：在预测�?终用于优化时，与其追求单�??测量的完美准�?��有时直接利用 stochastic structure �?decision optimization 更有效�?�\nUSENIX

(b) Σq95

VERIFIED：可作为 heuristic，但不能解释成�?��?��?�时 95% 上界”�?�\n
�?�?��

Q
.95
	​\n
(X+Y)
\n=Q
.95
	​\n
(X)+Q
.95
	​\n
(Y)

甚至 VaR/quantile �?��不具有一�?�� subadditivity。\n
�?以：

h
∑\n	​\n
q
.95,h
	​\n

�?好永久改称：

legacy scheduler risk score

而不要叫：\n
“future total q95”�?�\n
Meloni & Pranzo 2023 专门研究不确�?activity network �?makespan �?���?quantile �?superquantile；这恰好说明系统级�?险应该作用在 aggregate random variable 上，而非默�?把各 component quantile 相加。\n科�?直�?�车

不过现实 systems �?��实存在用�?quantile �?conservative runtime scheduling 的做法�??026 �?UARP �?HPC runtime prediction �?���?q50、q99 �?adaptive safety margin，并在四�?���?HPC 数据集上评价 scheduler success/utilization。它�?��“tail quantile 对调度有用�?�，但并不证明你�?Σq95 �??�?�� total risk。\nSpringer

(c) CVaR / Expected Shortfall

VERIFIED：比 VaR/q95 更�?�合作为明确�?tail-risk objective。\n
CVaR：\n
CVaR
α
	​\n
(T)=E[T�? 位于�?坏 α tail]

不仅�?��

�?5% 分位在哪？�?�\n
还问：\n
“一旦进入最�?5%，究竟有多慢？�?�\n
这与你的 10s+ heavy tail 很�?合�?�\n
而且 CVaR �?coherent risk measure，具�?subadditivity，所以：

CVaR
α
	​\n
(
h
∑\n	​\n
T
h
	​\n
)�?nh
∑\n	​\n
CVaR
α
	​\n
(T
h
	​\n
)

因�?如果你暂时只�?marginals：\n
h
∑\n	​\n
CVaR
.95
	​\n
(T
h
	​\n
)

至少有清楚的“保守上界�?�解释�?��?�但�?��非常保守。\n
生产/调度邻域�?��2023 �?Operations Research �?Wang & Yao �?CVaR 明确管理 downside risk；Meloni & Pranzo 2023 则直接针�?uncertain scheduling makespan �?quantile/superquantile。\nPubsOnLine
+1

因�?长期来看，我更喜�?��

CVaR
.95
	​\n
(
h
∑\n	​\n
T
h
	​\n
)
	​\n

而不�?��

h
∑\n	​\n
q
.95,h
	​\n

但前者需�?joint distribution/scenarios。\n
(d) Scenario sampling + stochastic / robust optimization

INFERENCE：这�?���?16-bin distribution �?有潜力的�?终用途�?�\n
例�? sample：\n
T
j,1:H
(s)
	​\n
∼P(T
j,1:H
	​\n
�?)

然后 scheduler 根据场景计算：\n
E
s
	​\n
[J(a,T
(s)
)]

或：

CVaR
α
	​\n
(J)

甚至 DRO：\n
a
min
	​\n
P∈U
max
	​\n
E
P
	​\n
[J(a,T)].

2023 �?Omega �?Yin et al. 直接研究 Wasserstein distributionally robust parallel-machine scheduling，并报告考虑 processing-time distribution uncertainty 的价值；Predict-then-Calibrate 则从理�?上把 calibrated uncertainty �?risk-sensitive/DRO optimization 接起来�?�\n科�?直�?�车
+1

但现在不能简单独立抽每个 slot：\n
T
1
	​\n
�?
2
	​\n
⊥⋯

因为同一 workflow 的未来节点共�?��

task complexity；\n
model；\n
video；\n
contention state；\n
相关性很�?��明显。\n
�?�?scenario planning �?Phase R 后续�?��，�?�不�?���?smoke 的�?�?步�?�\n
LAVA 2025 很�?�得你重点看：它不是�?��出一�?lifetime point，�?�是明确使用 learned distributions，并通过 runtime reprediction �?misprediction �??�应；这�?��你�?�完整资源分布到底有�?么用”最接近的近�?systems precedent。\nMLSys Proceedings

(e) Distribution distance / quantile-regression-style cost

INFERENCE：不适合作为你现在直接�?�任务的 scalar score。\n
Wasserstein distance 等在 DRO �?��有价值，但�?�常用于：\n
定义 ambiguity set；\n
而不�?��

�?�� job �?runtime distribution 距�?某个固定 distribution 更近就先执�?。\n
除非你们先定义明�?�� target distribution / SLA distribution，否则它缺少业务�?��。\n
(f) Decision-focused cost

VERIFIED：理论上成立；现在不�?��先级�?高�?�\n
近年�?DFL 直接把：

prediction→optimizer→decision loss

放到�?���?��，�?�不�?��化独立的 MAE。NeurIPS 2024 �?Gupta & Huang、ICML 2023 �?Maximum Optimality Margin、以�?JAIR 2024 的系统综述都�?��前较权威的入口�?�\nNeurIPS 会�?论文集\n+2
Proceedings of Machine Learning Research
+2

但你�?R1b 现在�?��次同时：

RuntimeQScore �?1%；\n
calibration 改善；\n
aggregate mean 几乎无偏。\n
此时还没证明传统 predict-then-optimize 失败。\n
�?以先�?scheduler，别急着 DFL。\n
Q4. 300 paired episodes 的最小可�?smoke

你的 300 �?episode 思路�??的�?�我建�?把它做成�?�?���?factorial experiment，�?�不�??? �?artifact 随便各跑�?下�?��?�\n
我建�?6 �?arm
Arm	Predictor	Consumer	回答的问题\nA0	J3	Σq95	历史 baseline
A1	R1b	Σq95	�?predictor effect
A2	R3a-U	Σq95	解冻带来的�?外效果；同时�?calibration-fail control
B1	R1b	ΣE[T]	�?consumer effect：risk-neutral
B2	R1b	ΣCVaR95(T_h)	conservative tail consumer
O	perfect runtime	oracle consumer	headroom reference，不�?��部署模型

同一批：

300

�?v03 episodes 全部�?��跑�?�\n
也就�?1800 �?simulation，但真�?的独�?workload 仍然�?�� 300 �?��因�?比较完全 paired。\n
如果你们 16-bin �?overflow �?��没有�?�?��信的 finite tail representative，那么：

先删�?B2。\n
不�?为了�?CVaR �??�一�?tail mean。\n
这几�?contrast 已经把问题拆�?

�??测改进：

A1−A0

R3a �?��值得：\n
A2−A1

mean vs legacy risk policy：\n
B1−A1

CVaR vs legacy risk policy：\n
B2−A1

而：

O−A0/O−A1

回答“还剩�?�?prediction headroom”�?�\n
Primary metric 不�?重新发明
判定：VERIFIED。\n
直接使用你们�?v03 scheduler 已经固定�?primary objective。\n
如果之前 primary �?mean flow time：\n
就用 mean flow time。\n
如果之前 primary �?makespan：\n
就用 makespan。\n
如果两个以前就是 co-primary：\n
两个都保持�?�\n
不�?因为 Phase R 方便突然换成：\n
resource regret；\n
Spearman；\n
new weighted score。\n
OracleGapClosed、decision disagreement 等只能做 diagnostic。\n
这一原则�?DFL/PtO 文献高度�?致：评价�?终方案应看原 decision objective/regret，�?�不�?��了新 predictor 再�?�一�?��理�?�\nML Anthology
+1

smoke 的判据\n
我建�??先冻结三�?��次�?�\n
�?��层：R1b �?���?��。\n
比较：\n
A1vsA0

要求：\n
�?scheduler primary 至少 non-inferior；\n
calibration/integrity �?PASS；\n
paired estimate �?CI 全部报告。\n
如果明显改善：\n
R1b 已具�?downstream value。\n
�?��层：5�? 秒是否�?�已经�?用�?��?�\n
�?oracle headroom：\n
J
A1
	​\n
−J
O
	​\n

如果 R1b 已经关闭绝大部分�?���?��调度 gap，那�?predictor �?��停�?继续优化。\n
如果 oracle 仍远好于 R1b：\n
“当时的�??没有完全解决”这�?��疑成立�?�\n
你们要继�?���?runtime information。\n
这里“绝大部分�?�具体是多少，没�?2023�?026 文献能替你们定；�?要用项目已有�?minimum practically important difference。�?果历史上从未定义，就必须在看�?300 �?��果之前由负责人写�?protocol。\n
�?��层：分布应�?怎么消费。\n
比较：\n
B1/B2vsA1

如果 ΣE 明显优于 legacy q95：\n
当前 q95 heuristic �?���?�?��错位。\n
如果 CVaR 更好：\n
tail-risk �?��要决策因素�?�\n
如果三�?�基�?��样：

scheduler �?distribution functional 不敏感，那么 full distribution 当前主�?价�?�在 prediction/calibration 和未来扩展，而非即时调度收益。\n
300 episodes �?�� smoke。�?�?paired CI 仍�?盖有实际意义的�?负变化，不能写�?�等价�?�；此时再用原有 full v03 benchmark protocol做确认，而不�?���?�� predictor。\n
Q5. 2023�?026 �?��用文献\n
下面我只放我核实到�?�?venue 的�?文；其中少数不是顶会，但与�?�runtime distribution �?scheduling”直接相关，我会标明。\n
分类	文献	对你�?��直接用�?�\n(i)	Mandi et al., “Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities,�?JAIR, 2024.	当前 DFL �?完整的综述之�?；支撑�?�prediction accuracy �?decision quality 必须分开评价”�?�\nML Anthology

(i)	Chunlin Sun, Shang Liu, Xiaocheng Li, “Maximum Optimality Margin: A Unified Approach for Contextual Linear Programming and Inverse Linear Programming,�?ICML, 2023.	�?downstream optimality structure 设�?预测 loss，支持后�?decision-aware �?��。\nProceedings of Machine Learning Research

(i)/(iii)	Chunlin Sun, Linyu Liu, Xiaocheng Li, “Predict-then-Calibrate: A New Perspective of Robust Contextual LP,�?NeurIPS, 2023.	与你�?��口�?计最相关：prediction �?risk/calibration 解�?�，再交�?robust/risk-sensitive optimizer。\nNeurIPS 会�?论文集\n
(i)	Vishal Gupta, Michael Huang, “Decision-Focused Learning with Directional Gradients,�?NeurIPS, 2024.	直接优化 downstream decision loss 的近期理论方法；用于�?�� DFL �?��。\nNeurIPS 会�?论文集\n
(i)	Sanket Shah, Bryan Wilder, Andrew Perrault, Milind Tambe, “Leaving the Nest: Going beyond Local Loss Functions for Predict-Then-Optimize,�?AAAI, 2024.	说明 PtO �?task-specific loss �?��也是学习/设�?�??，�?�普通�?测�?�?���?定�?齐决策�?�\nAAAI Publications

(i)	Alexander S. Estes, Jean-Philippe P. Richard, “Smart Predict-then-Optimize for Two-Stage Linear Programs with Side Information,�?INFORMS Journal on Optimization, 2023.	比较�???regression loss �?downstream objective-oriented training；直接支持�?�以决策价�?��?判�?��?�\nPubsOnLine

(iv)	Juan Carlos Perdomo, “The Relative Value of Prediction in Algorithmic Decision Making,�?ICML, 2024.	Q1 �?直接理�?参�?�之�?：�?测只�?��善最终决策价值的�?�?��杆，prediction improvement �?value 取决于决策环境�?�\nProceedings of Machine Learning Research

(ii)/(iv)	Hugo Barbalho et al., “Virtual Machine Allocation with Lifetime Predictions,�?MLSys, 2023 �?Outstanding Paper.	极其相关：云调度直接消费 lifetime prediction，并专门设�?�?prediction error 鲁�?的算法；说明“不完美预测仍可有系统价值�?��?�\nMLSys Proceedings

(ii)	Jianheng Ling et al., “LAVA: Lifetime-Aware VM Allocation with Learned Distributions and Adaptation to Mispredictions,�?MLSys, 2025.	与你�??�为�?么输出完整分布�?�最接近；利�?lifetime distributions + reprediction，并�?Google hyperscale cloud 部署。\nMLSys Proceedings

(ii)/(iv)	Abdullah Bin Faisal et al., “When will my ML Job finish? Toward providing Completion Time Estimates through Predictability-Centric Scheduling,�?OSDI, 2024.	GPU cluster；明�?���?completion-time predictability �?performance/fairness �?trade-off。\nUSENIX

(ii)/(iii)	Romil Bhardwaj et al., “Cilantro: Performance-Aware Resource Allocation for General Objectives via Online Feedback,�?OSDI, 2023.	用在线�?习模型和 confidence bounds 处理资源分配不确定�?�；�?��“consumer 应显式�?�?uncertainty”�?�\nUSENIX

(ii)/(iv)	Yarin Perry et al., “DOTE: Rethinking (Predictive) WAN Traffic Engineering,�?NSDI, 2023 �?Best Paper.	很强�?accuracy-vs-value 反例：不必显式追求未�?demand prediction，也能�?�过 stochastic optimization 获得接近 oracle 的决策质量�?�\nUSENIX

(ii)/(iv)	Chieh-Jan Mike Liang et al., “On Modular Learning of Distributed Systems for Predicting End-to-End Latency,�?NSDI, 2023.	连接 predictor MAE 与系�?tuning 结果，同时支持模块化、稳�?prediction interface。\nUSENIX

(ii)	Kaihao Ma et al., “PPS: Fair and Efficient Black-Box Scheduling for Multi-Tenant GPU Clusters,�?Parallel Computing, 2024.	直接 GPU cluster：基于运行时间分布�?�?job completion probability 并用�?predictive scheduling。venue 不�? OSDI/MLSys，但任务很贴近�?�\n科�?直�?�车

(ii)	Tiangang Li, Shi Ying, Yishi Zhao, Jianga Shang, “Batch Jobs Load Balancing Scheduling in Cloud Computing Using Distributional Reinforcement Learning,�?IEEE TPDS, 2024.	直接论证“只用期望会丢失随机�??信息”，利用 return distribution 做云 batch scheduling。\n数字对象标识�?n
(ii)/(iii)	Jiheon Choi, Sangyoon Oh, “UARP: Uncertainty-Aware Runtime Prediction for Preventing Scheduler Termination under Wallclock Constraints in HPC,�?Journal of Supercomputing, 2026.	�?runtime quantile→scheduler �?直接的新工作之一：q50/q99 + uncertainty margin。venue 较弱，但工程相关性非常高。\nSpringer

(iii)	Yunqiang Yin, Zunhao Luo, Dujuan Wang, T. C. E. Cheng, “Wasserstein Distance-Based Distributionally Robust Parallel-Machine Scheduling,�?Omega, 2023.	处理 processing-time distribution uncertainty；是 scenario/DRO scheduling 的直接参考�?�\n科�?直�?�车

(iii)	Carlo Meloni, Marco Pranzo, “Evaluation of the Quantiles and Superquantiles of the Makespan in Interval Valued Activity Networks,�?Computers & Operations Research, 2023.	对你�?Σq95 �??�?直接：研究的�?makespan aggregate �?quantile/CVaR，�?�非把组�?quantile 当作 aggregate quantile。\n科�?直�?�车

(iii)	Liao Wang, David D. Yao, “Production Planning with Risk Hedging Under a Conditional Value at Risk Objective,�?Operations Research, 2023.	权威 OR 期刊�?CVaR risk-sensitive planning 的近期实例，�?�� expected objective �?tail-risk objective 分�?。\nPubsOnLine

(iii)	Anastasios Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, Tal Schuster, “Conformal Risk Control,�?ICLR, 2024.	若将来需要给 scheduler 的�?险量加入有限样本 risk guarantees，这�?��现代�?calibration/risk-control �?��。\nICLR 会�?记录

其中我会让项�?��责人优先精�?�?��：Barbalho/MLSys�?3、LAVA/MLSys�?5、PCS/OSDI�?4、DOTE/NSDI�?3、Predict-then-Calibrate/NeurIPS�?3、Mandi/JAIR�?4。它�??好�?盖�?��?测�?不�?”�?�完整分布为�?么有价�?��?��?�接口解耦�?�和“为�?么最终必须看 downstream decision”�?�\n
�?终给项目负责人的三个直接回答

1. “现在还�?�� 5�? 秒，�?��吗？”\n
预测�??没有完全解决�?0s regime 仍明显低估�?�但近年文献没有�?�?��移�?的�?��?�?��多少才可调度”的阈�?��?�R1b 已在原�? RuntimeQScore 上改�?21%、校准也改善，所以已经�?�得进入 scheduler；是否�?��?用�?�必须看它关�?��多少 J3→perfect-runtime �?makespan/flow-time gap。\nMLSys Proceedings
+1

2. “输出完全变了，scheduler 怎么兼�?？�?�\n
不改�?scheduler API �?��：完�?16-bin distribution 作为 versioned source-of-truth，同时�?�?q50/q90/q95/mean �?canonical views；保�?legacy_sum_q95_v1，使新旧模型�?��接�?比�?�以�?consumer �?���?��升级，不�?要重新�?�?predictor。LAVA �?Predict-then-Calibrate 都支持这种�?�rich uncertainty �?downstream risk policy”的分层思路。\nMLSys Proceedings
+1

3. “这么�?细的分布到底怎么�?��”\n
立即有两�?��正的新用途：风险�??�调度用 ΣE[T]，它对任�?dependence 都严格可加；风险敏感调度�?��应�?�虑 total-cost CVaR / scenario optimization。现�?Σq95 保留作历�?baseline，但不�?再解释成 total-runtime q95。完�?16-bin distribution �?大的长期价�?�，�?? scheduler �?��选择 mean、tail risk、deadline exceedance、scenario/DRO，�?�不�?�� predictor 永久绑�?在一�?q95 上�?�\n科�?直�?�车
+1

�?以现在最合理的下�?步不�?�� 5.6 秒继�?��磨到 8 秒，而是先跑：\n
J3+Σq95,R1b+Σq95,R3a+Σq95,R1b+ΣE[T],R1b+ΣCVaR,Oracle
	​\n

在同�?�?300 �?paired v03 episodes 上�?�这�?��验一次就能回答负责人提出的三�?��题：新�?测到底有没有价�?��?�旧 q95 接口�?��还能工作、完整分布的额�?信息�?��真的值得 scheduler 使用�?
