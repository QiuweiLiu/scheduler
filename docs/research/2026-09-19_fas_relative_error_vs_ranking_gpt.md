# 相对误差 vs 排序（项目负责人质疑指标选择）— 对话 6aae601b

## 回答

这�?项目负责人的质疑�??的，而且会改变我上一�??下一步的优先级�?�\n
核心裁决�?��压缩成一句：

资源头�?先应该按“它声称预测的统计量�?��准确”来评价，最终再按�?�这些�?测是否改善调度决策�?�来裁判；单�?Spearman �?��辨识力诊�?��不应该成�?Phase R 的主 gate。\n
�?以我现在不建�?���?步继�?���?Spearman �?deeper unfreeze �?ranking loss。应该先把原 J3 的�?测验收指标完整补回，再直接做�?�?���?scheduler smoke。\n
Q1. “�?看相对�?�??�成立吗？\n判定：VERIFIED，但�??“相对�?�??�还不完整�?�\n
负责人指出的核心�??�??的：Spearman �?��答\n
“谁比谁大，顺序对不对？”\n
完全不回答\n
�?0s �??测成 1s 还是 9s？�?�\n
而资源头的直接任务显然�?先是后�?��?�\n
但这�?��布�?测器，不存在�?�?��能的“相对�?�??��?�你必须先�?清�?评价�?��统�?泛函。�?�?median、q95、mean 和完�?distribution，本来就应�?用不同的 scoring rule；�?测评价理论明�??�?loss 与所预测�?functional 匹配。\n泰安学术
+1

我建�?��现在�?始采用下面这套层级�?�\n
层级	指标	用�?�与判据
PRIMARY-P：�?测层	RuntimeQScore = mean(PB50, PB90, PB95)	�?重�?，因为这�?J3 原�?验收口径，所有旧/新头都可�?��比较。新头�?先不能比 J3 差\nPRIMARY-P	PB95 + Coverage95	调度器实际依赖尾部分位；PB95 判断 q95 数�?�准不准，Coverage 判断�?��真接�?95%
PRIMARY-S：系统层	固定 scheduler benchmark 的实际目标\t�?终部署�?判；例�?原协�?���?deadline miss / completion objective，必须沿用原调度协�?
SECONDARY	log-MAE(q50)	衡量跨数量级的点预测准确度，特别适合你这�?0.1ms�?0s+ 的跨�?nSECONDARY	CRPS / 当前离散头的 RPS	判断整个预测分布，�?�不�?��抽三�?quantile
SECONDARY	ΣE[T]/ΣY	判断总工作量/总资源量校准�?.9961 �?��常好的结果，但它不是尾部风险指标
SECONDARY	raw MAE(q50)、各 τ �?normalized pinball	给出�??尺度和无量纲尺度的解释\nDIAGNOSTIC	真�?�分�?median pred/true	找�??�?2s �?���?1s”这�?failure mode
DIAGNOSTIC	per-slot、per-family、per-role	找�?�?���?��里\nDIAGNOSTIC	Spearman / pairwise concordance / tail recall	判断 representation �?��能区分实例，不再负责模型验收

这里 CRPS/RPS 的�?色有�?点特殊：CRPS �?full-distribution �?proper scoring rule，同时体�?calibration �?sharpness，�?�合评估完整概率预测。\nOUP Academic
+1

但是 J3 �?��三个 quantile，并没有完整分布。所以你不能为了新模型方便，事后�?J3 �??�一�?��布然后拿 CRPS 比�?�\n
因�?：\n
J3 �?Phase R 的主�?��比较：原 RuntimeQScore。\nPhase R 内部模型之间：可以进�?步用 RPS/CRPS。\n
这是我�?为最干净的体系�?�\n
Q2. �?Spearman �?strong gate �?���?���?��
判定：VERIFIED，是。\n
准确�?点�?：\n
把�?�单�?���?slot runtime 的全�? Spearman”作为资源头替换的强制验�?gate，是指标错位。\n
它不�?���?��用的指标。\n
实际上它�?��发现了非常有价�?�的�??：\n
J3 能大致校准边际分布，�?node-level discrimination 很弱。\n
�?以当初引入它作为代码审查诊断�?��理的。\n
错�?发生在�?二�?：\n
�?diagnostic metric 升格成了“必�?+0.08，否�?strong FAIL”的 acceptance criterion。\n
因为你的 downstream scheduler 根本不是�?7195 �?slot 做全�?排序。\n
它实际做的是类似：\n
C
j
	​\n
=
h=1
∑\nH
	​\n
q
.95,jh
	​\n

然后在当前可选任务集合里依据 C
j
	​\n
 和其它状态做决策。\n
真�?�?��与调度相关的 ranking 应�?�?��

在实�?scheduler decision state �?���?candidate 的聚�?scheduler cost 排序�?��正确。\n
而不�?��

�?有�?频�?�所�?horizon、所�?node 混在�?起的 runtime Spearman。\n
两�?�不�?��回事。\n
决策导向学习领域也�?�?��为这种问题�?�强调：标准 predictive accuracy �?downstream optimization quality �?��错位，应当实际评估�?测�?�成的决策损失�?�\nAAAI Publications
+1

那现在降�?Spearman 算不�?moving the goalposts？\n
如果你偷偷删除这�?gate，然后�?布：

“R3a strong PASS！�?�\n
那当然会�?moving the goalposts。\n
正确处理方式不是删除历史，�?�是版本化评价协�??�\n
应�?明确留下：\n
Evaluation Protocol v1：R3a 在�?注册 strong gate �?FAIL，因�?Spearman �?���?+0.08。\n
这个事实永久保留。\n
然后另开：\n
Evaluation Protocol v2 / Metric Amendment

说明三个理由：\n
Spearman 不属�?J3 原�?验收指标；\n
它是 Phase R 审查期间新�?�?proxy；\n
经架构�?核后�??，它与实�?scheduler consumer 的统计量不匹配，因�?从下�?阶�?起降�?diagnostic。\n
而且必须�?��

�?test、跑新的 scheduler smoke、�?练下�?模型之前

�?v2 固定下来。\n
这样不是“把失败改成成功”，而是：\n
承�? v1 �?metric design 有问题，并�?�?��实验 prospectively �??。\n
这在论文里完全可以坦诚写。\n
甚至我建�?��留一句：

“An initially preregistered rank-correlation gate was retained in the audit record but subsequently demoted to a diagnostic metric because it did not directly correspond to the downstream scheduling cost.”\n
这个比假装从来没有那�?strong gate 干净得�?。\n
Q3. “准�?��全面改善，但排序基本没动”自洽吗？\n判定：VERIFIED，非常自洽�?�\n
这实际上已经�?��结果�?合理的解释�?�\n
�?��把它理解成：

J3：\n
真实�?�?2s
预测：∼0.8s

R1b / R3a：\n
预测：∼4�?s

于是：\n
log-MAE 大幅下降；\n
raw MAE 下降；\n
q50 总量偏差改善；\n
8�?2s ratio 大幅改善；\n
calibration 改善。\n
但是假�?这个 runtime regime 内：

真实：\n
6s,8s,10s,12s

模型都是：\n
5.5s,5.5s,5.5s,5.5s

那么 magnitude 非常明显地变好了，但 rank 没�?�么变�?�\n
这与你现在�?察到：\n
ΔSpearman�?.014

以及 pairwise concordance 几乎 0，非常一致�?�\n
�?scheduler �?��事还�?��事？

答�?�?��

INFERENCE：�?�常�?��事，但还不�?证明 scheduler 会改善�?�\n
如果调度器过去把：\n
�?�?���?10s 的任�?n
�??为：

1s

这当然会严重破坏 cost estimation。\n
把它恢�?�?5�?s，哪怕还不能准确区分：\n
8s vs 10s

scheduler 都可能已经明显受益�?�\n
但是如果某个决策状�?�里候�?�任务都�?���? regime：\n
A�?s

B�?0s

C�?2s

而模型全�?5.5s，\n
那么 scheduler 还是不知道应该先选谁。\n
因�?真�?�?要测的是：\n
decision-state discrimination
	​\n

不是 global slot Spearman。\n
也就�??，排序不�?��全没�?��而是之前排错对象了�?�\n
如果后面还想保留�?�?ranking diagnostic，我会换成：

scheduler 每个真实 decision state 内，candidate aggregate cost �?Kendall/Spearman、top-1 agreement �?decision regret。\n
这个比全�?单节�?Spearman 有意义得多�?�\n
Q4. “相对�?�??�到底�?怎么算？

针�?你的任务，我的排序是：\n
(b)>(e)>(a)>(c)�?d)
	​\n

但每�?��实回答不同问题�?�\n
�?��名：Normalized Pinball

�?quantile：\n
PB
τ
	​\n
=
N
1
	​\n
i
∑\n	​\n
ρ
τ
	​\n
(y
i
	​\n
−\nq
^
	​\n
τ,i
	​\n
)

然后定义：\n
NPB
τ
	​\n
=
s
PB
τ
	​\n
	​\n

其中 s 不�?用每�?���?��己的 y
i
	​\n
，�?�用�?�?��定�?练集尺度，例如：

s=mean(Y
train
	​\n
,Y>0)

这样它相当于：\n
“�?测�?�?��典型总体 runtime 尺度多少”�?�\n
�?重�?的优势是它仍然是 pinball loss 的�?比例缩放，因此不会改�?quantile 评价的目标�?�\n
对你尤其应�?报告：\n
NPB
.50
	​\n
,NPB
.90
	​\n
,NPB
.95
	​\n

和平均�?��?�\n
为什么我把它放�?�?？\n
因为 scheduler 明确消费 q95，�??quantile 应�?用与 quantile �?致的 loss 评价；使用与统�? functional 不匹配的 error function �?��产生错�?结�?。\n泰安学术
+1

而且它可以和 J3 完全�?��比较。\n
�?��名：CRPS / RPS

如果评价整个 16-bin distribution：\n
CRPS(F,y)=�?F(z)�?[y�?])
2
dz

非常合�?��?�\n
它相当于概率预测版本�?absolute error，还�?proper scoring rule。\nOUP Academic
+1

不过你的模型实际上是 ordered categorical bins，因此：

当前实现�?RPS 更原生�?�\n
如果要报 CRPS，需要明�?���?bin �?代表点以�?overflow 的分布定义，否则会引入人为假设�?�\n
�?以我会：

主代码里保留 RPS；\n
如果明确构�?�了合法 runtime CDF，再报告 CRPS。\n
如果特别关心尾部，还�?���?threshold/quantile-weighted CRPS；文�?��有保�?propriety 的加权方法�?�\n泰安学术
+1

�?��名：log-MAE

你的形式：\n
MAE
log
	​\n
=E�?og(1+
y
^
	​\n
)−log(1+y)�?n
我继�?��烈支持�?�\n
它本质上近似衡量乘法尺度�?��：\n
对于 runtime 较大：\n
�?og
y
^
	​\n
−logy�?
	​\n
log
y
y
^
	​\n
	​\n
	​\n

�?以：

100ms�?s

和\n
1s�?0s

得到类似惩罚。\n
这�?你跨几个数量级的 runtime 极其有用。\n
而且 +1ms 避免了零附近发散。\n
不过它仍然只�?���?point-estimate metric，不评价：\n
q95；\n
calibration；\n
full uncertainty。\n
�?以它应�?�?secondary，�?�不�?���? primary。\n
�?��名：按真值分�?pred/true

例�?：\n
<1ms

0.1�?s

1�?s

5�?s

8�?2s

>12s

每档报告：\n
median(
y
^
	​\n
/y)

以及 IQR。\n
这个非常适合你�?�\n
它就�?��这个�?��了：

8�?2s �?J3 �?�� 0.083。\n
但它�?���?diagnostic。\n
因为人为�?band 会影响结果，而且不同 band 样本数不�?样�?�\n
�?后：MAPE / sMAPE / MdAPE

我不会拿它们做主指标。\n
你的数据�?直就�?MAPE 的典型反例：

大量�?0.1ms；\n
�?���?0；\n
同时存在 10s+。\n
MAPE：\n
	​\n
y
y
^
	​\n
−y
	​\n
	​\n

�?y�? 时直接爆炸�?�\n
这不�?��瑕疵，�?�是经典 forecast-evaluation 文献长期强调的问题；MAPE �?0 时未定义，在接近 0 时会产生极�?值，sMAPE 同样存在明显�??。\n数字对象标识�?n+1

MdAPE 虽然�?median 减少了一些极�??�影响，但是：\n
很�?易把真�?重�?的慢任务灾难性低估隐藏掉。\n
�?以最多附录报告�?�\n
Q5. �?终�?判是不是调度结果？\n判定：VERIFIED—�?��?于系统价值，�??��?于�?��?测器�?��更准”，不是。\n
这两�?��题必须分�?：\n
resource predictor quality �?proper predictive metrics

和\n
VideoSeek scheduler utility �?downstream scheduling result

�?终部署当然看�?���??�\n
Predict-then-optimize / decision-focused learning 的核心结论之�?就是：�?测指标改善不保证优化决策改善，反过来也可能出现标准�?测�?�?���??�但优化决策更好的模型�?�\nAAAI Publications
+1

�?以你现在已经到了应�?�?scheduler smoke的时候�?�\n
我甚至�?为这比：

oracle-q(A)、继�?���?GRU、ranking loss

都更优先。\n
�?小可�?scheduler smoke

我建�?��时不�?��任何东西。\n
冻结三个 artifact：\n
A：J3 原�?资源头\n
这是 baseline。\n
B：R1b

回答：\n
仅换 distributional head 有没�?downstream benefit？\n
C：R3a-U

回答：\n
再加浅层 adaptation 有没有�?�?downstream benefit？\n
如果已有 zero-cost oracle runtime 接口，可以加：\n
D：True-runtime / Oracle-resource

�?���?headroom reference，不参与部署模型比较。\n
Episode 数\n
我建�?��

300 paired episodes
	​\n

作为 smoke，�?�不�??�?paper benchmark。\n
关键不是 300 这个神�?数字，�?�是：\n
完全 paired。\n
即同�?�?��

workload；\n
arrival trace；\n
deadline；\n
GPU 状�?�；

random seed；\n
分别�?A/B/C。\n
并且从既�?scheduler benchmark 的场�?��间分层抽取，不能因为知道结果再挑某些 scenario。\n
例�?：\n
6 representative strata×50 episodes

如果你现�?benchmark 已经定义了其�?strata，就直接复用原来的划分，不另造�?�\n
scheduler smoke 看什么？

�?重�?的�?则是：\n
沿用已发�?scheduler protocol 原本�?primary objective。\n
不�? Phase R 又发明一�?���?downstream 指标。\n
如果原来�?deadline miss rate �?completion time，那么全部按原定义跑。\n
另�?新�?三个 diagnostic：\n
decision disagreement rate

�?B/C �?J3 做出不同选择的比例�?�\n
candidate-cost rank agreement

实际 decision set �?�� cost 排序。\n
以及如果 simulator 有反事实成本：\n
decision regret=J(a
chosen
	​\n
)−\na
min
	​\n
J(a)

这是�?slot Spearman 更有意义�?rank/decision 指标。\n
�?�?��常重要的数�?细节

如果 scheduler 定义：\n
C=
h
∑\n	​\n
q
.95,h
	​\n

你可以称它：

scheduler risk score

但不要称它：

“�??runtime �?q95”�?�\n
因为�?�?��况下：\n
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

除非对联合依赖结构作很强的假设�?�\n
�?以不要�?求：

P(∑Y
h
	​\n
≤∑q
.95,h
	​\n
)=0.95

然后把偏离解释成 calibration failure。\n
�?��它只�?scheduler 人为定义的�?险聚合函数�?�\n
这进�?步�?明：

判断这个聚合到底好不好，�?直接的方法就�?�� scheduler。\n
smoke 判据

这是 exploratory smoke，所以我不建�?��新制造一�??�p<0.05 才能继续”的陷阱。\n
提前固定：\n
R3a-U 相�? J3 的主 scheduler objective 点估计不能恶化；

至少�?�?��系统指标出现实际改善；\n
各固�?scenario strata 不得出现明显系统性�??化；

paired episode 分布�?bootstrap CI 全部报告，不�?��均�?��?�\n
非劣 margin �?好直接�?用你之前 scheduler benchmark 已经认可的工程�?忍度。\n
如果从来没有定义过，我建�?���?smoke 结果前，根据�?scheduler �?seed/episode �?��波动定义�?�?margin，�?�不�?��在拍脑�?指定 1% �?2%。\n
如果预测指标大幅改善，但调度完全不改善呢？\n
这个结果完全�?��接受，�?�且非常有信�?��。\n
正确结�?�?��

“The revised resource model substantially improves runtime forecast quality, but these gains do not translate into measurable scheduling improvements under the current scheduler.”\n
然后继续查原因：

�?���?��

scheduler 对这部分�?��不敏感；

q95 聚合�?��守；

调度瓶�?�?GPU availability / topology；\n
候�?�任务�?�常不在同一�?��策边界附近；

或�?�真正影响调度的�?candidate-level discrimination，�?�不�?���?runtime calibration。\n
此时不能声称：\n
“新资源预测器提高了调度性能。�?�\n
但可以声称：

“提高了资源预测质量。�?�\n
如果 scheduler 甚至明显变差，那么即�?predictor 更准，也不应该直接替换生�?主系统里�?J3。\n
Q6. strong gate 怎么治理？\n
我的建�?非常明确：\n
不�?删除，不要重写历史，不�? retroactive PASS。\n
保留�?�??录：

Phase R Evaluation Protocol v1

R3a strong gate: FAIL
原因：Spearman improvement �?���?+0.08。\n
然后提交�?�?��的�?�带时间戳的：\n
Metric Amendment / Evaluation Protocol v2

其中写清楚：

变更原因：\n
Spearman �?Phase R 审查时新增的 node-level discrimination proxy；它不属�?J3 frozen acceptance protocol，且后续架构复核�??它并不直接�?�?scheduler 实际消费�?candidate-level aggregated risk score。\n
处理方式：\n
RuntimeQScore 恢�?�?cross-generation primary；\n
q95 pinball + coverage �?scheduler-relevant predictive primary；\n
downstream scheduler objective �?system-level primary；\n
Spearman 降级�?diagnostic；\n
�?v1 gate 和结果永久保留�?�\n
适用范围：\n
v2 �??�用�?amendment 之后的实验与�?���?�?test。\n
不�?�?v2 回头说：

“所�?R3a 当时其实 PASS。�?�\n
正确�?���?��

“R3a failed the original Phase-R strong proxy gate; the proxy was subsequently judged misaligned with the downstream task and was not retained as an acceptance criterion in protocol v2.”\n
谁应该批准？

�?好至少两�??色：

项目负责�?+ �?名没有参与当前模型调参的人�?�\n
比�?负责 scheduler/evaluation 的成员�?�\n
如果实际上只有你�?�?��在做，也没关系，就用 version-controlled decision record：\n
日期；\n
理由；\n
metric definitions；\n
evaluator commit hash；\n
当前已经看过�?�� validation result；\n
test = untouched；\n
后续 protocol。\n
�?重�?的是透明，�?�不�?��式上的�?�独立�?员会”�?�\n
我现在会怎么改下�?�?n
我会暂停：\n
oracle-qA �?deeper unfreeze �?ranking loss

这条优先级�?�\n
现在优先顺序变成：\n
�?零�?练补齐原始验收指标\n
�?R1b / R3a �?16-bin distribution 直接抽：

q50,q90,q95

然后计算完全相同的：

RuntimeQScore

以及：\n
PB
50
	​\n
,PB
90
	​\n
,PB
95
	​\n

和：

Coverage
50/90/95
	​\n

这是�?���?关键、�?�你给我的表里偏偏缺失的�?项�?�\n
如果 R3a �?��

�?J3 �?��定义�?RuntimeQScore

上都没有改善，那负责人�?��?测更准�?�的论点就�?重新�?查�?�\n
如果明显改善，那么证�?��非常强了。\n
�?锁定 Evaluation Protocol v2

不�? test。\n
不再�?Spearman pass/fail threshold。\n
�?�?300 paired-episode scheduler smoke

J3 vs R1b vs R3a-U

全部其它条件锁�?。\n
�?根据 smoke 决定研究方向

如果：\n
预测明显改善 + scheduler 也改善\n
�?Phase R 已经成功。\n
不用再为了把 Spearman �?0.64 做到 0.67 而继�?���?encoder。\n
如果：\n
预测改善 + scheduler 不变

再做 decision-state audit。\n
如果发现 scheduler 经常�?要区分同 regime candidate，�?�模型区分不了，这时才�?�得重启：\n
q(A) oracle / representation / ranking loss

这条线�?�\n
�?以这次我会明�?��正上�?�?��优先级：此前我把 Spearman 当成了过强的 proxy。它非常适合诊断“表示里有没有细粒度排序信息”，但不应�?凌驾�?RuntimeQScore、q95 calibration 和真�?scheduler utility 之上。\n
理�?依据也比较统�?：�?测某�?functional 时�?用与它一致的 scoring rule；完整�?率�?测�?�合 proper scores �?CRPS；MAPE 类指标在�?近零场景非常不稳定；而�?测进入优化器后，�?终必须�?外评�?downstream decision quality。\nAAAI Publications
+3
泰安学术
+3
OUP Academic
+3

如果�??我现在给你一�??�明天应该�?代码 Agent 做什么�?�的任务，我会�?�：先给 J3/R1b/R3a 统一重算 RuntimeQScore + PB50/90/95 + Coverage，然后直接生成三�?resource artifact �?paired scheduler smoke；先不�?练任何新模型�?
