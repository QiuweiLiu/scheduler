# 学习"消费函数"（GPT 方案，2026-09-16）

背景：我们的消费函数（逐步骤 runtime p95 求和）是手工设计的；用户要求探索"用学习得到更合理的消费方式"，
并且明确"旧 BC 是模仿专家、默认专家最好；现在要寻找超过专家的方案"。
本文件是 GPT 给出的可执行方案；正式执行前需预注册（见 PLAN）。

---

## 5. assistant
我认为你们现在最适合的方向，不是“重新做一个 RL scheduler”，而是把问题收缩成：

冻结预测器与调度器动力学，只学习一个把 future-action forecast 映射为候选动作代价的消费函数。

这比旧 PPO 更容易训练，也更契合你们已经证明的“消费方式比预测精度重要”。

1. 形式化：属于 decision-focused，但更准确叫 contextual/prescriptive policy learning。

经典 Decision-Focused Learning 往往是让“预测模型参数”直接接受下游决策损失；你们预测器已经冻结，所以不是最典型的 end-to-end DFL，而是其下游版本：给定 context x= future forecast + system state，学习决策代价 C
θ
	​

(x,a)，直接最小化调度后的 episode loss。JAIR 2024 的 DFL survey 和 contextual optimization 文献都把这种“预测不是终点，以决策质量为目标”的方法纳入更广义的 contextual optimization / policy optimization 框架。
ML Anthology
+1

建议参数化为：

C
θ
	​

(a)=C
current
	​

+
h=1
∑
H
	​

γ
h
	​

(s)[w
r
	​

Q
τ(s)
	​

(R
h
	​

)+w
l
	​

p
h
load
	​

Q
τ
l
	​

	​

(L
h
	​

)]+w
q
	​

Queue(a)

其中 τ,γ,w 可以依赖 slack、queue pressure、GPU state。重点是参数量控制在 8–20 个或一个很小的 MLP。

2. 学习范式排序

第一名：低维参数化消费函数 + CMA-ES / Bayesian Optimization。

训练信号：直接用 700 dev 上 mean completion（可加 miss penalty）作为黑盒 objective。

参数建议只学：

runtime τ；

horizon discount γ
h
	​

；

length weight；

queue pressure weight；

slack→risk 的两个系数；

load weight；

residual scale。

优点：完全绕开不可微 simulator；直接优化最终目标；不会受“专家天花板”限制。CMA-ES就是为低维、非凸、无梯度 black-box optimization 设计的。
CMA Evolution Strategy

样本复杂度：约 100–300 个参数候选即可有信息量；你们每臂约 7 分钟（700 eps），可先用 100–200 episode 作 inner search，再用完整 700 重排前 10 名。

风险：很容易过拟合 dev，所以参数必须少。

推荐度：★★★★★。

第二名：q95 + 学习残差 scorer。

不要从零学：

C
θ
	​

=C
q95
	​

+g
θ
	​

(z)

其中 g
θ
	​

 是 2×32 MLP 或 GAM，只修正 q95。

输入：

q95-sum、p50-sum、tail gap；

P(L≥h)；

slack ratio；

ready queue/GPU pressure；

current residency；

future GPU-step 数；

model transition count。

这样初始化时就是你们已验证最强策略，学习只能寻找“q95在哪些状态下应该加/减”。

训练可以用 CMA-ES 优化小网络，或者下面的 ranking supervision。

推荐度：★★★★★，而且论文最漂亮。

第三名：Counterfactual Learning-to-Rank。

这是我认为最可能真正“超过专家”的监督式方案。

对每个调度状态，不学“专家选了谁”，而是对每个 feasible candidate 做短 rollout / 完整 simulator continuation：

y(a)=episode cost after choosing a

然后训练 scorer 满足：

C
θ
	​

(a
i
	​

)<C
θ
	​

(a
j
	​

)⟺y(a
i
	​

)<y(a
j
	​

)

用 pairwise ranking loss/ListMLE。

这和 NeurIPS 2024 的 LLM Learning-to-Rank 思想一致：调度常常只需要正确排序候选请求，不需要准确预测绝对 cost。
NIPS 会议论文

相比 BC，优势是标签不是“专家动作”，而是动作后果，因此理论上可超过 teacher。

建议 30k–80k states，每状态随机评估 2–6 个候选即可。

推荐度：★★★★☆。

第四名：迭代 Policy Improvement / DAgger-style。

初始化 q95 或 ranking model：

rollout 当前策略；

收集它真正访问的 states；

对候选动作做 MPC/模拟反事实；

更新 ranking scorer；

重复 2–3 轮。

这是最自然的“模仿→超越专家”。

因为每轮 supervision 来自当前策略遇到的新状态上的更好动作，而不是固定 teacher trajectory。

风险是计算量高，但你们 simulator 很便宜，所以可行。

推荐度：★★★★。

第五名：Offline RL。

可以做 CQL/IQL 风格的 masked discrete-action Q-learning，并加入 counterfactual trajectories。近期离线调度工作也发现，混入多样、甚至带噪 expert data 可提升 policy，相比纯 online RL 更稳定。
arXiv

但你们现在不应优先：

动态 candidate set；

coverage 不足；

旧 PPO 已显示优化困难；

reviewer 很容易把故事拖成“RL训练细节”。

推荐度：★★。

3. 怎样真正“超越专家”

关键原则：

不要再训练 π(a∣s)≈π
expert
	​

(a∣s)。

那必然受到专家 ceiling。

三条可行路线：

第一，直接 episode-return black-box optimization。CMA-ES完全不需要教师，q95只是 initialization。

第二，反事实监督。教师只生成 visited states；label 来自：
“如果当时分别选 A/B/C，最终 completion 会怎样？”
因此监督信号可以发现专家本身没选到的更优动作。

第三，teacher ensemble + policy improvement。CP-SAT、MPC、q95、E2 都产生候选动作，然后 simulator rollout 决定谁在该 state 更好，而不是默认某一个 teacher 永远正确。

我尤其推荐：

q95 residual → counterfactual ranking → 一轮 policy improvement

这是从手工规则到学习消费函数最短的完整故事。

4. 最小可信论文实验

特征控制在约 20–30 维：

current：runtime/load、residency、memory；

system：free GPU、ready count、queue pressure、slack；

future：P(L=0..5)、Σp50/90/95、Σtail-gap、GPU-step count、load probability sum、model-switch count；

q95 baseline score。

模型：

Parametric：8–15 参数；

Learned：MLP 32×32，输出一个 scalar candidate cost。
不要再上 GRU/GNN。

协议：

700 dev 内按 source workflow/video group 再做 500/200 或 5-fold grouped tuning；所有超参、feature set、训练 budget 在这里确定。300 confirm 只跑最终冻结模型一次。

至少 3 training seeds。

正式比较：

E0

q95

CP-SAT/MPC

BC-expert

learned-parametric

learned-ranking/residual

Primary：mean completion。
Secondary：miss、p95 JCT、queue、策略计算开销。

必须同时报告：

dev → confirm generalization gap；

seed variance；

learned score 与 q95 的相关性；

消融掉 slack/queue/tail features 后性能；

inference latency。

成功标准建议预注册：

Strong success：
learned consumer vs q95 在 confirm paired CI upper < 0，同时 miss non-inferior。

这才能写：

learned consumption surpasses the strongest hand-designed consumer.

若只超过 BC/CP-SAT、不超过 q95：

learning recovers strong scheduling behavior but does not improve upon the best manually designed risk surrogate.

若和 q95 基本相等，而且参数最后退化成 τ≈.95、其他权重≈0：

这不是失败，反而说明：

q95 is near-optimal within a substantially broader learned consumer family.

5. 最大的坑

最危险的是700 dev 已经被看过很多次。因此不要再无限试 architecture。冻结：

2 个 model classes：低维 parametric + 32×32 residual；

固定 search budget；

固定 objective。

300 confirm 只能最终一次。

第二个坑是 reward hacking。建议训练目标固定：

J=mean completion+λ
miss
	​

max(0,miss−δ)

而不是不断改权重追结果。

第三个坑是不可解释。解决办法是把主模型设计成：

q95 + sparse residual

并报告 residual 对 slack/pressure/length 的偏移。这样即使 learned policy 只提高 2–4%，也能解释“q95何时过于保守、何时不够保守”。

如果只能选一条，我建议直接做：

q95 初始化的 8–15 参数 contextual consumer + CMA-ES → 再用 counterfactual ranking 做 residual refinement。

这条路线最可能在你们现有算力和数据规模下真正超过 q95，而且不会重蹈 PPO 的训练不稳定。
