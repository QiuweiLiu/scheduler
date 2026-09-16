---
research_backend: ChatGPT Web
requested_model: GPT-5.6 Sol
requested_reasoning: High
verified_before_submit: true
verified_after_response: false
timestamp: "2026-08-18 11:05"
status: incomplete
reason: "研究正文完整返回；回答后网页设置二次核验因 ab.chatgpt.com 网络超时未能完成。未发现自动降级提示；因此本报告只作为已独立核验来源支持的研究输入，不单独作为最终证据。"
---

# RL 与数学优化器改进路线：ChatGPT Web 研究记录

## Executive Summary

**[INFERENCE]** 对当前 VideoSeek 的节点级、多 GPU、动态到达调度，建议的主路线是：

`Rolling-Horizon CP-SAT → 同信息边界的专家动作 → 行为克隆（BC）→ Masked Actor-Critic/PPO + GAE`。

这里不是把“滚动时域、MILP、CP-SAT”当成三个互斥选项：滚动时域是在线控制外壳，CP-SAT 是每个窗口中的求解器；小规模 MILP 用于精确 sanity check 和公式审计。不要一开始做覆盖整个长任务的巨大 MILP。

**[VERIFIED/local]** 当前结果说明 PredOpt-H5 是最强的可部署候选，但 RL 的三 seed 方向不一致，因此 RL 不能冻结为最终方法。当前 `Optimizer-0` 是确定性一步评分器，不是已经实现的全局数学规划器。

**[INFERENCE]** 进入 CP-SAT 之前必须先修正当前 PredOpt-H 的目标函数：现在 priority 是词典序第一键，未来成本只有在 priority 相同的候选之间才有机会起作用。这会把“预测器没用”和“目标函数压制了预测器”混在一起。

## 研究输入与已知事实

研究 brief 只包含脱敏后的方法、指标和问题定义，没有发送密钥、私有文件、原始日志或私有路径。已验证的本地背景：

- 事件驱动的节点级调度，包含动态 job 到达、DAG 依赖、GPU 显存/缓存/负载、模型加载与驱逐、队列和 deadline。
- 部署策略只能使用当前状态、资源预测和有限步预测，不能使用真实未来。
- 1000 validation episodes、135 scenario cells、0 failed jobs、0 capacity violations。
- Myopic：215.12 s / 10.98% deadline miss；Optimizer-0：215.65 s / 10.98%。
- PredOpt-H5：210.64 s / 10.47%，约比 Myopic 改善 2.08%。
- TrueOpt-H5：183.85 s，Oracle：172.73 s；二者是特权信息参考，不是可部署方法。
- RL-0：204.31 s；RL-H5：202.29 s，但 3 个 seed 的方向不一致。
- 当前 RL 是 14 维候选 MLP + action mask + episode-end REINFORCE，没有 critic、专家预训练或 GAE。

## 关键结论

### 1. CP-SAT + Rolling Horizon 是主线，MILP 是审计基线

**[VERIFIED/source]** OR-Tools CP-SAT 提供整数规划求解、interval/optional interval、precedence、NoOverlap、Cumulative 等调度建模能力，且可设置时间限制和读取 solver status；这些原语直接覆盖 DAG 节点、GPU 分配、时序约束和容量约束。时间（毫秒）和显存（MB）可整数化。

**[INFERENCE]** 第一版应使用 event-triggered rolling horizon：在节点完成、job 到达、repair 分支产生或资源状态变化时，求解 `[t, t+H]`，只执行首个 dispatch action，然后在下一个事件重新求解。先与已有 `H∈{0,1,3,5}` 对齐，不直接扩大到 H=20。

**[VERIFIED/source + INFERENCE]** ICLR 2025 的 L-RHO 官方实现把 learning-guided rolling horizon 用于长时域柔性作业车间调度，并使用 OR-Tools；它支持“窗口求解、只执行一段、再滚动”的研究范式。它不能直接证明 CP-SAT 在本项目上一定比 MILP 快；求解器速度和窗口规模必须在本项目实测。

建议的 CP-SAT V1 变量/约束：

- `x[i,g]∈{0,1}`：节点 `i` 是否放到 GPU `g`；
- `s[i], e[i]`：节点起止时间；
- optional interval：把 assignment 与执行区间绑定；
- DAG 边 `i→j`：`s[j] ≥ e[i]`；
- GPU 串行版：每张 GPU 使用 `NoOverlap`；若以后允许同 GPU 并发，再评估 `Cumulative`；
- 当前 cache 作为确定性 snapshot；节点在非 resident GPU 上执行时计入 load cost；V1 限制 horizon 内重复 load/evict 的候选数量；
- 显存容量约束、deadline/tardiness、load 次数和 eviction 次数进入目标。

V2 再加入模型 residency interval `[load, eviction)`，避免第一版的组合复杂度过高。

### 2. 先修评分器，再生成专家数据

**[VERIFIED/local]** 当前 PredOpt-H 的排序 tuple 先比较 priority，再比较 current+future cost。不同 priority 的候选之间，future cost 无法翻转排序。

**[INFERENCE] Optimizer-A：PredOpt-v2** 应先做一个小改动的可解释对照：

1. current cost、future cost、runtime、load、eviction、deadline penalty 做尺度归一化；
2. priority 作为软权重 `w_j`，而不是默认的绝对词典序；真正硬 SLA 则建为约束；
3. 预测不确定性越大，future 项折扣越强；
4. 直接复跑 H0/H1/H3/H5。

这样可以先回答：当前小收益是评分器太弱，还是联合优化本身确实需要 CP-SAT。

**[INFERENCE] Optimizer-B：CP-RHO** 才是论文中应称为数学优化器的主 baseline。每次滚动求解只执行第一动作，并记录求解时间、status、time limit、gap、infeasible 次数和实际首动作。

**[INFERENCE] Optimizer-C：高预算专家** 可以离线慢求解，但必须与部署策略使用同一个 `DeployableObservation`，不能读取 `TrueOpt/Oracle` 的真实未来。专家应保存 `(state, action, objective, status, gap, solve_time)`，而不是只保存一个标签。

对每个合法动作强制一次首动作并优化后续窗口，可得到近似 `J_E(s,a)` 和 regret/margin；它比单一 argmin 标签能表达“两个动作几乎一样好”和“一个动作灾难性更差”的区别。OPTIMAL/低 gap 样本权重大，高 gap/UNKNOWN 样本降权或排除。

### 3. RL 的最小稳妥组合

**[VERIFIED/source]** PPO 使用 clipped surrogate 控制策略更新；GAE 通过 value function 和指数加权优势估计在偏差与方差之间折中；SB3-Contrib 的 MaskablePPO 直接支持 invalid-action masking。

**[INFERENCE]** 当前最小组合应是：

`专家数据 → BC 初始化 → masked actor-critic PPO + GAE → 稠密 event reward → 小且逐步衰减的 expert KL`。

不要同时加入 BC、logit prior、大 KL、PPO、辅助损失和 curriculum，否则无法解释收益来源。action mask 保留，它负责过滤 DAG 未 ready、容量不够和不可执行动作，避免 RL 浪费样本学习“不要 OOM”。

第一版网络不必立即改成 GNN：候选 MLP 先保留；actor 对每个候选编码，critic 读取 pooled candidate embedding + GPU state + queue state。如果实验证明同一个候选局部特征在不同 DAG/global context 下需要不同答案，再升级 attention/GNN。

建议的稠密奖励（与指标对齐）：

- 未完成 job 的流时间：两个 event 间 `r_flow = -Δt × Σ_j w_j`；
- deadline tardiness：对已经超过 deadline 且仍未完成的 job 按时间积分；
- job 第一次 miss 时一次性扣分；
- load 和 eviction 使用小的成本项；
- 保留 capacity violation 的硬惩罚。

**[VERIFIED/math]** `Σ_j(C_j-a_j)=∫N_unfinished(t)dt`，所以按 event 间隔给出的 flow reward 与 sum flow time 严格对齐，不只是经验性 shaping。

logit prior 放在第二阶段：先把廉价的 PredOpt-v2 score 标准化为 base logit，RL 学 residual；如果每次推理前都运行完整 CP-SAT，则方法应如实描述为“solver + residual policy”，不能再声称是纯快速 RL 替代器。

### 4. 公平比较与防泄漏

所有可部署方法（Myopic、PredOpt、CP-RHO、BC、RL）必须共享完全相同的 `DeployableObservation`：已到达 job、当前 DAG prefix、已执行节点、ready nodes、GPU state、queue、cache、冻结的 predictor 输出和有限未来。`TrueOpt/Oracle` 单独作为 upper-bound reference。

**[INFERENCE]** Expert 也只看 predictor future，不使用真实未来；Oracle-distilled RL 若未来研究，应单独标为 privileged ablation，不进入主方法训练。

预测器输出按 `episode × event × horizon` 预先缓存，所有策略读取同一份，避免各自重新推理造成随机差异。scheduler train/validation/test 按 template disjoint；最终 test 不能反复用于调参。使用 common random numbers 和成对差值；同时报告 micro/macro average、mean/p95/p99 completion、deadline miss、load、eviction、solver/policy latency、infeasible/capacity violation。

### 5. 建议实验矩阵与门禁

**P0：评分器审计（不新增模型）**

- PredOpt-v1 与 PredOpt-v2；H0/H1/H3/H5；
- 固定同一 workload、同一 predictor cache；
- 记录 priority dominance 被打破的比例以及 future cost 是否真正改变首动作。

**P1：CP-RHO V1**

- CP-SAT rolling horizon，H0/H1/H3/H5；
- V1 只做 cache snapshot + load/有限 eviction；
- 与 PredOpt-v2 和小规模 Myopic 对照；
- 记录 solve latency、timeout、status、gap、infeasible。

**P2：小规模 MILP sanity check**

- 只在小 workload/短 horizon 上与 CP-SAT 比 objective 和可行性；
- 不把“某次更快”写成普遍结论。

**P3：专家数据**

- 用 CP-RHO 产生同信息边界的首动作；
- 保存 hard label、soft distribution、regret/margin 和 solver metadata；
- 先验证 expert 自身相对 PredOpt-v2 的收益与实时成本。

**P4：学习策略**

- BC hard label；
- BC soft label；
- BC → masked PPO + GAE；
- 在同一 PPO 版本上加 decaying expert KL；
- 最后再测试 PredOpt-v2 logit prior。

**P5：鲁棒性与最终比较**

- 至少 5 个 RL seeds；
- predictor noise、arrival burst、cache miss/load cost、GPU count 和 unseen template；
- 只冻结在 final test 前通过门禁的配置。

建议 go/no-go：

- PredOpt-v2 必须证明 future cost 能实际改变决策，否则先修目标而不是上更大网络；
- CP-RHO 必须在相同信息边界下比 scorer 有稳定收益，且没有明显 deadline/latency 回归；
- BC 至少保留专家收益的大部分，动作 regret 优于 top-1 accuracy；
- PPO 至少 5 个 seed 中 4 个方向一致，且不损害 hard feasibility；
- 任一方法若依赖真实未来、未冻结 predictor 或未报告求解超时，不进入主表。

## 当前不应直接做的事情

- 不要把 TrueOpt/Oracle 直接变成 RL teacher；
- 不要直接把一个全长 MILP 作为在线调度器；
- 不要在未修正 priority lexicographic 之前用 RL 解释“预测收益小”；
- 不要把一次 seed 的 RL-H5 胜出写成结论；
- 不要把浏览器研究建议当成已运行实验，也不在本轮修改代码或启动远端作业。

## Sources（独立核验）

1. [VERIFIED] L-RHO 官方实现（ICLR 2025）：[mit-wu-lab/l-rho](https://github.com/mit-wu-lab/l-rho)。README 明确包含 rolling-horizon 训练数据收集、窗口/步长/时间限制和官方 OR-Tools 依赖。
2. [VERIFIED] Google OR-Tools：[CP-SAT Solver](https://developers.google.com/optimization/cp/cp_solver)；官方页面说明求解状态、时间/内存限制下的 UNKNOWN 等行为。[Constraint Programming](https://developers.google.com/optimization/cp) 页面列出适合 CP-SAT 的 scheduling 场景。
3. [VERIFIED] Schulman et al.：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，arXiv:1707.06347。
4. [VERIFIED] Schulman et al.：[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)，arXiv:1506.02438。
5. [VERIFIED] Ross, Gordon, Bagnell：[A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)，PMLR 15 (2011)，DAgger 原始论文。
6. [VERIFIED] Stable-Baselines3 Contrib：[Maskable PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) 官方文档。

## Provenance and unresolved items

- 本报告由已登录的 ChatGPT Web `GPT-5.6 Sol + High` 研究回答整理而来；发送前设置已可见核验。回答后页面设置的二次核验调用因 `ab.chatgpt.com` 网络超时未完成，因此 metadata 保守标为 `status: incomplete`。
- 关键 OR-Tools、L-RHO、PPO、GAE、DAgger、MaskablePPO 资料已用独立网页检索核验；关于“CP-SAT 在本项目上优于 MILP”“某个 H 一定更好”“RL 会稳定提升”等均仍标为 `UNVERIFIED`，必须通过本地实验。
- 本轮没有更新 `.project/STATE.md` 或 `.project/PLAN.md`，没有实现代码、下载依赖、删除数据或启动远端实验。
