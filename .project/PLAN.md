# Plan

**Last Updated:** 2026-09-15

## Current Objective

按系统论文定位推进：**"Predicting What Agents Will Do Next: Future-Action-Aware Scheduling for Multi-Model Video Agents"**。
贡献排序（GPT 重分析，`docs/research/2026-09-15_fas_system_story_gpt.md`）：
- **C1 主贡献**：future-action-aware scheduling system（把"请求会不会继续、还剩多少节点、风险多大"作为一等调度状态）。
- **C2 机制贡献**：不是越精细越好——oracle factorial 表明价值来自 continuation/horizon；精确 node identity 与 point resource 非瓶颈；runtime-tail 才是有效消费形式。
- **C3 supporting**：forecast→artifact→scheduler→oracle/ablation 评测管线（J 预测器是系统组件，不单独声称 SOTA）。

**新颖性边界（必须准确表述）**：Parrot（OSDI 2024）已把**已知** application DAG 暴露给 serving；
我们的对象是 **agent 运行时动态展开、未来尚不存在的控制流**，因此预测 future action sequence。
Related Work 定调句：*Prior systems exploit known workflow semantics, current execution state, or request-level workload predictions;
our target is the latent future control flow of dynamically unfolding agents.*

## 实验总表（GPT，2026-09-15；详见 docs/research/2026-09-15_fas_experiment_table_gpt.md）

| 实验 | 目的 | 设计/臂 | 基线 | 通过判据 | 成本/依赖 | 优先级 |
|---|---|---|---|---|---|---|
| 真实 GPU replay | 回应"仿真 artifact" | replay 单位 = episode（保留到达序列/依赖/deadline/决策点）；2×同型号 ≥24GB GPU；固定 3–5 个代表模型的可运行 stack；100–200 episodes；E0/E2/q95/Oracle-H5；重复 3 次 | E0/E2/q95/Oracle | q95 对 E0/E2 真机仍显著；策略排序不反转；sim↔real 误差 <10–15% | 2 GPU；1×3060 6GB 只能做节点级 latency/replay fidelity，不能替代 | **P0** |
| 投稿基线矩阵 | 回应"只打赢弱 baseline" | FCFS；SJF/LTR-style；E0；E2；q95；STS-style；SOLA/SlackFit-style；Parrot-like known-DAG；Oracle-H5/H10 | 同左 | q95 至少显著优于 FCFS/SJF/E0/E2/state-aware；已知图/oracle 为上界 | 8–9 臂，现有 simulator 数小时 | P0/P1 |
| 6-cell pressure sweep | 回应"只在一个压力点有效" | arrival {0.7,1.0,1.3}× × deadline {tight,loose}；每 cell 跑 E0/E2/q95/Oracle | E0/E2 | q95 对 E0 ≥5/6 cell 不劣且有获益；对 E2 ≥4/6 正收益；高压力区收益不消失 | 24 arms ≈ 4 小时 | P1 |
| Forecast-quality sensitivity | 回应"单 checkpoint 偶然命中" | length 混入 10/25/50% 噪声；runtime-tail 从 q95 收缩向 p50 25/50/75%；content 随机替换 10/25/50% | 原 q95 | 剂量-反应式退化；content 破坏影响小反而支持机制 | 12–20 arms，2–4h | P1 |
| q95 相关尾风险机制 | 回应"q95 只是调参/长度惩罚" | matched length 下比 p50/p90/p95；构造 ρ=0/0.5/1 相关性；控制节点 tail 共振 | length-only、CVaR | 优势随 contention/deadline pressure（及 ρ）系统增强；否则降级为 heuristic | <1 天 | P1+ |
| 接口 factorial | 回应"到底消费了什么" | length-only / runtime-tail-only / length+tail / content+length+tail | E0 | length+tail ≥ 单因素；content 不应成为隐藏混淆 | 大部分已有 | 已有/补齐 |
| H10-lite | 回应"为何只有 H5" | H5 backbone/near head + 6–10 步 coarse auxiliary head；3 seeds | frozen H5 | 前 5 步 NI≤2%；关闭 oracle H5→H10 gap 的 ~30% | 重训，3060 可做 | P2 |

**三档冻结**：
- **最小可发表集** = 现有主实验 + FCFS/SJF/state-aware/termination-aware 基线 + 6-cell sweep + forecast-quality sensitivity（→ ICPP/CCGrid/Cluster/FGCS/JPDC）。
- **推荐集** = 最小集 + 2-GPU 真实 replay + sim-real fidelity + q95 相关尾风险机制（→ MLSys/ATC/EuroSys 认真尝试）。
- **完整集** = 推荐集 + H10-lite + 第二 workload（电梯 Agent）+ 冻结后一次性 `T_final`。

**真机执行细节**：不要追求完整原模型栈；两张 24GB GPU 上固定一个代表性 stack（planner/LLM + VLM + detector/segmenter + CPU tool），保留模型切换/异构节点/并发/依赖即可；模型集合实验前冻结。

## 工业场景（电梯检测 Agent + 私有数据）：定位为第二 workload / case study

- **净加分前提**：它必须构成不同的 workload distribution（分支触发、模型组合、链长、SLO、Repair 分支概率不同）→ 直接解决"单 workload family"质疑。
- **减分风险**：私有数据不可复现；只展示案例 = demo；为电梯调参反而削弱通用接口主张。
- **做法**：不把私有数据当主 benchmark；**冻结现有 Future-Action 接口与 q95 consumer**（只重新生成该 workload 的预测 artifacts）；公开脱敏 DAG/action trace、模型类别、runtime/load trace、到达/deadline 分布、分支统计、schema + replay driver（视频不可公开则给脱敏 trace + synthetic trace generator）。
- **最小工业实验**：E0 / E2 / q95 / known-future oracle 四臂，在真实到达 trace replay 上报告 completion/miss/queue；有两张卡就直接真机跑。
- **投入顺序**：第一优先仍是 2-GPU replay；最理想是把电梯 Agent 作为**同一 2-GPU replay 的第二 workload**——一次工程同时解决"无真机"和"单 workload"。

## Steps（GPT 给的最小补充实验）

1. **P0 真实 GPU replay / 小 prototype（最大短板，优先）**
   - 最低配置：2 GPU + 几十~几百条 replay workload；比较 E0 / E2 / q95 / oracle；
   - 验收：q95 相对 current-node baseline 在真实 replay 上有明确正收益；仿真对策略排序不反转；报告 sim-vs-real 延迟误差。
   - 前置：确认可用硬件（本地 1×RTX 3060 6GB；远端 autodl 盒子的 GPU 情况待确认）与可跑的模型子集。
2. **P1 workload-pressure sweep（无需新数据）**
   - 用 600 视频池改 arrival rate / deadline multiplier / GPU contention / model mix → low/med/high load × tight/normal deadline ≈ 6 cells；
   - 主比较 E0 / E2 / q95 / oracle；验收：future-aware ≥5/6 cell 不劣，q95 ≥4/6 显著优于普通 future-cost。
3. **P1 forecast-quality sensitivity**
   - 不重训：对 frozen artifact 人工控制 length error / runtime quantile calibration / content corruption，画 scheduling gain vs forecast quality 曲线；
   - 验收：连续可解释退化 + oracle 端点。
4. **P2 H10-lite**（非必要条件；H10 oracle 已说明 horizon 有未利用价值；朴素 H10 失败可作诚实边界）。
5. **投稿策略**：纯仿真 → ICPP/CCGrid/Cluster（TPDS/TCC 需更多 sensitivity+分析）；
   补真实 replay + simulator fidelity + pressure sweep → 认真冲 MLSys，ATC/EuroSys 值得尝试；
   OSDI/NSDI/SOSP 还需多 workload/多栈 + 更强算法或理论。

## Story（GPT 定稿骨架）

- **卖点句**：Agent 调度的关键缺失信息不是"当前节点有多贵"，而是"这个请求接下来会做什么、还会走多远"；我们预测尚未生成的未来控制流，并证明这种未来状态能被风险感知调度器直接转化为端到端收益。
- **三段式**：问题（动态展开的 workflow 对 scheduler 是 latent-future problem；Parrot 要求显式暴露，我们预测尚未存在者）→ 洞察（不是越精细越好：oracle resource≈table、真值内容更差；continuation/horizon 主导；runtime-tail 消费有效）→ 结果（−17.1s / −18.8s confirm / H10 oracle −10.2s）。
- **问题正式写法**：*Dynamic agent execution hides near-future control flow from the resource manager. Decisions based only on currently-ready nodes ignore downstream execution externalities. Existing systems either assume the workflow is known, optimize current execution state, or predict a scalar request/job length. We ask whether the latent future action sequence itself can serve as a scheduling state.*
- **慎用表述**：不能说"首个 workflow-aware serving / state-aware / slack-aware / heterogeneous workflow serving / prediction-driven scheduling / 首次利用 termination uncertainty"。

## Baselines（投稿必需）

- **最低集合**：FCFS、SJF/LTR-style（预测剩余工作量）、E0（当前节点贪心）、E2（future-action+静态成本）、q95（future-action+尾部成本）、STS-style termination-aware、Oracle-H5/H10。
- **最加分两个**：Parrot-like known-DAG oracle（给真值未来图但用同一资源模型）+ SOLA/SlackFit-style state-aware（只用当前 slack/queue）。
- 真实 replay 的最低集合：FCFS / current-state / LTR-like / q95（Orca/vLLM continuous batching 不是公平的端到端多模型 baseline）。

## Venue 目标（GPT 主观概率，纯仿真 → 补真实 replay）

- 期刊：FGCS 55%→70%、JPDC 45%→60%、TCC 30%→50%、TPDS 25%→45%、TC 15%→30%（TOCS 10%→25%；TACO <10% 不建议）。
- 会议：CCGrid/Cluster 45%→60%、ICPP 35%→50%、HPDC/ICDCS 20%→35%、SoCC/Middleware 20%→35–40%、IPDPS 15%→28%、
  **MLSys 12%→25–35%（高档最匹配）**、ATC 10%→20–28%、EuroSys 6%→12–18%、NSDI 4%→8–12%、SC 5%→10–15%、OSDI/SOSP <3%→5–8%。

## Reviewer 预演（必须准备防御）

- R1：这是 simulator artifact → 2-GPU replay + sim-vs-real error（没有它 OSDI/EuroSys 很难防）。
- R2：单 workload + 单 predictor → pressure sweep（≥5/6 cells）+ artifact corruption/sensitivity。
- R3：q95-sum 只是调参 heuristic → frozen confirm + 多方对比 + 相关性/压力机制分析；否则降为 implementation choice。


## 评审行动项（GPT 2026-09-15，见 docs/research/2026-09-15_fas_results_review_gpt.md）

**表述修正（写作层）**
- oracle 臂改名：`trueopt_h5` → **OracleInfo-H5-Point**；`oracle_topology_h5_tab` → **OracleTopo-H5-Table**；
  留 "oracle upper bound" 给真正的理想策略。禁止写 "prediction outperforms oracle"；
  正确表述：*our risk-aware policy outperforms point-cost policies even when the latter are given oracle H=5 topology/resource information,
  indicating decision-rule design can dominate information accuracy.*
- 压力 sweep 表述为 **contention-sensitive**（tight/loose 中性，不能写成 deadline-risk-specific）。
- FCFS 实为 **Priority-FCFS**（priority 外生固定，所有策略一致）——论文需注明。
- `sjf_pred` 改名 **Length-SJF**（不声称复现 LTR）。
- 敏感性中 content 的结论只能写 "content fields have no measurable marginal effect under the q95 resource-consumption path"。

**代码修正（实现层）**
- `sjf_pred`：把 `per_step = 当前节点表成本` 改为 train-only **lane 级平均成本**（`R = current + Σ_{{h=1..L}} c̄_lane`），避免"每步都冷加载"的系统高估。
- `state_aware`：改为 **laxity-first**：`score = (priority, laxity, current, ready_time, ...)`，`laxity = (deadline − now − current)/window`；
  不再用 miss_flag 把负 slack 任务排到最后（除非明确采用 drop/late-job 策略并单独定义）。
- `pool = fitting or candidates` 是全体策略共用（非臂间不公平）；本仿真在派发层做准入/驱逐（每集 38–45 次驱逐、全部运行 0 容量违规），
  需在论文中说明该 fallback 由后续准入层兜底。

**新增对照实验（成本低，C2 因果性必需）**
1. **scale-matched p50**：`C = k·Σ p50_h`，k 在 train/dev 上固定使平均 future cost 与 q95 相同 → 比较 raw p50 / scaled-p50 / p90 / p95；
   若 q95 ≫ scaled-p50 才能排除"只是把未来权重放大"的替代解释。
2. **tail-shuffle**：保留每 episode 的 (p95−p50) gap 分布但在节点/链间打乱 → 若真 q95 明显优于 shuffled-q95，说明尾部幅度必须与具体 future state 对齐。
3. **oracle-topology + J3 predicted q95**：只替换 topology 真值、消费者完全一致，与 predicted topology + q95 对比；
   若更差，说明预测结构充当 implicit regularizer，需要解释。


## 消费策略重写清单（GPT 代码审查，2026-09-15）

**必须暂停的表述**："q95 战胜所有分布消费方法"、"预测资源优于静态表"（E2→pred-p50 实际 +16.0s，pred-p50→q95 才是 −33.7s）。

**P0（重跑后才能正式引用）**
1. 修缓存键：`_SCEN_COST_CACHE` 加入 artifact/stats 版本（现仅 node_id/horizon/samples/coupling，扰动 artifacts 会读到旧结果）；
   `_JRES_COST_CACHE` 去掉 `id(cost_fn)`，改用显式 (policy_name, parameters) 并含 train_stats。
2. 修表查询：`_step_estimate_cost`/`_table_step_components` 改为真正的层级查找（model_lane count≥3 → lane count≥3 → fail），
   不再把 exact/model_lane/lane 三层统计量混在一起取中位数。
3. 建立 **runtime-only 正交消费族**（future load=0；相同 current_cost、长度链、fallback）：
   `Pred50-R`、`Pred90-R`、`Pred95-R`、`ScaledPred50-R`（k 在 dev700 预冻结使平均 future cost ≈ p95）、`ShuffledTail`（跨 step/node 打乱 p95−p50）。
   判据：Pred95-R ≫ ScaledPred50-R 且 ≫ ShuffledTail 时 C2 才成立。
4. Stage0 校验：分位数单调（p50≤p90≤p95）、有限非负、缺失/fallback 率必须报告。

**P1（要保留 CVaR 结论时必须）**
5. artifacts 保留完整 H 个 slot（现在链被 argmax 截断 → 采样只能变短）；6. 重写分位数→分布重建（现 u≤0.5 从 0 线性升到 p50、u>0.95 外推 9×(p95−p90) 均为人为假设）；
7. N≥128；8. 真正采样 scenario_probability / model / length 分布；CVaR 目标改为 (1−κ)E+κCVaR 或纯 CVaR。

**P2**
9. comonotone 全分量耦合（含 load）；10. adaptive slack 计入 current cold-load；11. load occurrence 语义统一（表 load 是条件正时长，预测臂是 occ≥0.5 门控）。


## 定位修正与 BC 重跑建议（GPT 定向文献检索 2026-09-15）

**最危险的 prior**：TIE — *Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions*（ICML 2026，正式录用）：
把**单个请求的输出长度**建成 heavy-tailed log-t 分布，用 Tail Inflated Expectation 代替点长度进入 SJF。
→ 因此不能写 "first uncertainty-aware predictive scheduler"。
**安全表述**：*To our knowledge, this is the first systematic study of how multi-step future-action uncertainty should be consumed for scheduling dynamically unfolding agent workflows.*

**Related Work 三层结构**：
1. future workflow **已知**（Parrot OSDI'24、Katz ATC'25）；
2. future uncertainty 是**单一标量或终止分布**（LTR NeurIPS'24、TIE ICML'26、STS TCC'24）；
3. 开始进入 **agent future**，但用途是 cache / downstream state（PBKV arXiv'26、FATE arXiv'26）；
4. 我们：多步 action sequence + 每步资源分布 + 随机终止，研究**结构化 forecast 的消费函数**。

**学习型调度器趋势**（RLTune SoCC'25 把 RL 限制到 priority、placement 交给 MILP；ILETS FGCS'24 用 BC 初始化 DRL 缓解样本效率/振荡）：
→ "RL 全面战胜启发式"不成立；我们的旧结果（BC≈专家但不如启发式、PPO 明显差）不是异常。未见成熟的 "future-agent-prediction → learned policy" 路线。

**BC 重跑建议（值得做；PPO 非必须）**：
- 目的：堵住 "为什么不用 learned policy 自动学会消费" 的审稿问题；
- 要求：teacher 与 BC 看**完全相同的 future artifacts**；teacher 取当前最强的优化器或 q95；~200k–300k masked decisions、3 seeds；
  **按 episode/workflow 切训练/验证**（不能按 decision 随机切，否则泄漏）；指标除一致率外还要 completion/miss、teacher gap、q95 gap、inference overhead；
- 评测：paired 到相同 episodes，至少含 E0 / E2 / q95 / expert / BC；
- 两种结果都有价值：BC≈expert 但输 q95 → "gap 不是模仿失败，而是手工的未来风险代理与调度目标更对齐"；BC>q95 → "future-action prediction enables both analytic and learned consumers"。


## 学习消费函数（Phase 14 计划，GPT 2026-09-16）

**目标**：冻结预测器与调度器动力学，只学习 `C_θ(future forecast, system state, action) → 标量代价`，以下游 episode 指标为训练信号。

**排序（GPT 建议）**
1. **低维参数化消费函数 + CMA-ES/贝叶斯优化**（★★★★★）：参数 8–15 个（runtime τ、horizon 折扣 γ_h、长度权重、队列压力权重、slack→risk 系数、load 权重、残差尺度）；
   inner search 用 100–200 episodes，前 10 名再到完整 dev700 重排；这是绕开不可微仿真器的直接路径。
2. **q95 + 学习残差 scorer**（★★★★★，论文最漂亮）：`C_θ = C_q95 + g_θ(z)`，g 为 2×32 MLP/GAM；
   输入 z = q95-sum、p50-sum、tail gap、P(h)、slack ratio、队列压力、驻留、GPU-step 数、模型切换数；初始化即当前最强策略。
3. **Counterfactual Learning-to-Rank**（★★★★）：标签不是"专家选谁"，而是对候选做 rollout 得到"选它的 episode 代价"，用 pairwise/ListMLE 排序损失；理论可超过教师；30k–80k states × 2–6 候选。
4. **迭代策略改进（DAgger 式）**（★★★★）：rollout 当前策略 → 收集真实访问状态 → 反事实评估候选 → 更新 → 重复 2–3 轮。
5. **Offline RL**（★★，暂不优先）。

**超越专家的三条路线**：① 直接黑盒优化 episode 回报（无需教师）；② 反事实监督（发现专家没选的更好动作）；③ 教师集成 + 策略改进。

**最小可信实验**：特征 20–30 维；模型只用"低维参数化"或"MLP 32×32"两类；dev700 内按 workflow 分组 500/200 调参，300 confirm 只跑一次；3 seeds；
对照 E0 / q95 / CP-SAT / MPC / BC-expert / learned-parametric / learned-residual；主指标 mean completion，次指标 miss、p95、queue、决策开销；
必须报告 dev→confirm 泛化差、seed 方差、与 q95 的相关性、特征消融、推理延迟。

**预注册成功判据**：
- 强成功：learned vs q95 在 confirm 上 paired CI 上界 < 0 且 miss 非劣 → "learned consumption surpasses the strongest hand-designed consumer"；
- 只赢 BC/CP-SAT：learning recovers strong behavior but does not improve on the best hand-designed surrogate；
- 与 q95 持平且参数退化到 τ≈0.95 → "q95 is near-optimal within a much broader learned family"（同样是有效结论）。

**坑**：dev700 已复用多次 → 冻结模型类别/搜索预算/目标函数；奖励作弊 → 固定 `J = mean completion + λ·max(0, miss − δ)`；可解释性 → 主模型用 "q95 + 稀疏残差" 并报告残差随 slack/pressure/length 的变化。


## 事件链风险聚合：理论定位与表述（GPT 2026-09-16）

**核心结论**：r95 在数学上 = **共单调耦合（comonotonic coupling）下总时长的 VaR**：
若各步 runtime X_1..X_H 共单调，则 `VaR_α(Σ X_h) = Σ VaR_α(X_h)`（Dhaene et al. 2002；Kaas et al. 2002 的 convex-largest sum 定理）。
→ 我们的规则有正式名称：**comonotonic upper-tail aggregation surrogate / tail-comonotonic VaR aggregation**。

**表述纪律**：
- 不要写"每一步的最坏情况同时发生"（p95 不是 max）；准确说法是"**各步处于同一高分位 rank 状态**"。
- 不要写成"正相关必然拉长平均工期"：PERT 文献（Banerjee & Paul 2008）表明串行和只影响方差/尾部，并行 max 下正相关甚至可能降低期望 max。
- 强调"相关性通过调度排序/资源争用被放大成系统性能差异"（PMS 2026 的结论：相关性对风险/方差的效应远大于对均值的效应）。

**必引三篇（正式发表）**：
1. Dhaene et al., *The Concept of Comonotonicity in Actuarial Science and Finance: Theory*（IME 2002）；
2. Kaas et al., *A Simple Geometric Proof that Comonotonic Risks Have the Convex-Largest Sum*（ASTIN Bulletin 2002）；
3. Hua & Joe, *Tail Comonotonicity…*（IME 2012，尾部共单调的渐近可加性）。

**其它支撑文献**：Ringer 1971（PERT 依赖）、Banerjee & Paul 2008（PERT bias）、Yang 2007（copula/NORTA）、
Boys et al. 1997（dependent processing times）、Ruszczyński & Shapiro 2006 / Ruszczyński 2010（多期嵌套风险）、
Chung et al. 2009（time diversification：独立 σ~√H vs 完全相关 σ~H —— 正是我们"独立采样会把尾部平均掉"的解释）。

**最近邻近期工作**：**LLMSched（ICDCS 2025，正式）**——把 compound-LLM workflow 建成不确定 DAG（Bayesian network），
利用阶段间相关性与 mutual information 优先执行"能降低后续不确定性"的 stage；**但没有比较 comonotonic VaR / 独立 sampling CVaR 等聚合方式**。

**贡献表述建议**：
> r95 is a comonotonic upper-tail aggregation surrogate: it sums per-stage VaR_0.95, equivalent to the VaR_0.95 of the aggregate latency under a comonotonic coupling of the marginal stage-latency distributions.
> 经典风险理论给出共单调聚合的极端依赖解释；我们的贡献是**在动态 Agent workflow 调度中实证发现**该 tail-aligned conservative coupling 优于独立/场景化 CVaR，
> 并用 scale-matched 与 tail-shuffle 对照证明收益来自**步骤特异的尾部对齐**而非简单放大。

**"首个"的安全写法**：*to our knowledge, the first systematic evaluation of dependence-aware multi-step uncertainty aggregation for dynamic agent-workflow scheduling*（投稿前再做最终查新）。


## 事件链形式化与证明路线（GPT 2026-09-16）

**新文献**（正式发表优先）：调度侧最近的是"先逼近整个 makespan 分布再算 VaR/CVaR"（Liu & Urgo IJPR 2024；Meloni et al. IJPE 2022；Computers & OR 2023），
不是"逐步边际 quantile 相加做决策代理"；依赖不确定性有 Branda（C&OR 2018，copula in ambiguity set）、Novak et al.（EJOR 2022）；
**De Vecchi–Nendel–Streicher（Mathematical Finance 2026）**：即便只有很弱的正依赖信息，允许的极端尾部仍可达到完全相关水平 → 为"保守尾部耦合"提供现代依据。

**四个命题（论文骨架）**
- **A 精确性**：若存在公共 U，使 R_ah = F_ah^{{-1}}(U)（同一视频/会话的"快慢状态"作用于所有步骤），则 VaR_α(S_a) = Σ VaR_α(R_ah) = C_α(a)。
  **是定理不是近似**（Dhaene/Cheung 的 quantile representation；Kaas 的 convex-largest）。
  可检验：同 workflow 跨步 Spearman/Kendall、90/95% co-exceedance、upper-tail dependence。
- **B 上界纪律（重要）**：只有 **ES/CVaR** 才有"共单调给出上界"的严格结论（S ≤_cx S_com ⇒ ES_α(S) ≤ ΣES_α(R_h)）；
  **VaR 一般没有次可加性**，所以 **不能写"r95 是任意依赖下的上界"**。若要任意依赖下 VaR 上界需 union bound：α_h = 1−(1−α)/H（H=5、总体 95% → 单步 ~99%）。
  → r95 是"有结构假设的保守代理"，不是无条件鲁棒界。
- **C 只需上尾共单调**：Cheung 的 upper-comonotonicity / Hua & Joe 的 tail-comonotonicity ⇒ α 高于阈值后 VaR/TVaR/ES 重新可加。
  **可直接用数据检验**：画 `VaR_q(ΣR_h) / ΣVaR_q(R_h)` 随 q=.5/.75/.9/.95/.99 的曲线；若随 q 上升趋近 1 = 极强机制证据。
- **D 排序目标更弱**：只需存在共享严格增函数 g 使 E[S_a|x]=g(C_α(a))，则 C_α(a)<C_α(b) ⟺ E[S_a]<E[S_b]；
  单机两作业用标准 interchange argument 可证。但我们的多 GPU 系统不满足全局条件 → 写成 **local ranking lemma**，不声称 q95 全局最优。
  （scale-matched 与 tail-shuffle 恰好支持 D：价值来自 candidate-specific 的尾部排序而非绝对尺度。）

**比 r95 更好的候选（按成本）**
1. **最便宜：upper-tail hybrid** `C = Σp50_h + β·Σ(p95_h − p50_h)`，β 由**实测 tail co-exceedance** 决定（不再手调）；
   若 β≈1 且性能≈r95 → "r95 不是偶然 heuristic，而是数据中的 common-tail dependence 推出的近共单调最优消费"。
2. **中等：dependence-adaptive tail aggregator**：dev 上估 upper-tail dependence λ、用一因子/t-copula 生成联合 runtime → 算 VaR_0.95(Σ) 作为分数（H≤5，CPU 足够）。
3. **重：dependence-ambiguity DRO**：边际固定、Kendall/Spearman/tail-dependence 落在经验置信区间，求 worst-case ES（理论强、实现重）。

**立刻可做的机制实验**：命题 C 的比值曲线（便宜、直接）。


## 验证设计（GPT 评审，2026-09-16）：公共慢化因子与尾部共动

**叙事修正（重要）**：主实验的"风险"不是重复执行的 aleatoric 噪声，而是**预测不确定性**（预测误差 + 跨样本异质性）；
不要叫 epistemic（除非模型是 Bayesian/ensemble），准确说法 = **conditional predictive uncertainty**。
论文建议句：*The simulator replays fixed measured node runtimes; the predictive quantiles represent forecast uncertainty across heterogeneous
executions, not repeated-sampling noise. Tail-aware consumption therefore hedges forecast error and latent workload heterogeneity.*
三类不确定性要分开：execution uncertainty（我们没有）/ forecast uncertainty（我们有）/ latent shared heterogeneity（检验 1–3 验证）。

**检验 1（混合效应方差分解）**：`log1p(runtime) ~ type + stack + baseline + stack×baseline + b_video + b_run(video) + ε`；
报告 ICC_video 与 ICC_run；**识别限制**：每 video×stack×baseline 只有一次 run → 只能说 "run-level shared heterogeneity"，不能说因果慢化；
稀有 type 合并到 family；bootstrap 单位 = **video**（连带 4 条 run 与全部节点）。

**检验 2（残差共动）**：只移除 type+stack+baseline（版本 B 再去 video），**不能移除 run 随机效应**；
run×type 先聚成一个标准化残差（median/mean），再做 type-pair Spearman/Kendall；不要把同 run 内节点当独立样本。

**检验 3（尾部共动）**：χ_q = P(Z_A>Q_A(q) | Z_B>Q_B(q))（独立时应 ≈1−q）；另报 Lift_q = P(both)/(1−q)²；
q=.8/.9 为主、.95 为 exploratory；video-cluster bootstrap B≥2000 + 分层置换零假设；主结果报 pooled/weighted tail lift + CI。
统计依据：Ledford–Tawn 1997（JRSS-B）、Asimit et al. 2016。

**执行顺序**：ICC → 残差共动 → 尾部 lift；**只有至少中等证据**才做相关重采样扩展（单因子 λ_t 模型 + 经验 CDF
quantile transform，先验证能复现 ICC/Spearman/co-exceedance，再做 ρ-strength ∈ {0,0.5,1} × (independent-CVaR vs r95) 的机制 sweep）。

**若检验 1–3 很弱 → 转向替代机制（仍可成立）**：
① **排序 margin 放大**（p95 增大候选分数间距 → 误差不易翻转排序；测 Δscore 与排序准确率）；
② **异方差难度信号**（p95−p50 是"难预测"的 proxy；测 tail-gap 与真实 |R−Q₀.₅|）；
③ **隐式 future-importance 加权**（候选级排序准确率：q95 vs p50/scaled-p50/shuffled-tail）。
此时诚实写法：*r95 is effective as a forecast-error-aware ranking surrogate, even though strong physical tail dependence is not observed.*

**最该引用的方法文献**：Werner 2021（Frontiers，DES 服务时间尾部依赖/共同潜因子，最直接先例）、Yang 2007（NORTA/copula 任务时长相关）、
Ledford & Tawn 1997（JRSS-B，joint tail dependence）。


## 公开仓库审阅修复清单（GPT 2026-09-17，仓库 QiuweiLiu/scheduler）

**P0（必须修，修完需重跑相关基线）**
1. **CP-RHO 量纲 bug**：`end_cost = 1000×weight×end_ms` 而 `future_cost = weight×future_ms` → future 项被缩小约 1000×；
   fallback 又整体除以 100000，语义不一致 → 修权重后**重跑 cp_rho_h3/h5**。
2. **runner resume 无指纹**：现仅按 `(episode_id, policy)` 跳过，换代码/输入/artifact 后会**静默混入旧结果** →
   加 git SHA + 输入/代码/artifact hash 指纹，mismatch 直接 fail。
3. **normalizer manifest 失效**：`normalize_full5_artifacts.py` 改了 gzip 内容却复制旧 `b05_artifact_manifest.json` → 重算哈希。
4. **tail-shuffle 生成脚本未公开**（现在只在 `.scratch/phase11_stats.py`）→ 移入 `scripts/` 并写入 README。
5. **requirements**：README 引用的 `configs/requirements.txt` 实际不存在 → 提供真实 `requirements.lock` / `environment.yml`。

**P1**
6. T1/T3 的 bootstrap/permutation 统一 ≥2000，并给**双侧/下尾** p 值（当前只检验正 lift，无法证明负尾依赖）。
7. 补 T1 的 rich fixed-effect 敏感性（type×stack、type×baseline）。
8. 补 **fixed-argmax-L** 的 scenario/comon 对照（现版本 length 仍独立抽样 → 严格说是 conditional-comonotone）。
9. 公开 confirm300 的 ID 清单与 paired-bootstrap 统计脚本。

**P2**
10. 加 LICENSE、seed/provenance 总表、一键 `reproduce_main.sh`；
11. README 修正：改为"主仿真可复现；原视频/原始 trace/权重不可再生"。

**表述纪律（补充）**：MOM 只能称描述性矩估计（非无偏方差分量估计）；场景族结论需限定 "under our lognormal reconstruction"。

## Constraints

- 只在 dev700 调参；confirm300 已用于 q95 确认，再用需降级证据等级。
- 正式实验前写门禁；负结果不重跑；不读 `T_final`；不在 `S_*` 上拟合。
- 进论文前核验 GPT 引用的全部论文（Parrot/SOLA/SuperServe/Katz/Vidur/FATE）。

## Current Blockers

- P0 的真实 GPU replay 需要硬件确认（本地单卡 6GB / 远端 GPU 可用性）—— GPT 判定这是提档第一名实验。
- 第二名实验 = 6-cell pressure sweep（E0/E2/q95/oracle，成本低）。
