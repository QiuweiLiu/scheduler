---
research_backend: ChatGPT Web
requested_model: GPT-5.6 Sol
requested_reasoning: High
verified_before_submit: true
verified_after_response: false
timestamp: "2026-08-21"
status: valid
scope: "动作空间、调度紧迫性、基线可分性与压力 workload 的研究"
---

# 动作空间与调度紧迫性：论文、当前证据和网页版 GPT 交叉研究

## 0. 结论先行

用户的判断基本成立，但需要把问题说得更精确：**当前 workload 不是完全没有压力，而是典型决策点的候选动作仍偏窄、deadline 的紧迫性没有被独立控制，导致预测式策略和学习策略能利用的“决策差异”被压缩。**

这不等于“当前结果全部无效”。R7 的 Myopic 与 Oracle 仍有明显差距；问题是现有 workload 没有把这种差距稳定地转化成“预测器/调度器必须做出不同选择”的场景。尤其是 PredOpt-H5 只比 Myopic 好约 2%，而 TrueOpt-H5 和 Oracle 的差距明显更小，说明当前评分器/信息边界/候选池的组合仍然可能让简单排序器占据优势。

本次网页版 ChatGPT 的独立建议与论文核验后的主结论一致：后续不应先扩大模型或先重训 RL，而应先建立三个门禁：

1. **动作空间门禁**：每个决策事件实际有多少个合法的 `ready-node × free-GPU` 动作，而不是只看 GPU 数量；
2. **紧迫性门禁**：队列、deadline slack、到达速率和资源占用是否持续把系统推入需要取舍的状态；
3. **可分性门禁**：Oracle、Myopic、有限前瞻和学习策略是否在同一信息边界下产生足够不同、且可复现的动作与结果。

只有这三个门禁通过，才值得把 CP-RHO、BC/PPO 或更大的 GNN 当成主实验。

## 1. 本次网页版 ChatGPT 研究的范围和可信度

我在已登录的 ChatGPT Web 中新建了研究对话，发送前肉眼确认模型为 **GPT-5.6 Sol、推理强度“高”**。brief 只描述了脱敏后的调度问题、当前 2-GPU/16-64 jobs/动态到达/多种 baseline，以及“动作空间和 urgency 可能不足”的疑问；没有发送 API key、远端路径、原始日志或私有文件。

网页版 GPT 返回的核心建议被拆成三类：

- **VERIFIED-like research guidance**：生产调度和 LLM serving 工作普遍把到达、队列、异构资源、SLO/deadline、DAG/依赖和缓存状态作为调度难点；
- **INFERENCE**：本项目应把 `ready-node × free-GPU` 分支数、deadline slack、负载、cache miss/eviction 和 runtime tail 做成显式 workload 因子；
- **UNVERIFIED proposal**：例如“候选数中位数至少 4–6”“p90 至少 10”“利用率 85–95%”等是实验起始建议，不是任何论文规定的通用阈值，必须用本项目 pilot 校准。

回答正文完整返回；回答后没有再次打开模型菜单，因此本报告不把“回答后的 UI 二次核验”写成已完成，但没有看到自动降级或配额提示。网页版 GPT 的建议只作为研究输入，下面的事实以论文、官方文档和当前项目产物为准。

## 2. 当前项目的可复核事实

### 2.1 R7 workload 与策略矩阵

远端 `r7_scheduler_matrix_validation_1000_final`：

- 1,000 validation episodes、135 scenario cells、10 policies、10,000 result rows；
- 34,160 jobs、0 failed jobs、0 capacity violations；
- workload 训练侧为 20,000 episodes、640 templates、约 74 万 jobs；
- GPU topology 仍是 2 张卡的同构/小显存/混合三类；
- arrival pattern 有 Alibaba replay、burst、poisson；pressure 采样覆盖 0.5、0.7、0.85、0.95、1.05；
- initial cache state 有 cold、skewed、warm；
- aggregate 仍有 GPU eviction，说明资源/驻留成本不是零，但还没有证明它在多数决策点改变了首动作。

当前完整矩阵的关键均值如下（completion 单位为 ms）：

| 策略 | mean completion | deadline miss | mean queue | evictions | 相对 Oracle completion gap |
|---|---:|---:|---:|---:|---:|
| Oracle | 172,732.90 | 6.08% | 78,938.06 | 39.81 | 0 |
| Myopic | 215,116.40 | 10.98% | 123,365.33 | 41.06 | +24.54% |
| Optimizer-0 | 215,646.79 | 10.98% | 123,494.74 | 44.64 | +24.84% |
| PredOpt-H1 | 213,049.76 | 10.68% | 122,116.32 | 41.21 | +23.34% |
| PredOpt-H3 | 212,387.92 | 10.70% | 121,549.71 | 41.15 | +22.96% |
| PredOpt-H5 | 210,635.10 | 10.47% | 120,369.82 | 39.91 | +21.94% |
| TrueOpt-H3 | 195,021.86 | 8.56% | 126,669.97 | 50.82 | +12.90% |
| TrueOpt-H5 | 183,845.15 | 7.30% | 110,661.43 | 46.22 | +6.43% |

因此，不能简单说“基础模型太强所以所有结果没有 gap”：Myopic 到 Oracle 的 gap 很大；更准确的说法是，**可部署的有限前瞻策略相对 Myopic 的可见收益很小，而拥有特权真实未来的 TrueOpt-H5 已经接近 Oracle**。这正是需要先审计候选动作和 urgency 的地方。

### 2.2 动作空间的直接诊断

模拟器在每个 dispatch event 中按以下方式形成候选池：

```text
候选动作 ≈ 当前 ready 的 GPU 节点 × 当前 free GPU
先筛 predicted-fit；如果没有 fit，则退回全部候选
```

对现有 `simulation_events.jsonl.gz` 的流式诊断（该文件是中断写入的 partial evidence，不作为最终指标）得到：

| 策略 | candidate count p50 | p90 | max | ready-node p50 | free GPU mean |
|---|---:|---:|---:|---:|---:|
| Myopic | 3 | 11 | 52 | 2 | 1.149 |
| Oracle | 2 | 7 | 46 | 2 | 1.152 |
| Round-robin | 3 | 13 | 76 | 2 | 1.146 |

这说明候选池并非永远只有一个动作，但典型事件仍然是 2–3 个候选，只有尾部事件才出现较宽分支。注意三种策略的事件文件不是最终完整审计证据，且不同策略的运行状态/随机复现需要继续核对；这里仅用于说明“不能只看 GPU=2”以及“中位数分支偏窄”。

### 2.3 当前 baseline 的解释

- `Myopic`：根据当前可见成本/资源状态选一步；不使用未来预测；
- `PredOpt-H1/H3/H5`：使用有限步预测，但仍受当前 score 的尺度和排序规则影响；
- `TrueOpt-H*`：使用真实未来的特权参考，不能视为可部署方法；
- `Oracle`：全局完美参考；
- `RL-H5`：当前三 seed 方向不一致，paired report 的 final-freeze gate 为 false，不应当作已经稳定胜出的方法。

所以，如果某个 stress workload 让 PredOpt/CP-RHO 明显拉开 Myopic，而 RL 仍不稳定，这不是失败，反而能区分“信息边界/优化器有效”和“学习器训练稳定性”两个问题。

## 3. 论文和官方资料告诉我们的事情

### 3.1 生产 GPU 调度不是只有“两个 GPU 选一个”

Tiresias 的生产分析把不可预测运行时、all-or-nothing 执行和长队列延迟作为核心问题，并用基于部分信息或无信息的排队策略降低 JCT；它还用 trace-driven simulation 与完美知识参考比较。这支持我们把 runtime tail、到达和 queue pressure 单独做成因子，而不是只改变 job 数量。[Tiresias（USENIX NSDI’19）](https://www.usenix.org/conference/nsdi19/presentation/gu)

Alibaba 的公开 GPU trace 覆盖真实生产的 GPU/机器/工作负载；其 GPU v2020 记录了两个月、6,500+ GPUs，GPU v2023 强调异构资源和 fragmentation，仓库还列出 2026 的大规模 GPU trace。论文和数据集把低利用率、长排队、难以放置的高端 GPU、异构机器不平衡和 CPU bottleneck 作为实际调度挑战。[Alibaba Cluster Trace](https://github.com/alibaba/clusterdata)；[MLaaS in the Wild（USENIX NSDI’22）](https://www.usenix.org/conference/nsdi22/presentation/weng)

Gavel 明确把异构加速器上的“任务性能不同”形式化为优化问题，并通过 round-based scheduling 提高异构集群上的有效吞吐；这说明 GPU 类型/容量和 job-to-GPU 性能差异应当进入动作价值，而不是只用 GPU index 打破平局。[Gavel（USENIX OSDI’20）](https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak)

### 3.2 workload 生成要保留经验分布，而不是随意造数字

Google 的公开 cluster-scheduler-simulator 说明其 workload 从经验参数分布生成，至少包含初始集群状态、job size、job inter-arrival time 和 runtime；它明确建议从公开 trace 提取这些分布。对我们来说，Alibaba replay 可以提供 arrival/queue 的真实形状，视频 trace 提供 node/runtime/memory 的任务形状，二者应分层合成而不是相互覆盖。[Google Cluster Scheduler Simulator](https://github.com/google/cluster-scheduler-simulator)

Pegasus 的官方 workflow gallery 保留了真实 workflow runs，并覆盖 Montage、CyberShake、Epigenomics、LIGO 等不同 DAG 结构；它支持“宽度/深度/分支/汇合”是独立结构因子，而不是把每个 job 都做成同一条线。[Pegasus Workflow Gallery](https://pegasus.isi.edu/workflow_gallery/)

### 3.3 urgency 的核心是 SLO/slack/queue，不是简单 deadline multiplier

Kubernetes 的官方调度文档把 pending queue、priority ordering、backoff 和 resource-pressure 下的 preemption 写成一等机制：高优先级 Pod 会排到低优先级前面，资源不足时可驱逐低优先级 Pod。这说明“谁已经等多久、谁的 deadline/priority 更紧、是否会挤占别人”应该显式进入 workload 与评估，而不是只给所有 job 一个同质化 deadline。[Kubernetes Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)

JITServe 针对包含多 agent/多调用依赖的 LLM pipeline，把 latency-sensitive、deadline-sensitive 和 compound dependencies 作为不同 SLO 类型，并在信息不完整、响应长度/依赖逐步显现的情况下做 goodput 调度。这和我们的节点级 agent trace 很接近：应至少区分“必须快速触发下一工具”的 job 与“最终完成时间重要”的 job。[JITServe（USENIX NSDI’26）](https://www.usenix.org/conference/nsdi26/presentation/zhang-wei)

Sarathi-Serve 研究 throughput–latency tradeoff，并在 tail-latency 约束下用 chunked-prefill/stall-free scheduling 改变可行动作与批处理组成；它支持我们同时报告 mean、P95/P99 和 deadline miss，而不是只报告平均完成时间。[Sarathi-Serve（USENIX OSDI’24）](https://www.usenix.org/conference/osdi24/presentation/agrawal)

### 3.4 cache/模型驻留会形成长时域耦合

vLLM 的官方 prefix-caching design 明确描述了 KV-cache block 的 hash、allocation、free queue 和 eviction：如果没有足够 block，新的请求不能分配；缓存命中/驱逐改变后续请求成本。对我们来说，model residency、load/eviction 和 memory headroom 应成为影响下一节点的真实状态，而不是只作为事后统计。[vLLM Automatic Prefix Caching](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)

Gandiva 和 Pollux 也说明了同一件事的另一面：Gandiva 利用 job 内可预测性、time-slicing 和迁移改善 latency/利用率；Pollux 联合优化资源分配和 job 内训练状态，并以 goodput 而非单一 JCT 作为目标。[Gandiva（USENIX OSDI’18）](https://www.usenix.org/conference/osdi18/presentation/xiao)；[Pollux（USENIX OSDI’21）](https://www.usenix.org/conference/osdi21/presentation/qiao)

## 4. 对“动作空间不够”的严格定义

不要把动作空间定义成“GPU 数量”。本项目至少要记录四个量：

```text
raw_actions       = ready_gpu_nodes × free_gpus
feasible_actions  = raw_actions 中满足显存/容量/模型加载约束的动作
decision_width    = feasible_actions 的数量
action_disagreement = 不同策略在同一 event 对首动作的分歧率
```

还要把以下比例按 episode、scenario cell 和 policy 分开统计：

- `feasible_actions == 1` 的事件比例；
- `feasible_actions > 1` 的事件比例；
- `decision_width` 的 p50/p90/p99；
- ready node 数、free GPU 数、队列长度的联合分布；
- 同一状态下 PredOpt、CP-RHO、Myopic、Oracle 的首动作分歧；
- 候选之间的 objective margin（最优和次优差多少）。

**推荐门限不是论文事实**：先用 pilot 观察，不要直接把“p50 必须 10”写进论文。一个合理的起始判据是：主 stress cell 不应几乎全是单合法动作；候选 p50 至少比当前 2–3 更宽，p90 应稳定出现多动作事件，并且 action disagreement 与 objective margin 都非零。最终阈值应由 pilot 的置信区间和可行性共同决定。

## 5. 对“调度紧迫性不够”的严格定义

当前 `deadline_multiplier=(1.5, 2.0, 3.0)` 是一个容易生成 workload 的参数，但不是充分的 urgency 定义。建议记录并控制：

```text
slack_ms(t) = deadline_ms - now_ms - predicted_remaining_ms
slack_ratio(t) = slack_ms(t) / max(predicted_remaining_ms, ε)
queue_age_ms   = now_ms - ready_since_ms
load_ratio     = admitted_gpu_demand / effective_gpu_capacity
backlog        = ready nodes + arrived-but-not-started jobs
```

紧迫性 pilot 可以用三档，但用当前 trace 的经验分位数校准：

- loose：大多数 job 的 slack 明显为正；
- mixed：一部分 job slack 接近 0，短 job 和长 job发生取舍；
- tight：一部分 job 进入负 slack，priority/SLO 与总完成时间冲突。

这里的“tight”不是故意让所有 job 必然 miss，而是让 miss rate、queue age 和策略分歧处于可测区间。若一改 multiplier 就让所有方法都 100% miss，说明 workload 失真，不是压力足够。

## 6. 建议的 workload 重构（先 pilot，后正式）

### 6.1 不先增加视频，先重组已有模板

现有 640 templates 足够做压力诊断。视频/trace 数量主要决定行为和资源模板的多样性；动作空间和 urgency 更多由 DAG 混合、到达过程、GPU topology、deadline/SLO、runtime tail 和 cache state 决定。**本轮不需要重新采视频，也不应先解封 `T_final`。**

### 6.2 因子

| 因子 | 当前 | stress pilot 的建议 | 目的 |
|---|---|---|---|
| GPU topology | 2 GPU，同构/小显存/混合 | 保留 2 GPU 对照；新增 4 GPU/异构容量 cell | 提高 node×GPU 配对和碎片效应 |
| DAG width | 现有模板混合但需量化 | 低/中/高 ready width；优先重采样宽 DAG 与汇合 DAG | 提高合法动作数 |
| 到达 | replay、burst、poisson | 用 replay 做基线，用相关 burst/短间隔 burst 形成持续 backlog | 提高队列竞争 |
| 负载 | pressure 0.5–1.05 | 先做低/中/高三档，按实际 admitted demand/容量校准 | 让 free GPU 和 queue 状态真正变化 |
| urgency | multiplier 1.5/2/3 | 用 slack 分位数构造 loose/mixed/tight | 让 deadline penalty 参与动作 |
| runtime | trace-derived | 保留真实分布，同时单独抽取 p50/p90/p99 heavy-tail cell | 防止短 job score 永远压过长 job |
| memory/cache | cold/skewed/warm，已有 eviction | 低/高模型多样性和 cache churn 对照；记录 load/evict 状态 | 让同一节点在不同 GPU 上的未来成本不同 |
| job mix | 16/32/64 jobs | 混合 short/long、low/high memory、narrow/wide DAG | 避免只靠 job 数改变压力 |
| SLO/priority | 目前较粗 | 至少两类 priority/SLO，报告 fairness 与 miss | 形成真正的取舍 |

### 6.3 规模建议

不要一次做完整的 3×3×3×2×2=108 格。建议先做 12–18 个诊断 cell：

1. 2 GPU baseline（当前 workload 复现）；
2. 2 GPU + 高 ready width；
3. 2 GPU + tight slack；
4. 2 GPU + heavy-tail runtime；
5. 2 GPU + cache churn；
6. 4 GPU heterogeneous + 高 ready width；
7. 上述几项的中高负载组合；
8. 至少 2 个 Alibaba arrival replay 片段作为外部到达形状。

每个 cell 先跑 100–200 episodes、RR/Myopic/PredOpt-H5/Oracle 四个策略。只有通过门禁的 cell 才扩到 1,000 validation episodes 和后续 RL/CP-RHO。这样可以避免用数万 episode 掩盖 workload 结构不产生分歧的问题。

## 7. 基线和信息边界必须这样对齐

主比较建议保留：`RR → EDF/SJF → Myopic → PredOpt-H1/H3/H5 → CP-RHO-H1/H3/H5 → Oracle`；RL/BC 在 workload 通过门禁后再加入。所有可部署方法共享同一个 `DeployableObservation`：已到达 jobs、DAG prefix、ready nodes、GPU/cache state、队列、冻结的预测器输出；Oracle/TrueOpt 单独标为 privileged upper bound。

专家标签也不能读真实未来。CP-RHO 若作为 BC/PPO teacher，必须使用和部署预测器相同的 horizon 与信息边界，并保存 solver status、time limit、objective gap、首动作和 candidate table。否则会把“专家知道未来”误写成“学习器利用预测器”。

目标函数也要拆开报告：mean completion、P95/P99 completion、deadline miss、mean/p95 queue、GPU utilization、load/eviction、fairness、policy/solver latency。单看 mean completion 会让短 job 优先策略显得过强。

## 8. 必须先通过的四个 gate

### Gate A：可分动作空间

- action width 的 episode/cell 分布完整；
- 单合法动作比例不能接近 100%；
- 至少有一批事件存在 2 个以上、且 objective margin 不为零的合法动作；
- 不同策略的 action disagreement 可复现。

### Gate B：真实压力

- arrival/backlog/queue age 随 pressure 单调或至少可解释地变化；
- load ratio、free GPU、显存 headroom 和 evictions 有明确变化；
- tight cell 不能只是“所有 job 都 miss”，而应处于可比较的 miss 区间。

### Gate C：基线可分性

- Oracle 与 Myopic 有稳定 gap；
- PredOpt/CP-RHO 相对 Myopic 的改善在多个 cell 方向一致；
- 若 PredOpt-H5 几乎和 Myopic 完全相同，要先检查 score 的 priority 词典序是否压制 future cost；
- RL 只有在 BC/CP-RHO 的收益稳定后才进入训练，不用一次 seed 决定结论。

### Gate D：因果和泄漏

- predictor 只看 episode prefix 和冻结 future artifact；
- workload 生成不把 Oracle 的真实 future、完成时间或最终答案写进部署 observation；
- train/val/test template 分离；
- 同一 episode/seed 下所有策略使用 common random numbers；
- `simulation_events`、candidate table 和 solver metadata 能回溯到同一 workload hash。

## 9. 之后的执行顺序（本报告不执行）

1. 只读导出当前完整矩阵和 partial event 的 action-width、slack、queue、cache、runtime-tail 统计；
2. 用已有 640 templates 生成 12–18 个 stress pilot cell，不改 canonical workload；
3. 仅跑 RR/Myopic/PredOpt-H5/Oracle，检查四个 gate；
4. gate 通过后，加入 EDF/SJF、CP-RHO，并做 solver latency/timeout/optimality 审计；
5. 再生成同信息边界的 BC/专家数据，最后重做 RL 多 seed 稳定性；
6. 所有方法和 workload 冻结后才解封并运行 `T_final`。

## 10. 参考来源

1. [Tiresias，USENIX NSDI’19](https://www.usenix.org/conference/nsdi19/presentation/gu)
2. [Gavel，USENIX OSDI’20](https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak)
3. [MLaaS in the Wild，USENIX NSDI’22](https://www.usenix.org/conference/nsdi22/presentation/weng)
4. [Alibaba Cluster Trace Program（官方仓库）](https://github.com/alibaba/clusterdata)
5. [Google Cluster Scheduler Simulator（官方仓库）](https://github.com/google/cluster-scheduler-simulator)
6. [Pegasus Workflow Gallery（官方）](https://pegasus.isi.edu/workflow_gallery/)
7. [Kubernetes Pod Priority and Preemption（官方文档）](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)
8. [vLLM Automatic Prefix Caching（官方设计文档）](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
9. [Gandiva，USENIX OSDI’18](https://www.usenix.org/conference/osdi18/presentation/xiao)
10. [Pollux，USENIX OSDI’21](https://www.usenix.org/conference/osdi21/presentation/qiao)
11. [Sarathi-Serve，USENIX OSDI’24](https://www.usenix.org/conference/osdi24/presentation/agrawal)
12. [JITServe，USENIX NSDI’26](https://www.usenix.org/conference/nsdi26/presentation/zhang-wei)

## Provenance

- 本报告记录了 ChatGPT Web 的独立研究建议，并将其与当前本地/远端项目产物及公开一手资料逐条对齐。
- 本轮只读调研；未修改源码、未修改数据、未启动新实验、未解封 `T_final`。
- 数值事实以当前 R7 矩阵报告为准；partial event 的 action-width 只作诊断，不作为最终指标。
