---
research_backend: ChatGPT Web + primary-source verification
requested_model: GPT-5.6 Sol
requested_reasoning: High
verified_before_submit: true
verified_after_response: false
timestamp: "2026-08-23"
status: partial
delegated_status: invalid
delegated_reason: request_error
---

# 调度动作空间扩展：论文调研与 VideoSeek 适配建议

## 0. 研究边界与证据状态

本轮研究的问题是：其他 GPU/深度学习/LLM/DAG 调度工作如何把动作从“选一个 ready 节点放到一个空闲 GPU”扩展到有实际调度意义的动作空间，并且如何避免组合爆炸。

网页版 ChatGPT 已在发送前核验为 `GPT-5.6 Sol + 高`，但提交后页面返回 `RequestError: Something went wrong`，没有产生可引用的回复。因此本文件的论文结论**不使用 ChatGPT Web 的无效结果**，只使用下面列出的论文原文、官方项目页和官方文档。事实分级：

- **VERIFIED**：原始论文/官方项目页明确写出，且给出链接。
- **INFERENCE**：根据论文机制映射到 VideoSeek 的推论。
- **UNVERIFIED**：需要在 VideoSeek 的 profile/workload 上实测，不能当成效果承诺。

## 1. 先区分两件事：动作空间变大 vs. 压力变大

当前 R7 诊断中，候选池是 `ready GPU nodes × free GPUs`，Myopic action-width 的 p50 约为 3，p90 为 11；arrival 时的 deadline slack 全为正，1.5×/P90 runtime 的中位 slack 仍约 102 s。说明当前主要问题不是“算法不会选”，而是很多时刻根本没有足够多的互相竞争的选择。

需要区分：

1. **增加压力（pressure）**：更多并发 job、更突发的到达、更少/更忙的 GPU、更紧 deadline。它增加 ready 节点数和等待，但没有增加动作的维度。
2. **增加动作维度（action-space expansion）**：除了选节点和 GPU，还能选 GPU 集合/拓扑、并行度、batch、时间片、抢占/恢复、驻留/预取或资源份额。

论文通常先让 workload 有足够竞争，再增加少量真实动作维度，并用分层/参数化策略避免一次性枚举所有组合。

## 2. 论文中最常见的扩展方式

| 机制 | 论文中的动作粒度 | 如何增加选择 | 如何控制组合爆炸 | 对 VideoSeek 的含义 |
|---|---|---|---|---|
| **多 GPU placement / 拓扑** | 选择一个 job 的 GPU 集合、服务器集合或是否集中放置 | 从单个 `free_gpu` 变成 GPU set / placement class / topology class | 先选 job 的 placement class，再由 placement 模块落到物理 GPU；或只保留可行集合 | 适合 node-level DAG；应把“快卡/慢卡/同机/跨机”作为可测的 placement 选项 |
| **异构加速器与共置** | 给 job 分配某种 accelerator 类型、共置组合或时间比例 | 同一个节点在 A100/V100/慢卡、独占/共置之间有不同 throughput | 先优化 allocation matrix，再按 round 实现，而不是枚举全部时间序列 | 适合 H4/H8；需要每个 `(node_type, gpu_type, co-location)` 的 runtime/load profile |
| **并行度 / replica / worker 数** | 选择 job 当前使用的 worker/PS 数或并行度上限 | 一个 ready node 可以选择 1、2、4、… 个并行槽；也可改变 replica 数 | 离散候选 + 性能曲线；或选择上限，系统逐步分配 | 比单纯增加 GPU 更贴近“动作由调度器决定”；需要实测节点在不同并行度下的 runtime 和显存 |
| **batch / microbatch / token budget** | 在一次 dispatch/iteration 选择 batch、microbatch 或 token 上限 | `(node, gpu)` 变为 `(node, gpu, batch)` | 只保留 profile 中有真实测量的档位；用最大 batch/最大 token 约束 | 当前 YOLO `{1,8,16,32,64}` 是合理起点，但若 ready node 少，单加 batch 不会显著增宽实际候选池 |
| **时间片、抢占、恢复、迁移** | 在迭代/节点边界 pause、resume、migrate 或换 GPU | 动作不再只是“启动谁”，还决定“暂停谁、何时恢复、恢复到哪” | 只允许在 checkpoint/节点边界发生；将 checkpoint、swap、recompute 作为真实成本 | 当前抢占是自动事件且会丢弃重算；要变成显式动作必须先测 checkpoint/restore/swap 成本 |
| **模型驻留 / cache / prefetch / eviction** | 选择预加载、保留、驱逐或等待 demand-load | 同一 GPU 上有“先加载哪个模型/保留哪个模型”的选择 | 只在 cache 状态变化或 load lane 空闲时决策；加入容量和 wasted-prefetch 约束 | 适合作为第二阶段动作；不能把未来节点直接当成当前可见信息 |
| **GPU fraction / memory share / space sharing** | 给任务分配 GPU 时间比例、显存份额或共置 lane | 从整数 GPU 变成资源份额或共置组合 | 以少量离散 share bins 或优化器 allocation matrix 表示 | 如果要体现显存挤占，这是比“再加几个 GPU 标签”更真实的方向；需要干扰/共置 profile |
| **分层/参数化动作** | 先选 job/node，再选资源类别，再选具体资源或参数 | 将一个指数级联合动作拆成多个小动作 | action mask、分层 policy、候选评分、round-based allocation | 是 VideoSeek 最应该采用的实现形式；既保留 node 粒度，也避免 `node × GPU × batch × preempt × cache` 全枚举 |

## 3. 代表性工作与可核验结论

### 3.1 Tiresias：空间 placement + 时间调度 + 显式抢占

**VERIFIED。** Tiresias 把问题拆成两个主模块：调度器决定哪些 job 获得资源，placement 模块决定多 GPU job 放在哪些机器/GPU。论文明确讨论了空间维度（需要多少 GPU）和时间维度（运行多久），并用离散化的 2D Gittins/LAS 处理优先级变化，避免连续抢占。placement 会根据模型通信/结构决定集中放置还是降低碎片；恢复时加载 checkpoint。论文还写明：作业到达、完成或资源变化时会触发调度，调度器可以把运行中 job 放回等待队列再重新分配 GPU。

来源：

- [Tiresias 原始 NSDI'19 论文](https://www.usenix.org/system/files/nsdi19-gu.pdf)
- [Tiresias 官方 GitHub（含离散时间 simulator 和 placement）](https://github.com/symbioticlab/tiresias)

**对 VideoSeek 的推论。** 最适合迁移的是“placement class + node dispatch”：先决定 `ready_node` 属于 `same_gpu / same_server / cross_server / fast_gpu / slow_gpu` 哪一类，再选择具体 GPU；不要第一步就枚举所有 GPU 集合。抢占必须以节点边界或真实 checkpoint 为前提。

### 3.2 Gandiva：时间片、迁移、packing 和 grow-shrink

**VERIFIED。** Gandiva 利用深度学习训练的 mini-batch 迭代边界，在多个 job 之间做 GPU time-slicing，并根据 profile 迁移 job 到更合适的 GPU；论文还包含 packing 和 intra-job grow-shrink。它的核心不是增加很多静态候选，而是把“运行一段时间、暂停、迁移、重新共置”变成可执行系统原语。

来源：[Gandiva 原始 OSDI'18 论文与官方页面](https://www.usenix.org/conference/osdi18/presentation/xiao)

**对 VideoSeek 的推论。** 若要研究节点级调度，Gandiva 的可迁移版本是：节点在完成当前 micro-step/工具边界后可以 `continue`、`pause` 或 `migrate`。当前 simulator 的“节点重算”不是 Gandiva 式 checkpoint 抢占，不能直接宣称已经复现。

### 3.3 Gavel：异构 GPU、共置和时间比例 allocation

**VERIFIED。** Gavel 不把动作定义为单个 GPU 标签，而是先求每个 job 在不同 accelerator、共置配置上的 allocation；allocation 还可以表示“在资源配置 A 上运行 60% 时间，在配置 B 上运行 40% 时间”。然后用 round-based scheduler 近似实现该 allocation。论文明确支持 heterogeneity、colocation、placement 和多种全局目标，官方仓库提供 simulator 和 policies。

来源：

- [Gavel 原始 OSDI'20 论文](https://people.eecs.berkeley.edu/~matei/papers/2020/osdi_gavel.pdf)
- [Gavel 官方 GitHub](https://github.com/stanford-futuredata/gavel)

**对 VideoSeek 的推论。** 这是目前最适合数学优化器-H 的动作抽象：决策变量是 `allocation(node, gpu_class, co-location)` 或小份额，而不是每个时间点枚举所有物理 GPU 序列。需要先补齐异构卡、共置干扰和 placement profile。

### 3.4 Pollux / Optimus / DL²：动态资源数量和资源-性能曲线

**VERIFIED。** Pollux 根据在线测量的 goodput，动态增加或减少 job 的资源，并联合调整 batch size、学习率等 job 参数；官方页面链接了开源 AdaptDL。Optimus 在线拟合“资源数量 → 训练速度”的性能模型，动态调整资源和 placement。DL² 则把“每个 job 当前应有多少 worker/PS”作为动作，在离线行为克隆后在线 RL 微调，并明确指出资源分配动作空间会随资源数呈指数增长，因此使用 job-aware exploration。

来源：

- [Pollux 原始 OSDI'21 论文与 AdaptDL 链接](https://www.usenix.org/conference/osdi21/presentation/qiao)
- [Optimus 原始 EuroSys'18 论文记录](https://hub.hku.hk/handle/10722/259647)；[官方代码](https://github.com/kzhang28/Optimus)
- [DL² 原始论文](https://arxiv.org/abs/1909.06040)

**对 VideoSeek 的推论。** “并行度/worker 数”比把 GPU 数简单放大更有论文依据，但不能直接套训练 job 的收敛假设。VideoSeek 可以把它变成有限离散的 node concurrency 或 microbatch 档位，并由资源预测器给每一档 runtime/load/peak。

### 3.5 Decima / Clotho：DAG 节点选择 + 并行度上限的分层动作

**VERIFIED。** Decima 明确讨论了两种极端：一次性给所有 executor 分配任务会产生指数级动作空间；每次只选一个 stage 虽然动作小，但序列很长。它折中为二维动作 `(stage, parallelism_limit)`：先选一个可运行 DAG stage，再选该 job 的并行度上限，系统按上限分配 executor；仍有空闲资源时重复调用。候选 softmax 只对当前 runnable stages 做 action mask。Clotho 同样以 DAG embedding 选择下一任务，并同时学习 parallelism level 和 execution order。

来源：

- [Decima 原始 SIGCOMM'19 论文](https://people.csail.mit.edu/hongzi/content/publications/Decima-Sigcomm19.pdf)
- [Clotho 原始 MLSys'18 论文](https://mlsys.org/Conferences/doc/2018/129.pdf)

**对 VideoSeek 的推论。** 这与我们的“node-level 核心创新”最接近：将动作从 `(ready_node, free_gpu)` 扩成 `(ready_node, parallelism_or_batch_limit)`，再由 simulator 分配实际 GPU。这样既能看到未来节点预测对下一步的影响，也避免直接枚举所有 GPU×节点组合。

### 3.6 Salus / AntMan：GPU lane、显存共享和 opportunistic co-location

**VERIFIED。** Salus 提供 fast job switching 与 memory sharing 两个底层原语，在 iteration 边界执行调度，支持 time-sharing、preemption、fairness、priority 和 packing。AntMan 与训练框架协同，在运行期间动态调整显存和计算单元，让 opportunistic job 使用空闲 GPU 周期。它们的动作不是“选择另一个空闲 GPU”，而是选择是否共置、给谁多少计算/内存、何时让出资源。

来源：

- [Salus 原始 MLSys 论文](https://proceedings.mlsys.org/paper_files/paper/2020/hash/d9cd83bc91b8c36a0c7c0fcca59228f2-Abstract.html)；[官方 GitHub](https://github.com/symbioticlab/salus)
- [AntMan 原始 OSDI'20 论文](https://www.usenix.org/system/files/osdi20-xiao.pdf)

**对 VideoSeek 的推论。** 这是“显存压力”最真实的扩展，但成本也最高：需要共置 slowdown、显存峰值、切换/恢复开销的 profile。当前仅有单节点 YOLO batch 曲线，不能据此声称已经支持 Salus/AntMan 式共享。

### 3.7 Orca / FastServe / Triton / vLLM：iteration、token budget 和动态 batch

**VERIFIED。** Orca 把 serving 的调度粒度从 request 降到 iteration，并使用 selective batching；FastServe 进一步把抢占粒度降到每个 output token，并用多级反馈队列和 proactive offload。Triton 的 dynamic batching 明确把最大 batch、preferred batch、最大排队等待时间和优先级队列作为配置；vLLM 暴露每次 iteration 的最大 token 数和最大 sequence 数。

来源：

- [Orca 原始 OSDI'22 论文](https://www.usenix.org/conference/osdi22/presentation/yu)
- [FastServe 原始 NSDI'26 论文](https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang)
- [Triton 官方 dynamic batching 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
- [vLLM 官方 scheduler 参数文档](https://docs.vllm.ai/en/stable/cli/run-batch/)

**对 VideoSeek 的推论。** 对 planner/answer worker，未来可以将“token budget/sequence cap/等待窗口”作为资源动作；对 YOLO/视频工具，优先使用已有真实 batch 曲线，不要把 LLM token action 直接套到视觉节点。

## 4. 对当前 VideoSeek 的具体判断

### 4.1 当前已经做对的部分

- **VERIFIED（本项目）**：4/8 GPU topology pilot 已证明增加物理 GPU 能显著增加候选池；压力 pilot 已证明增加 ready frontier/到达压力能把 action-width p50 推到约 12。
- **VERIFIED（本项目）**：YOLO batch profile 已有 batch 1/2/4/8/16/32/64 的 runtime/load/peak 原始测量；调度扩展当前只激活 `{1,8,16,32,64}`，并把 batch 2/4 保留为测量证据。
- **VERIFIED（本项目）**：当前 100 集扩展 pilot 中实际 batch-action-width p50 约 2.8，说明“加入 batch 维度”本身没有解决候选不足；主要瓶颈仍是 ready node 数和并发压力。

### 4.2 最适合先做的三层动作空间

#### Layer A：扩大真实竞争，不改动作定义

先建立 `W-wide` workload：4/8 GPU、突发到达、每个 episode 同时存在 8–16 个可运行 ready frontier，保持模型/priority/objective 不变。它回答“当前策略是否因为候选太少而看不出差异”。这一步属于 pressure calibration，不应冒充新的动作维度。

#### Layer B：节点级参数化动作（首选正式扩展）

把动作改成：

```text
(ready_node, placement_class, batch_or_parallelism_limit)
```

其中：

- `placement_class` 先只取 `same_gpu / fast_gpu / slow_gpu / cross_gpu` 中有真实 profile 的类别；
- `batch_or_parallelism_limit` 对 YOLO 使用 `{1,8,16,32,64}`，对其他节点先用单一值；
- simulator 再把参数化动作解析成一个或多个物理 GPU assignment；
- action mask 只保留当前 ready、容量可行、无前序冲突的候选。

这对应 Decima 的 `(stage, parallelism_limit)`、Gavel 的 allocation/placement 抽象，能够保留 node-level 创新而避免全组合枚举。

#### Layer C：显式时间动作（第二阶段）

在真实 checkpoint/节点边界 profile 完成后再加入：

```text
continue(node, gpu)
pause(node)
resume(node, placement_class)
prefetch(model, gpu)
evict(model, gpu)
```

每个动作必须记录 checkpoint/save/restore、load、eviction、wasted-prefetch 和重算成本。当前“自动抢占 + 丢弃重算”只能作为诊断对照，不应当被称为论文式 preemption。

## 5. 建议的 stress pilot cells

| cell | 目的 | 主要改变 | 通过条件 |
|---|---|---|---|
| W0 | 当前 baseline | 2 GPU、当前 arrival/deadline/cache | 复算旧结果 |
| W1 | 只增加 pressure | 4/8 GPU + ready frontier 8–16 + burst arrival | action-width p50 ≥ 8，且无容量违规 |
| W2 | placement dimension | H4/H8，快慢 GPU与同机/跨机 topology | 记录每类真实 runtime/load，策略排序出现可解释差异 |
| W3 | batch/parallelism dimension | 真实 batch profile，激活 1/8/16/32/64 | 每个档位有 profile；batch 选择不引入 future 字段 |
| W4 | time dimension | checkpoint-boundary pause/resume/migrate | checkpoint/restore 成本守恒；不能把重算当免费 |
| W5 | combined | W1 + W2 + W3，最后才加入 W4 | Myopic、PredOpt、CP-RHO 的决策和完成时间有可解释 gap |

每个 cell 先跑 100–200 个 paired episodes；只有 action-width、queue age/slack、cache/load 和 baseline-separation 四个 gate 都通过，才扩为正式 1,000 episodes。`T_final` 和已有 640 trace 不覆盖，使用独立 workload 派生版本。

## 6. 结论

1. 论文并不是简单把 GPU 数或 batch 档位机械相乘；更常见的做法是增加一个真实资源维度，然后用 placement class、allocation matrix、parallelism limit、时间片或分层 policy 处理组合爆炸。
2. 对 VideoSeek，优先顺序应是：**先增加并发竞争（pressure）→ 再加入 placement/parallelism 参数化动作 → 最后加入显式 preemption/prefetch/cache 动作**。
3. 当前最值得先实现的是 Decima/Gavel 风格的分层动作：`ready node → placement class → batch/parallelism limit`。它比继续增加虚拟 GPU 标签更有论文依据，也最容易与行为预测器的“下一个 node/未来几步”衔接。
4. 具体收益目前均为 **UNVERIFIED**。正式实现前必须补齐 placement、共置干扰、checkpoint/restore 和不同并行度的真实 profile，并在同一 arrival/seed/信息边界下比较。

