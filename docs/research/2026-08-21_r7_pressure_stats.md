# R7 workload pressure diagnostic statistics

## Scope and evidence status

这是一次只读诊断，不是新的正式实验。统计输入：

- workload：远端 `results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl`；
- events：远端 `results/processed/r7_workload_20260817/sim_validation_1000/simulation_events.jsonl.gz`（压缩后约 1.68 GB）；
- workload jobs：34,160；event lines：4,633,945；
- 现有审计已说明该 event gzip 在写入时中断，因此下面的 event-based 数值只能作为 **diagnostic/partial evidence**，不能替代最终完整事件日志；workload 中的 arrival/deadline 字段是完整的。

## 1. Action width

调度器候选池是 `ready GPU nodes × free GPUs`，并记录 `candidate_count` 和 `feasible_candidate_count`。在这份日志中两者数值相同，说明本批次没有观察到显存 fit 过滤进一步缩小候选池；动作宽度主要受 ready 节点数和空闲 GPU 数量限制。

| policy | dispatch events | mean | p50 | p90 | p95 | p99 | max | one-feasible fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| myopic | 248,756 | 4.83 | 3 | 11 | 15 | 27 | 52 | 22.34% |
| oracle | 248,756 | 3.50 | 2 | 7 | 10 | 17 | 46 | 26.63% |
| round_robin | 248,839 | 5.49 | 3 | 13 | 18 | 34 | 76 | 21.68% |

解释：当前不是“永远只有一个动作”，但典型事件只有 2–3 个动作，较宽的分支主要出现在尾部。`oracle`、`myopic` 和 `round_robin` 是独立策略运行，状态轨迹不同，因此不能把三行直接当成同一时刻的候选池比较；它们用于诊断量级。

## 2. Deadline slack

### 2.1 到达时的完整 workload slack

对每个 job 计算：

```text
arrival_slack = deadline_ms - arrival_ms - predicted_remaining_runtime
```

分别使用 scheduler-visible 的 P50/P90 job runtime。结果：**所有 job 的 arrival slack 都为正，负 slack 比例为 0。**

| deadline multiplier | P50 slack（P50 runtime） | P50 slack（P90 runtime） | 负 slack 比例 |
|---:|---:|---:|---:|
| 1.5 | 192.17 s | 102.29 s | 0% |
| 2.0 | 294.46 s | 204.59 s | 0% |
| 3.0 | 499.05 s | 409.17 s | 0% |

这不是说最终不会 miss；最终的 miss 主要由排队、前序节点、GPU 串行和模型加载逐步消耗 slack。它说明当前 deadline 在 job 到达时没有制造“立刻必须取舍”的紧迫性，尤其 2×/3× 档非常宽松。

### 2.2 dispatch 时的节点级估计

事件中没有同时保存完整的 `deadline_ms` 和“该 job 剩余所有节点的预测执行时间”，所以无法从当前 event 文件精确计算完整 job-level dynamic slack。我们做了一个保守的节点级估计：

```text
node_slack_estimate = deadline_ms - dispatch_time - current_node.runtime_p50/p90
```

GPU node dispatch 的中位估计仍约为：myopic 327.5 s、oracle 315.6 s、round-robin 307.6 s。该数值不能当作完整 job slack，但进一步说明当前 workload 的 deadline 余量很大，紧迫性更多发生在排队尾部而不是到达瞬间。

## 3. Queue age

`node_start.queue_ms` 是节点从 ready 到 start 的实际等待时间，因此是当前日志中最直接的 queue-age 字段。

| policy | all-node mean | all-node p50/p90/p95/p99 | GPU mean | GPU p50/p90/p95/p99 |
|---|---:|---|---:|---|
| myopic | 11.82 s | 0 / 9.34 / 36.50 / 303.36 s | 16.84 s | 0 / 19.38 / 66.69 / 390.02 s |
| oracle | 7.36 s | 0 / 5.75 / 17.65 / 144.55 s | 10.49 s | 0 / 9.24 / 29.94 / 217.50 s |
| round_robin | 14.68 s | 0 / 43.29 / 80.98 / 199.32 s | 20.91 s | 3.36 / 61.37 / 103.57 / 229.36 s |

结论：压力确实存在，但主要集中在 tail。Myopic/Oracle 的 GPU queue 中位数为 0，说明很多 dispatch 是立即开始的；round-robin 的 GPU queue 中位数约 3.36 s、P90 约 61.4 s，明显更拥挤。

## 4. Cache miss / model load

当前事件日志没有一个名为 `cache_miss=true` 的单一字段，因此报告两个可复核的代理量：

- `load_positive_rate`：GPU `node_start` 中 `load_ms > 0` 的比例，最接近“这个节点发生了模型加载”；
- `model_load_event_rate`：`model_load_start` 事件数除以 GPU node starts，表示加载操作事件的频率，可能因一次调度涉及多个加载操作而高于前者。

| policy | GPU starts | load>0 starts | load-positive rate | model_load_start events | load-event rate | eviction events |
|---|---:|---:|---:|---:|---:|---:|
| myopic | 248,756 | 23,137 | 9.30% | 33,395 | 13.42% | 20,348 |
| oracle | 248,756 | 23,119 | 9.29% | 32,573 | 13.09% | 18,537 |
| round_robin | 248,838 | 29,585 | 11.89% | 58,576 | 23.54% | 39,954 |

结论：cache/load 不是零，且 round-robin 明显更差；但因为 candidate_count 与 feasible_candidate_count 相同，当前 cache/memory 压力大多没有把候选动作直接筛掉，更多体现为额外 load/eviction 时间。正式实验应把 `cache_hit`、`load_ms`、`evicted_models` 和可行性过滤原因直接写入完整事件日志。

## 5. Runtime tail

这里使用 GPU `node_start.truth_runtime_ms` 作为实际 runtime，使用 `scheduler_view.runtime_p50_ms` 作为预测 runtime。

| 量 | P50 | P90 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|
| actual GPU runtime | 6.04 s | 29.10 s | 53.48 s | 72.42 s | 82.52 s |
| predicted GPU runtime | 5.49 s | 41.93 s | 46.58 s | 56.48 s | 73.67 s |

runtime 确实有长尾，但当前日志显示预测分布和真实分布的尾部并不完全对齐：预测 P90 偏高，而真实 P95/P99 又高于预测对应分位数。后续应按 template/model/node_type 校准 tail，而不是只用全局均值。

## 6. 目前能得出的判断

1. **action-width 偏窄**：中位合法动作 2–3 个，约 22%–27% dispatch 只有一个合法动作；预测器很多时候没有足够的选择空间。
2. **deadline slack 偏宽**：即使以 P90 runtime 估计，最紧 1.5× 档到达时的中位 slack 仍约 102 s；2×/3× 更宽松。
3. **queue pressure 存在但集中在尾部**：Oracle 能明显降低 queue tail，说明调度确实有作用；但大量事件 queue=0，导致平均决策环境不够紧。
4. **cache 压力存在但未充分改变动作可行性**：load/eviction 事件不少，不过当前日志没有显示显存 fit 过滤候选。
5. **runtime 有长尾**：适合用 heavy-tail cell 检验预测器，但要先校准预测 P50/P90 与真实 P95/P99 的关系。

## 7. stress pilot cell 是什么

`cell` 可以理解成“一个固定的测试工况小组”，不是新视频，也不是新的模型。每个 cell 固定一组 workload 条件，然后在完全相同的 jobs/arrival/seed 上比较 RR、Myopic、PredOpt-H5、Oracle。

例如：

```text
S0 baseline：2 GPU + 当前 arrival + 当前 cache + 当前 deadline
S1 wide：2 GPU + 更多同时 ready 的 DAG 节点
S2 tight：2 GPU + 让 slack 接近 0 的 deadline
S3 cache：2 GPU + 更高模型切换/缓存 churn
S4 tail：2 GPU + 保留真实 runtime 长尾
S5 combined：4 GPU 异构 + wide DAG + tight slack + burst arrival
```

每个 cell 先做 100–200 个 episode。先看四件事：候选数是否变宽、slack 是否真的变紧、queue/cache 是否增加、四个 baseline 是否拉开。只有通过门禁的 cell 才扩成正式 1,000 episode；这样可以避免做几十万条 episode，却仍然在一个“多数动作无需选择”的环境里训练。

## 8. 建议的下一步

1. 保留当前 R7 结果作为 baseline，不覆盖；
2. 用现有 640 templates 先生成 12–18 个 diagnostic stress cells；
3. 为完整日志补充 `deadline_ms`、`remaining_predicted_ms`、`slack_ms`、`cache_hit`、`evicted_models` 和 candidate table；
4. 先只跑 RR/Myopic/PredOpt-H5/Oracle；
5. 通过 action-width、urgency、baseline-separation gates 后，再进入 CP-RHO 和 RL。

本次统计未修改代码、数据、模型或 workload，也未执行 `T_final`。
