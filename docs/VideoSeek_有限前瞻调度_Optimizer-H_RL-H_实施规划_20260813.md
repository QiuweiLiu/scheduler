# VideoSeek 有限前瞻调度、Optimizer-H 与 RL-H 实施规划

版本：2026-08-13  
状态：实施前审查完成、待确认执行；已完成接口与环境核验，尚未修改调度代码、训练 RL 或运行 C2 确认性测试  
本地项目：`/Users/liuqiuwei/Documents/调度`  
远端项目：`/root/autodl-tmp/scheduler`  
远端执行 Python：`/root/miniconda3/bin/python`

## 1. 任务边界卡

### 1.1 本轮目标

把以下四件事拆成可以逐步实现、逐步验收且不会混淆因果关系的正式路线：

1. 将 B05 的 H=1/3/5 概率未来真正接入节点级调度器；
2. 实现共用同一信息边界和约束的数学优化器 `Optimizer-H`；
3. 实现成对对照的 `RL-0 / RL-H`；
4. 在相同 workload episode 上，正式比较预测式策略、无未来策略和 Myopic。

### 1.2 本轮只交付什么

- 一份可执行的实现和实验规划；
- 明确代码入口、数据接口、算法定义、泄漏边界、单测、门禁和产物路径；
- 将文档同步到远端项目的 `docs/`。

### 1.3 本轮不做什么

- 不修改 `workload_v02_simulator.py` 或其他调度代码；
- 不安装依赖；
- 不训练 RL；
- 不运行新的 validation、C1 test/replay 或 C2 confirmatory test；
- 不删除已有约 2.3GB 的 validation 事件日志；
- 不把当前 `scheduling_future_smoke.py` 的排序实验包装成正式有限前瞻优化器。

### 1.4 最终验收标准

后续执行完成时，至少应满足：

1. Myopic、Optimizer-0、PredOpt-H、TrueOpt-H、RL-0、RL-H 共用同一个事件引擎、资源账本和动作合法性检查；
2. `PredOpt-H` 和 `RL-H` 在决策时不能读取未来真值；
3. `Optimizer-0` 与 `PredOpt-H` 除未来输入外完全相同，`RL-0` 与 `RL-H` 亦同；
4. C1 validation 冻结超参数、H、求解时限和随机种子后，才运行 C1 retrospective；所有配置和代码再次冻结后，C2 confirmatory 只打开一次；
5. 正式结果使用同 episode 配对统计，报告置信区间、调度开销和失败/回退率。

## 2. 当前已核验的基础

### 2.1 数据与 workload

| 项目 | 当前事实 |
|---|---|
| Agent 模板 | 768 条模板、9,366 个节点、64 个视频 |
| C1 workload | train 20,000、validation 1,000、test/replay 6,750 episodes |
| C1 test/replay 结构 | 135 个场景 cell，每 cell 50 episodes |
| 场景轴 | arrival pattern × pressure × GPU topology × initial residency |
| 到达模式 | Alibaba PAI 到达间隔回放、burst、Poisson |
| GPU 拓扑 | 24GB×2、32GB+24GB、32GB×2 |
| 压力档 | 0.50、0.70、0.85、0.95、1.05 |
| 初始驻留 | cold、warm、skewed |

C1 validation 的 1,000-episode preliminary 已产生调度空间：

- Myopic mean completion：199,097.66ms；
- 当前历史 `oracle` mean completion：155,534.70ms；
- 相对 gap：28.01%；
- Myopic deadline miss：11.86%，历史 `oracle`：4.65%。

这说明 workload 已经不是“所有策略都一样”的无压力情况，但该结果仍是 preliminary，不能直接作为论文主表，原因见 2.5。

另有一个必须诚实记录的历史事实：`workload_v0_2_formal_20260812` 已经在 6,750 个 test episode_id 上完整运行过 round_robin、myopic 和历史 oracle，共 20,250 行结果。它与当前 C1 test 的文件 SHA256 不同，但 6,750 个 episode_id、source video/template、scenario cell、seed 和 GPU topology 全部相同；job payload 只有 41/6,750 完全相同，说明 C1 是同一实验设计的资源/初态修订版，而不是全新盲测。因此 C1 test/replay 只能作为冻结后的同分布回溯评估，不能宣称为完全未暴露的确认性测试。

正式主结论另设 `C2 confirmatory`：从未参与行为/资源预测训练、validation、C1 或历史调度结果的视频中冻结目标 32 个新视频（最低 24 个），生成独立 Agent traces/templates，并复用同一 135-cell × 50 episodes = 6,750 episode 的 workload 设计。当前远端扫描到 363 个视频文件、351 个唯一 stem，其中 287 个不属于 C1 的任何 64 个视频；这只证明候选池充足，不证明它们都未参与其他训练，R0a 必须与所有 trace/feature/model registry 交叉审计后才能冻结 C2。

C2 采集采用预先冻结的确定性规则：

1. 候选先按 content hash/感知 hash 去重，再排除所有 predictor/resource/trace/workload registry 中出现过的视频；
2. 只用视频元数据按公开 dataset、时长桶和内容域分层，用固定 seed 排序选 32 个主样本和 8 个有序替补；
3. 在看任何 C2 调度指标前，先根据现有 runner/config 审计 baseline×stack 的可执行组合，冻结每视频采集矩阵；目标至少覆盖 `langgraph_react`、`star`、`st_fixed` 和 stack A/B，但不虚构当前代码不支持的组合；
4. 每个组合固定 seed/重复数；失败只按预先排序替补或预先定义的同视频重试规则处理，不因模型表现人工挑样本；
5. 若最终不足 24 个完整视频，C2 门禁失败，不把不完整样本悄悄混入主表；
6. C2 template/episode manifest 在运行任何策略前冻结 SHA256，随后只允许 runner 读取。

### 2.2 有限未来预测产物

正式产物位于：

```text
results/processed/scheduling_future_v1_20260812/prediction_artifacts/
  b05_node_h1.jsonl.gz
  b05_future_h3.jsonl.gz
  b05_future_h5.jsonl.gz
  resource_predictions_v1.jsonl.gz
  b05_artifact_manifest.json
  resource_artifact_manifest.json
```

B05 三个 horizon 均覆盖 9,366 个当前节点，每个节点最多有 3 条概率场景。每个未来步骤含：

```text
role, action_family, raw_action, model_id, execution_lane,
role_probability, family_probability, scenario_probability,
step_offset, prototype_source
```

B05 manifest 已记录：train-only encoder fit、三 seed checkpoint、holdout 复算一致和标签突变防泄漏通过。

但实施前代码审查发现，现有 artifact **不能直接进入正式调度**：

- `expand_scenarios()` 在未来 offset 预测 execute family 时，会优先从 `actual_tool_by_cutoff[current_index]` 取同一真实 run 的对应 future tool sample；
- 当前 cutoff 没有 tool sample 时，主循环还会以 `tool_by_run[run_id][0]` 回退到同一 run 的第一条工具样本；
- 9,366 行里有 4,749 行标为 `same_run_tool_template`，这不是 train-only prototype；
- 当前 leakage audit 只突变 target 标签，没有测试“删除当前 cutoff 之后所有同 run 行，预测是否不变”。

因此现有 H=1/3/5 文件降级为 `B05 diagnostic artifact v1`。正式调度必须重建 `causal-v2`：observed prefix 只能包含 `event_index < current ready node` 的已完成事件；当前节点只暴露调度前已知的 role/action/model/lane 身份。synthetic rollout 后续 family base 只能来自 train-only role/family schema prototype，不能读取该 validation/test run 的当前结果或未来行。

### 2.3 资源预测产物

现有 `resource_predictions_v1.jsonl.gz` 对每个当前节点给出：

```text
runtime_p50_ms, runtime_p90_ms, load_p50_ms,
peak_memory_p95_mb, uncertainty
```

它目前只能作为 schema 和连接覆盖率的诊断产物，不能直接用于正式比较：其训练契约不是 C1 的统一 48/8/8 video split，且必须重新审计所有 prefix 特征在“当前节点执行前”是否可见。正式阶段应按统一 split 重 fit，产出 `resource_predictions_scheduler_v2.jsonl.gz` 及 manifest。调度器必须读取这一冻结 v2 产物，不能继续把 simulator 内部的 `train_resource_stats()` 当作正式资源预测器。

### 2.4 远端运行环境

已实际核验：

- `/root/miniconda3/bin/python` 可用；默认 shell 中的 `python` 不可用；
- SciPy 1.17.1，`scipy.optimize.milp` 可用，底层可使用 HiGHS；
- PyTorch 2.7.0+cu128 可用；
- 未安装 Gymnasium、Stable-Baselines3、OR-Tools、CVXPY、PuLP、Gurobi；
- 因此 Optimizer-H 优先使用 SciPy MILP，RL-H 优先使用原生 PyTorch，不把安装新框架作为前提；
- 远端 `/root/autodl-tmp` 当前仅余约 13GB，正式阶段禁止再默认保存全量未压缩事件日志。

### 2.5 当前实现为什么还不能称为正式结果

当前有两个不同原型：

1. `scripts/scheduling_future_smoke.py`：
   - Pred-H 只把未来 family median runtime 加进候选排序；
   - 不联合优化 node×GPU、模型驻留、显存、驱逐和求解器约束；
   - 因此它只能叫 `PredScore-H diagnostic`，不能叫 Optimizer-H。
2. `src/tracing/analysis/workload_v02_simulator.py`：
   - 事件循环、DAG、到达、GPU 驻留和队列逻辑更完整；
   - 但策略只有 round_robin、myopic 和读取完整后缀的历史 oracle；
   - 正式资源预测仍被内部 train-only quantile 代替；
   - GPU 执行 load 使用了 `max(真实 load, 预测 load)`，使执行真值和预测输入发生耦合；
   - policy 直接拿到含完整 Template 的 Job 对象，尚未形成可审计的信息隔离层。

实施前审查还发现六个正式实验阻断项：

1. **静态 prefix 与 raw-DAG ready 状态不一致**：当前 768/768 模板都有分叉/汇合；在 6,693 个 GPU 节点中，1,828 个（27.31%）并不以所有更早 `sequence_index` 节点为祖先，涉及 636 个模板。它们可能在某些更早兄弟节点未完成时 ready。B05 artifact 却按原始线性 trace 的所有更早事件构造 prefix。此时直接按 node_id 查 artifact 会把尚未完成的兄弟节点当成已观测信息。
2. **未来步骤映射读取了完整模板**：当前 artifact 生成器的 `prototype_map()` 会在当前 template 的全部节点中，为预测 role/family 选 model/lane；对 validation/test 来说，这利用了完整 realized template 的工具存在性，正式 provider 不能沿用。
3. **episode 内含完整 job 规模估计**：workload 的 `scheduler_visible.predicted_job_runtime_p50/p90` 来自整条模板。如果交给 H=0/Myopic，会相当于提供剩余路径总量提示。正式 StateView 必须排除这些字段。
4. **预测器 split 与 workload split 不一致**：按 run_id 实查，C1 workload 的 566/94/108 templates 与 B05 数据 split 的交叉表并非对角；C1 validation 94 条全部落在 B05 train，C1 test 108 条中有 64 条在 B05 train、34 条在 B05 validation、仅 10 条在 B05 test。视频级也不一致：C1 test 的 8 个视频里，5 个属于 B05 train、2 个属于 B05 validation、1 个属于 B05 test。现有 checkpoint 不能作为 C1 test-retrospective 的 holdout predictor，更不能用于 C2 confirmatory。
5. **32GB 容量标签大于实测卡容量**：episode topology 使用 32,760MiB，但 C1 设备实测总显存是 32,230.812MiB。若直接用 topology 数值做准入，会多出约 529MiB 的虚假空间。
6. **行为预测时间点晚于调度决策**：现有 B05 样本的语义是“当前事件已经成功执行后，预测下一事件”；例如 tool-family 样本的 compute/evidence prefix 已包含当前 planner 的 runtime 和结果。调度器却要在当前 node 刚 ready、尚未执行时决定 GPU。直接使用当前 node 对应 B05 行，会读取决策时尚不存在的当前执行结果。

因此正式工作必须复用第二个事件引擎，但重构其 policy/resource 边界；不能继续维护两套互相不一致的 simulator。

当前 28.01% gap 只能证明“旧 raw-DAG workload 有调度空间”。修正 causal-prefix 和预测/真值隔离后必须重测，不能把旧 gap 原样继承到正式结果。

正式实验必须建立唯一 `video_split_registry_v1`，显式标记 C1 的 48 train / 8 validation / 8 test-retrospective 视频，以及独立的 C2 confirmatory 视频，并按该 registry 重新：

1. 构建行为预测训练/validation/test rows；
2. 训练 B05 seed 11/22/33；
3. 训练或至少严格重 fit 资源预测器及所有 group/prototype 统计；
4. 生成 causal-v2 future/resource artifacts；
5. 生成 RL train/validation 环境。

旧 B05 指标和 checkpoint 只作历史参考。禁止为了复用旧模型而改变 C1 视频归属；C1 test-retrospective 和 C2 confirmatory 视频都禁止参加 encoder、vocabulary、normalization、prototype、temperature calibration 或 checkpoint 选择。

统一 registry 冻结前先只看输入元数据，按 baseline/stack、视频域、时长和模板长度审计 C1 48/8/8，并用同样元数据规则选择 C2。允许读取 C2 的文件名、video_id、schema 和输入 hash 以冻结清单，但不得运行策略、计算 C2 指标或据结果重选视频。C2 还必须做 content hash/感知 hash 去重，并和全部 C1/历史 predictor/trace registry 取差集；仅凭文件名不同不算未见视频。

资源预测器也不能默认复用。现有正式资源报告的 `fit_splits` 是 `development_train + development_validation`，最终评估另用 40-video `final_holdout_v1`；它与 C1 48/8/8 workload 不是同一评估契约。虽然其报告声明 video_id 不作为特征、fit 统计 train-only，但正式调度仍必须按统一 registry 重 fit，并只用 C1 train 视频训练；C1 validation 只选模型和风险参数，C1 test-retrospective 只作回溯评估，C2 confirmatory 只评一次。

## 3. 要验证的五个问题

| 问题 | 正确对照 | 能说明什么 |
|---|---|---|
| 数学优化本身是否有用 | Optimizer-0 vs Myopic | 联合 node×GPU×cache 决策是否优于逐项贪心 |
| 预测未来是否有用 | PredOpt-H vs Optimizer-0 | 在同一个数学优化器中，仅增加有限预测未来的因果收益 |
| 真有限未来是否可利用 | TrueOpt-H vs Optimizer-0 | workload 和优化目标是否存在可被 H 步信息回收的空间 |
| RL 是否能利用有限未来 | RL-H vs RL-0 | 在同一网络、训练预算和动作空间内，未来特征是否有效 |
| 端到端系统是否更好 | PredOpt-H / RL-H vs Myopic | 实际预测式系统相对现有在线基线的总体收益 |

必须特别注意：`PredOpt-H vs Myopic` 同时改变了算法和信息，不能单独用于宣称“预测带来收益”。主因果证据必须来自 `PredOpt-H vs Optimizer-0` 和 `RL-H vs RL-0`。

## 4. 统一调度契约

### 4.1 只保留一个事件引擎

以 `src/tracing/analysis/workload_v02_simulator.py` 的事件循环为唯一事实来源，拆出三个接口：

```text
ExecutionTruthProvider  仅事件引擎可见：真实 runtime/load/workspace/status
SchedulerStateView      所有在线策略可见：当前 ready/prefix/GPU/预测资源
SchedulerPolicy         输入 StateView，输出合法 DispatchAction 列表
```

`scheduling_future_smoke.py` 只保留为历史 diagnostic，不再扩写成第二套正式引擎。

### 4.2 SchedulerStateView

每个决策时刻只包含：

- 当前模拟时间；
- 已到达 job 的 service class、arrival、deadline、queue age；
- 当前已完成节点形成的 causal prefix；
- 当前 ready GPU 节点的 role/action/model/lane；
- 每张 GPU 的容量、当前是否空闲、模型驻留、可驱逐缓存；
- 当前节点的冻结资源预测；
- 由选定 FutureProvider 返回的有限未来场景；
- 候选动作的预测显存可行性。

任何 DAG 深度/进度特征只能由已完成 prefix 和当前节点已知前驱计算；不能读取完整未来图后预先计算全图 centrality、remaining nodes、critical path 或真实 out-degree。

明确禁止进入 StateView：

- 当前或未来节点的真实 runtime/load/workspace；
- 未执行后缀、真实 successors、最终答案、未来错误状态；
- template_id、video_id 作为模型特征；
- C1 test-retrospective/C2 confirmatory 标签，或任何由它们 fit 的统计量。
- workload 中的 `scheduler_visible.predicted_job_runtime_p50_ms`、`predicted_job_runtime_p90_ms`、`predicted_gpu_runtime_p50_ms` 和 `deadline_multiplier`。

template_id/node_id 只允许由 provider 内部用于查表和写审计日志，不作为 Optimizer 权重或 RL 特征。

GPU 容量通过冻结的 `gpu_capacity_registry_v1.json` 解释：

- topology 中的 32760 是 GPU class 标签，主 C1 class 的物理预算映射到实测 32230.812MiB；
- topology 中的 24576 是未在本机实测的模拟 24GB class，必须明确标为 extrapolated；
- admission 使用 `min(episode_class_budget, registry_budget)`，不能使用更大的那个；
- memory p95 作为工作集风险边界，不再通过 validation 调一个有利于结果的隐形 capacity margin；
- 映射后重新审计 initial residency 和每个单节点至少在目标 topology 的一张 GPU 上可运行。

显存 schema 必须先冻结：确认 `peak_memory_p95_mb` 表示“该节点运行时 GPU 总峰值”还是“不含 resident weights 的增量 workspace”。若是总峰值，准入不能再直接做 `resident + peak`；应先用测量 manifest 拆出可审计 workspace，或用 `max(existing_resident, node_total_peak)` 这类经实测验证的公式。该语义未确认前，R1 capacity gate 不得通过。

### 4.3 DispatchAction

所有策略共用以下动作语义：

```json
{
  "node_id": "ready-node",
  "gpu_index": 0,
  "evict_models": ["model-x"],
  "predicted_total_memory_mb": 12345.0,
  "source_policy": "pred_opt_h3"
}
```

规则：

1. 只能调度已到达、所有前驱完成且状态为 ready 的 GPU 节点；
2. CPU/API 节点仍由事件引擎自动推进，不占 GPU action；
3. 主实验每张 GPU 同时最多一个 active compute node；可同时驻留多个模型；
4. 主实验不允许策略主动空转：存在合法 ready 动作时必须填充空闲 GPU；
5. 同一决策时刻可返回一组 node×GPU 匹配；Myopic 顺序构造、Optimizer 联合求解、RL 自回归构造，但最终都经过同一个合法性检查；
6. 下一到达、节点完成或缓存状态改变后重新决策，即 rolling horizon 只提交当前动作。

### 4.4 为什么主实验暂时仍采用单 active node/GPU

C1 已证明若干模型组合可以真实并发，并给出显存峰值，但当前 C1 没有可靠的“固定输入/固定输出长度”并发 slowdown 系数。直接用混合 case 的 wall time推导速度倍率会把生成长度差异当成争用。

因此：

- 主实验使用已验证的单 active node/GPU 假设；
- C1 resident/workspace 数据用于模型驻留、显存准入和驱逐；
- GPU 压力来自多任务到达、队列、两张 GPU、模型切换、冷加载、容量异构和 deadline；
- 并发共置作为后续 `contention-aware` 鲁棒性实验，必须先补固定 token、固定帧数的 duration-normalized calibration；
- 未标定组合不得猜 slowdown，也不得默认可共置。

这不会否定节点级调度创新，但论文中必须清楚写出执行互斥假设。

### 4.5 投机预加载的边界

有限未来除了改变 ready node 的顺序、GPU 放置和缓存驱逐，还可以在 Agent 正执行 CPU/API 节点时，趁 GPU 空闲提前加载下一候选模型。该机制单独记为 `+Preload`，不能捆绑进基础 PredOpt-H 后只给一张总表。

P0 预加载规则：

1. 只允许 `preload(model_id, gpu_index)`，不允许提前执行尚未 ready 的工具节点，也不产生或提交工具输出；
2. 只有该 GPU 空闲且当前没有可在其上立即运行的 ready GPU 节点时，才允许预加载，不以预测为理由延迟已 ready 的真实工作；
3. 候选模型必须来自有限未来场景，且模型概率、supported mass 和显存账本通过冻结门槛；
4. 预加载真实占用 GPU 时间和显存，不按瞬时完成处理；预测错误造成的无用加载、驱逐和等待全部进入指标；
5. 调度器只看到 load p50 预测；事件引擎使用冻结的 model-level load truth 执行；两者物理隔离；
6. 只有 C1 或 train truth 表中存在合法 resident/load 记录的模型才可预加载，unknown 模型不猜；
7. `Optimizer-0+Preload` 和 `RL-0+Preload` 具有相同模型候选空间，但没有未来概率，可学习或使用 train-only popularity；这样能区分“预加载机制本身”与“有限未来让预加载更准”。

预加载的独立比较为：

```text
PredOpt-H vs PredOpt-H+Preload
Optimizer-0+Preload vs PredOpt-H+Preload
RL-0+Preload vs RL-H+Preload
TrueOpt-H+Preload vs PredOpt-H+Preload
```

基础 PredOpt-H 若不能通过 True-H/信息边界门禁，不允许用 `+Preload` 的收益掩盖问题。

## 5. 有限前瞻如何真正接入

### 5.0 首先修正 causal-prefix 契约

B05 是在原始 Agent 的线性事件前缀上训练和生成的。主实验不能一边让同一 job 内的节点按不完整 raw parent edge 越序执行，一边读取包含完整线性历史的静态预测；也不能在当前 node 执行前读取“当前 node 执行完成后”的特征。

主实验采用一个不覆盖原模板的 causal-chain overlay：

```text
job_templates_causal_chain_v1.jsonl
```

对每条 realized trace：

- 保留原有 raw predecessor 作为审计字段；
- 给 sequence_index>0 的节点额外加入前一 sequence node 依赖；
- 从而保证同一 job 内只有原始执行前缀完成后下一节点才 ready；
- 多 job 之间仍可并发，调度器仍在节点粒度决定跨 job 顺序、GPU 放置、模型驻留和驱逐；
- 复用同一批 episode 的 template_id、arrival、deadline、GPU topology 和初态，不重新抽样 test job。

新的行为训练数据必须改成 decision-time 样本。对当前 ready node `c`：

```text
observed_prefix = 已完成且 event_index < c 的事件
revealed_current = c 的 role/action/model/lane 身份
forbidden_current = c 的 runtime/load/status/output/新证据
target_h = c 之后的第 1/3/5 个未揭示节点
```

这样 H 始终表示“当前 ready node 之后”的未来步数，不把已知当前节点冒充预测命中。B05 网络结构可复用，但 role/family 数据构造、prefix aggregate 和 cutoff 必须按上述决策时间重建并重新训练。

每次 future/resource 查表前还必须做动态校验：

1. 取该 job 实际已完成节点的 source_event_id，按 canonical event index 排序；
2. 用 `causal_prefix_registry` 补齐已完成的 init/control rows，计算“当前 node 之前”的完整 prefix hash；
3. 当前 ready node identity 单独进入 `current_identity_hash`，不进入 observed-prefix hash；
4. 两个 hash 和预期 event count 都一致才允许读取；
5. 不一致时整个决策退化到 H=0，并记录 `prefix_mismatch`；
6. 在 causal-chain 主实验中 prefix mismatch 必须为 0，否则是实现错误。

raw-DAG 保留为后续扩展，但只有两种合法做法：

- 使用实际已完成的 partial-order prefix 在线重新运行行为/资源预测器，并单独验证这种 OOD prefix 的质量；或
- 对 prefix 不一致的所有竞争动作整次退化为 H=0，报告 effective-lookahead rate。

禁止在 raw-DAG 上直接按 node_id 使用原静态 B05/resource artifact。正式执行顺序因此变为：先构建 causal-chain overlay 并重跑 Myopic/FullTruth-Greedy preliminary；确认修正后仍存在可利用的调度空间，再实现 Optimizer-H。

同时生成新的正式行为产物：

```text
b05_causal_v2_h1.jsonl.gz
b05_causal_v2_h3.jsonl.gz
b05_causal_v2_h5.jsonl.gz
b05_causal_v2_manifest.json
causal_prefix_registry_v1.json
```

`causal-v2` 的硬规则：

1. observed rows 必须满足 `event_index < current ready node index`；
2. 当前 node 只提供已揭示身份，不提供 runtime/status/output/evidence delta；
3. synthetic offset 1..H 不得访问真实 run 的任何 current/post-current role/tool row；
4. 缺 tool-family base 时只使用 train-only decision-time prototype，不能使用 same-run first row；
5. model/lane 通过 5.3 的 train-only stack registry 绑定；
6. 删除/打乱当前 node 及其之后的所有真实结果行，H1/H3/H5 输出必须逐位不变；
7. 改写当前 node runtime/status/output 后，输出必须逐位不变；
8. v1/v2 文件并存，正式 runner 只接受 schema=`b05-causal-v2` 和冻结 SHA256。

`causal_prefix_registry_v1` 还需保存每个 compute node 之前的 canonical 非 compute/control rows（例如 init/run-control）、observed-prefix hash 和当前身份 hash。运行时 observed hash 只覆盖“当前 node 之前”的完整已完成 prefix，不把当前 ready node 加入；当前身份 hash 只覆盖 scheduler 已知的 role/action/model/lane，不含当前结果。

### 5.1 FutureProvider 的四种模式

| mode | 输入 | 用途 |
|---|---|---|
| `none` | 不打开任何 future artifact | Myopic、Optimizer-0、RL-0 |
| `pred_h` | B05 H=1/3/5 概率场景 | 可部署的 PredOpt-H、RL-H |
| `true_h` | 当前节点之后真实执行序列的前 H 个节点 | 有限信息上界，仅评估 |
| `full_truth` | 完整剩余真实序列 | 信息上界，不是可部署策略 |

表中的 B05 特指 decision-time `b05-causal-v2`，v1 诊断文件不得被正式 runner 接受。H 的统一定义是“当前 ready 节点之后，按 canonical sequence_index 的后续 H 个未揭示节点”，不是模板图的 H 层。主 causal-chain workload 中顺序唯一；raw-DAG 扩展必须使用实际完成 prefix 在线预测，不能把未发生分支、尚未完成兄弟节点或当前节点执行结果伪造成已观测真值。

### 5.2 当前节点资源连接

当前 ready 节点必须以 `node_id` 精确连接按统一 split 重 fit 的 `resource_predictions_scheduler_v2.jsonl.gz`。现有 v1 仅用于重建前的 schema/coverage 对照：

- 9,366 个模板节点都必须一一匹配；
- 重复、缺失、NaN 或负数直接阻断；
- GPU 节点使用 runtime p50/p90、load p50、memory p95；
- CPU/API 节点可使用 runtime 预测参与未来 release-time 估计，但 GPU load 和 GPU memory 强制为 0；
- 当前 CPU 样本中出现约 19GB 的预测 memory 是已知语义错位，不能进入 GPU 显存账本。
- v2 artifact 允许使用“上一已完成节点的 observed runtime”；只有 prefix hash 一致时这些历史量才是因果可见。raw-DAG prefix 不一致时必须在线重算或使用不含 observed-runtime 的 scheduler-safe 资源版本，不能静态查表。

执行真值也必须修正当前 simulator 的语义：

- GPU node truth service time 使用模板中已测的 `runtime_ms`，不能再用 `runtime_ms - load_ms` 后又额外添加 `max(true_load,predicted_load)`；
- 模型已驻留时仍执行完整测得 runtime；模型冷时只额外加入一个 model-level cold-start penalty，且该 penalty 必须由 C1/训练测量表冻结；
- scheduler 只看到 cold-load prediction，事件引擎只使用 cold-load truth；
- 如果现有 trace 的 `runtime_ms` 已包含冷加载，则先通过采集 schema/audit 确认并拆分，不能重复加 load；
- 50-episode parity 只验证事件状态机；正式指标以修正后的 truth semantics 为准，不强求复制旧数值。

### 5.3 反事实未来步骤的工具与资源连接

B05 synthetic step 没有真实 node_id。正式 provider 只信 artifact 中的 role/family 概率、step offset 和 scenario probability；丢弃当前由完整 template 产生的 `model_id`、`execution_lane` 和 `prototype_source`，重新用 train-only registry 绑定工具实现。

新增两个只由 train split 构建并冻结的表：

```text
stack_tool_registry_v1.json
future_resource_prototypes_v1.json
```

`stack_tool_registry_v1` 按已知的 `model_stack_id + role + action_family` 映射到一个或多个 `(raw_action, model, lane)` 实现及其 train-only 概率。它只使用 train template 和已声明的 stack 配置；validation/test template 的 realized future 不参与。无法映射的 family 保留 unsupported。

映射优先级是：已冻结的 collection config/model-stack contract > train trace 中与配置一致的观测 > unsupported。不能仅凭 train 中“最常见模型”覆盖显式 stack 配置，也不能从 validation/test trace 反推。registry manifest 保存每条映射的配置文件路径、SHA256、support count 和冲突记录。

候选键按以下顺序匹配，并记录实际命中层级：

1. 已声明 collection/model-stack 配置中的 `(model_stack_id, role, action_family)`；
2. train trace 中与配置不冲突的 `(model_stack_id, role, action_family)`；
3. 只有在全体 train stack 唯一映射时，才回退到 `(role, action_family)`；
4. 否则标记 unsupported。

禁止用预测 artifact 自带的 future `model_id/lane` 反向作为 registry key，因为它们正是需要由 train-only 配置重新绑定的输出。若一个 key 有多个合法实现，保留 train-only 概率分布，不根据当前 validation/test template 选择其中之一。

每个 prototype 的 runtime/load/memory 来自冻结资源预测在 train 模板上的分位汇总，不读取 validation/test 真值。Pred-H 与 True-H 都使用这套相同 prototype：True-H 只揭示真实未来的 role/action/model/lane 身份，不读取对应 future node 的专属预测行、runtime 或执行结果。若仍无法连接，保留 `unsupported`，不能填零或用 test 全局均值。

### 5.4 unknown/unsupported 的主实验规则

当前原始 B05 artifact（尚未经过 train-only registry 重绑定）的完整可用场景概率质量并不相同：

| H | mean supported probability mass | zero-mass rows | mass≥0.8 rows |
|---:|---:|---:|---:|
| 1 | 0.883 | 1 | 8,118 |
| 3 | 0.653 | 1,774 | 5,470 |
| 5 | 0.478 | 3,583 | 3,596 |

因此不能把未知步骤忽略成零成本。主规则为：

1. 先用 train-only stack registry 和 resource prototype 表重新解析，并重新生成 coverage 报告；
2. 一条场景只要仍含 unsupported step，就从可用场景集合中移除；
3. 计算每个候选节点剩余的 `supported_probability_mass`；
4. 若同一决策时刻任一竞争候选低于阈值 `tau`，整个决策降级为 H=0，避免“缺预测的候选反而成本更低”；
5. 可用场景在剩余概率质量上重新归一化；
6. 每次降级记录 `future_unusable_reason`、原概率质量和有效 horizon；
7. `tau` 只在 validation 的预注册集合 `{0.6, 0.8}` 中选择，随后冻结。

正式报告必须同时给出 nominal horizon 和 effective-lookahead rate。若 H=5 大部分时间退化成 H=0，这就是模型能力边界，不能隐藏。

### 5.5 决策审计行

每个发生竞争的决策至少记录：

```text
episode_id, policy, decision_index, time_ms,
candidate_count, free_gpu_count, chosen_actions,
horizon, effective_horizon, supported_probability_mass,
future_artifact_sha256, resource_artifact_sha256,
solver_status, solver_time_ms, mip_gap,
fallback_reason, information_boundary
```

## 6. 数学优化器 Optimizer-H

### 6.1 方法定位

Optimizer-H 是一个滚动的、有限未来感知的联合 node×GPU×cache 准入优化器。它不调度尚未 ready 的预测节点；预测未来只用于评估“现在把哪个 ready 节点放到哪张 GPU、保留或驱逐哪些模型”对后续的影响。

### 6.2 候选动作构造

对每个 ready GPU 节点 `i`、空闲 GPU `g` 和合法驱逐子集 `E` 构造一个 action candidate：

```text
a = (node i, gpu g, evict subset E)
```

ActionBuilder 使用同一组冻结信息：

- 当前模型是否已驻留；
- C1 resident model size；
- 当前节点预测 workspace p95；
- GPU capacity；
- 被驱逐模型未来重新加载的预测成本；
- H 步场景下未来模型序列与冷加载机会。

只枚举能在预测 p95 账本下通过容量约束的动作。由于当前模型种类少，枚举合法驱逐子集比在 MILP 中重新实现复杂 cache 状态更直接，也便于单测穷举验证。

### 6.3 MILP 变量与约束

每个候选动作一个二元变量：

```text
x_a ∈ {0,1}
```

约束：

1. 同一 ready 节点最多选择一次；
2. 同一空闲 GPU 最多接收一个当前节点；
3. 同一 GPU 不能选择相互矛盾的驱逐方案；
4. 按 4.2 冻结后的显存 schema 计算 post-action peak，不超过 capacity；
5. 先在同一候选二分图上求最大可行 matching 基数 `K*`，再约束 `Σ_a x_a = K*`；实现上可先做一次“最大化派发数”的 MILP，再固定 `K*` 最小化调度代价。不能简单用 `min(空闲 GPU 数, ready 节点数)`，因为异构容量和模型驻留会使某些 node×GPU 不可行；
6. 只能选择 ActionBuilder 已标记为 scheduler-visible 和 capacity-valid 的动作。

### 6.4 H 步动作代价

每个动作的代价都以毫秒为主单位：

```text
cost(a,H) =
    current_runtime_p50
  + current_cold_load_p50
  + expected_future_runtime_p50(H)
  + expected_future_cold_load_p50(H | post_action_cache)
  + deadline_tardiness_penalty(p90)
  - bounded_queue_age_credit
```

其中：

- future 项按场景概率求期望；
- p90 只用于 deadline/risk，平均完成时间项仍用 p50；
- memory p95 是硬约束，不以软惩罚代替；
- future cold-load 项已经包含被驱逐模型下一次需要时的 reload；不能再单列同一 reload cost 重复收费；
- queue-age credit 有上限，防止饥饿，又不让极老大任务永远压过 deadline；
- priority service class 使用预先冻结的 job weight。

正式目标：

```text
min Σ_a x_a · cost(a,H)
```

这不是简单排序：多张 GPU、缓存状态和驱逐方案通过同一个 MILP 联合选择。

准确命名为 `rolling-horizon assignment MILP`：MILP 联合优化当前决策 epoch 的 node×GPU×cache matching，H 步概率未来进入 action continuation cost；它不会对整条 episode 建立全局 start-time/precedence MILP，也不声称全局最优。若论文需要与完整 stochastic scheduling MILP 比较，另做小实例 Oracle，不改变本方法命名。

### 6.5 四个严格配对版本

| 策略 | 未来路径 | 未来资源 | 其他部分 |
|---|---|---|---|
| `opt_0` | 无 | 无 | 相同 MILP、权重、动作空间、时限 |
| `pred_opt_h{1,3,5}` | B05 概率场景 | train-only prototype 预测 | 与 opt_0 相同 |
| `true_opt_h{1,3,5}` | 真实后续 H 节点身份 | 与 Pred-H 相同的 train-only prototype 预测 | 与 opt_0 相同 |
| `true_opt_h*_truth_resource` | 真实后续 H 节点 | 执行真值 | 仅诊断 A4，不可部署 |

主因果消融使用 `true behavior identity + shared prototype resource`，这样 True-H 和 Pred-H 的差异只来自行为路径，而不是偷偷给 True-H 更好的 future node 专属资源标签。只有 `true_opt_h*_truth_resource` 诊断可以读取执行真值，并必须单独标为不可部署上界。

### 6.6 求解器、时限和回退

- 使用已安装的 `scipy.optimize.milp` / HiGHS；
- 不安装 Gurobi、OR-Tools 或其他依赖作为 P0 前提；
- validation 先在固定 100 episodes 上测试 time limit `{10,25,50}ms`；
- 选择能达到 `incumbent_rate≥99%` 且 p95 求解时间最低的时限；
- 到时有 incumbent：提交 incumbent，记录 gap；
- 到时无 incumbent、infeasible 或数值错误：显式回退同一 StateView 上的 Myopic；
- 不能默默返回第一候选、零成本动作或随机动作；
- 正式报告给出 fallback rate、求解 p50/p95 和 overhead-accounted 指标。

### 6.7 Optimizer-H 门禁

1. 小型随机实例上，MILP 与穷举最优动作完全一致；
2. H=0 不打开 future 文件；
3. Pred-H 改写未来真值后决策不变；
4. True-H 改写 H 之后的后缀后决策不变；
5. `true_opt_h` 必须在 validation 上至少有一个 H 明显优于 opt_0；否则停止调 Pred-H，先检查目标函数和 workload；
6. opt_0 不能因 bug 大幅劣于 Myopic；若劣于超过预注册容忍度，先修 optimizer，不进入 C1 retrospective 或 C2 confirmatory。

### 6.8 Optimizer-H+Preload

基础 Optimizer-H 通过后，ActionBuilder 增加 `preload(model,gpu)` 候选。预加载动作的目标系数为：

```text
expected_saved_cold_load
- immediate_preload_cost
- expected_wrong_preload_eviction_cost
```

其中收益和误加载风险均按有限未来场景概率计算。MILP 在“当前无可运行 ready node 的空闲 GPU”上最多选择一个 preload；只提交本时刻动作，下一事件重算。对 H=0 使用 train-only model popularity 生成同样候选，不能让 H=0 因动作空间缺失而天然吃亏。

## 7. RL-H

### 7.1 为什么必须成对实现 RL-0 与 RL-H

只训练一个 RL-H 再与 Myopic 比较，会把“神经网络策略学习”和“有限未来信息”混在一起。必须使用同一网络、同一动作生成器、同一 reward、同一训练预算：

- `RL-0`：future feature block 全零，`has_future=0`；
- `RL-H`：同一位置输入 B05 H 步特征；
- `TrueRL-H`：同一位置输入真实有限 H 步，仅作 validation 信息上界和接口门禁；
- 两者参数量相同，区别只有未来信息内容。

`TrueRL-H` 也必须使用与 RL-0/RL-H 相同的网络、训练 decision budget、seed 和 checkpoint 规则；不能直接把真值塞给已经训练好的 RL-H 后只做推理。它只得到真实未来身份和同一套 train-only prototype resource，不得到 future runtime/result 真值。其用途是回答“这套状态、动作和 reward 是否真的能学会利用有限未来”，不作为可部署方法。

### 7.2 RL observation

#### 当前节点特征

- role、action family、model、lane 的 embedding；
- runtime p50/p90、load p50、memory p95；
- queue age、arrival、deadline slack、service class；
- 当前 DAG 深度、已完成节点数、当前 prefix 长度；
- 当前资源不确定度。

#### GPU 与 node×GPU 特征

- capacity、resident set 汇总、free memory；
- target model cache hit；
- cold-load cost、驱逐字节数、驱逐后 headroom；
- 当前动作 p95 fit margin；
- 未来 H 步同模型复用概率和预期缓存节省。

#### 有限未来特征

对每个 offset 1..H 固定长度编码：

- 6 类 role 概率；
- action-family、model、lane 概率或 embedding 加权和；
- expected runtime p50/p90、load p50、memory p95；
- termination probability；
- scenario entropy、supported probability mass；
- effective horizon 和 fallback 标记。

实现时统一按 `H_max=5` 分配特征槽，H=1/3 未使用的 offset 置零并附 mask；RL-0 把全部未来槽置零。这样 RL-0、RL-H1/H3/H5 与 TrueRL-H 的输入维度和参数量都一致，避免把网络大小差异误当成前瞻收益。

RL 不直接读取原视频或 template_id。多模态内容已经由行为预测器压缩成概率未来；这样调度器接口可迁移到电梯 Agent，而不绑定 VideoMME 的原始视觉特征。

### 7.3 网络结构

P0 使用原生 PyTorch 的 masked actor-critic：

```text
candidate action feature ─┐
global/GPU pooled state ──┼─> shared MLP(128/256) ─> action score
future feature block ─────┘

all candidate embeddings ─> mean/max pooling ─> critic MLP ─> V(state)
```

- Actor 对 ActionBuilder 生成的合法 `(node,gpu,evict-set)` 候选打分；
- `+Preload` 消融中，Actor 同时对合法 `preload(model,gpu)` 候选打分；RL-0 与 RL-H 仍共享相同候选集合定义；
- 非法动作在 softmax 前 mask；
- 自回归选择动作并更新空闲 GPU mask，直到填满本时刻可用 GPU；
- P0 不先上 GNN；只有 MLP/集合编码器能稳定复现且 RL-H 仍无法利用 True-H 时，才把图编码器列为单独消融，避免同时改变太多模块。

### 7.4 Reward

用事件间隔的未完成任务面积近似总 flow time：

```text
r_t = -[
    weighted_unfinished_jobs × Δt
  + λ_deadline × overdue_jobs × Δt
  + invalid_or_failure_penalty
]
```

理由：`∫ unfinished_jobs(t)dt` 等于所有 job flow time 之和，和 mean JCT 目标一致，比“每完成一个任务奖励 +1”更密集。模型加载/错误预加载已经通过延长 Δt 自然增加 flow-time 面积，不能再额外加同一 load_ms 导致双重收费；load/eviction 只作为报告指标，若要加入能耗或显式换模成本必须做单独 reward 消融。显存是硬动作约束；出现引擎真值容量违例时该 episode 判为无效门禁，不靠随意设置巨额 OOM 奖励修补。

### 7.5 训练协议

1. 训练只采样 20,000 条 train workload pool；
2. 所有 normalization、embedding vocabulary 和 prototype 只 fit train；
3. validation 1,000 条只做 checkpoint/hyperparameter 选择；
4. C1 test-retrospective 与 C2 confirmatory 绝不参与训练、early stopping 或 reward 调整；
5. 先用 50 train episodes 做环境/梯度 smoke；
6. 再以固定 decision budget 训练，初始预算 2M decisions，必要时扩至 5M，但 RL-0/RL-H 必须相同；
7. seed 固定为 11、22、33；
8. 每 250k decisions 在 validation 固定子集评估；
9. 超参搜索预算对 RL-0/RL-H 相同，候选只包括 hidden size、learning rate、PPO clip、entropy coefficient；
10. checkpoint 按 validation 的预注册指标选择，不能人工挑 test 表现好的 seed。

RL 训练环境不能使用“在全部 train 视频上 fit、又回头预测这些相同 train 视频”的 in-sample future/resource 值。对 48 个 train 视频按视频分组做 K-fold cross-fitting，生成 OOF causal-v2 future/resource artifact 供 20,000 条 train workload 使用；每个 fold 的 encoder、vocabulary、normalization、prototype 和 predictor 都只 fit 其余 fold。validation、C1 test-retrospective 和 C2 confirmatory 都使用全体 48 train 视频 fit、经 validation 冻结的正式模型。OOF artifact 只服务 RL 训练，不进入 evaluation 主表。

实现采用 PyTorch-only masked PPO，避免在本阶段安装 Gymnasium/SB3。若 PyTorch-only PPO smoke 无法通过可重复性和 action-mask 单测，再单独请示是否引入成熟框架，不能静默切实现。

### 7.6 RL 门禁

1. reward identity toy test：事件面积与离线重算的总 flow time 一致；
2. action mask 100% 阻止未 ready、超容量、重复 node/GPU 动作；
3. 固定 seed 的 10-episode rollout 完全可复现；
4. RL-0 与 RL-H 网络参数量、训练步数、优化器和 checkpoint 规则一致；
5. TrueRL-H diagnostic 若仍不优于 RL-0，说明 RL 接口/reward 尚不能利用未来，不能把 RL-H 的失败归因于行为预测器；
6. 三 seed 至少方向一致，并报告 seed 方差，不能只挑最好 seed。

## 8. Myopic 与其他参照的冻结定义

### 8.1 Myopic

保留当前可解释逻辑：

1. priority service class 优先；
2. 当前 `runtime_p50 + cold_load_p50` 较小者优先；
3. cache hit 优先；
4. ready time 和稳定 node/gpu index 打破平局；
5. 不读取任何未来 artifact。

正式实现只把其资源输入替换成冻结的 `resource_predictions_scheduler_v2`，并让它通过统一 ActionBuilder 和显存账本，不能为了让新方法显得更好而削弱 Myopic。

### 8.2 历史 oracle 的命名

当前 `workload_v02_simulator.py` 中名为 `oracle` 的方法，只是读取完整后缀后做逐步贪心，不是全局最优解。历史文件不改名，但新结果中称为：

```text
FullTruth-Greedy reference
```

只有在完整 episode 上建立全局求解，并给出 optimal 或可核验的 MIP gap/下界后，才使用 `Oracle`。正式主结论不依赖 Oracle 这个名称。

## 9. 正式实验矩阵

### 9.1 Validation 阶段

必须运行：

```text
RR（附录）
Myopic
Optimizer-0
PredOpt-H1 / H3 / H5
TrueOpt-H1 / H3 / H5
FullTruth-Greedy reference
PredOpt-H*+Preload（基础 PredOpt-H gate 通过后）
```

RL 阶段运行：

```text
RL-0 seed 11/22/33
TrueRL-H1/H3/H5 pilot（先 seed 11；有正收益后对 H* 补 seed 22/33）
RL-H1/H3/H5 pilot
RL-H* seed 11/22/33
RL-0+Preload / RL-H*+Preload（第二层消融）
```

H* 只由 validation 选择。`tau`、deadline 权重、queue-age 上限、solver time limit、RL 超参和 checkpoint 同时写入 freeze manifest。

### 9.2 C1 retrospective 与 C2 confirmatory 阶段

冻结后先在 C1 的 6,750 episodes 运行同分布 retrospective；确认所有预注册配置、实现和产物 hash 不再改变后，才在 C2 的 6,750 episodes 一次性运行 confirmatory。两者主要策略相同：

```text
Myopic
Optimizer-0
PredOpt-H*
TrueOpt-H*
RL-0 × 3 seeds
RL-H* × 3 seeds
```

H1/H3/H5 全曲线可在 C1 retrospective 作为次要结果；C2 只运行 H* 主策略矩阵，防止确认性测试后挑最好 horizon。

### 9.3 主要比较与解释

| 比较 | 解释 |
|---|---|
| Optimizer-0 − Myopic | 优化器收益，不是预测收益 |
| PredOpt-H* − Optimizer-0 | 数学优化器中有限预测未来的净收益 |
| TrueOpt-H* − Optimizer-0 | H 步信息可利用上界 |
| PredOpt-H* − Myopic | 完整预测式优化系统收益 |
| RL-H* − RL-0 | RL 对有限预测未来的净收益 |
| RL-H* − Myopic | 完整预测式 RL 系统收益 |
| PredOpt-H* − TrueOpt-H* | 行为预测误差造成的可回收差距 |
| PredOpt-H*+Preload − PredOpt-H* | 投机预加载机制的净收益与误加载代价 |
| PredOpt-H*+Preload − Optimizer-0+Preload | 相同预加载动作空间下有限未来的净收益 |

## 10. workload 合法性与正式统计

### 10.1 打开正式比较前的 workload gate

除当前 28.01% aggregate gap 外，还必须补：

1. 135 cells 全部存在且每 cell episode 数正确；
2. 按 arrival pattern、pressure、GPU topology、initial residency 分层报告 Myopic/FullTruth gap；
3. 报告真正发生选择的 decision opportunity rate：`candidate_actions > available_gpu_slots` 或存在多个可行 node×GPU 映射；
4. 报告 queue wait、GPU utilization、cache hit、cold load、eviction 分布；
5. 确认 gap 不是只由一个视频、一个 baseline 或一个模型栈贡献；
6. validation deterministic replay 保持通过；
7. 当前预测资源与执行真值彻底分离后，gap 仍存在。

若 TrueOpt-H 对 Optimizer-0 没有改善，说明即使给真实有限未来，当前目标/动作空间也用不上未来，应停止 Pred-H 主表并修正 formulation，而不是继续调 predictor。

### 10.2 指标

主要指标：

- mean job completion time / mean JCT；
- global P95 job completion time；
- deadline miss rate。

次要指标：

- mean queue time、makespan、throughput；
- GPU utilization；
- model cold-load time、cache hit rate、eviction count/bytes；
- predictor、solver、RL inference p50/p95 overhead；
- optimizer fallback rate、effective-lookahead rate；
- capacity invariant violation count。

### 10.3 配对统计

1. 所有策略使用完全相同的 episode_id、arrival、template 抽样、GPU topology 和初态；
2. 保存 job-level 结果，不能用“每 episode 的 p95 再平均”代替全体 job P95；
3. 在 135 cells 内进行配对、分层 episode bootstrap，10,000 次，固定 bootstrap seed；
4. 同时报告 absolute delta、relative delta、95% CI、episode win rate 和 cell win rate；
5. H1/H3/H5 的次要多重比较用 Holm 修正；H* 在 validation 预先选定后作为 C1/C2 的主检验；
6. RL 报告每个 seed 的配对结果及跨 seed 汇总，不能只给均值；
7. 每套 6,750 episodes 都是 workload 重放，不等于 6,750 条独立语义 trace，论文中必须区分。

指标分母冻结为：

- mean/P95 JCT：所有到达 job；失败/死锁 job 不能从分母删除，另报 failed-job rate，并将未完成 job 视为 gate failure；
- deadline miss：所有设置 deadline 的到达 job；`deadline_ms` 本身可见，`deadline_multiplier` 不可见；
- completed-job-only 指标只可放诊断附录，不作主指标；
- P95 从全部 job-level completion values 一次计算，并额外做按 episode/cluster 的不确定性重采样。

因为同一语义 template 会在大量 workload episode 中被重复抽样，统计独立性不能把所有 job 当作独立样本。主 bootstrap 的随机化单位是配对 episode，并在 scenario cell 内重采样；C1 retrospective 估计的是“给定这 8 个 C1 test 视频时，到达、任务混合和资源状态变化下的收益不确定性”，不声称语义总体泛化。另做 8 次 leave-one-video-out C1 workload 敏感性（不用于调参），检查结论是否被单个视频驱动。C2 以目标 32 个独立视频提供更可信的语义外推证据，但 6,750 episodes 仍主要提高到达/资源场景精度，而不是变成 6,750 个语义独立样本。

### 10.4 主成功门槛

对 `PredOpt-H* vs Optimizer-0` 和 `PredOpt-H* vs Myopic`：

- mean JCT 相对改善的 95% CI 不跨 0；
- deadline miss 的绝对恶化上界不超过 0.5 个百分点；
- P95 JCT 不出现超过 1% 的显著恶化；
- capacity violation=0；
- fallback、effective horizon 和 overhead 完整报告。

RL-H 的信息收益以 `RL-H* vs RL-0` 为主；要求三 seed 方向一致，并报告是否在计入推理开销后仍成立。

上述数值门槛是待在 validation 开始前冻结的预注册提案，不是已取得的实验结论；若需调整，只能在打开 validation 结果前改版本并记录理由，不能看到结果后追改。

## 11. 调度开销必须怎样计入

预计算 artifact 查表很快，但真实系统仍需运行行为预测器。因此每个策略做两套报告：

1. `schedule_quality_only`：不把决策 wall time推进模拟时钟，用于隔离调度质量；
2. `overhead_accounted`：把 predictor + policy/solver 的实测决策时间作为 dispatch delay 推进时钟，作为系统主结果。

要求：

- B05 对 H1/H3/H5 在固定 1,000 个 prefix 上测在线推理 p50/p95；
- artifact replay 使用该冻结预测开销，而不是把查表时间冒充在线推理时间；
- Myopic、Optimizer、RL actor 都测实际 decision wall time；
- True-H/FullTruth 只作为信息上界，不拿“无预测开销”与 deployable 策略做系统公平比较。

## 12. 代码落地路径

遵循“复用事件循环、最少新接口”的原则，计划只新增必要模块。

### 12.1 修改现有文件

`src/tracing/analysis/workload_v02_simulator.py`

- 保留现有 CLI 兼容；
- 把 policy 分支改成注入式 policy callback；
- 分离 SchedulerStateView 与 ExecutionTruthProvider；
- 正式执行 duration 只使用执行真值，预测值只进入 scheduler view；
- 支持 summary-only、sampled-decisions 和 smoke-full-events 三档日志。

`tests/test_workload_v02_simulator.py`

- 保留已有 admission/Myopic 回归测试；
- 增加 legacy parity、执行真值隔离和日志级别测试。

`scripts_behavior_r2_v5/r2_v5/r2b/final_holdout_runner.py`

- 复用其现有 `fit_b05()` 和 seed 11/22/33 训练路径；
- 参数化 dataset/output root，使其读取统一 split registry 生成的 decision-time dataset；
- 不改 B05 网络结构，不复用旧 split checkpoint 冒充新 holdout。

原始 decision-time dataset builder 需要在现有 `scripts_behavior_r1_v5/build_r1_v5.py` 的旁路版本中实现：不覆盖 v5 数据；对每个 current node 用 `event_index < current` 重算结构计数和 evidence，只保留 current identity，目标从 current 之后开始。若实现能以参数化 mode 清晰复用现有 builder，则不新建第二个大文件。

`scripts/scheduling_future_b05_artifacts.py`

- 改为显式接收新 checkpoint/dataset/split registry；
- 以 decision-time current identity 为 rollout 起点，offset 1 就预测 current 之后的第一个节点；
- 删除 same-run future tool row 和 template-specific future prototype；
- 输出 causal-v2、prefix registry、post-cutoff mutation audit；
- v1 输出保持不覆盖。

正式资源模型重训应复用现有
`resource_predictor_v1_final_q99_20260811` 对应训练/选择流程，而不是退回
`src/tracing/analysis/reproduce_resource_predictor.py` 的简单 median adapter；后者只作为小型审计基线。实现前先定位生成 `models.pkl` 的准确入口和配置，若无法唯一追溯则 R0 阻断并补 manifest，不能暗猜命令。

### 12.2 新增文件

```text
src/tracing/analysis/scheduling_future_provider.py
  B05/True-H/None provider、prefix registry、train-only stack/resource prototype、unknown gate

src/tracing/analysis/scheduling_optimizer_h.py
  ActionBuilder、Optimizer-0、PredOpt-H、TrueOpt-H、HiGHS MILP

src/tracing/analysis/scheduling_rl_h.py
  RL observation、masked actor-critic、PPO trainer、checkpoint I/O

scripts/run_scheduling_future_v1.py
  统一 smoke/validation/C1-retrospective/C2-confirmatory runner、分片、resume、freeze 校验

configs/scheduling_future_v1.json
  信息模式、H、tau、权重、solver、seed、日志级别

configs/video_split_registry_v1.json
configs/gpu_capacity_registry_v1.json
  跨预测器/workload 的唯一视频 split 与物理容量映射

tests/test_scheduling_information_boundary.py
tests/test_scheduling_optimizer_h.py
tests/test_scheduling_rl_h.py
```

若实现时发现现有代码已有等价接口，应复用并减少文件，不为了匹配本文档强行创建抽象。

## 13. 逐步执行顺序与门禁

### R0：preflight、冻结登记与派生 overlay

动作：

- R0a 只读：冻结所有输入 SHA256、行数、schema；
- R0a 只读：审计 64 个视频在 B05/resource/workload 的 split 交叉表，以及 baseline/stack、视频域、时长、模板长度分布；
- R0a 只读：计算 135-cell historical preliminary gap、decision opportunity rate 和模型栈贡献；
- R0a 只读：检查 9,366 node 的 B05/resource/C1 template 精确连接，定位资源模型训练入口和配置；
- R0a 只读：审计 raw-DAG 与静态 linear-prefix 的不一致率，并输出 proposed registry/overlay diff；
- R0b 派生写入：经 R0a 验收后，新建而不覆盖 `video_split_registry_v1`、GPU capacity registry 和 causal-chain overlay；
- R0b 派生写入：把 32GB class 映射到 C1 实测容量并明确标记 24GB extrapolation；
- R0b 派生写入：从 train split 生成 stack-tool/resource prototype coverage 报告，不写正式预测 artifact；
- R0b preliminary：只在 validation 输入上重跑 Myopic/FullTruth，确认修正 overlay 与 truth semantics 后是否仍有调度空间；不运行 C1 retrospective 或 C2 confirmatory。

门禁：原 workload 内部 split isolation、跨预测器/workload split 对齐方案、coverage、deterministic replay、磁盘预算通过。当前跨组件 split 审计已明确失败，因此正式 R2 前必须重训/重建，不允许直接复用旧 checkpoint。

### R1：统一 state/action/truth 边界

动作：

- 在现有事件引擎中引入 SchedulerStateView 与 policy callback；
- 用 adapter 复现原 Myopic/RR；
- 修正预测资源与执行真值耦合；
- 加三档日志。

门禁：固定 10/50 episodes 的 legacy parity、DAG、arrival、capacity 和 replay 测试通过。

### R2：有限未来 provider

动作：

- 按统一 48/8/8 video registry 重建行为/资源数据并重训 B05/resource predictor；
- 为 RL train workload 生成按 video 分组 cross-fitting 的 OOF causal-v2 future/resource artifact；
- 修复 B05 rollout 的 same-run future-row 读取，生成 causal-v2 H1/H3/H5；
- 实现 none/pred_h/true_h/full_truth；
- 生成并冻结 train-only stack registry 和 resource prototype；
- 实现 supported mass、tau、H0 fallback；
- 记录 effective horizon。

门禁：三方 video split 对角一致、test-video mutation/removal 不影响 train artifact、current/post-current row deletion invariance、current result mutation invariance、prefix/current-identity hash 一致、无真值泄漏、H 截断、概率归一、unknown 非零回退全部通过。

### R3：Optimizer-0

动作：

- 实现 ActionBuilder 和 SciPy MILP；
- 与穷举对拍；
- 先跑 50 episodes，再跑 validation 固定子集；
- 确认 Optimizer-0 相对 Myopic 的行为合理。

门禁：solver incumbent≥99%、capacity violation=0、回退可解释。

### R4：TrueOpt-H，再接 PredOpt-H

顺序不能反：

1. 先跑 TrueOpt-H1/3/5，验证 formulation 真能使用有限未来；
2. True-H 有收益后，接 PredOpt-H；
3. 运行 H×tau×time-limit 的小预算 validation；
4. 冻结 H*、tau、权重和 solver time limit。

门禁：True-H gate 通过；Pred-H 所有决策均有信息边界审计行。

### R5：Optimizer 正式 validation 与 freeze

动作：

- 在完整 1,000 validation episodes 上跑配对策略；
- 生成 job-level、episode-level、cell-level 和 overhead 报告；
- 锁定配置、源码 SHA256、artifact SHA256；
- 写 `validation_freeze_manifest.json`。

门禁：未运行 C1 retrospective 或 C2 confirmatory；所有主参数 frozen。

### R5b：投机预加载消融

仅在 R4/R5 基础门禁通过后：

- 加入不执行工具的 model preload action；
- 对拍 preload load-time、resident memory、错误预加载和驱逐账本；
- 在 validation 比较 H=0 popularity preload、Pred-H preload、True-H preload；
- 冻结 preload probability threshold 和预算；
- 将其作为独立机制列，不覆盖基础调度结果。

### R6：RL 环境、RL-0 与 RL-H

动作：

- 复用同一 ActionBuilder 和事件引擎；
- 先 reward/action-mask/determinism 单测；
- 训练 RL-0，再用相同预算训练 RL-H；
- 三 seed validation；
- 只选一个 H* 进入 C1 retrospective 和 C2 confirmatory 主比较。

门禁：RL-0/RL-H 配对公平、三 seed 可复现、无 test 参与选择。

### R7：C1 retrospective 与一次性 C2 confirmatory

动作：

- runner 校验 freeze manifest 和输入 hash；
- 先分片运行 C1 6,750 retrospective episodes，生成明确标为 retrospective 的表；
- C1 完成后只允许修复会使整轮作废的实现错误；若发生修复，重新冻结并重跑 C1，不得参考 C2；
- 按 R0 冻结的主样本/替补顺序和 baseline×stack 采集矩阵生成 C2 Agent traces；失败处理只能执行预注册规则；
- 用冻结的 C1-train predictor 对 C2 decision-time prefix 做推理，生成 C2 causal-v2/resource artifact；不得用 C2 label 做校准、重训或改阈值；
- 构建并冻结 C2 causal-chain templates、135-cell workload、输入 SHA256 和 split/去重审计；正式 runner 运行前只做 schema、coverage、capacity 和 deterministic input replay，不在 C2 上试跑任何策略；
- 再分片运行 C2 6,750 confirmatory episodes，每个策略、seed、episode 配对；
- 汇总前检查所有 shard 完整；
- C2 生成主 bootstrap CI 和论文主表，C1 作为同分布回溯证据。

门禁：禁止看到 C2 后改 H、tau、权重、reward、checkpoint 或 C2 视频；需要改动时该轮 C2 降级为 exploratory，另收集全新 confirmatory set，不能覆盖或伪装成同一次测试。

### R8：contention-aware 鲁棒性实验（后续）

仅在主结果稳定后：

- 固定 token 数、帧数、batch、并发起止窗口重做 slowdown calibration；
- 为每个允许共置的模型对生成 memory peak + duration multiplier；
- 事件引擎支持 active-set 变化时更新 remaining work 和 finish event；
- 未测 pair 禁止共置；
- 将 serial-GPU 主实验作为控制，不覆盖。

## 14. 单测与泄漏审计清单

### 14.1 信息边界

- H=0 provider 不打开 B05 文件；
- 正式 runner 拒绝 B05 diagnostic v1，只接受 causal-v2 schema/hash；
- 删除或打乱同 run 的所有 post-cutoff row 后，causal-v2 输出不变；
- 改写当前 ready node 的 runtime/status/output/evidence delta 后，causal-v2 输出不变；
- causal-v2 的 H1 target 是当前 ready node 之后的节点，不是当前节点本身；
- validation/test tool/model/lane prototype 不参与 train-only registry；
- B05、resource predictor、workload 和 RL 共用同一 video split registry；
- 删除 C1 test-retrospective/C2 confirmatory 视频后，所有 train-fitted artifact SHA256/参数不变；
- causal-chain 每次决策的 actual prefix hash 与 artifact 一致；
- Pred-H 修改真实 future runtime/path 后动作不变；
- Pred-H 修改 B05 概率后动作按预期改变；
- True-H 修改第 H+1 步后动作不变；
- RL observation 不含 template_id/video_id/真实后缀；
- StateView 不含完整 job runtime estimate、deadline multiplier 或由完整 DAG 得出的剩余图特征；
- train normalization 不读取 validation/test；
- reward 可读取动作后的真实结果，但真实结果不能进入动作前 observation。

### 14.2 资源与状态机

- CPU/API GPU memory 恒为 0；
- GPU current prediction 与 execution truth 分离；
- raw-DAG prefix 不一致时禁止静态查询带 observed-runtime 的 resource artifact；
- truth runtime/load 语义不重复计费，预测 load 不得改变执行真值；
- resident + workspace 账本一致；
- 32760 class 必须映射到 C1 实测 32230.812MiB，并重验 initial residency；
- 未 ready/未到达/前驱未完成节点永远不能调度；
- 同一节点不能重复 dispatch；
- cold load、cache hit、eviction 后驻留正确；
- preload 真实耗时、错误预加载和被抢占/驱逐结果正确；
- 没有未来完成事件时不得死锁。
- 任一 job 未完成/failed 时主 episode 判 gate failure，不能从 JCT 分母删除。

### 14.3 优化器

- MILP 与穷举对拍；
- 不可行和 timeout 回退原因可复现；
- H=0 与 H>0 只有 future coefficient 不同；
- 选中动作数、node uniqueness、GPU uniqueness 正确；
- 相同输入、seed 和 HiGHS 配置 deterministic replay。

### 14.4 RL

- action mask 无非法动作；
- RL train observation 只读取 OOF future/resource artifact，不能读取同视频 in-sample 预测；
- candidate permutation 不改变策略分布（稳定排序或集合编码）；
- RL-0/RL-H 参数量相同；
- checkpoint 含 config/artifact hash；
- fixed-seed rollout、loss、validation selection 可复查；
- reward 离线重算一致。

## 15. 产物与磁盘策略

统一写入，不覆盖历史结果：

```text
results/processed/scheduling_future_v1_20260812/
  configs/scheduling_policy_v1.json
  manifests/future_resource_prototypes_v1.json
  manifests/validation_freeze_manifest.json
  smoke/engine_parity/
  smoke/optimizer_h/
  smoke/rl_h/
  validation/optimizer_h/
  validation/rl_h/
  c1_retrospective/optimizer_h/
  c1_retrospective/rl_h/
  c2_confirmatory/optimizer_h/
  c2_confirmatory/rl_h/
  tables/
  overhead/
```

日志策略：

- 10/50 episode smoke：可保存完整 gzip events；
- 1,000 validation：保存 episode/job summary，decision 日志按 cell 分层抽样；
- C1/C2 各 6,750 episodes：只保存 gzip episode/job summary、solver/RL overhead 聚合和不超过 1% 的 decision sample；
- 不再生成类似 2.3GB 的默认未压缩全量 events；
- 任何清理旧日志的动作另行请示。

## 16. 停止规则与问题定位

| 观察 | 应采取的动作 |
|---|---|
| Optimizer-0 明显劣于 Myopic | 检查目标、动作构造和求解器；不接 Pred-H |
| TrueOpt-H 不优于 Optimizer-0 | 未来没有进入可用决策，或 workload/目标无有限前瞻价值；停止调 predictor |
| TrueOpt-H 有益、PredOpt-H 无益 | 检查 B05 预测质量、prototype resource join、supported mass 和置信度门控 |
| PredOpt-H 质量好但计入 overhead 后无益 | 优化预测/求解时延或异步化；不能只报免费预测结果 |
| RL-0 已不稳定 | 先修 reward、mask、训练方差；不解释 RL-H |
| RL-0 稳定、TrueRL-H 有益、RL-H 无益 | 未来预测质量或编码问题 |
| 只有少数高压力 cell 有收益 | 将结论限定为高压力条件，不能写成普遍收益 |
| H=5 effective-lookahead 极低 | 报告 H=5 的覆盖边界，不能以 nominal H 冒充实际 H |
| capacity invariant 失败 | 整轮结果作废，修账本后重跑 validation |

## 17. 执行优先级结论

推荐顺序为：

```text
统一引擎与信息边界
  → Optimizer-0
  → TrueOpt-H
  → PredOpt-H
  → 完整 validation 与 freeze
  → RL-0 / RL-H
  → C1 retrospective
  → 一次性 C2 confirmatory
  → contention-aware 扩展
```

原因：先用 True-H 验证“有限未来能否被当前动作空间和目标函数利用”，可以在训练 RL 之前排除 formulation 问题。Optimizer-H 是主线，RL-H 是第二条利用同一有限前瞻接口的策略路线；RL 不应阻塞最先形成的可解释因果证据。

## 18. 本文档完成后的第一个执行门禁

在获得下一次代码修改授权后，只执行 R0 和 R1：

1. 先完成 R0a 只读 workload/split/coverage 审计并交付审计结论；
2. R0a 验收后才写入不覆盖历史文件的 split/capacity registry 与 causal-chain overlay；
3. 重构现有 simulator 的 StateView/Policy/Truth 边界；
4. 复现 Myopic/RR 的 10/50 episode parity；
5. 不实现 Optimizer、不训练 RL、不运行或读取 C1 retrospective/C2 confirmatory 结果；
6. R1 验收通过后，再请示进入 R2/R3。

这一步是最小、可回滚且能排除后续所有信息泄漏与双 simulator 漂移的实现起点。
