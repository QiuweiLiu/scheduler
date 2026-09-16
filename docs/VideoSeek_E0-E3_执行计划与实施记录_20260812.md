# VideoSeek E0–E3 执行计划与实施记录

版本：2026-08-12  
项目：面向未来执行图未知 Agent 的节点级 GPU 调度  
执行环境：远端 `/root/autodl-tmp/scheduler`，单卡 RTX 4080 SUPER 32GB

## 0. 本轮边界

本轮把《VideoSeek 下一阶段实验验证与论文组织计划》中的 E0–E3 落地。顺序固定为：

1. 冻结输入、数据 split、资源契约和代码快照；
2. 将 B05 Masked GRU 封装为可重复调用的预测产物；
3. 用相同的 scheduler-visible prefix 生成 H=1/3/5 的有限未来预测；
4. 接入冻结资源预测产物，完成预测/真值隔离和小规模 smoke；
5. 只有通过无泄露、覆盖率和 deterministic replay 门禁，才进入统一 Optimizer-0 / Pred-H / True-H 的实现。

本轮不做：全量 locked test、RL、Grounded-VideoLLM 官方复现、重新采集主线 trace、删除旧结果、初始化 Git 仓库。

## 1. 已冻结的输入

| 输入 | 远端路径 | 用途 |
|---|---|---|
| 768 templates / 9,366 nodes | `results/processed/workload_v0_2_smoke_fixed_20260812/job_templates.jsonl` | 节点级执行图 |
| formal workload | `results/processed/workload_v0_2_formal_20260812/` | 后续 validation/locked test |
| B05 data contract | `results/processed/behavior_nn_v1_r1v5_candidate/data/` | 全量 64 视频 prefix 覆盖 |
| B05 train/validation/checkpoints | `results/processed/behavior_nn_v1_r2/final_holdout/`、`results/processed/behavior_nn_v1_r2/runs/` | 冻结 encoder 与模型参数 |
| resource contract | `results/processed/resource_predictor_v1_final_q99_20260811/` | runtime/load/peak-memory 预测 |
| current simulator | `tracing/analysis/workload_v02_simulator.py` | 旧 W6 参考，不能直接冒充 E1 |

数据和产物不可覆盖。新实验统一写到：

```text
results/processed/scheduling_future_v1_20260812/
  configs/
  manifests/
  prediction_artifacts/
  smoke/
  validation/
  locked_test/
  leakage_audit/
  overhead/
```

## 2. 信息边界

所有策略的输入都是：当前 prefix、ready nodes、arrival/deadline、GPU 状态、resident model、资源预测。

- `Optimizer-0`：不读取未来 artifact；
- `Pred-H`：额外读取 B05 生成的 H 步概率场景；
- `True-H`：读取同一 H 步的真实后缀，仅作为有限未来上界；
- `Full-info reference`：读取完整剩余路径，仅作为信息上界；
- `Oracle` 这个名称只在完整未来被全局求解/枚举验证后使用。旧 W6 的真实后缀贪心结果保留为历史动机，不改名、不混入主结果。

硬约束：Pred-H 文件不得包含未来真值、target 标签、完整后缀、observed runtime/load/memory；True-H 和 Full-info 必须由不同 provider 读取。

## 3. E0：冻结与 registry

### 动作

1. 检查文件存在、JSONL 可解析、计数和 schema version；
2. 记录每个输入文件的 size、SHA256、行数；
3. 记录 Python、PyTorch、CUDA、GPU、SciPy 和可用求解器；
4. 记录当前远端不是 Git worktree，因此 `git_commit=null`，改用源码快照哈希；
5. 生成 `manifests/input_freeze.json` 和 `configs/e0_freeze.json`。

### 门禁

- templates=768，nodes=9,366；
- workload train/validation/test=20,000/2,000/6,750；
- locked test cells=135，每 cell=50 episodes；
- B05 角色/工具数据可解析；
- 资源契约 schema 为 `scheduler-resource-contract-v1`；
- 远端磁盘空间足够保存本轮小产物；不能因为空间不足产生未压缩全量事件日志。

## 4. E1 前置：B05 冻结推理

### 动作

1. 只用 B05 的 train split fit `StaticPreprocessor` 与 `SequencePreprocessor`；
2. 用 B05 seed 11/22/33 checkpoint 加载 role/family 两个 head；
3. 对 canonical role samples 逐条推理，三 seed 概率取平均；
4. 将 `source_event_id` 映射到 9,366 个调度节点；
5. 对每个节点保存当前 prefix、role 概率、family 概率来源和 coverage；
6. 用最终 holdout 样本复算已有 B05 predictions，检查浮点容差内一致；
7. 把可复算失败、缺少 prefix、无法映射的样本显式写入 audit，不用零向量静默替代。

### 产物

```text
prediction_artifacts/b05_node_h1.jsonl.gz
prediction_artifacts/b05_node_h1_manifest.json
leakage_audit/b05_recompute_report.json
```

### 门禁

- 9,366 节点全部匹配 canonical source event；
- role/family 概率有限、非负、和为 1；
- holdout 复算在固定容差内一致；
- 预测输入不包含 target、未来事件、answer text、video id feature；
- 预测 artifact 与 truth artifact 物理分离。

## 5. E2：H=1/3/5 有限未来场景

### 场景生成

对每个当前节点的 causal prefix：

1. 用 B05 role head 预测下一 role；
2. role=execute 时，用 semantic-tool prefix 预测 action family；
3. 用统一 role/action-family 映射得到 model slot 和 execution lane；
4. 对预测 prefix 追加预测动作，重复 rollout；
5. 用合法角色转移约束过滤非法分支；
6. beam 保留 top-K 分支，并保存概率质量、终止概率和 `synthetic_rollout=true`；
7. H=1/3/5 分开写出，H=8 只在 validation 做开销试验。

### 产物字段

```text
prefix_hash, horizon, scenario_id, scenario_probability,
step_offset, role, action_family, model_slot, execution_lane,
predecessor_offset, terminal_probability, source_model, seed_ensemble
```

### 评估

- next-role/family top-1、NLL、Brier、ECE；
- path hit@1/3/5；
- top-K coverage；
- 终止长度误差；
- error 随 H 的累积曲线。

## 6. E3：资源预测冻结与组合消融准备

1. 通过现有 resource predictor `models.pkl` 对 candidate node 输出 runtime p50/p90、load p50、peak-memory p95、uncertainty；
2. 不再把当前 simulator 的 `train_resource_stats()` 当作正式预测器；
3. runtime/load/VRAM 真值只进入 simulator truth；
4. unknown 保持显式状态，使用已有 contract 的层级回退并记录 fallback level；
5. 生成 A0/A1/A2/A4 的输入 artifact：
   - A0：无未来；
   - A1：Predicted behavior + Predicted resource；
   - A2：True behavior + Predicted resource；
   - A4：True behavior + True resource。

A3（Predicted behavior + True resource）暂不伪造反事实节点的真实资源标签，待 E6 真实 GPU calibration 后再实现。

## 7. Smoke 与后续调度器

E0–E3 门禁通过后，才实现统一 rolling-horizon optimizer：

- H=0/1/3/5 共用同一目标、约束、solver 和 time limit；
- 每次只执行第一步 dispatch，再重新求解；
- timeout 有 incumbent 用 incumbent，无 incumbent 回退 Myopic，并记录原因；
- 必须先通过 DAG、arrival、VRAM、no-future-leakage、deterministic replay 单测；
- 先在 30–50 个 validation episodes smoke，再跑全 validation；
- validation 冻结 horizon、权重、风险系数、solver time limit 后，才打开 locked test。

## 8. 执行记录

### 2026-08-12 初始执行

- 原始计划 DOCX 已只读审阅，未覆盖；
- 远端确认：单张 RTX 4080 SUPER，剩余空间约 16GB；
- 远端当前已有下载清理进程，主线 trace/workload 未被停止；
- 远端项目根目录不是 Git worktree，本轮使用快照哈希；
- 发现 B05 candidate 数据与 9,366 个模板节点可 100% source-event 对齐，作为 E1 推理输入；
- 发现旧 W6 simulator 只有 `round_robin/myopic/oracle`，且 oracle 是真实后缀贪心，保留为历史结果，不直接当作 E1 Oracle。

### 2026-08-12 E0–E3 实施记录

- E0 已通过。远端脚本 `scripts/scheduling_future_freeze.py` 生成
  `results/processed/scheduling_future_v1_20260812/`，冻结结果为
  `templates=768`、`nodes=9,366`、`workload_test_episodes=6,750`、
  `locked_test_cells=135`。
- B05 适配器为 `scripts/scheduling_future_b05_artifacts.py`。它只在 train
  split fit encoder，加载 seed 11/22/33 checkpoint；holdout 复算改用官方
  batch=64 推理路径，避免 CUDA packed-GRU 的 batch-size-1 数值差异。
- B05 20 节点 smoke 已通过：holdout recompute、标签突变防泄露、概率归一化
  均通过，并写出 H=1/H=3/H=5 gzip artifact。优化后的 rollout 将一次 H=5
  再截取 H=1/H=3，且缓存 observed-prefix 概率；这不改变模型、beam 或信息边界。
- 全量 B05 已在远端后台执行，日志为
  `results/processed/scheduling_future_v1_20260812/full_b05/b05_full_optimized.log`；
  正式 artifact 只有任务成功后才写入主 `prediction_artifacts/`。
- E3 已完成。`scripts/scheduling_future_resource_artifacts.py` 使用冻结
  `resource_predictor_v1_final_q99_20260811/models.pkl` 和其正式 selector，生成
  `prediction_artifacts/resource_predictions_v1.jsonl.gz`（9,366 行）。
  资源输入不读取当前节点 runtime/load/peak/queue/resident/error/answer 等目标；
  上一已完成节点 runtime 仅作为 observed-prefix 特征。所有 9,366 节点没有
  `input_scale`，因此显式保留 unknown（-1），没有静默填零；这已记录在
  `prediction_artifacts/resource_artifact_manifest.json` 和
  `leakage_audit/resource_leakage_report.json`。
- 当前允许进入下一步的门禁：E0、E3 和 B05 smoke 已通过；全量 B05 写出并解析
  前，不进入 E4 locked test 或调度器正式对比。
- 全量 B05 已完成并验收：三份 gzip 均为 9,366 行，coverage=1.0，holdout
  recompute=true，leakage=true。正式路径为
  `prediction_artifacts/b05_node_h1.jsonl.gz`、`b05_future_h3.jsonl.gz`、
  `b05_future_h5.jsonl.gz`。
- 调度 smoke 脚本为 `scripts/scheduling_future_smoke.py`。首轮报告曾因校验器
  忘记登记 CPU/API 节点 `node_start` 而阻断；修正状态机后，10 episode 和 50
  episode 两轮均通过。50 episode 共产生 248,479 个事件，DAG/到达/状态机和
  deterministic replay 均通过；报告在
  `validation/scheduler_50ep/scheduling_smoke_report.json`。
- 50 episode 聚合结果（这里只作为 validation smoke，不是 locked-test 结论）：
  `pred_h0` 平均完成时长 146,593.65ms，`pred_h1` 146,517.70ms，
  `pred_h3` 146,708.66ms，`pred_h5` 146,889.53ms；`true_h5` 的 deadline
  miss rate 为 0.035。不同 horizon 已产生可见但很小的差异，说明调度器确实
  读取了有限前瞻；差异大小不能替代正式统计检验。
- 因为本轮的资源输入尺度在模板节点上全部 unknown，E3 资源 artifact 仍只
  能作为“契约和泄露边界已验证”的预测输入；要声称资源预测改进或进入论文
  主表，仍需 E6 的真实 GPU calibration。当前不打开 135-cell locked test。

### 2026-08-12 C1：真实单卡资源/争用标定

- C1 使用远端 `NVIDIA GeForce RTX 4080 SUPER`（约 32,231 MiB 显存）、
  `finetooling` 环境和一个已有抽帧，固定 `repeats=3`，每个 case 以约
  100ms 频率记录 GPU 利用率、显存、功耗和采样序列；本轮明确不是 Agent
  trace，也没有主动制造 OOM。
- 最终 manifest 为
  `results/processed/scheduling_future_v1_20260812/calibration/resource_contention_v1/calibration_manifest.json`，
  共 12 个 case，12/12 success，`failed_cases=0`，结束时无标定或模型
  worker 残留进程。远端日志为同目录上一级的
  `calibration_stdout.log`。
- YOLO11x batch 基线的 GPU 采样峰值（MiB）为：batch 1/8/16/32/64 =
  1,065/1,883/3,099/5,123/9,219；对应 worker 的 CUDA peak reserved
  约为 476/1,294/2,510/4,534/8,630 MiB。batch64 达到 100% GPU 利用率，
  但没有 OOM。
- 单模型峰值（GPU 采样）为：Qwen3-VL-8B 20,033 MiB，Qwen2.5-VL-3B
  7,923 MiB，Qwen3-4B 8,385 MiB。模型请求均为 3 次，记录了 model load、
  inference 和 worker peak memory。
- mixed case 采用“YOLO batch32 后台持续推理 + Qwen worker”的真实重叠，
  不再把两个串行阶段的最大值冒充争用。峰值为：
  `Qwen2.5-VL-3B+YOLO11x=12,820 MiB`、
  `Qwen3-4B+YOLO11x=13,282 MiB`、
  `Qwen3-VL-8B+YOLO11x=24,938 MiB`；三者均达到 100% GPU 利用率且成功。
- 双 VLM 并发（Qwen2.5-VL-3B + Qwen3-4B）峰值为 16,046 MiB。期间修复
  了同一 client 多线程懒启动造成重复 worker 的竞态，并验证最终仅各有一个
  worker；早期的导入错误、竞态和非重叠 manifest 均保留为带说明的备份，
  不作为正式结果。
- C1 结果现在可以给 workload simulator 提供“实测资源层”：YOLO batch
  1/8/16/32/64、三种单模型、三种真实 mixed 组合和双模型组合；仍需在
  2+ GPU 或真实多租户进程上做后续扩展，不能把单卡 calibration 直接写成
  多 GPU 调度收益。

### 2026-08-12 C1 回填 workload 与 validation preliminary

- `scripts/build_c1_resource_profile.py` 从最终 C1 manifest 生成独立 profile：
  `results/processed/scheduling_future_v1_20260812/calibration/resource_contention_v1/c1_resource_profile.json`。
  profile SHA256 为
  `ea63a9f35adf4063fe39d592607783c38502953f305cd7ddf7499dc1f02bac17`，
  明确区分模型驻留平台、活动 workspace 和 confirmed overlap peak，不把
  混合峰值倒灌成单节点资源标签。
- `scripts/build_c1_workload.py` 调用已有 v0.2 builder，输出目录为
  `results/processed/scheduling_future_v1_20260812/workload_c1_v1/`，没有覆盖
  2026-08-04 的 workload。768 templates/9,366 nodes 保持不变，
  train/validation/test 为 20,000/1,000/6,750 episodes；Alibaba 到达回放、
  burst、poisson，各 5 档压力、3 种 GPU topology、cold/warm/skewed 初态和
  135 个场景 cell 均有覆盖，三份 summary 的 split isolation、scenario
  matrix、residency capacity gate 均为 true。
- C1 validation 50-episode smoke 通过，随后完成 validation 全量 1,000
  episodes，三策略报告为
  `workload_c1_v1/validation/simulator_1000ep/simulation_report.json`；事件
  文件约 2.3GB，包含 861,282 个 node dispatch 和 398,341 个 node wait
  事件，`scheduler_truth_separated=true`、`queue_generated_by_events=true`。
- preliminary 汇总（不是 locked-test 结论）：
  `round_robin` 平均完成 200,579.86ms、deadline miss 12.87%；
  `myopic` 199,097.66ms、11.86%；`oracle` 155,534.70ms、4.65%。
  Myopic 相对 Oracle 的平均完成时长 gap 为 28.01%，说明实测 C1 资源 profile
  和高压力 workload 已产生可见策略差距；仍需按 seed 分层、加入预测策略并
  检查 simulator 的单 GPU node-slot 假设后，才能进入正式 P4 主表。
- 当前下一门禁：先补全 C1 workload 的 deterministic replay/跨 seed 汇总，
  再实现预测器有限前瞻接入和 optimizer 对照；暂不读取或开启 6,750 条
  test workload 的 locked-test 结果，也不宣称已完成 2+ GPU 实验。
- 追加验收：同一 50-episode smoke 重跑的 `simulation_results.jsonl` 与
  `simulation_report.json` SHA256 均完全一致，deterministic replay 通过；
  `tracing/workloads/audit_workload_v02.py` 对 20,000 train + 1,000 validation
  的模板边、到达、资源 contract、容量、split 和 135 cells audit 通过，
  输出为 `workload_c1_v1/c1_workload_audit.json`。由于磁盘只剩约 13GB，
  尚未复制或运行 test locked workload。

后续每完成一步，追加：执行时间、命令、产物路径、门禁结果、失败原因和是否允许进入下一步。

## 9. 停止条件

任一条件触发时停止扩大实验：

- B05 复算不一致；
- source-event 覆盖率低于 100%；
- Pred-H 能读到 truth/target；
- H=0/Pred-H/True-H 不能 deterministic replay；
- 资源预测缺失被静默替换为零；
- smoke 出现 DAG、arrival、显存账本错误；
- 磁盘空间不足以保留 manifest、日志和压缩结果。

## 10. 验收标准

本轮完成的最低标准：

1. 有一份本文件对应的远端执行副本；
2. E0 manifest 可复查；
3. B05 H=1/3/5 artifact 可解析；
4. leakage audit 明确通过或明确阻断原因；
5. smoke 可 deterministic replay；
6. 没有覆盖 2026-08-12 之前正式结果；
7. 任何未通过的门禁都记录为 blocked，不包装成结果。
