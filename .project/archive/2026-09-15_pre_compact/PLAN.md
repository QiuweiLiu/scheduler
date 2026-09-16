# Plan

## Current Objective

在不改变 R7/T_final 边界的前提下，把 topology predictor 正式归入 predictor block，并验证它能否在与行为/资源预测器相同的视频数据体系上预测未来 DAG layers。正式拓扑训练/验证只使用 `P_dev`/`P_holdout_diag`；`S_train`/`S_val`/`T_final` 只在模型冻结后用于调度器集成评估。先完成拓扑标签覆盖与因果输入门禁，再比较统计、传统机器学习和序列神经网络；PPO、WAIT/RESERVE、抢占、多 GPU 和 `T_final` 继续封存。

## Execution Boundary (2026-09-04)

- 后续实验优先在独立本地副本 `/Volumes/Lenovo/scheduler` 执行，不依赖远端计算环境。
- 远端 `/root/autodl-tmp/scheduler` 已完整同步到外接盘；远端保留为源/备份，除非另行确认，不再把新实验默认提交到远端。
- 本地运行前先准备并验证依赖环境、checkpoint 加载和数据路径；不改变 `P_dev/P_holdout_diag`、`S_train/S_val`、`T_final` 的既定边界。

## Steps

1. [x] R0：本地与远端只读审计，确认根目录、磁盘、进程、数据和缺失项。
2. [x] R1：初始化 Project OS v1，建立唯一 canonical control plane。
3. [x] R2：本地/远端 `tracing/` → `src/tracing/`，同步脚本路径、导入入口和 `PYTHONPATH` 约定。
4. [x] R3：补齐项目边界、源码布局、旧 collection 与新 collection 的权威说明；旧计划不与当前计划并列。
5. [x] R4：建立远端 inventory、trace collection、source hash 和容量登记；不复制视频/模型到 Git。
6. [x] R5：完成逐视频 provenance、600 视频 source/archive 覆盖、容量 gate 和分批上传；旧 `P_dev` 已在 2026-08-17 经用户明确授权后按范围例外清理，其他 holdout/scheduler 视频继续按删除闸门保留。
7. [x] R6：固化 C1/C2 split 与 causal-v2 契约，重新生成 prefix/resource/workload 派生数据，并分别记录 train/validation/C2。
8. [x] R7-0：冻结600视频边界，修复截断ID，生成唯一分组注册表；不在ID未同步前生成正式workload。
9. [x] R7-1：在同一300-video predictor pool上对齐行为/资源预测器，冻结StateView、Truth和FutureProvider接口；旧64-video资源模型只保留legacy baseline。模型输出完整性、接口泄漏和10/50 contract smoke已通过；真实事件循环接入放入R7-2。
10. [x] R7-2：把冻结接口接入唯一事件循环，完成120个scheduler-train和40个scheduler-validation视频的640条trace整理、节点级压力workload和压力gate；workload审计通过。
11. [x] R7-3：完成RR、Myopic、Optimizer-0、PredOpt-H1/H3/H5、TrueOpt-H1/H3/H5的1,000-episode validation矩阵；TrueOpt-H优于Optimizer-0，PredOpt-H1/H3/H5随前瞻步数改善，进入RL阶段。
12. [x] R7-4：完成成对的RL-0/RL-H5三seed训练与验收；六个run均通过，validation保留为候选选择依据。
13. [x] R7-5：成对统计已完成，`H*=5` 为当前 PredOpt 候选；RL seed-consistency 未通过，暂不冻结 RL，`T_final` 继续封存。
14. [x] R8-P0：在冻结的 `S_train/S_val` 输入上审计 PredOpt-v2 加权目标；初始候选和三组 `S_train` 校准权重均未被提升，未读取 `T_final`。
15. [x] R8-P0b：记录 CP-RHO 前置依赖和模型契约；远端 `finetooling` 已固定安装 `ortools==9.9.3963`，最小 CP-SAT 实例返回 `OPTIMAL`，进入 P1。
16. [x] R8-P1：实现并完成 event-triggered rolling-horizon CP-SAT V1，只执行首动作；记录 solver status、gap、timeout、fallback、决策延迟和资源指标。1,000-episode S_val 矩阵通过执行/审计 gate，但 CP-RHO V1 因高 timeout/fallback 且性能不优于 PredOpt-v2，不提升为最终候选。
17. [x] R8-P2：在短窗口、小 workload 上完成同信息边界的 MILP sanity check；fix2 的 20/20 两种求解器均为 `OPTIMAL`，node-level 首动作一致率=1.0，目标相对差最大 `2.6319902025099073e-13 <= 1e-6`，复现元数据/候选与排程审计齐全；不作为在线主方法。canonical 输出为远端 `results/processed/r8_optimizer_rl_upgrade/p2_milp_sanity_20260819_fix2/`。
18. [x] R8-P3a：创建 BC 数据收集脚本 `r8_bc_collect.py` 和训练脚本 `r8_bc_train.py`；在 `choose_action` 注册 `bc_h` 策略；创建 EXP-20260819_r8_bc 实验目录。
19. [x] R8-P3b：远端 S_train 300 episodes 收集 246,567 条决策；BC 训练 50 epoch，val_accuracy=86.86%，S_val 评估 gap=+2.2%。
20. [x] R8-P3c：BC gate passed（动作一致率 86.86% >= 60%，completion +2.2% <= +10%），进入 P4。
21. [ ] R8-P4：将 episode-end REINFORCE 升级为 BC 初始化的 masked actor-critic PPO + GAE + 稠密 event reward；至少 5 个 seed 验收稳定性。BC 权重作为 actor 初始化，critic 随机初始化；在 S_train 上训练，S_val 上评估。
22. [ ] R8-P5：冻结一个 optimizer 配置和一个 RL 配置后，才在 80 个 `T_final` 视频运行最终 paired matrix；不因结果不佳替换视频。
23. [x] R8-P6：在不改变 R7/T_final 的前提下，加入 opt-in per-tool batch-size action、paid prefetch/load lane 和 node-level recompute preemption；完成同一 validation 前100集的 paired pilot、契约测试和容量/失败 gate。当前结论写入 `experiments/EXP-20260821_scheduler_extensions/`，不替换 R7 主表。
24. [ ] R8-P7（可选）：每个 YOLO batch 的真实 runtime/load profile 已于 2026-08-22 聚合并写入扩展配置；仍需按同一 episode 重新运行 pilot/完整1,000集并重新冻结目标，未完成前不得把本轮100集 pilot 当作 T_final 结果。
25. [x] R8 Transition Profile Simulator Pilot：将 `EXP-20260824_transition_profile` 的 cold-load/eviction proxy 以 opt-in 方式接入单 GPU slot simulator；完成严格 GPU class 过滤后的 55 episode paired pilot、10 episode event audit、契约/容量/失败 gate。该 pilot 不替代 R8-P7，不解封 `T_final`，不引入多 GPU 语义。
26. [x] R8-P8：完成 100 集 common-state Action-Value Audit；15,364 个 dispatch states、strict-feasible action enumeration、model/role/cache/deadline/width strata 和输入完整性 gate 全部通过。
27. [x] R8-P8b：完成 100 个 common states 的 full-event forced-first-action audit；717 个 candidate branches、0 failures、100/100 reference consistency 通过。确认现有 `Q_local_H5` 不是 full-event value proxy。
28. [x] R8-P8c：冻结 full-event Q/value contract，审计 deployable surrogate 与 full-event value 的相关性；100 集/15,364 decisions 的 aligned H5 audit 与 100 状态/774 branches 的 full-event audit 均通过执行门，但未缩小 full-event gap；WAIT/RESERVE 不加入，`T_final` 不解封。
29. [x] R8-P9a：创建 opt-in `PredMPC-3/5`、terminal/no-terminal 消融和唯一实验目录；完成最终 visible-window 信息边界版远端 10 集 smoke，60 rows、0 failed jobs；结果显示未超过 Myopic/PredOpt-H5，不能升级为正式候选。
30. [x] R8-P9b：完成 100 集 Pred/True aligned-H5 逐 decision、逐 candidate score decomposition 和 future-topology audit；候选过滤、priority/tie-break、score 加和均通过，但发现 Pred 的 5-event synthetic future 与 True 的 5-DAG-layer future 单位不一致，暂不进入 300/1,000 集选择验证，不解封 `T_final`。
31. [x] R8-P9c：在不改变动作空间和 GPU 语义的前提下，统一 Pred/True 的 H5 future topology unit 为 DAG layers，并以 train-only conditional empirical multi-node baseline 重跑同一 100 集逐动作审计、拓扑审计和 10 集 smoke；结果为 diagnostic passed，`aligned_predopt_h5_layer` 不提升为正式策略，learned topology predictor 与正式矩阵仍待后续独立确认。
32. [x] R8-P9d：将 topology predictor 重规划到 predictor block；以 `P_dev`/`P_holdout_diag` 的同一视频边界构建 topology labels，完成标签覆盖、因果输入、模型比较和冻结后调度集成前验收；不得使用 `S_train/S_val` 拟合或选择拓扑模型。标签、empirical、tabular、shared causal GRU 和 all-future-node content `N0/A/B/D` 训练/诊断已完成；A 按 validation structure score 选中。冻结后校准、`S_train/S_val` 只推理集成和 future-cost/resource calibration 仍待完成。
33. [x] R8-P9e：在同一 P9d 数据边界上完成 reduced future-node decoder 消融；未来每个节点只预测 `role + action_family`，同时保留 `next_role + next_family_if_execute` 辅助行为头，并比较 `N0/A/B/D`。`B` 按 validation structure score 选中；不得覆盖 P9d 四字段实验，不使用 `S_train/S_val` 拟合或选择，不解封 `T_final`。
34. [x] R8-P9f：只加载冻结 P9e `B` checkpoint，完成 train-only probability calibration、P_dev/test/holdout 分层误差审计和 role/family future-runtime/resource 可辨识性诊断。结构误差相对 empirical baseline 改善，但 calibration 不提升、runtime coverage 不完整且 reduced node contract 无法辨识 load/memory；因此标记 `completed_diagnostic`，关闭 scheduler integration，不读取 `S_train/S_val/T_final`。

## R8-P9d Predictor-aligned Topology Predictor

### Data boundary

- 正式 predictor 数据沿用 `data/manifests/video_split_registry_r7_v1.json`：`P_dev=300` 个视频（内部 `train=240`、`validation=30`、`test=30`）和 `P_holdout_diag=40` 个视频。
- topology、behavior、resource 三个预测器共享视频 ID、视频级 split 和可见输入边界，但各自拥有独立目标：行为是 next role/family，资源是 runtime/load/peak，拓扑是未来 DAG layers/width/prototype。
- `S_train=120`、`S_val=40`、`T_final=80`、`T_backup=20` 仍是调度器专用视频；不得用于 topology predictor 的拟合、超参选择或阈值校准。冻结后可以在这些视频上只做推理和调度集成评估。
- 现有 `EXP-20260901_topology_predictor_baseline` 使用 R7 `S_train/S_val`，因此降级为 scheduler-side diagnostic；不修改、不覆盖其历史产物。

### Execution gates

1. **Label availability gate**：先逐视频核对 `P_dev/P_holdout_diag` 是否有完整 trace/template 和可重建的 `predecessor_node_ids`、父子边或等价事件顺序；若只有 role/tool/resource 行而没有 DAG 边，停在数据准备阶段，不用 `S_*` 补标签，也不直接读取未来模板后缀生成“预测输入”。
2. **Derived dataset gate**：在 `data/processed/` 建立独立 topology dataset，在 `data/manifests/` 登记源、视频 split、run/template coverage 和版本；原始视频、原始 trace 和现有 R7 artifact 只读。features 与 future labels 分开保存，按 `video_id + run_id + prefix/current_node` 对齐。
3. **Causal contract gate**：模型输入只包含当前可见 history、current node/task、stack/baseline、冻结行为预测输出和 train-only 资源预测输出；禁止 future events、future state、remaining steps、successor IDs/edges、执行 runtime/load/memory/status、resource truth 和 `video_id` 作为模型特征。P_dev 上的上游预测特征优先使用 OOF/交叉拟合结果，避免把上游模型的训练内预测当作泛化特征。
4. **Baseline gate**：先用同一 `P_dev/train` 拟合 conditional empirical multi-node layer baseline，在 `P_dev/validation` 选择 schema/超参，在 `P_dev/test` 做诊断；`P_holdout_diag` 全程冻结，留到最终验收。输出 H=5、identity-free、多节点 layer、top-3 scenario probabilities。
5. **Model comparison gate**：按复杂度递增比较三类模型：
   - conditional/statistical baseline：作为最低可复现基线；
   - tabular ML：预测 layer count、width 和 prototype 分布，优先复用远端已有 LightGBM/XGBoost；
   - B05 风格序列网络：复用 history encoder 思路，但新增 topology heads/beam decoder，不能把 B05 的 role/family head 冒充 topology head。
   只有在当前 prefix graph 确实属于可见输入且标签对齐时，才评估 GNN；否则不引入 GNN。
6. **Predictive acceptance gate**：在未见的 `P_holdout_diag` 上比较 layer-count MAE/bias、node-count/width MAE、layer non-empty accuracy、top-3 coverage、NLL/calibration 和按 model/role/stack 分层误差。学习模型必须相对 empirical baseline 在主要指标上有明确改善，不能只靠 scheduler smoke 选模型。
7. **Scheduler integration gate**：拓扑模型冻结后，才把预测结果推理到 `S_train/S_val`，接入现有 `future_h5_layers`/aligned H5 入口；只用执行 truth 做 audit，不反馈训练。先做小规模 contract/smoke，再决定是否扩大调度评估；不解封 `T_final`。
8. **Reproducibility gate**：正式训练前创建唯一 `EXP-xxx`，记录 split hash、label version、feature contract、seed、远端环境、命令和 checkpoint hash；远端只同步必要的小型派生数据/代码/配置，先做磁盘预算，不复制原始视频。

### Current recovery point

`P_dev/P_holdout_diag` 的 topology label coverage、provenance、derived-dataset 和 causal-input gates 已通过。远端已生成独立的
`/root/autodl-tmp/scheduler/results/processed/topology_predictor_p9d_v1`；features 与 labels 按
`video_id + run_id + current_node_id` 分 split 保存，`S_train/S_val/T_final/T_backup` 与输出视频零交集。
本轮已完成两个唯一正式实验：`EXP-20260902_p9d_topology_empirical_baseline` 和 `EXP-20260902_p9d_topology_tabular`。两者均完成 train-only fit、validation selection、test diagnostic 和冻结后 holdout 验收；empirical 选择 `current_full`，tabular 选择 `lgbm_small`。tabular 显著改善结构误差，但原型/分布偏移仍未通过完整 topology predictor gate；下一步才考虑 shared causal GRU。`T_final` 继续封存。
本轮随后完成第三个唯一正式实验 `EXP-20260902_p9d_shared_causal_gru`：4 个预注册变体×3 seeds 均完成 train-only fit、validation epoch selection、test diagnostic 和冻结后 holdout；按 validation 结构分数选择 `topology_only`，但 `shared_multitask` 在 holdout 的结构-行为平衡更好，首层 consistency 未显示清晰增益。该结果闭合 learned topology 的 diagnostic stage，但 predictive acceptance 尚需 calibration/分层误差审计，scheduler integration gate 尚未执行；`T_final` 继续封存。

本轮已完成第四个唯一正式实验 `EXP-20260903_p9d_future_content_multitask`：`N0/A/B/D × 3 seeds` 共 12 个 run 全部完成。A 按 validation `layer_count_mae + full-H width_vector_mae` 选中；D 的 frozen holdout structure 最好，但不得事后改选。B/D 的 validation next-family F1 均低于 N0，辅助行为头的非劣性尚未通过；future content 在 holdout 存在明显分布落差。实验通过数据/泄漏/输出契约诊断，但尚未冻结可部署候选或接入 scheduler；`T_final` 继续封存。

本轮已完成第五个唯一正式实验 `EXP-20260903_p9d_future_role_family_multitask`：在同一 P9d 数据边界上将未来节点内容收缩为 `role + action_family`，并保留 `next_role + next_family_if_execute` 辅助头。`N0/A/B/D × 3 seeds` 全部完成，B 按 validation structure score 选中（`0.12847`）；B 的 holdout structure score/exact=`0.27986/0.82923`，未来节点 role/action-family F1=`0.90151/0.40360`。两字段 macro-F1 不与四字段总分直接比较；共同 `action_family` 在 holdout 为 `0.40360`，未显示单独提升。该实验通过数据/泄漏/输出契约诊断，但尚未冻结可部署候选或接入 scheduler；`T_final` 继续封存。

本轮已完成第六个唯一正式实验 `EXP-20260903_p9f_predictor_acceptance_audit`：只加载 P9e `B`/seeds=`11/22/33`，在 `P_dev/train` 拟合诊断 temperature，在 validation/test/holdout 生成 calibration 与分层误差，另用 train-only role/family P50 profile 评估 future-runtime proxy。holdout layer/width/node MAE=`0.1442/0.1357/0.3807`，exact=`0.8292`，均优于 empirical baseline；但四个 head 的 holdout NLL 均不因 temperature 改善，runtime coverage 仅 `0.8449/0.8435/0.9152`，且 top-1 runtime MAE 均值=`118,798.7 ms`、存在系统性低估。由于 reduced future-node 输出不含 `model_id/node_type/input_scale/cold_warm`，load/memory 不可辨识，P9f 只作为 `completed_diagnostic`，不进入 scheduler integration；`S_train/S_val/T_final` 继续封存。

## R7 Video Grouping

同一视频的所有trace、预测结果和派生workload必须留在同一组。当前边界清单见 `data/manifests/video_split_registry_r7_v1.json`；准确视频ID以已冻结的远端 predictor split 文件为准，不在本地缺少该文件时猜测ID。

| 组 | 视频数 | 用途 | 信息边界 |
|---|---:|---|---|
| `P_dev` | 300（240/30/30） | 行为/资源预测器开发、校准 | train拟合，validation选型，旧test只做诊断 |
| `P_holdout_diag` | 40 | 预测器冻结后的未见诊断 | 不参与拟合、调参或scheduler训练 |
| `S_train` | 120 | 调度器训练、workload生成 | 只读取冻结预测器输出和engine truth |
| `S_val` | 40 | 选择压力档位、H*和RL/优化器超参 | 不读取最终测试结果 |
| `T_final` | 80 | 最终系统测试 | 在所有模型、策略和参数冻结后一次性运行 |
| `T_backup` | 20 | 预注册替补 | 仅在T_final发生可证明的外部失败时启用 |

当前600视频中，`S_train/S_val/T_final/T_backup` 合计260个视频；旧计划中的`U_reservoir=281`已废弃，不再作为当前事实。

## R8 Optimizer/RL Upgrade Boundary

- 现有行为预测器、资源预测器、future artifacts、640 templates、20,000 train episodes 和 1,000 validation episodes 只读复用；本阶段不下载视频、不重采 trace。
- `S_train` 只用于目标权重/solver 参数校准、专家数据和 BC/PPO 训练；`S_val` 只用于一次性选择 H、solver budget 和 RL 配置。`T_final=80`、`T_backup=20` 保持封存。
- 所有 deployable 方法共用 `DeployableObservation`、冻结 predictor cache、episode seed 和 workload；真实执行 truth 只能由 engine provider 使用。
- CP-SAT V1 的实际实现边界是当前-ready候选的 assignment、GPU NoOverlap，以及预测 runtime/load/memory/future 的软目标；ready-pool 本身沿用事件引擎的 DAG 可达性，但 V1 没有把未来节点 precedence、显存硬约束或 cache residency 生命周期显式放进优化模型。完整约束、GNN、H>5 暂缓。
- 若 CP-SAT 在 time limit 内没有可行解，只能使用预注册的 PredOpt-v2 首动作并记录 `solver_fallback=1`；不得静默替换或丢弃样本。
- T_final 前不把 Oracle/TrueOpt 用作专家训练信息；它们只保留为 privileged upper-bound reference。
- PredMPC-P9a 是 opt-in diagnostic branch：当前实现只滚动当前 ready-node candidate window，future artifacts 只提供 continuation cost，不暴露真实 template suffix 或未来 arrival；smoke 未超过 PredOpt-H5，因此 P9b 仍待 action-score/terminal audit，不直接扩大样本。

### 分组原则

1. `P_dev`和`P_holdout_diag`沿用最新`behavior_nn_v1_r2`视频级划分；资源预测器必须对齐到同一300-video development pool。
2. `S_*`和`T_*`只能从300+40之外的260个视频中选，不能把预测器见过的视频重新包装成调度测试。
3. 采集后报告path length、branch/retry、action-family、runtime/VRAM和uncertainty；这些是诊断字段，不能反向改组。
4. `T_backup`只在预注册的文件损坏、采集失败或不可恢复运行失败时替换，并记录替换原因；不能因结果不好而替换。

### R7 执行顺序与最小比较矩阵

1. 修复21个截断视频ID并通过分组不交叉、600计数和来源完整性gate。
2. 用300个`P_dev`对齐资源预测器；只在`S_train/S_val`采集调度trace和workload，80个`T_final`视频保持封存。
3. 统一事件引擎只暴露当前ready节点、资源估计和未来provider；执行时长、显存和失败状态只来自Truth。
4. 优化器阶段运行：`Myopic`、`Optimizer-0`、`PredOpt-H1/H3/H5`、`TrueOpt-H1/H3/H5`。若`TrueOpt-H`不优于`Optimizer-0`，停止扩展并先修正workload/目标。
5. RL阶段只有在优化器门禁通过后运行：`RL-0`与`RL-H5`，各补3个seed；validation阶段比较并登记候选，最终只冻结一个`H*`和一个RL配置。
6. 配置冻结后，在`T_final`只运行Myopic、Optimizer-0、PredOpt-H*、TrueOpt-H*、RL-0、RL-H*；RR只作附录控制，R6只作legacy compatibility replay。

## Acceptance Criteria

- [x] `.project/` 六类 canonical 文件和 `AGENTS.md` 存在。
- [x] 本地和远端只有 `src/tracing` 作为当前 tracing 源码路径。
- [x] Python 包仍以 `tracing` 导入，命令约定为 `PYTHONPATH=src`。
- [x] 768 条旧轨迹有独立、不可覆盖的 v1 登记；扩展 collection 有独立 ID。
- [x] 视频、模型、原始 trace 不进入 Git。
- [x] R5 容量预算和 600 视频 source/archive gate 已完成；C1/C2 实验 split 在 R6 固化，不与历史混杂 split 直接复用。
- [x] R5 provenance gate：v1/v2 每个目标视频具备原始 SHA-256、字节数、ffprobe 尺寸/时长、官方 archive 定位和 retention 状态；派生视频具备 `parent_sha256`。旧 `P_dev` 300 个视频已按 2026-08-17 用户明确授权的范围例外删除并标记 `deleted_after_gate`；其余 300 个 holdout/scheduler 视频仍为 `deletion_eligible=false`，未完成归档恢复测试前不删除。
- [x] R6 causal-v2 gate：C1 48/8/8 与 C2 pending/backup registry 固化；prefix 输入无 future/answer/current-truth 泄漏；H1/H3/H5 概率归一；资源预测只用 C1 train；workload job 不含整条任务预测 runtime 或 source video refs；所有派生输出写入独立 versioned 目录。
- [x] 最新行为预测器数据口径已冻结：`behavior_nn_v1_r2` 使用 300 个 development 视频，另设 40 个完全未见 final holdout；holdout 不参与拟合、特征/阈值选择或调度策略调参。
- [x] R7-0 数据边界 gate：600视频分组注册表完整，300/40 predictor split与260个scheduler视频无交集，21个截断ID已修复并有映射证据。
- [x] R7-1 predictor gate：行为和资源预测器共享P_dev，资源输出完整性、冻结接口、Truth隔离和10/50 contract smoke均通过。
- [x] R7-2 pilot gate：8 条 predictor-unseen trace 全部成功且通过 validator；2 条可恢复解析错误保留并有成功 retry，source manifest/planner input 无 answer/gold 泄漏。
- [x] R7-2 trace expansion gate：640 条正式 trace 全部落盘，`run_status.json`/`trace.jsonl`/`run_manifest.json` 各 640 个，摘要 640/640 success、0 failed；`star` 与 `langgraph_react` 各 320 条。
- [x] R7-2 workload gate：S_train/S_val的到达、拥挤、显存压力和等待/OOM标签达到预注册阈值；640 templates、21,000 episodes、135 scenario cells、777,840 jobs，0 template/episode errors、0 capacity violations。
- [x] R7-3 optimizer/future matrix gate：1,000 episodes × 10 policies = 10,000 rows，0 failed jobs、0 capacity violations；PredOpt-H1/H3/H5和TrueOpt-H1/H3/H5均完成。
- [x] R7-4 RL gate：`rl_0`/`rl_h5` 各 3 个 seed 共 6 个 run，均为 1,000 train + 1,000 validation、1,000 validation rows、0 failed jobs、1,001 heartbeat lines、stderr 为空；runner 与 aggregate report 均已哈希核验。
- [x] R7-5 paired-statistics gate：1,000 validation episodes、135 scenario cells、10,000 fixed-seed stratified bootstrap；H1/H3/H5 的 completion/deadline 比较完成 Holm 校正，RL 每个 seed 的 paired comparison 已保留；episode-level summary 不冒充 global job-level P95。
- [ ] R7-3 final gate：T_final只在冻结配置下运行，主结果按视频/配对episode统计，R6兼容性结果单独标记。

## Blockers

- 本地没有 `pytest`；需要远端 finetooling 环境或补齐测试依赖后才能完成完整测试套件。
- 远端磁盘空间已因清理旧 `P_dev` 释放约 28.44 GB；R5 视频扩展已停止，后续派生数据和 workload 仍必须先做空间预算。
- C2 尚未生成新 Agent traces；C2/电梯数据继续保持独立 collection，不得混入 300/40 VideoMME 主线；若启动 C2，仍需先冻结 predictor/resource，再按 `video_split_registry_r6_v1_causal_v2.jsonl` 采集 32 个主候选。
- R7-0 的300/40准确ID文件、21条修复证据和600条边界注册表已在本地/远端核对一致；该 gate 已通过，后续不再以旧的DNS/ID缺失状态阻塞主线。
- R7-2/R7-3 已把冻结的行为/资源输出、三类 provider 和压力 workload 接入唯一事件循环；R6 artifacts 仍只是可审计兼容性输入，不与新主线混合。
- RL-0/RL-H5 六个远端 seed 已完成验收，但 RL-H5 相对 RL-0 的 seed 方向不一致；在决定重跑 RL 或明确排除 RL、并冻结最终策略前，不启动 T_final。
- 远端磁盘约剩8–9G；正式workload和模型训练必须先做空间预算，不复制原始视频和大模型。
- R8-P0 已证明三组简单加权评分未改善完成时间；`finetooling` 已安装并核验 `ortools==9.9.3963`，CP-RHO P1 仍需通过最小 workload smoke 和 S_val 资源/延迟门禁。
