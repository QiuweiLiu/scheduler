# Forecast-aware scheduling — Phase 2 正式结果（1,000 episodes，2026-09-11）

状态：正式矩阵完成；**已通过独立审计（15/15 Δ+CI 独立复算一致；行数与 manifest sha 闭环）与独立评审（CONDITIONAL → 补丁已并入）**。结论方向明确（未来结构预测显著改善调度；资源直接替换显著有害；跨 provider 归因受制品形态限制）。

## 0. 设计与治理

- 数据：R7 冻结 workload 的 validation episodes 前 1,000 个（`workload_validation_r7.jsonl`）；`T_final` 封存。
- 策略（确定性，无需 seed 变化）：E0 `myopic`、E2 `predopt_h5`（J 拓扑 + 静态表）、E2-legacy `predopt_h5`（B05 旧 provider）、E3a `predopt_h5_jrt`（只换 runtime）、E3b `predopt_h5_jres`（完整资源，决策兼容语义）、E4 `oracle`。
- **运行命令（复现用）**：
  ```sh
  PYTHONPATH=src python scripts/r7_scheduler_matrix.py \
    --templates results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl \
    --episodes results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl \
    --future-artifacts outputs/sstar_predictor_artifacts_sched/prediction_artifacts \
    --output-dir outputs/phase2_r7_j_1000 --limit 1000 \
    --policies myopic,predopt_h5,predopt_h5_jrt,predopt_h5_jres,oracle
  # legacy 对照：--future-artifacts results/processed/r7_scheduling_future_20260817/prediction_artifacts --policies predopt_h5
  ```
- **环境**：Windows 本地；`D:\anaconda\envs\scheduler\python.exe`（Python 3.10.21）；纯 CPU（策略为确定性）；无 GPU/torch 依赖（RL/CP 分支未用）。
- **artifact 链**：J 侧 = `python scripts/pack_j_predictor_artifacts.py --config experiments/EXP-20260911_p9d_j_predictor_acceptance/config.json --anchors-file outputs/sstar_predictor_anchors/features_sstar.jsonl.gz --output-root outputs/sstar_predictor_artifacts` → 经 `.scratch/build_sched_compatible_artifacts.py` 改名/补 manifest 到 `outputs/sstar_predictor_artifacts_sched/prediction_artifacts/`（`b05_artifact_manifest.json` sha256 `0d0c9668…`，与 J 侧 matrix report 的 `future_artifacts_manifest_sha256` 一致）；B05 侧 = `results/processed/r7_scheduling_future_20260817/prediction_artifacts`（manifest sha256 `aa943b60…`）。
- **资源臂语义**：E3b 的加载项仅在 GPU step 的预测发生概率 ≥ **0.5**（`load_threshold`）时计入条件时长；E3a 保持表的加载项。**与规划偏离说明**：规划中的 E1（resource-only）在本 simulator 中无法与"未来步"分离，Phase 2 以 E3a/E3b 分解替代（h1 代理已见于 Phase 1）。
- 统计：episode 配对 + 2,000 次 bootstrap（episode 为单位，n=1,000）。
- 资源适配层为新增策略（`predopt_h*_jrt/_jres`），既有策略未改动；所有运行 0 failed jobs、状态 passed。
- 成本：J 侧 5 臂 × 1,000 集 ≈ 50 分钟；B05 对照 1 臂 ≈ 9.5 分钟（本地 CPU）。

## 1. 聚合结果

| 臂 | mean completion | p95 | queue | miss | evictions |
|---|---|---|---|---|---|
| E0 myopic（无未来） | 215,116 | 465,788 | 123,365 | 10.98% | 41.06 |
| **E2 predopt_h5（J 拓扑）** | **198,052** | **414,235** | **107,790** | **8.64%** | 45.08 |
| E2-legacy（B05 拓扑） | 210,635 | 455,764 | 120,370 | 10.47% | 39.91 |
| E3a runtime 替换 | 214,064 | 462,016 | 123,046 | 10.92% | 42.29 |
| E3b 完整资源 | 214,049 | 462,506 | 123,020 | 10.88% | 42.25 |
| E4 oracle（上界） | 172,733 | 311,027 | 78,938 | 6.08% | 39.81 |

## 2. 配对统计（A − B，95% CI，2,000 bootstrap）

| 对比 | completion Δ | queue Δ | miss Δ |
|---|---|---|---|
| **E2 − E0（未来结构 vs 无未来）** | **−17,065 [−18,692, −15,458]** | −15,575 [−17,238, −13,969] | **−2.34pp [−2.62, −2.06]** |
| **E2 − E2-legacy（新 vs 旧 provider）** | **−12,583 [−13,962, −11,323]** | −12,580 [−14,000, −11,258] | **−1.83pp [−2.06, −1.60]** |
| E3a − E2（runtime 替换） | **+16,012 [+14,480, +17,611]** | +15,256 [+13,707, +16,878] | **+2.28pp [+2.01, +2.55]** |
| E3b − E3a（完整资源增量） | −15 [−161, +125] | −26 [−170, +115] | −0.03pp [−0.08, +0.01] |
| E4 − E2（距上界） | −25,319 [−27,372, −23,291] | −28,852 [−30,942, −26,778] | −2.56pp [−2.84, −2.27] |

注（多重比较）：5 组对比 × 3 指标未做事前校正；主效应在 Bonferroni（99% CI）下仍显著（最小 |下界| 11,323ms 远大于区间半宽），唯一含 0 的 E3b−E3a 本就判 null。

## 2b. J vs B05 artifact 形态对照（关键混淆披露）

| 维度（覆盖 template 节点 8,935） | J（sstar） | B05（legacy） |
|---|---|---|
| 场景数 | **1**（prob=1.0） | 3（首场景概率质量均值 0.606） |
| 每节点步数分布 | {0:1272, 1:656, 2:582, 3:581, 4:621, 5:5223} | {5:8935} |
| **空步（0 步）比例** | **14.2%**（触发 0 成本回退） | 0% |

→ E2 vs E2-legacy 的差异中，**除预测质量外还混入了制品形态**（单确定链 + 14% 零长度截断 vs 固定 3×5 场景展宽）。因此该对比应表述为"**新 provider 制品级更优**"；要把收益完全归因到"预测质量"需要形态匹配对照（见 §6 后续工作）。J 内部（E2/E3a/E3b）形态一致，不受此混淆影响。

## 3. 结论（本实验可支撑的表述）

1. **未来行为（结构）预测显著改善调度**：相对无未来 −17.1s（−7.9%），相对旧 provider −12.6s（−6.0%，制品级对比，见 §2b）；三项指标（完成时间、排队、违约）方向一致且 CI 不含 0。→ 对研究问题的回答：**是，显著有影响**。
   - 注：E2 的 evictions 高于 E0（45.08 vs 41.06），收益并非全面占优——调度器以更多换出换取整体完成时间；该现象未单独归因（后续工作）。
2. **资源预测的直接替换显著有害**（+16.0s，+8.1% 相对 E2），且加载块增量为零（E3b≈E3a）。原因（此前已量化）：静态表是同域（同 workload、含 position 分层）的强基线；J 预测在 S_* 上 OOD（role 0.912、runtime pinball 1045ms vs 域内 845ms）；且 current(表)/future(预测) 尺度混用（0.34×）。**不能就此否定资源预测的潜在价值**，但"直接替换"路线在本设定下被否定。
3. **上界空间仍大**：oracle 比 E2 好 25.3s（E2 相对 oracle 高 14.7%）→ 未来信息利用远未饱和。

## 4. 边界与限制

- 仅 validation episodes（1,000/1,000 全量）；`T_final` 未动；holdout 未动。
- 策略确定性；统计单位为 episode（bootstrap 配对）；episode 复用有限模板池，未做 cluster 调整（主效应量级下不影响方向）。
- **预测器种子边界**：Phase 2 只用了主候选 `J3:seed11`；seed22/33 未进调度矩阵，不得把结论泛化为"预测器族"的性能。
- 资源替换结论依赖于本 simulator 的成本语义（identity lookup + 非 cache-aware 的 H5 路径）与 S_* OOD 域；换域/换成本模型可能不同。
- 未在 S_* 上做任何拟合/选择（标定仅限语义修正，不含数据拟合）。
- 拓扑臂的差异仅来自 artifacts（node_id 覆盖率 8,935/8,935；resource 块对 E0/E2/legacy 不生效）；但 J/B05 的制品形态不同（§2b），跨 provider 归因受限。

## 5. 产物

- 运行：`outputs/phase2_r7_j_1000/`（5,000 行）、`outputs/phase2_r7_b05_1000/`（1,000 行）；聚合与 manifest sha 见各自 `scheduler_matrix_report.json`。
- 统计脚本：`.scratch/phase2_stats.py`（seed 20260911，B=2,000；已知其 `paired()` 首参未用的 latent bug，当前所有 A 侧均为 J，数值不受影响）。
- 形态对照与哈希核验：`.scratch/patch_verifications.py`。
- Phase 1/1b pilot：`outputs/phase1_pilot_100/`、`outputs/phase1b_pilot_100/`。
- 适配层：`predopt_h1/h3/h5_jrt`、`predopt_h1/h3/h5_jres`（新增）。

## 6. 后续工作（评审建议）

1. **形态匹配对照**（把 E2-vs-legacy 从"制品级"提升到"质量归因"）：例如将 B05 展宽聚合为单链，或将 J 零长度节点填充/裁剪到与 B05 同形态，重跑同一 1,000 集。
2. **预测器种子稳健性**：J3 seed22/33 进调度矩阵（成本同 Phase 2）。
3. 资源臂若要翻案：需要域匹配的资源模型（当前 OOD）或按目标函数定制的消费方式（cache-aware / uncertainty-aware）。
4. `T_final` 维持封存；若推进正式结论，先冻结形态匹配对照与种子策略。
