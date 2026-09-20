# EXP-20260920 · resource-v2 六臂调度 smoke（v03，300 配对集）

**状态：已执行完毕，结果与预注册假设相反；等待独立审查。**

- 运行日期：2026-09-20
- 执行机：Windows，`D:\anaconda\envs\scheduler\python.exe`，`PYTHONPATH=src;scripts`
- 运行耗时：6 臂 × 300 集 = **586 秒**
- 预注册：`.project/EXPERIMENT_GATE.json` → `experiments.EXP-20260919_j_series_resource_dist_v1.smoke_preregistration_20260920`
  （`status = FROZEN_BEFORE_RESULTS`，含 6 臂 immutable identity、contrasts、抽样规则、统计设计、9 条 preflight、5 个阻塞 P0）
- 审查链：`docs/research/2026-09-20_fas_resource_v2_packer_review_gpt.md`（提交 `78f000a` 的 P0 清单）

---

## 1. 问题

资源头 v2（R1b：16 格 mass-balanced 离散分布）在原 J3 验收协议下 **pinball −21.0%**、
校准同时改善（cal err 0.0301 → 0.0245），
而 R3a-U（额外解冻 `res_hidden`）pinball −23.6% 但**违反原 calibration integrity gate**（cov90 0.803）。

本实验回答：**这些预测侧改进是否转化为真实调度收益？** 以及**消费泛函是否是约束**？

## 2. 臂与不可变身份

| 臂 | 预测 artifact | 消费策略 ID | 模拟器 policy |
|---|---|---|---|
| A0 | `j3_seed11@<base sha>` | `legacy_sum_q95_v1` | `sameshape_h5_p95` |
| A1 | `resource_v2_r1b_seed11@7d45f2f7…` | `legacy_sum_q95_v1` | `sameshape_h5_p95` |
| A2 | `resource_v2_r3a_u_seed11@3a63af41…` | `legacy_sum_q95_v1` | `sameshape_h5_p95` |
| B1 | 同 A1 | `sum_conditional_runtime_mean_plus_legacy_load_v1` | `sameshape_h5_condmean` |
| B2 | 同 A1 | `sum_marginal_step_cvar95_plus_legacy_load_v1` | `sameshape_h5_stepcvar95` |
| O | `oracle_truth_v03` | `samekey_true_future_h5_v1` | `sameshape_h5_truth` |

- **A2 标 `eligible_for_model_selection = false` / `role = diagnostic_only`**：它违反冻结的 Phase-R
  校准完整性门（cov90 0.803），即使调度得分最好也不得成为 winner。
- **O 臂命名**：joint-future-truth headroom，**不是** resource-head oracle
  （它同时消除 runtime 误差与 topology/length 误差）。

## 3. 抽样（预注册，已冻结）

- 135 个 cell 各取 2（=270）+ 30 个 cell 各取第 3 个；
- extras 经构造满足：arrival 各 +10、load 各 +6、GPU topology 各 +10、state/deadline 各 +10；
- 最终边际**精确命中**：arrival 100/100/100、load 60×5、GPU 100/100/100、state 100/100/100；
- 清单 `resource_v2_smoke_v1_episode_ids.txt`，SHA256 `c98556c8003aabb484b4d4caa41049094c24b2ba01c9d56014407f2ba7d07430`；
- **未**使用 `validation_000000:000299` 顺序切片。

## 4. 结果

主指标 = `mean_completion_ms`（越低越好）。

| 臂 | 完成时间 | 相对 A0 |
|---|---|---|
| O（joint-future-truth） | **84246.2** | −1543.0 |
| **A0（J3 + 旧 q95）** | **85789.2** | — |
| B2（R1b + ΣCVaR95） | 86158.1 | +368.9 |
| A1（R1b + 旧 q95） | 86585.8 | +796.6 |
| B1（R1b + ΣE[T]） | 86631.2 | +841.9 |
| A2（R3a-U + 旧 q95） | 86880.6 | +1091.4 |

预注册 contrasts（paired episode bootstrap，B=2000，δ_NI = **485 ms**）：

| contrast | point | CI95 | P(Δ≤0) | 预注册判定 |
|---|---|---|---|---|
| **A1−A0**（预测器效应） | **+796.6** | [471.0, 1104.7] | 0.000 | **显著更差，且超过 +485** |
| A2−A1（解冻附加效应，诊断） | +294.9 | [−18.1, 624.8] | 0.030 | inconclusive |
| B1−A1（均值 vs 旧风险） | +45.4 | [−290.6, 394.3] | 0.385 | non_inferior |
| B2−A1（边际 CVaR95 vs 旧风险） | **−427.7** | [−687.7, −145.2] | 1.000 | non_inferior（**未达 −485 的实质改进**） |
| O−A1（剩余 headroom） | **−2339.6** | [−2856.3, −1855.0] | 1.000 | **material_improvement** |
| A1−O | +2339.6 | [1860.2, 2856.8] | 0.000 | 未饱和（practically_saturated = false） |

## 5. 核心结论

> **更好的预测器在原验收协议下显著更优（pinball −21%、校准改善），但在真实调度器上显著更差
> （A1−A0 = +796.6 ms，CI 排除 0）。预测侧质量提升没有转化为调度收益。**

配套结论：

1. **A2（更多解冻）不比 A1 好**（+294.9，CI 跨 0）→ 与 R3a 的预注册判定一致。
2. **B1（ΣE[T]）与 A1 不可分**（+45.4）→ 风险中性消费没有改变局面。
3. **B2（ΣCVaR95）是唯一改善 A1 的消费方式**（−427.7，CI 排除 0），但仍不如 A0。
4. **相对真值上界仍有 2339.6 ms 的 headroom**（CI 排除 0，未饱和）→ 不是"已经没有空间了"。

## 6. 机制（已量化，`artifacts/mechanism_conservatism.json`）

**调度器实际消费的是保守性，不是分布精度。**

把每个臂**真实喂给调度器**的每节点未来成本分数（`Σ ≤5 步`的消费量）算出来：

| 臂 | 消费的统计量 | 每节点均值 |
|---|---|---|
| B2 | `cvar95_ms` + legacy load | **48020.3** |
| A0 | `p95` + legacy load | 37885.1 |
| A1 | `p95` + legacy load | 36559.9 |
| A2 | `p95` + legacy load | 32702.3 |
| B1 | `runtime_mean_ms` + legacy load | 15078.8 |

- **三个 p95 臂的调度排序 A0 < A1 < A2 被这个分数完全预测**（分数越大越好）；
- Spearman(消费分数, 完成时间) = **−0.800**（5 臂，负号 = 越保守越好）；
- **B2 比 A1 更保守**：CVaR95 / p95 = 中位 **1.323×**，**100% 的节点 > 1** → 与 B2 优于 A1 一致；
- **B1 是严格单调关系的例外**：它比 A1 少保守得多（0.372×）却仍略优于 A1，
  所以"保守性"能完整解释三个 p95 臂的排序与 B2>A1，但**不能**解释全部五臂排序。

**根因**：J3 的 q95 是**直接训练的分位头**；R1b 的 q95 由**离散 CDF 反演取 bin 代表值**
（`quantile_rule = cdf_inversion_bin_representative`）。
同一个消费规则看到**系统性不同**的数字：R1b 的 p95 **比 J3 低 6.2%**
（每节点 paired 中位比 **0.938**，p10 0.775 / p90 1.444，61.5% 的节点 < 1）。

这正是接口审查警告过的陷阱的实证：**换预测器会静默改变被消费的量**，
因此 `A1−A0` **不是**纯粹的"预测质量"对比。

## 7. 复现

```bash
export PYTHONPATH=src:scripts
python scripts/pack_resource_v2_artifacts.py \
  --anchors-file outputs/sstar_predictor_anchors/features_sstar.jsonl.gz \
  --base-pack outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts \
  --arm r1b --output-root outputs/resource_v2_artifacts/r1b
python scripts/resource_v2_preflight.py --arms j3 r1b r3a_u r3a_f
python scripts/resource_v2_scheduler_smoke.py --stage select
python scripts/resource_v2_scheduler_smoke.py --stage run
python scripts/resource_v2_scheduler_smoke.py --stage report
```

打包后的 H5 包各约 11.7 MB，未随仓库发布；用上面的命令可从冻结 checkpoint + 已发布的头文件重建
（每臂约 30 秒）。

## 8. 门禁状态

- 9 条 preflight 全部 PASS（`artifacts/preflight_report.json`、`artifacts/overlay_preflight.json`）：
  节点集一致、每行恰好 1 个 scenario、每个 step 都已升级、无静默回退、
  probs→各视图重推导一致（视图误差 3.64e-12）、非 resource 字段未变、
  artifact/head/bin/base SHA 已冻结、overlay loader 可用、
  `sameshape_h5_p95` 回归测试逐位不变（`test_p95_arm_is_exactly_the_q95_champion_consumer`）。
- 执行期发现并修复的**两个真实缺陷**（由新加的验证器抓出）：
  1. 打包器在 **float32** 上算视图 → CVaR 跨界权重近乎相消导致误差 6.99e-02 → 改全程 float64（现为 0.0）；
  2. r3a checkpoint 不带 bin spec → 改为从**权重本身**推导维度契约并逐项交叉断言。

## 9. 措辞纪律（不得越界）

- **不得**写"R1b 预测更准所以调度更好"——两者都不成立。
- **不得**写"更好的预测器提升了调度"。
- B2 必须写成"各预测未来步边际 CVaR95 的加和风险分数"，
  **不得**写成 `CVaR_0.95(Σ_h T_h)`（我们没有联合分布）。
- O 必须写成 joint-future-truth headroom，**不得**写成 resource-head oracle。
- 本次是 **dev/验证集**上的 300 集 smoke，**未接触 J test split**。
