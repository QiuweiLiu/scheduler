# N1 验证结果与当前状态（2026-08-11）

## 结论边界

文档前半部分记录的是固定 validation split 上的统计 screening；第 5 节补充了之后锁定执行的 final holdout。两者都不等价于生产环境一般化优越性证明。

- 主候选：B05 Masked GRU。
- 统计单位：validation video（30 个），每个模型先做 seed-mean，再按 video 配对 bootstrap。
- joint 分母：1,861 个可评分 key；168 个 execute 行没有 family，按既定契约排除。
- bootstrap：5,000 次，seed 42，percentile 95% CI，superiority margin 0.005。
- test targets：未读取、未用于选择、未预测。

## 1. OOF 完整性修复

远端修改：

- scripts_behavior_r2_v5/r2_v5/r2b/oof_cache.py
- scripts_behavior_r2_v5/r2_v5/r2b/models_meta.py
- scripts_behavior_r2_v5/r2_v5/r2b/models_residual.py

新增 fail-closed 检查：

1. manifest 自身 canonical SHA256；
2. 当前 role/tool 数据、split manifest、feature contract 的来源哈希；
3. role/family train/validation 数量；
4. 5 折覆盖、概率有限性和行和为 1；
5. train 主键集合精确匹配 (run_id, target_source_event_id)。

当前缓存通过严格正测：

- role train 13,754；
- family train 4,712；
- manifest SHA256：
  83b0268c2875d34da279114712d93018e3884ba45a2a2bdf1e5204e339676d86。

篡改 manifest 的负测被拒绝。远端 R2 全套 154 个测试通过。

说明：严格 loader 是在旧 B08/B09/B10 formal run 之后加入的；它验证了当前 OOF 缓存，但旧 run 的 environment source hash 仍代表运行时的旧代码版本。最终重跑前应使用版本化的新代码重新生成相关 formal artifact。

## 2. Paired video bootstrap

产物：

- results/processed/behavior_nn_v1_r2/comparisons/n1_paired_bootstrap.json
- results/processed/behavior_nn_v1_r2/comparisons/n1_paired_bootstrap.md
- results/processed/behavior_nn_v1_r2/comparisons/paired_bootstrap.json

| 比较 | mean diff | 95% CI | Holm tail p | 当前判断 |
|---|---:|---:|---:|---|
| B05 vs B01 | +2.92pp | [+1.86,+4.11]pp | 0.0016 | screening support |
| B05 vs B03 | +2.17pp | [+1.06,+3.37]pp | 0.0080 | screening support |
| B06 vs B01 | +2.86pp | [+1.74,+4.08]pp | 0.0016 | screening support |
| B06 vs B03 | +2.11pp | [+0.79,+3.49]pp | 0.0160 | screening support |
| B07 vs B01 | +2.89pp | [+1.68,+4.26]pp | 0.0016 | screening support |
| B07 vs B03 | +2.14pp | [+0.90,+3.47]pp | 0.0102 | screening support |
| B10 vs B01 | +2.33pp | [+1.28,+3.54]pp | 0.0020 | screening support |
| B10 vs B03 | +1.58pp | [+0.04,+3.18]pp | 0.0834 | not confirmed |

主比较 B05 vs B01/B03 的 CI 下界都高于 0.005；这支持进入下一步，但不打开 final holdout，也不使用“最终显著优于”措辞。

## 3. B05 zero/shuffle 消融

产物：

- results/processed/behavior_nn_v1_r2/comparisons/n1_sequence_ablation.json
- results/processed/behavior_nn_v1_r2/comparisons/n1_sequence_ablation.md
- results/processed/behavior_nn_v1_r2/comparisons/n1_ablation/runs/

每个消融均为 B05、seed 11/22/33、train-only preprocessing、validation early-stop。

| 消融 | full - ablation | 95% CI | 解释 |
|---|---:|---:|---|
| zero history | +2.14pp | [+1.01,+3.31]pp | 清空历史后稳定下降 |
| shuffle history | +0.61pp | [+0.08,+1.24]pp | 保留历史但打乱顺序仍有小幅稳定下降 |

因此当前证据支持：

- B05 的提升不只是静态特征或随机 seed；
- 历史内容有增量作用；
- 历史顺序本身也有增量作用，但效应小于“是否有历史”。

这仍不是因果证明，也不能外推到未见视频或 Phase 4 workload。

## 3.5 B10 zero/shuffle 消融

产物：

- results/processed/behavior_nn_v1_r2/comparisons/n1_b10_sequence_ablation.json
- results/processed/behavior_nn_v1_r2/comparisons/n1_b10_sequence_ablation.md
- results/processed/behavior_nn_v1_r2/comparisons/n1_ablation_b10/runs/

每个消融均为 B10、seed 11/22/33，严格使用当前 OOF cache 和 train-only preprocessing。

| 消融 | full - ablation | 95% CI | 解释 |
|---|---:|---:|---|
| zero history | +2.15pp | [+1.00,+3.41]pp | 清空历史后稳定下降 |
| shuffle history | +0.60pp | [-0.19,+1.48]pp | 顺序破坏的影响方向为正，但当前区间跨 0 |

因此 B10 的历史存在性贡献得到 screening 支持；仅凭当前验证集，不能确认 B10 对历史顺序的独立贡献。

## 4. Finalist CV（development-only）

产物：

- results/processed/behavior_nn_v1_r2/comparisons/finalist_cv/finalist_cv.json
- results/processed/behavior_nn_v1_r2/comparisons/finalist_cv/finalist_cv.md
- results/processed/behavior_nn_v1_r2/comparisons/finalist_cv/fold_manifest.json
- results/processed/behavior_nn_v1_r2/comparisons/finalist_cv/failures.json

本轮只使用 candidate 的 train+validation：role 15,783 行、family 5,327 行；test role 1,520 行、family 562 行被排除。按从 task_id 解析出的 canonical video id 做固定 5-fold，development pool 共 270 个视频；fold manifest SHA256 为 ae478f8e08f3e5cbaa189933d0292bce41e7bc4096c34839fcddf908927c344a。

总计 55 个独立运行：B01 5 个确定性运行；B03 15 个 primary（seed 11/22/33）+10 个 smoke（seed 44/55）；B05 同样为 15+10。每个运行都保存 config、环境、数据清单、预测和 fold metadata，并在进入汇总前由 evaluator 独立复算。

| 模型 | primary n | primary joint mean±std | sensitivity n | sensitivity joint mean±std |
|---|---:|---:|---:|---:|
| B01 | 5 | 0.8355 ± 0.0059 | 5 | 0.8355 ± 0.0059 |
| B03 | 15 | 0.8395 ± 0.0060 | 25 | 0.8389 ± 0.0061 |
| B05 | 15 | 0.8754 ± 0.0060 | 25 | 0.8750 ± 0.0064 |

primary seed-mean 的同折差异如下；B03 comparator 为 seed 11/22/33 的折内均值：

| fold | B05 | B01 | B05−B01 | B03 | B05−B03 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8658 | 0.8334 | +3.24pp | 0.8349 | +3.09pp |
| 1 | 0.8723 | 0.8401 | +3.22pp | 0.8448 | +2.75pp |
| 2 | 0.8822 | 0.8424 | +3.98pp | 0.8457 | +3.65pp |
| 3 | 0.8779 | 0.8339 | +4.39pp | 0.8409 | +3.70pp |
| 4 | 0.8788 | 0.8274 | +5.14pp | 0.8312 | +4.76pp |

完整性验收：status=completed、failures=[]、55 个 job key 唯一；55 份 metadata 使用同一 fold manifest SHA；所有 predictions 的 test 行数为 0，test_used=false。这些结果只支持把 B05 作为 finalist screening 候选，不能替代 final holdout，也不能宣称生产环境一般化优越。

## 5. Final holdout（2026-08-11，已完成）

正式 holdout 使用下载后锁定的 40 个未见 VideoMME 视频，并对每个视频运行 3 个 baseline：st_fixed、star、langgraph_react，共 120 条轨迹；本轮轨迹采集全部成功（120/120，失败 0）。

产物：

- results/raw/final_holdout_v1/phase2_batch_summary.jsonl
- results/processed/behavior_nn_v1_r2/final_holdout/enrichment/enrichment_summary_v0_1.json
- results/processed/behavior_nn_v1_r2/final_holdout/dataset/dataset_report.json
- results/processed/behavior_nn_v1_r2/final_holdout/evaluation/final_holdout_report.json

数据验收：

- development：role 15,783 行、family/tool 5,327 行；
- final holdout：role 1,380 行、family/tool 526 行；
- 5,853/5,853 条 role→family join 完整；split、task、current-role、cutoff 关系均通过；
- test_targets_used=false，训练只使用 development_train_validation_relabelled_train，评估只使用 final_holdout_v1_relabelled_validation；
- holdout 选择、下载和文件校验均在训练/评估前完成，未根据 holdout 调参。

锁定 finalist CV 主运行的 epoch 后，单次 final holdout 结果如下。主指标是联合 role+family pipeline accuracy：

| 模型 | 运行数 | fixed epoch | joint accuracy mean±std | family top-1 mean | role top-1 mean |
|---|---:|---:|---:|---:|---:|
| B01 | 1 | — | 0.7406 | 0.5494 | 0.9123 |
| B03 | 3 | 18 | 0.7377±0.0026 | 0.5615 | 0.9046 |
| B05 | 3 | 19 | **0.8046±0.0048** | **0.6521** | **0.9348** |

B05 相对 B01 的 holdout 平均提升为 +6.40pp，相对 B03 为 +6.69pp。这是当前锁定 holdout 上的结果，不外推到其它 agent 或生产 workload；报告 failures=[]，7 个运行全部成功。final report SHA256：38ba1bea8eb1fc4e22a8eaa7f21c54a60da62b74a7ba7867a334e67cf2ab92f6。

## 6. 尚未完成

1. 资源预测器的训练、校准和独立 holdout；
2. 并发 workload、显存挤占、等待/OOM 重放；
3. 将行为预测输出接入调度器并完成系统级验收；
4. 真实电梯 graph/verify 数据接入后的跨场景迁移验证。

下一步只进入资源预测与调度 replay；行为模型、特征契约、holdout 结果和 B05 的 19 epoch 配置保持冻结，不再用 holdout 选择模型或阈值。
