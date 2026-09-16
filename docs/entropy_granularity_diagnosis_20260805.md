# 粒度消融诊断计划(2026-08-05)

- 文档状态:已完成(2026-08-05 执行完毕,结果见第 7 节)
- 执行地点:远端 /root/autodl-tmp/scheduler
- 原则:只读分析,不改任何训练数据、不打标签;标签映射仅发生在统计时,不写回数据。偏离本文档时在"偏离记录"如实补充。

## 1. 目的

检验"预测目标粒度"对可预测性(熵降 R)的影响:7 类细粒度 canonical activity 是否因为"标签过细"而拉低了可预测性;粗粒度(5/4/3 类)下 R 与条件增量的变化,为"调度用粗粒度、研究用细粒度"的分层设计提供量化依据。

## 2. 背景

- 细粒度 7 类一阶 R=0.4041;+planner+baseline 后 0.5014;+question_type 后 0.5334(step0-3 可靠区间)。
- 既往 coarse routing(observe/summarize/answer/end)test Top-1 0.9589,显著高于细粒度 0.8157。
- 粒度变粗预计同时缓解"样本稀释"(每条件键样本增多),使 step4+ 的结果更接近真实值。

## 3. 数据

- 输入:`results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl`(8,268 行 / 1,144 runs)
- 目标字段:`target_next_activity`(7 类: sample_seek / spatial_qa / temporal_qa / object_detection / summarize / answer / __END__)
- 约定:unknown/MISSING 条件值作为独立类别,不丢弃行。

## 4. 方法

### 4.1 粒度映射方案(统计时应用,不改数据)

| 粒度 | 映射 |
|---|---|
| 7 类(参照) | 原样 |
| 5 类 | spatial_qa + temporal_qa → vision_qa |
| 4 类 | sample_seek + spatial_qa + temporal_qa + object_detection → observe;其余(summarize/answer/__END__)不变 |
| 3 类 | sample_seek + spatial_qa + temporal_qa + object_detection → visual;summarize + answer → text;__END__ 不变 |

映射只作用于预测目标;条件特征(planner/baseline/question_type 等)不变。

### 4.2 诊断指标(每个粒度跑同一套增量步骤)

沿用增量顺序 step0-step9(与 `docs/entropy_evidence_diagnosis_20260805.md` 第 4.3 节一致),报告:边际熵 H_marg、H_cond、R、n_transitions、每条件键平均样本数;每条件键样本 <20 标 ⚠️ 不下结论。

### 4.3 对比口径

- 主对比:各粒度下 step0(一阶)、step2(+planner+baseline)、step3(+question_type)的 R 与每键样本数。
- 稀释缓解度:各粒度下 step4-9 的每键样本数是否回升至 >=20。

## 5. 执行步骤

1. 扩展 `scripts/entropy_reduction_diagnostic.py`:内置粒度映射(--granularity 7|5|4|3,缺省全跑),统计时映射 target_next_activity。
2. 上传远端 `scripts/`。
3. 运行:
   ```
   cd /root/autodl-tmp/scheduler
   /root/miniconda3/envs/finetooling/bin/python scripts/entropy_reduction_diagnostic.py \
     results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl \
     --granularity all --out results/processed/entropy_reduction_granularity_20260805.json
   ```
4. 结果与解读写入本文档第 7 节。

## 6. 偏离记录

(执行中发现与原计划不一致之处,如实记录于此)

**2026-08-05 执行后补充**:

1. 无执行偏离:映射方案、诊断指标、执行命令均按第 4/5 节执行;`--granularity all` 一次跑完 4 个粒度,产物 `results/processed/entropy_reduction_granularity_20260805.json`。
2. 结果性发现(非偏离,记录供参考):3 类方案(visual/text/end,即 summarize+answer 合并)实测 R 不升反降,说明 summarize→answer 的转移是重要结构边界,该合并方案不佳;5 类(spatial+temporal 合并)几乎无变化。
3. 稀释缓解有限:粒度变粗仅将 step3 每键样本从 41.9(7 类)提升至 64.2(4 类),step4 起仍 <20;粒度变粗不是解决稀释的手段。

## 7. 结果

(执行后填写:各粒度 R、稀释缓解情况、结论)

**执行时间**:2026-08-05(远端),全量数据 8,268 行 / 1,144 runs,转移样本 7,124。

### 7.1 各粒度关键指标(可靠区间 step0-step3)

| 粒度 | 映射 | H_marg | step0 一阶 R | step2(+planner+baseline) R | step3(+question_type) R | step3 每键样本 |
|---|---|---|---|---|---|---|
| 7 类 | 原样 | 2.268 | 0.4041 | 0.5014 | 0.5334 | 41.9 |
| 5 类 | spatial+temporal→vision_qa | 2.070 | 0.4057 | 0.4931 | 0.5186 | 47.5 |
| **4 类** | **sample_seek/视觉问答/detection→observe** | **1.487** | **0.4706** | **0.5522** | **0.5651** | **64.2** |
| 3 类 | 视觉→visual;summarize+answer→text | 1.318 | 0.4000 | 0.4559 | 0.4777 | 89.0 |

### 7.2 稀释情况(step4-9)

所有粒度下 step4 起每键样本均 <20(4 类:step4=15.7,step9=1.9),step4-9 的 R 仅记录不下结论;4 个粒度的 step7-9 高 R(0.75-0.89)均为稀释假象。

### 7.3 结论

1. **4 类粒度是"可预测性甜点"**:一阶 R 0.4706,较 7 类 +6.7pp;+planner+baseline 后 0.5522(+5.1pp);+question_type 后 0.5651。样本充足,全部可靠。与既往 coarse routing(Top-1 0.9589)表现一致,验证"分类太细确实拉低可预测性"的假设。
2. **3 类过粗,不可取**:summarize+answer 合并使 R 从 4 类的 0.4706 跌回 0.4000,说明"总结→回答"的转移是重要结构;合并破坏了它。
3. **5 类 ≈ 7 类**:spatial/temporal 的区分不贡献总体可预测性,但影响 planner 条件的增量(0.4666→0.4534,合并后损失部分 planner 信息)。
4. **对设计的指导**:调度 admission/routing 用 **4 类 coarse**(observe/summarize/answer/end);细粒度 7 类仅保留为研究输出(H3/H5 路径);question_type 在 4 类下仍有增量(+1.3pp,step2→step3),值得加入调度侧特征。
5. **稀释仍是 step4+ 的障碍**,后续对 domain/coverage/progress 等字段需单条件/成对复测(见上一文档偏离记录第 3 条)。
