# 熵降证据增量诊断计划(2026-08-05)

- 文档状态:已完成(2026-08-05 执行完毕,结果见第 7 节)
- 执行地点:远端 /root/autodl-tmp/scheduler
- 原则:只读分析,不改任何训练数据、不打标签;若执行偏离本文档,在"偏离记录"如实补充。
## 1. 目的

检验 8,268 行 prefix 数据中,各类条件证据对"下一个动作可预测性"(熵降 R)的**边际贡献**,为预测器特征选择提供依据。

## 2. 背景

- 当前 locked 基线 v0.4(test Top-1 0.8157);神经模型(GRU/LSTM/GNN)在 locked test 上未超过。
- 全量熵降实测:一阶 0.4041,二阶 0.4765,一阶+planner+baseline 0.5014。
- 待回答:哪些证据(任务文本/观测证据/进度/raw 动作/NN 先验)还有增量;文本与连续值如何在马尔可夫框架下进入条件。

## 3. 数据

- 输入:`results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl`(8,268 行 / 1,144 runs)
- 字段覆盖(2026-08-05 实测):question_type 56.3%、domain 71.5%、temporal_scope 96.1%、state 类 86.2%(缺失全为 __END__ 行,符合预期)、raw_prefix_tail2 100%、compute 数值 100%、teacher_probs 100%、question/options 文本 100%
- 约定:unknown / MISSING 一律作为独立类别参与统计,不丢弃行。

## 4. 方法

### 4.1 熵降定义

R = 1 - H(A_{t+1} | 条件) / H(A_{t+1}),熵用 log2(bit)。

### 4.2 条件构造规则

- 离散字段:直接进入条件键。
- 连续字段:按数据分位数三等分(低/中/高)。
- 嵌套字段:扁平化为顶层 key(如 task_structure.question_type → question_type)。
- teacher_probs:按最大值分档(高置信 >=0.8 / 中 0.5-0.8 / 低 <0.5 / 缺失)。

### 4.3 增量测试顺序(一次只加一类,防组合爆炸)

| 步 | 条件(在上一基础上加) | 目的 |
|---|---|---|
| 0 | 一阶马尔可夫 A_t(基线) | 参照 |
| 1 | + planner_model_id | 复现已知信息 |
| 2 | + baseline | 复现已知信息 |
| 3 | + question_type | 任务文本标签的信息 |
| 4 | + domain | 任务元数据 |
| 5 | + temporal_scope | 任务元数据 |
| 6 | + coverage_ratio(分箱) | 观测证据 |
| 7 | + progress_ratio(分箱) | 进度 |
| 8 | + raw_prefix_tail2 | raw 动作历史 |
| 9 | + teacher 置信档 | NN 先验下界 |

### 4.4 稀释监测

每条件组平均样本数 = n_transitions / 唯一条件键数。低于 20 时该步结果标 ⚠️(不可靠),仍记录但不下结论。

## 5. 执行步骤

1. 扩展 `scripts/entropy_reduction_diagnostic.py`:连续分箱、嵌套扁平化、teacher 分档、增量顺序报告。
2. 上传远端 `scripts/`。
3. 运行:
   ```
   cd /root/autodl-tmp/scheduler
   /root/miniconda3/envs/finetooling/bin/python scripts/entropy_reduction_diagnostic.py \
     results/processed/neural_prefix_dataset_v0_9_options_sourcefix_20260805/prefixes_visual_context.jsonl \
     --out results/processed/entropy_reduction_evidence_incremental_20260805.json
   ```
4. 结果与解读写入本文档第 7 节。

## 6. 偏离记录

(执行中发现与原计划不一致之处,如实记录于此)

**2026-08-05 执行后补充**:

1. 无代码层面偏离:增量顺序、分箱规则、teacher 分档均按第 4 节执行。
2. 方法论发现(需记录):按"累计加条件"方式执行时,**step4 起每条件组平均样本数即跌破 20 阈值**(10.6),step7-9 低至 1.9-3.4。按 4.4 节规则,这些步骤的 R 值仅记录、**不下结论**(小样本下条件熵估计系统性偏大,R 虚高)。因此 step4-9 的 R 是"信号存在性"参考,不是真实可预测性。
3. 修正建议(后续执行):对 step4-9 中感兴趣的字段(domain、temporal_scope、coverage_ratio、progress_ratio、raw_prefix_tail2、teacher_bucket),改用**单条件独立测量 + 两两成对测量**,避免组合爆炸;连续字段可改用 2 档(中位数切分)进一步缓解稀释。

## 7. 结果

(执行后填写:每步 R、H_cond、样本量、稀释情况、结论)

**执行时间**:2026-08-05(远端)
**命令**:见第 5 节;产物 `results/processed/entropy_reduction_evidence_incremental_20260805.json`

| 步 | 条件(累计) | R | H_cond | n_transitions | 每条件组均样本 | 可靠性 |
|---|---|---|---|---|---|---|
| step0 | 一阶基线 A_t | 0.4041 | 1.351 | 7124 | 1187.3 | ✅ |
| step1 | +planner | 0.4666 | 1.210 | 7124 | 647.6 | ✅ |
| step2 | +baseline | 0.5014 | 1.131 | 7124 | 296.8 | ✅ |
| step3 | +question_type | **0.5334** | 1.058 | 7124 | 41.9 | ✅ |
| step4 | +domain | 0.5836 | 0.944 | 7124 | 10.6 | ⚠️ 稀释 |
| step5 | +temporal_scope | 0.5946 | 0.919 | 7124 | 8.2 | ⚠️ 稀释 |
| step6 | +coverage_ratio | 0.6259 | 0.848 | 7124 | 6.0 | ⚠️ 稀释 |
| step7 | +progress_ratio | 0.7829 | 0.492 | 7124 | 3.4 | ⚠️ 稀释 |
| step8 | +raw_prefix_tail2 | 0.8661 | 0.304 | 7124 | 2.0 | ⚠️ 稀释 |
| step9 | +teacher_bucket | 0.8839 | 0.263 | 7124 | 1.9 | ⚠️ 稀释 |

**结论**:

1. **可靠区间(step0-3,样本充足)**:
   - 一阶基线 R=0.4041(与既往一致);
   - planner +6.3pp、baseline +3.5pp(复现已知信息);
   - **question_type +3.2pp(R 0.5014→0.5334),这是首次结构化进入统计条件的新证据,有真实增量** → 值得加入 v0.4 backoff 链。
2. **step4-9 不可靠(每条件组样本 <20)**:domain/temporal_scope/coverage/progress/raw/teacher 的 R 数值为稀释假象,不能作为真实可预测性,也不应据此判定特征价值。
3. **下一步(按偏离记录第 3 条)**:对 step4-9 字段做单条件/成对复测,验证是否存在真实增量;优先 domain(step4 起始稀释,但单条件时样本充足)。
