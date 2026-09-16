# 粗粒度行为预测器与神经网络实验总规划

- 文档状态：执行中；B00--B03 已完成正式 train-to-validation 复现；B05 finalist CV 已完成（development-only），final holdout、资源预测与调度 replay 仍未执行
- 远端项目根目录：`/root/autodl-tmp/scheduler`
- 正式文档路径：`/root/autodl-tmp/scheduler/docs/behavior_nn_experiment_plan.md`
- 当前主线：粗粒度行为预测；资源预测与调度模拟后置
- 适用数据：当前 300 个 VideoMME 视频的 1,240 个 Agent run，以及后续新增的未见视频和真实电梯 Agent trace
- 取代范围：本文件取代旧文档中“NN 探索结束”“残差模型已是生产候选”“调度相关指标全面领先”等结论；旧文件仅保留为实验历史，不删除、不覆盖

## 0. 任务边界与最终目标

### 0.1 本阶段目标

本阶段只研究行为预测器：在严格只使用当前时刻及历史 prefix 的条件下，输出下一角色、下一 execute 动作族、未来多步路径、终止/失败概率和不确定性。后续资源预测器再把行为概率映射成 runtime、显存和负载，调度器最后消费行为与资源预测结果。

正式架构边界如下：

```text
Agent prefix/state
      ↓
behavior predictor
      ├─ p(next_role)
      ├─ p(action_family | execute)
      ├─ top-K future paths, H=3/5
      ├─ p(terminate within H)
      ├─ p(retry/failure within H)
      └─ uncertainty/calibration
      ↓
resource predictor（后续阶段，不在本文件执行）
      ↓
scheduler（后续阶段，不在本文件执行）
```

### 0.2 本文件不授权的事项

- 不启动训练、下载、trace 采集、模型服务或调度仿真；
- 不修改原始 trace；
- 不删除或覆盖旧实验结果；
- 不把 VideoMME 中不存在的 `verify` 样本人工伪造出来；
- 不把资源预测或调度收益写成已经验证的结论；
- 不把旧 fine-grained v0.4 与当前 coarse-role 指标直接横向比较。

### 0.3 总验收标准

1. 每个正式数字可追溯到数据 hash、split、代码版本、配置、seed 和逐样本预测；
2. 所有 scaler、vocab、transition graph、OOF teacher 只由对应训练折拟合；
3. 统计单位以视频为主，不把同视频内大量相关事件当成独立样本；
4. NN 的增益必须通过无序列、乱序历史和简单校准基线，不能只与一个弱 NN 比；
5. 模型选择只使用 development/validation，最终未见视频只评估一次；
6. 未满足门槛时停止相应路线，并保留 XGB 作为正式基线。

## 1. 已核对的当前状态

### 1.1 远端环境

| 项目 | 当前事实 |
|---|---|
| 主机 | `autodl-container-41d846924e-d62d0601` |
| 项目 | `/root/autodl-tmp/scheduler` |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| GPU 显存 | 32,760 MiB，总体空闲约 32,230 MiB（规划审查时快照） |
| 数据盘 | 250 GiB，已用约 218 GiB，可用约 33 GiB，使用率 87% |
| 执行原则 | 视频、数据处理、训练和评测全部在远端完成；结构化 NN 与视觉编码器不并发驻留 |

磁盘和 GPU 状态属于动态值，执行前重新读取；本文件中的数值不能替代正式运行前检查。

### 1.2 当前 coarse-role 数据

| split | 事件 | run | 视频 |
|---|---:|---:|---:|
| train | 13,754 | 988 | 240 |
| validation | 2,029 | 144 | 30 |
| test | 1,520 | 108 | 30 |
| 总计 | 17,303 | 1,240 | 300 |

已核对：三个 split 的视频和 run 交集均为 0；run 内 `event_index` 从 0 连续编号；同一 run 不跨 split。

当前 next-role 支持：

| 角色 | 全量支持 |
|---|---:|
| plan | 6,749 |
| execute | 6,552 |
| aggregate | 2,762 |
| terminate | 1,240 |

角色 schema 仍保留 `init / plan / execute / verify / aggregate / terminate` 六类，但本 VideoMME 任务的正式目标只评价有真实 target 的 `plan / execute / aggregate / terminate`。`init` 主要是当前首节点，`verify` 没有真实目标样本，不能混入 macro-F1 分母制造虚假结果。

当前 test 的 execute 动作族支持：

| 动作族 | test 支持 |
|---|---:|
| select_frames | 370 |
| visual_qa | 73 |
| temporal_ops | 94 |
| summarize | 24 |
| detect | 1 |
| other | 0 |

因此 `detect` 和 `other` 暂不具备独立泛化结论条件；必须报告 support，不能只报一个宏平均。

### 1.3 现有多模态资产

- 已有 300 个有效 VideoMME 视频；
- 已有 prefix-only InternVideo 表示和视觉上下文数据；
- `neural_prefix_dataset_v0_9_options_sourcefix_20260805` 包含 8,268 条 prefix 级记录，但它与 17,303 行 role dataset 不是天然一一对应；
- 现有 prefix 数据覆盖 1,144 个左右的历史 run，而 role 数据有 1,240 个 run；正式融合前必须重新审计 join coverage；
- 视觉缺失与真实零向量必须用单独 mask 区分；
- 第一轮复用缓存表示，不重新抽取全量视觉特征。

### 1.4 当前结果的正确解释

最新残差实验的关键数字：

| 指标 | XGB | XGB+GRU residual |
|---|---:|---:|
| family Top-1 | 0.8060（453/562） | 0.8025（451/562） |
| macro-F1 | 0.7376 | 0.7427 |
| temporal recall | 34/94 | 39/94 |
| 自定义加权代价总和 | 约 331 | 约 319 |
| pipeline join | 0.8507 | 0.8474 |

残差模型多识别对 5 个 temporal，但少识别对 5 个 select 和 2 个 visual，净少对 2 个样本。它证明了“改变决策边界可改善 temporal 召回”的可能性，尚未证明 GRU 学到了有用的动态历史，更不能称为全面领先或生产候选。

## 2. 旧结果清算与有效性分级（R0）

执行新实验前先生成正式 result registry，旧文件不删除。

| 实验/数字 | 分级 | 原因 | 后续动作 |
|---|---|---|---|
| XGB role 0.9322 | 暂定基线 | split 正确，但元数据与最终评估契约仍需修复 | R2 重跑锁定 |
| XGB family 0.8060 | 暂定基线 | 使用当前完整特征；仍受元数据覆盖影响 | R2 重跑锁定 |
| XGB joint 0.8507 | 暂定基线 | join 有意义，但 route/layer 旧实现分母不正确 | 用新 evaluator 重算 |
| FT-Transformer v2 0.8060 | 部分有效 | 静态结构已修，但共享元数据仍可能错误 | R2 重跑 |
| BiLSTM 0.8043 | 作废待重跑 | 右 padding 后直接读隐藏状态 | 正确 mask 后重跑 |
| GRU 0.8096 | 作废待重跑 | 同上 | 正确 mask 后重跑 |
| 全 NN join 0.8059 | 作废待重跑 | 角色/动作 GRU 均有 padding 问题 | 正确 mask 后重跑 |
| XGB+GRU residual 0.8025 | 诊断性线索 | padding 未修；未与无序列 residual 比较 | R2/N1 重跑 |
| OOF stacking 0.7900 | 部分有效 | GroupKFold 已修；共享元数据仍错误 | 作为简单 ensemble 对照重跑 |
| MLP+FM 0.7705 | 历史负结果 | 真 FM 已实现且无增益 | 核心 MLP 变化 <1pp 时不重跑 |
| focal/class weight | 历史负结果 | 曾降低总体表现 | 不作为第一轮主实验 |
| fine-grained v0.4/64 视频 | 历史独立口径 | 标签空间、样本和 split 不同 | 只在附录单独报告 |

R0 产物：

```text
results/processed/behavior_nn_v1/registry/legacy_result_registry.json
results/processed/behavior_nn_v1/registry/legacy_result_registry.md
```

每条记录至少保存：原始文件、任务语义、数据版本、split、已知缺陷、是否可引用、替代结果路径。

## 3. 统一数据契约与实现修复（R1，所有训练的硬前置）

### 3.1 canonical sample key

禁止再以 `video_id` 单独关联 task/prefix。目标是保留以下稳定键：

```text
run_id
task_id
source_event_id
target_source_event_id
event_index
video_id              # 仅用于分组和审计，不作为输入
source_trace_sha256
```

首选 join key：`run_id + source_event_id/target_source_event_id`。如果旧 role dataset 没有 source ID，必须从原始/compute event 重新派生，不能以事件顺序猜测后静默接受。`run_id + event_index` 只能作为经审计确认后的兼容键，并需保存 alignment report。

### 3.2 单条样本的允许输入

#### 当前状态与任务

- current role、current raw/canonical action、status；
- question type、domain、temporal scope、answer type、required modalities；
- question/options 的文本或冻结表示，但不含答案、正确选项和 ground truth；
- prefix position、真实 prefix 长度、是否首步；
- planner/model stack/baseline 作为 provenance 分支，必须可完全关闭。

#### 历史事件

- 严格早于目标事件的完整 role/action/status 序列；
- retry/error 计数和已发生的 recovery；
- evidence coverage、frames seen、OCR/object/temporal evidence 统计；
- 已观察到的模型驻留、历史 runtime 等只能作为单独 resource-history 消融，不能与主迁移模型混为一体。

#### 视觉输入

- 只允许 prefix 中已经被 Agent 看到或采样的帧；
- 目标事件及未来帧全部排除；
- 保存 embedding source、frame indices、cutoff event 和 `vision_available`；
- 不允许先编码整段视频再按 prefix 截断。

### 3.3 明确禁止的输入

- video ID/hash、run ID、task ID 的数值或 embedding；
- answer/answer label/correct option；
- next/future action、future state、remaining steps 作为输入；
- target runtime/VRAM/status；
- 用 test 视频拟合的 scaler、vocab、class prior、transition graph、RAG index；
- 目标事件本身产生的帧或 evidence；
- 从完整 trace 生成后无法证明 prefix cutoff 的表示。

### 3.4 padding 和序列编码修复

统一序列接口：

```text
tokens: [B, T, D]
lengths: [B]
attention_mask: [B, T]
```

- GRU/LSTM：使用 `pack_padded_sequence`，或显式 gather 每条样本真实最后位置；
- Transformer：使用 attention mask，PAD 不参与注意力或池化；
- 不再把 execute 以外事件全部删除；“完整事件轴”和“execute-only 压缩轴”只能作为两种显式消融；
- max length 由 train prefix length P99 决定，上限先设 64；超过部分只截断最早历史并记录 truncation rate；
- PAD token 固定且不允许通过 padding 数量泄漏长度，真实长度作为单独显式数值输入。

### 3.5 元数据修复

已发现旧脚本按 `video_id` 连续覆盖 metadata，64/300 个视频存在有效 domain 被后续空记录覆盖为 unknown 的情况，影响 729/1,520 个 test 事件。修复要求：

1. task metadata 按稳定 task/run/event 键关联；
2. 空记录不得覆盖非空记录；
3. 一对多 join 必须显式报错或保留候选，不静默取最后一条；
4. 输出 exact/missing/ambiguous join 数量；
5. 每个模型实际使用的字段写入 feature contract，避免“赋值了但没有进入特征”的假增强。

### 3.6 视觉缺失修复

- 视觉向量和 `vision_available` 同时进入模型；
- 首步无视觉、文件缺失、提取失败、真实空证据分别编码；
- 对 available-only 子集和全量-with-mask 分别报告；
- 任何 zero-fill 结果都保存 `missing_reason`；
- shuffled-vision 负对照与正式特征使用相同 missing mask。

### 3.7 R1 自动化测试

#### 因果与泄漏测试

1. 修改目标之后 suffix，当前样本序列和静态特征 hash 不变；
2. 修改答案/正确选项，不影响输入；
3. 目标事件及以后 frame index 不出现在视觉 manifest；
4. train/validation/test 的 video/run 无交集；
5. fold-specific scaler/vocab/graph 只包含 fold train 来源；
6. OOF 每条训练预测来自未见过该视频的模型。

#### padding 测试

1. 同一真实序列增加任意数量 PAD，logits 在容差内不变；
2. 长度 0/1/max/P99+1 均可推理；
3. batch 内不同长度与逐条推理输出一致；
4. Transformer mask 和 GRU packed sequence 得到一致的有效 token 数。

#### join 与评估器测试

1. metadata 空记录不能覆盖有效记录；
2. 造一个非 execute 角色预测错误样本，joint 必须记错；
3. executor route 的分母只能是 execute 子集；
4. 无支持类不进入 observed-class macro-F1，同时 fixed-schema macro-F1 单独报告；
5. support、confusion matrix、per-class recall 可从 predictions 重算。

### 3.8 R1 验收门槛

- 所有上述测试通过；
- canonical 样本解析率 100%，或每个 rejected row 有明确原因；
- exact join 覆盖率和视觉覆盖率写入审计报告；
- future/answer/target leakage violation = 0；
- padding invariance violation = 0；
- 新 evaluator 的手工 fixture 全部通过；
- 未通过前禁止进入 R2。

## 4. 重新设计数据划分和测试集策略

### 4.1 现有 test 的重新定位

当前 30 个 test 视频已被多轮观察和用于架构/代价讨论，不再是完全未污染的最终测试集。后续将现有 300 视频全部视作 development pool：

- R2 仍用旧 240/30/30，仅用于与历史结果对齐；
- N1 以后用按视频分组的 development CV；
- 最终模型冻结后，另取从未用于开发的新视频或真实电梯 trace 进行一次性评估。

### 4.2 development CV

候选设计：5-fold stratified group CV，group=`video_id`，尽量平衡 domain、question type 和动作族支持。若当前 sklearn 不支持所需 stratified group 行为，则实现可审计的固定 split manifest，不在每次运行中随机重分。

每折必须独立拟合：

- scaler；
- categorical vocab；
- transition graph；
- class priors；
- OOF XGB teacher；
- temperature/calibration 参数。

屏幕筛选阶段：旧 train/validation，不看旧 test；3 seeds。

开发确认阶段：排名前三配置进行 5 folds × 3 seeds；结果按视频 paired bootstrap 汇总。

### 4.3 最终未见集合

模型结构、输入字段、损失、阈值、ensemble 权重全部冻结后，再准备 30–50 个未参与开发的视频。执行前：

1. 固定 manifest 和 source hash；
2. 估算下载体积，数据盘至少预留安全余量；
3. 优先覆盖 temporal、object/spatial、general/summarization 问题；
4. 采集方式和 baseline 配置在看结果前冻结；
5. 最终集合只评估一次，不用于继续调参。

如果暂时不新增视频，则只能报告 development CV，不能宣称最终泛化性能。

### 4.4 迁移评估

除 IID video split 外，增加：

- leave-one-baseline/planner-out；
- leave-one-model-stack-out；
- provenance-on vs provenance-off；
- 普通模板位置 vs branch-ambiguous/hard subset。

如果 provenance-on 很高、provenance-off 和 held-out planner 大幅下降，必须把模型描述为 workflow-specific predictor，不能宣称跨 Agent 迁移。

## 5. 修正复现实验（R2：补完旧实验）

### 5.1 统一任务

#### Head A：next role

输入全部 prefix/state，输出 observed target roles：`plan / execute / aggregate / terminate`。

#### Head B：next action family

只在真实下一角色为 execute 的样本训练与计算 oracle-gated 指标；输出 `select_frames / visual_qa / temporal_ops / summarize / detect / other`，报告 observed-class 和 fixed-schema 两种 macro 指标。

#### End-to-end joint

角色预测为 execute 时使用 family head；真实非 execute 时只要求角色正确；真实 execute 时角色和 family 均正确才算 joint correct。

### 5.2 最小模型矩阵

| ID | 模型 | 当前状态 | 目的 | 必跑 seed |
|---|---|---|---|---|
| B00 | Global/position/Markov prior | formal v2 三变体已完成 | 衡量模板可预测性 | deterministic（seed=0） |
| B01 | XGB role + XGB family | formal validation 完成（joint=0.8501） | 锁定树基线 | deterministic（seed=0） |
| B02 | Logistic/linear baseline | formal validation 完成（joint=0.7829） | 简单可解释下限 | deterministic（seed=0） |
| B03 | Static MLP | formal validation 3 seeds 完成（joint=0.8510±0.0006；未证明优于 B01） | 无序列 NN 基线 | 11/22/33 |
| B04 | FT-Transformer v2 | 0.8060 暂定 | 静态交互 NN | 11/22/33 |
| B05 | Masked GRU | 旧结果作废 | 修复序列主线 | 11/22/33 |
| B06 | Masked LSTM | 旧结果作废 | RNN 对照 | 11/22/33 |
| B07 | Causal Transformer | 旧实现口径不统一 | 非循环序列对照 | 11/22/33 |
| B08 | Group-OOF XGB+linear meta | 0.7900 暂定 | 简单 stacking | deterministic |
| B09 | XGB+MLP residual | 未做 | 检验普通校准/静态 residual | 11/22/33 |
| B10 | XGB+GRU residual | padding 未修 | 检验序列 residual | 11/22/33 |
| B11 | XGB+Transformer residual | 未做 | 检验长历史 residual | 11/22/33，B10 通过后 |
| B12 | cost-sensitive XGB decision | 未做 | 最简单长尾/代价基线 | validation only tuning |

FM、focal、inverse class weight 不在第一轮重复。若修复后 B03 与旧结果差异超过 1pp，才追加这些负对照，判断旧结论是否失效。

### 5.3 公平参数预算

先以同量级容量比较：

- MLP：2 层，width 128/256；
- GRU/LSTM：1–2 层，hidden 128/256；
- Transformer：2 层，d_model 128/256，4 heads；
- dropout 0.1/0.2；
- AdamW，初始 lr 候选 `3e-4 / 1e-3`；
- batch 先 64；
- max length 使用 train P99，最大 64；
- early stopping 只看 validation；
- 参数搜索限于预定义小网格，不能依据 test 手工增删。

XGB 先复用当前 `n_estimators=300, learning_rate=0.05, max_depth=6` 作为复现锚点，再只在 development validation 上测试有限网格；重跑每个 NN seed 时复用同一 fold-safe OOF logits，不重复训练 XGB。

### 5.4 R2 产物

```text
results/processed/behavior_nn_v1/
  data/
    canonical_samples.jsonl
    split_manifest.json
    feature_contract.json
    leakage_report.json
    join_report.json
  baselines/
    xgb_oof_logits.npz
    xgb_oof_manifest.json
  runs/
    Bxx_<model>_<seed>/
      config.json
      environment.json
      dataset_manifest.json
      metrics.json
      predictions.jsonl
      confusion_matrix.json
      checkpoint.pt            # 仅 NN
      run.log
  comparisons/
    corrected_vs_legacy.json
    paired_bootstrap.json
```

固定文件名不得静默覆盖；每个 run 使用实验 ID、seed 和配置 hash。最终汇总从逐样本 predictions 重算，不手工抄数字。

### 5.5 R2 验收与停止条件

R2 只回答“修复后谁最好”，不宣称跨场景泛化。

- XGB/MLP/FT/GRU/LSTM/residual 均使用同一 canonical dataset 和 evaluator；
- 每个 NN 至少 3 seeds；
- 输出均包含 class support、per-class recall、NLL/ECE 和 joint；
- 如果 B05/B06 与 B03 无差异，序列路线进入 N1 反事实验证，不直接增加更大模型；
- 如果 B10 不优于 B09，残差增益归类为静态校准，不再声称 GRU 使用历史；
- 如果任何模型只在自定义 cost 上变好而 joint/macro/NLL 均变差，不能升级候选。

## 6. 序列信息因果验证（N1）

### 6.1 研究问题

`完整历史序列是否提供了当前结构化状态和 XGB logits 之外的增量信息？`

这是继续投入 RNN/Transformer、公开预训练和多步生成的必要条件。

### 6.2 输入反事实矩阵

| ID | 输入 | 说明 |
|---|---|---|
| S0 | 当前状态，不含历史 | 静态下限 |
| S1 | 当前状态 + 最近一个事件 | 一阶历史 |
| S2 | 当前状态 + 完整 causal prefix | 序列主实验 |
| S3 | S2，但历史顺序在同长度桶内打乱 | 检查是否真用顺序 |
| S4 | S2，但历史 token 全置为统一 unknown | 检查是否只依赖静态分支 |
| S5 | execute-only 压缩历史 | 与旧脚本兼容的消融，不作默认输入 |
| S6 | 完整历史但隐藏 event_index/position | 检查是否只记步骤模板 |
| S7 | 完整历史但隐藏 baseline/planner/stack | 检查迁移能力 |

先只用 masked GRU；若 S2 相对 S0/S3/S4 有稳定增益，再对 LSTM/Transformer 复核。若 GRU 不通过，不扩大 backbone 搜索。

### 6.3 hard subset

总体角色序列高度模板化，必须另外报告真正有决策难度的子集：

- 相同 `(baseline, current_role, position_bucket)` 在训练中对应多个 next role；
- 相同结构 prefix 对应不同 family；
- temporal_ops 与 visual_qa 混淆；
- 长循环、非典型长度、提前终止；
- error/retry 后的恢复；
- provenance-off 和 held-out planner；
- prefix length P75 以上。

hard subset 的定义和阈值只由 train 统计生成，不能看 test 错例后手工挑样本。

### 6.4 序列有效性的正式判据

只有同时满足以下条件，才允许写“序列建模有效”：

1. S2 相对 S0 在 joint/macro-F1/NLL 至少一项有 paired video CI 改善，且 primary joint 不显著下降；
2. S3/S4 相对 S2 明显退化；
3. 长 prefix 或 hard subset 增益大于简单模板子集；
4. XGB+GRU residual 优于 XGB+MLP residual；
5. 至少 3 seeds 和 development folds 方向一致。

未满足时停止序列 backbone 扩展，正式行为模型回到 XGB 或 XGB+静态校准。

## 7. 层级多任务与详细输出（N2）

### 7.1 模型结构

```text
task/static encoder ──────┐
event-prefix encoder ─────┼─ fusion ─ shared state
optional evidence encoder ┘                │
                                           ├─ next_role head
                                           ├─ conditional family head
                                           ├─ direct H=2/3/5 heads
                                           ├─ terminate-within-H head
                                           ├─ retry/failure-within-H head
                                           └─ uncertainty/calibration
```

建议损失：

```text
L = L_role
  + λ_family * I(next_role=execute) * L_family
  + λ_horizon * Σ_h L_h
  + λ_terminate * L_terminate
  + λ_retry * L_retry
```

辅助任务权重仅在 validation 小网格选择；先报告单任务 head，再报告多任务，避免辅助标签反而损害主任务却被总体 loss 掩盖。

### 7.2 详细推理输出契约

```json
{
  "next_role_probs": {},
  "action_family_probs_if_execute": {},
  "topk_future_paths": [
    {"roles": [], "families": [], "joint_probability": 0.0}
  ],
  "expected_execute_count_h3": 0.0,
  "terminate_probability_h3": 0.0,
  "retry_probability_h3": 0.0,
  "entropy": 0.0,
  "calibrated_confidence": 0.0,
  "fallback": false,
  "model_version": "",
  "feature_contract_version": ""
}
```

资源预测器以后只消费候选角色/动作概率和必要上下文，不依赖隐藏层内部实现。

### 7.3 多步路线

对比三种方法：

1. 每个 horizon 独立 direct head；
2. autoregressive rollout，把预测回填；
3. shared encoder + multi-horizon direct heads。

图约束只由 fold-train 构建，比较：无约束、soft penalty、hard mask。输出：

- per-horizon role/family/joint；
- H=3/H=5 path exact；
- normalized edit distance；
- top-K path coverage；
- legal path rate；
- terminate/retry AUROC/AUPRC（支持足够时）；
- 不同 prefix 长度分桶。

### 7.4 N2 进入门槛

- N1 已证明序列信息有效，或静态 residual 已经在概率质量上稳定改善；
- H=3/H=5 标签完全在 supervision 侧读取；
- failure/retry 支持不足时只输出支持报告，不训练伪标签；
- 多任务不能让 next-role joint 的 paired CI 显著下降。

## 8. 多模态 NN 探索（N3）

### 8.1 核心假设

视觉和任务内容更可能改善 `temporal_ops vs visual_qa vs select_frames` 等 family 决策，而不一定改善高度模板化的 next role。因此多模态的主评测放在 family、hard subset、多步路径和不确定性，不以 role Top-1 单一否决。

### 8.2 消融矩阵

| ID | structured | task text | evidence stats | InternVideo prefix | provenance | XGB logits |
|---|---|---|---|---|---|---|
| M0 | ✓ |  |  |  | off/on 两版 |  |
| M1 | ✓ | ✓ |  |  | off/on |  |
| M2 | ✓ |  | ✓ |  | off/on |  |
| M3 | ✓ |  | ✓ | ✓ | off/on |  |
| M4 | ✓ | ✓ | ✓ | ✓ | off/on |  |
| M5 | ✓ | ✓ | ✓ | shuffled | 与 M4 相同 |  |
| M6 | ✓ | ✓ | ✓ | zero + missing mask | 与 M4 相同 |  |
| M7 | ✓ | ✓ | ✓ | ✓ | off/on | ✓ residual |

M5 是必需负对照：如果视觉向量随机配对后结果不下降，说明模型没有利用视频内容。M6 检查收益是否来自 missingness 模式而不是视觉语义。

### 8.3 融合结构

第一轮不用大 VLM 反向训练：

- task text：冻结小型文本表示或现有 teacher 表示；
- InternVideo：复用 prefix embedding，线性 projection 到 128/256；
- structured/event encoder：GRU 或通过 N1 的 backbone；
- fusion：先 concat+MLP，再比较 gated fusion/FiLM；
- 每个 modality 有 availability mask 和 dropout，防止模型过度依赖单一来源。

只有 concat 基线显示真实增益后才引入 cross-attention；避免一次同时改变表示、融合和损失导致无法归因。

### 8.4 多模态评测分桶

- temporal、spatial、object、general question；
- 有视觉 prefix / 首步无视觉 / 提取失败；
- low/high evidence coverage；
- 短/长视频；
- same-structure-different-family hard subset；
- provenance-off；
- held-out planner/stack。

### 8.5 Qwen3-VL-8B 分支门槛

当前 GPU 足以做冻结推理，但第一轮不加载 Qwen3-VL-8B。仅当 M3/M4 相对 M0 在 validation/CV 上有稳定内容增益时，才测试：

1. 冻结 8B；
2. 只读取 prefix-visible 帧/证据；
3. 缓存 representation，不更新 8B；
4. 单独报告 feature extraction runtime、peak VRAM 和 load cost；
5. 与 InternVideo 使用同一后端预测头；
6. 如果表示提取成本超过预测收益，标记为 research-only。

### 8.6 N3 成功判据

- M3/M4 优于 M0，paired video CI 支持；
- M5 明显劣于 M4；
- 增益出现在内容相关 hard subset，而非仅 provenance 模式；
- overall joint 不显著下降；
- missingness 分桶没有隐藏失败；
- 视觉提取成本单独报告。

## 9. 公开数据预训练与迁移（N4）

### 9.1 当前 pilot 的限制

现有公开事件预训练约 2,746 个 prefix 样本，只能证明流程能跑和部分 GRU 权重可加载，不足以支持“大规模公共预训练提高迁移”的结论。旧实验只迁移 recurrent weights，label/token embedding 未迁移，收益极小且使用旧 fine-grained 数据。

### 9.2 正式数据门槛

进入 N4 前先形成 dataset audit：

- 至少数万条合法 prefix 样本；
- 至少数千个独立 case/run；
- case 级 split；
- 包含循环、分支、失败、retry、终止；
- 许可证、来源、下载时间、原始和转换 SHA256；
- activity 映射到公共抽象角色/事件，不直接混用领域 label；
- public 数据与 VideoMME/elevator final holdout 完全隔离。

候选数据按作用分开：

- process logs（BPI/Sepsis 等）：学习循环、分支、终止；
- tool-agent traces（ToolBench/API-Bank/Mind2Web 等）：学习 plan/tool/result/retry 结构；
- 不把两类来源无标记混在一起，必须支持来源消融。

### 9.3 预训练任务

- next abstract role/event；
- masked event reconstruction；
- direct H=3/H=5 future prediction；
- terminal/retry classification；
- 可选 case-level contrastive learning。

### 9.4 迁移矩阵

| ID | 初始化 | 迁移方式 |
|---|---|---|
| P0 | random | 自有数据从头训练 |
| P1 | public process | 冻结 encoder，只训 head |
| P2 | public process | 全量低学习率微调 |
| P3 | public tool-agent | 冻结 encoder |
| P4 | public tool-agent | 全量微调 |
| P5 | mixed public | 冻结 warm-up 后解冻最后一层 |

token embedding、role head 与来源 ontology 不一致时重新初始化；每次记录实际 loaded/skipped keys，不能只写“使用了预训练”。

### 9.5 N4 停止条件

- N1 未证明序列编码器有效：停止公共序列预训练；
- 数据规模未过门槛：保留 pilot，不扩大结论；
- pretrained 相对 scratch 在 3 seeds/多折不稳定：不进入主模型；
- 收益只来自 provenance：不声称通用迁移；
- public 预训练导致自有 hard subset/rare class 退化：停止或只作附录。

## 10. 图约束与 GNN（N5）

### 10.1 当前 VideoMME 只做图约束

现有大多数轨迹是 `plan → execute → plan` 的链式/循环结构。立即训练通用 GNN 很可能只复刻 n-gram/GRU，不具备结构创新意义。因此当前只做：

- fold-train transition graph；
- no graph / soft penalty / hard mask；
- unknown context fallback；
- legal path rate 与错误拒绝率。

图 mask 不得由 validation/test future 构建，也不能通过把所有未知样本回退到单一先验虚增合法率。

### 10.2 电梯真实轨迹到位后做实例图 GNN

电梯 Agent 的实例图：

- 节点：event、requirement、evidence、candidate action；
- 边：temporal、parent、retry_of、requirement dependency、evidence support、planner→tool→validator；
- 模型：GraphSAGE/GAT/Graph Transformer 小模型；
- 输出：next role、branch probability、remaining loop、terminate/retry；
- 对照：GRU/Transformer 使用相同节点特征；
- `verify` 必须来自真实 validator 行为。

实例 GNN 进入门槛：存在足够真实 branch/verify/retry edge；每种关键边有训练和测试支持；图在 prefix cutoff 后不会看到未来节点。

## 11. 指标、统计和模型选择规则

### 11.1 primary 指标

1. `role_top1`、`role_macro_f1`；
2. `family_top1_oracle_gate`、`family_macro_f1_observed`；
3. `joint_pipeline_accuracy`；
4. `joint_nll` 或层级 NLL；
5. H=3/H=5 path exact/top-K coverage。

### 11.2 secondary 指标

- per-class precision/recall/F1/support；
- MRR、Brier、ECE；
- selective accuracy at 80/90/95% coverage；
- normalized path edit distance；
- legal path rate；
- inference latency、feature extraction latency、参数量、峰值显存；
- hard subset、prefix length、planner/stack/task type 分桶。

### 11.3 cost 指标

`cost_err` 暂时只作 secondary diagnostic，不作为当前模型选择第一标准。最终 cost matrix 必须在看 final holdout 前冻结，优先由 fold-train 中 action family 的 runtime/VRAM/resource-class 差异推导；人工设置时必须写清业务依据并做权重敏感性分析。

### 11.4 模型选择顺序

使用 validation 上的字典序规则，避免临时发明综合分数：

1. 首先最大化 joint accuracy；
2. 距最优 0.5pp 内时，选择更高 family macro-F1；
3. 再选择更低 NLL/ECE；
4. 仍相近时选择更简单、推理更快、依赖更少的模型。

cost、某一个长尾 recall 或 test 表现不能覆盖 primary 规则。

### 11.5 统计协议

- NN screening：3 seeds；finalist：5 folds × 3 seeds；最终 ensemble：冻结后 5 seeds；
- 以 video_id 做 paired bootstrap，建议 5,000–10,000 次；
- 报告差值均值和 95% CI；
- 同视频事件不可按独立 Bernoulli 计算普通置信区间；
- detect/verify/retry 等小类必须报告支持视频数和事件数；
- test/final 只执行冻结配置，不再选择阈值。

### 11.6 三档结论门槛

#### 升级为主模型

- joint 提升至少约 2pp，或 paired CI 明确优于 XGB；或者
- joint 满足非劣界（建议 lower CI > -0.5pp），同时 NLL/ECE、多步和长尾有稳定改善；
- 至少 3 seeds 和跨 planner/stack 测试方向一致；
- 没有依赖未来、ID 或错误 metric denominator。

#### 保留为研究候选

- 总体 Top-1 持平；
- 但多步、校准或 hard subset 有稳定增益；
- 通过 shuffled-history/shuffled-vision/no-sequence 对照；
- 还未经过新 final holdout。

#### 停止路线

- 只在定制 cost matrix 上变好；
- joint/macro/NLL 大部分下降；
- shuffled history/vision 不影响结果；
- 只在一个 seed 或一个视频折有效；
- 需要引入高成本模型但预测收益不稳定；
- 数据支持不足却依靠过采样制造 headline 数字。

## 12. 资源与执行安排

### 12.1 计算原则

- 所有数据构建、训练、评测都在远端；
- R1 单元测试和数据审计先在 CPU；
- XGB OOF 使用 CPU，完成后缓存；
- NN 训练使用单卡，seed 串行；
- InternVideo/Qwen 特征提取与 NN 训练不并发；
- 不同时驻留多个 VLM；
- 每个 run 记录 wall time、GPU peak allocated/reserved、CPU RSS 和退出状态；
- 长任务使用已批准的可恢复会话方式前，需单独确认，不默认创建后台进程。

### 12.2 磁盘原则

- 当前约剩 33 GiB；
- 第一轮复用已有 embeddings；
- checkpoints 只保留 best 和必要 seed；
- predictions 优先 jsonl/parquet 压缩，但保留可重算性；
- 新视频下载前先生成 size budget；
- 临时特征放统一 `_tmp`，清理仍需确认；
- 不复制 27 GiB 视频库制作不同版本。

### 12.3 分阶段运行预算

以下是调度顺序，不是时间承诺：

1. R0/R1：无正式训练，完成 schema、validator、fixtures；
2. R2：先 B00–B10，3 seeds；
3. N1：只用 GRU 做 S0–S7，过门后才扩展 LSTM/Transformer；
4. N2：只对 N1 胜出 backbone 做多任务/多步；
5. N3：先缓存 InternVideo 消融，再决定 Qwen；
6. N4：先扩公开数据审计，再预训练；
7. N5：VideoMME 只做 graph mask，实例 GNN 等电梯数据；
8. 冻结后才采集/评估 final holdout。

## 13. 统一实验产物与复现契约

唯一正式根目录：

```text
results/processed/behavior_nn_v1/
```

建议结构：

```text
behavior_nn_v1/
  README.md
  registry/
  data/
  baselines/
  runs/
  comparisons/
  final_candidate/
  final_holdout/          # 仅模型冻结后生成
```

每个 run 必须包含：

```text
experiment_id
timestamp
host/environment
script_sha256 / code provenance
dataset_sha256
split_manifest_sha256
feature_contract_version
config
seed
train/validation/test counts
class support
best epoch and selection metric
metrics
predictions
confusion matrix
runtime / resource usage
failure status
```

任何汇总文档只引用这些正式 artifact，不能引用 stdout 截图、手抄表格或被覆盖的固定 JSON。

## 14. 完整执行清单与阶段门

### Gate A：允许训练前

- [x] legacy result registry 完成
- [x] canonical sample schema 确认
- [x] stable join key 覆盖报告完成
- [x] split/group leakage 检查通过
- [x] padding invariance 测试通过
- [x] future/answer/visual cutoff 测试通过
- [x] evaluator fixture 通过
- [x] artifact root 和 config schema 确认

#### Gate A R2 验收记录（2026-08-10）

- [x] scripts_behavior_r1_v5/v5_verify_final.py 返回 all_ok=true，Gate A 8/8 通过
- [x] 当前 R2 全量测试套件 scripts_behavior_r2_v5/tests 通过 154 项（含 B01/B02 static 与 B03 neural 测试）
- [x] 全量 sample 预处理审计：role=17,303、tool=5,889，静态维度 71/66，序列最大长度 16/13，unknown-category 行数为零，history 索引严格早于 cutoff
- [x] B00 validation-only smoke 与 formal 三变体比较已完成；尚未进行任何 test 评估
- [x] B01/B02 formal validation 已完成；尚未进行任何 test 评估
- [x] B03 formal validation 三 seed 已完成；尚未进行任何 test 评估

#### B00 validation-only smoke 记录（2026-08-10）

- [x] artifact：results/processed/behavior_nn_v1/runs/B00_markov1_smoke__bf3ddf61ca3e
- [x] 仅 train 拟合（role=13,754；family=4,712）并仅 validation 评估（role=2,029；family=615）；test_rows_modelled=0
- [x] 7 个 canonical 文件齐全，predictions.jsonl 可独立重算并与 metrics.json 完全一致
- [x] 仅管线 smoke：role top-1=0.7782、family top-1=0.6683、joint=0.6889；其生成早于完整 provenance 补齐，不作为 formal 比较来源

#### B00 formal validation 比较（2026-08-10）

- [x] formal source artifacts：B00_global_formal_v2__09d2fc055122、B00_position_formal_v2__28a74b270a8e、B00_markov1_formal_v2__799d407817c2（均在 results/processed/behavior_nn_v1/runs/）
- [x] 三者均只用 train 拟合、只在 validation 评估，test_rows_modelled=0；joint 分母均为 1,861
- [x] validation joint：global=0.4250、position=0.5476、markov1=0.6889；对应 role top-1=0.3898/0.5885/0.7782
- [x] v2 中每份均记录 7 个实际代码 hash、dataset/split/feature-contract hash、UTC 时间和 wall/CPU/RSS；predictions.jsonl 均由 evaluator 独立重算并与 metrics.json 完全一致
- [x] 旧 B00_*_formal artifact 保留但标记为 pre-provenance，不作为 formal 比较来源；仅锁定 B00 比较记录，不勾选 Gate B

#### B01/B02 formal validation 比较（2026-08-10）

- [x] artifacts：B01_xgb_formal__907103f4f291、B02_logistic_formal__10145d8115b2（均在 results/processed/behavior_nn_v1/runs/）
- [x] 两者均仅用 train 拟合（role=13,754；family=4,712）、仅在 validation 评估（role=2,029；family=615）；test_rows_modelled=0，predictions 的 split 唯一为 validation
- [x] B01（XGBoost）validation：role top-1=0.9202、family top-1=0.7610、joint=0.8501；role/family NLL=0.2546/0.5675
- [x] B02（LogisticRegression）validation：role top-1=0.8605、family top-1=0.7496、joint=0.7829；role/family NLL=0.3693/0.5860
- [x] 两份均记录静态特征维度 role/family=71/66、代码/data/split/contract provenance 与 wall/CPU/RSS；family 固定 6 类宽度，当前训练缺失的 other 列显式为零
- [x] 两份 predictions 均由 evaluator 独立重算并与 metrics.json 完全一致；B01 当前仅是 B00--B02 同口径 validation 锚点，不构成 final 或跨场景泛化结论

#### B03 Static MLP formal validation 比较（2026-08-10）

- [x] artifacts：B03_static_mlp_seed11__b7bd16a7e2fe、B03_static_mlp_seed22__57b07a95b8d9、B03_static_mlp_seed33__e22f985cffab（均在 results/processed/behavior_nn_v1/runs/）
- [x] 三份均仅用 train 拟合（role=13,754；family=4,712）、仅在 validation 评估（role=2,029；family=615）；history 仅含 train+validation 的 15,783 条 role 事件，test_rows_modelled=0，predictions 的 split 唯一为 validation
- [x] 三份均独立验证 config/checkpoint、固定类别宽度（role/family=4/6）、代码 hash 与当前 B03 源码一致；逐样本 predictions 均由 evaluator 复算并与 metrics.json 完全一致
- [x] joint validation（seed 11/22/33）=0.8506/0.8506/0.8517，均值±样本标准差=0.8510±0.0006；role top-1=0.9179±0.0010，family top-1=0.7713±0.0025
- [x] 相对 B01（joint=0.8501），B03 joint 均值仅高 0.0009（0.09 个百分点），与 B03 seed 波动同量级且 role top-1 低于 B01；当前结论为“B03 与 B01 持平，未证明稳定优于 B01”，不作为后续模型升级依据

### Gate B：允许宣称“修正复现完成”

- [ ] XGB、MLP、FT、GRU、LSTM、OOF、MLP residual、GRU residual 同口径完成
- [ ] 每个 NN 3 seeds
- [ ] 逐样本 predictions 可重算全部指标
- [ ] 新旧差异表完成
- [ ] 作废数字不再进入 headline

### Gate C：允许继续投入序列 NN

- [ ] full history 优于 static/no-history
- [ ] shuffled history 明显下降
- [ ] GRU residual 优于 MLP residual
- [ ] hard subset/long prefix 有一致增益
- [ ] provenance-off 不出现不可接受崩塌

### Gate D：允许继续投入多模态

- [ ] structured backbone 已锁定
- [ ] visual join/missing mask 审计通过
- [ ] InternVideo 正向增益稳定
- [ ] shuffled visual 负对照下降
- [ ] 额外 feature extraction 成本已记录

### Gate E：允许公开预训练升级为主实验

- [ ] public 数据规模和 case split 过门
- [ ] ontology mapping 审计通过
- [ ] scratch/frozen/fine-tune 对照完整
- [ ] 3 seeds/多折方向一致

### Gate F：允许最终结论

- [ ] 模型和阈值已冻结
- [ ] 新 final manifest 未参与开发
- [ ] final 只评估一次
- [ ] paired video CI 完成
- [ ] 跨 planner/stack 结果完成
- [ ] 限制与失败如实记录

## 15. 规划自检：发现的问题与本版修正

本规划在落地前按 correctness、ML leakage、metric、reproducibility、resource 和 architecture 六个方向反向审查，发现并修正以下问题：

### [P0] 不能继续使用当前 test 做最终选择

旧计划仍围绕固定 30-video test 比较模型，但该集合已经被多轮观察。修正：把 300 视频降为 development pool，最终另设未见集合；没有新集合时只能报告 CV。

### [P0] padding 问题不只存在于旧 benchmark_action_family

`nn_bilstm_retest.py`、`nn_both_gru.py` 和 `nn_residual_model.py` 同样右 padding 后读取最终隐藏状态。修正：所有序列数字统一作废待重跑，R1 加 padding invariance 自动测试。

### [P1] metadata join 会污染接近一半测试输入

旧脚本按 video 最后一条 prefix 覆盖 metadata，有效 domain 会被空记录覆盖。修正：稳定 event/task join、空值禁止覆盖、ambiguous join 报告。

### [P1] route/layer 指标分母会被非 execute 样本抬高

旧实现中大量非 execute 样本的 executor 为 `None`，容易得到表面 0.95。修正：route 只在 execute 子集，正式主指标使用 joint pipeline。

### [P1] residual 的 temporal 改善可能只是概率校准

旧计划直接从 XGB 对比 GRU residual，缺少无序列 residual 和 cost-sensitive XGB。修正：强制加入 B09/B12，并以 shuffled-history 证明序列贡献。

### [P1] cost matrix 与资源阶段存在循环论证

资源模型尚未锁定时直接用人工 5/3 权重选模型，容易按结果调权。修正：cost 降级 secondary，权重在 final 前由 train-only 资源差异或明确业务规则冻结。

### [P1] provenance 可能让模型只记住 baseline 模板

现有 role 路径高度由 baseline、step 和 model stack 决定。修正：provenance-on/off、position-off、leave-one-planner/stack-out 和 hard subset 作为强制实验。

### [P1] 当前公开预训练数据规模不足

约 2,746 个 prefix 只能做 pipeline pilot。修正：设置数万 prefix/数千 case 门槛，按 process/tool-agent 来源消融；N1 未通过时停止 N4。

### [P1] 当前链式 VideoMME 不足以证明 GNN 优势

旧计划可能过早训练实例 GNN。修正：VideoMME 只做 train-only graph decoding；GNN 等真实电梯 branch/verify/evidence edge。

### [P2] 多模态覆盖与行为数据不一一对应

prefix v0.9 与 role dataset 的 run/row 覆盖不同。修正：N3 前单独做 join coverage、missing reason 和 available-only/all-with-mask 报告，不默认补零即完成。

### [P2] 旧 artifact 固定文件名易覆盖且缺乏 provenance

旧 JSON 常只保存 metrics。修正：统一 `behavior_nn_v1` 根目录，每个 run 保存 config/hash/predictions/environment，不静默覆盖。

### [P2] 计划模型过多，可能在基础错误未修时浪费算力

修正：使用 Gate A–F；GRU 先验证序列，过门后才扩展 Transformer、多模态、公共预训练和 GNN。

经上述修正，本规划不存在已知的直接 future/target leakage 设计；剩余未知主要是 canonical event join 的实际覆盖、新视频最终集的可获得性、rare class 支持和真实电梯 graph 数据量，这些必须在相应 gate 用实测解决，不能提前假定。

## 16. 推荐的下一次执行边界

下一次只执行 B04（FT-Transformer）进入正式训练前的实现与验证：

1. 只读检查旧 FT-Transformer 的输入、padding、metadata 和评估路径，不复用旧数值；
2. 锁定 B04 的 train-only 特征契约、CUDA/seed/hyperparameter 配置与 artifact 字段；
3. 为 split、padding、固定类别宽度、无 test 可见性和预测可复算新增最小单元测试；
4. 在通过全量测试后只做一次 GPU smoke，审计其 config、code hash、prediction split 和 metrics；
5. 仅在上述审计通过且得到明确确认后，串行运行 11/22/33 三个 formal seed；
6. 在 B04 三 seed 完成前不进入序列、残差、多模态或 test 集评估；
7. B03 仅是无序列 NN 对照，不能以 0.09pp validation 差异宣称胜过 B01。

这样可以避免再次出现“训练全部跑完后才发现输入或指标有问题”的情况。



## 2026-08-11 当前状态更新（N1）

此前文档中“下一次只执行 B04”的段落属于 2026-08-10 formal 运行前的历史状态，现由本节覆盖。

当前已完成：

- B04-B10 19 个 formal validation run；
- OOF cache 严格来源/主键校验；
- N1 paired video bootstrap（8 个比较，5,000 reps，seed 42）；
- B05 zero/shuffle 三 seed 消融；
- v5 visual provenance 统一复核（candidate phase1_report、artifact_manifest 与 31 个 hash）；
- 远端 R2 测试 154/154 通过；
- finalist CV：development-only、canonical video id 分组的 5-fold，55 个独立运行（B01=5，B03=25，B05=25），test 未读取。

当前 screening 结论：

- B05 相对 B01 和 B03 的 paired video CI 下界均高于 0.005；
- B05 full 相对 zero history 的差异为 +2.14pp，95% CI [+1.01,+3.31]pp；
- B05 full 相对 shuffle history 的差异为 +0.61pp，95% CI [+0.08,+1.24]pp；
- finalist CV primary joint：B01=0.8355±0.0059（n=5），B03=0.8395±0.0060（n=15），B05=0.8754±0.0060（n=15）；B05 在 5 个折上相对 B01/B03 的同折提升均为正；
- finalist CV sensitivity（seed 44/55 smoke）未进入 primary 选择：B03=0.8389±0.0061（n=25），B05=0.8750±0.0064（n=25）。

finalist CV 产物：

- results/processed/behavior_nn_v1_r2/comparisons/finalist_cv/finalist_cv.json
- results/processed/behavior_nn_v1_r2/comparisons/finalist_cv/finalist_cv.md
- fold manifest SHA256：ae478f8e08f3e5cbaa189933d0292bce41e7bc4096c34839fcddf908927c344a

这些结果属于 development validation screening；锁定配置后的 final holdout 已在本文件后续章节完成，不能回看 holdout 继续选择模型或阈值。当前仍未完成：

1. 资源预测器训练/校准；
2. 并发 workload、资源挤占和调度 replay；
3. 将行为预测结果接入调度器并完成系统级验收；
4. 真实电梯 graph/verify 数据接入后的跨场景迁移验证。

正式结果说明见：

docs/n1_validation_results_20260811.md


## 2026-08-11 final holdout 已完成

N1 的 finalist CV screening 和锁定的 final holdout 均已完成。40 个未见视频分别运行 st_fixed、star、langgraph_react，得到 120/120 条成功轨迹；holdout 数据为 1,380 条 role 行和 526 条 family/tool 行，join 完整且 test_targets_used=false。

final holdout 固定使用 finalist CV 主运行的 epoch：B03=18、B05=19。联合 role+family accuracy：B01=0.7406（n=1），B03=0.7377±0.0026（n=3），B05=0.8046±0.0048（n=3）；B05 相对 B01/B03 分别提升 6.40pp/6.69pp。7 个运行均成功，正式报告位于 results/processed/behavior_nn_v1_r2/final_holdout/evaluation/final_holdout_report.json。

从现在起行为预测器的配置、特征契约和 holdout 结果冻结。剩余实验顺序为：资源预测器训练/校准 → 并发 workload 生成 → 显存挤占与等待/OOM replay → 调度器系统级验收；不得回看 holdout 进行模型选择。
