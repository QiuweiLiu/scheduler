# 预测器诊断与重建记录

更新时间：2026-08-04（768 条主线集合已冻结；resource/robustness 子门槛单独报告）

> 当前权威状态：正式集合为 640 core + 128 resource = 768 条 run；由 768 个模板生成
> train/validation/test = 20,000/1,000/1,000 episodes。canonical 修复后的 v0.4
> locked test 为 fine Top-1/Top-3=`0.8157/0.9803`，非终止子集为
> `0.7891/0.9791`；coarse scheduler routing 为 `0.9589/0.9946`。当前正式产物见
> `results/processed/trace_predictor_core_fixed_canonical_v04_20260804.json` 和
> `results/processed/prediction_baselines_core_fixed_canonical_v03_20260804.json`。

## 为什么会出现接近 0.5 的概率

这不是“模型知道一半、另一半纯猜”这一种解释，而是三个因素叠加：

1. 旧 `TransitionModel` 在空前缀时没有位置条件。它把同一基线在所有阶段的
   `spatial_qa / summarize / answer / __END__` 计数混在一起，再用统一的
   Dirichlet 平滑；首步因此会出现接近均匀的概率。
2. 旧输入只保留 canonical action 前缀，缺少显式的 `position`、模型栈和
   state_t 证据。相同的 `last_action` 在“刚开始”和“快结束”时被当成同一状态。
3. 旧 enrichment 没有读取采集器已经写出的 `states/state_XXXX.json`，因此
   coverage、frames_seen、modality、retry/error、progress 等可用的观测被丢弃。

另外，Pythia 适配器原来把 `prefix + target` 当成完整路径，导致路径覆盖率
   不是对真实未来后缀的检验；这会让路径指标偏乐观或难以解释。

## 已执行的修复

- `tracing/analysis/reproduce_prediction_baselines.py`
  - 增加 `baseline + position` 和 `position + history` 条件计数；空前缀不再
    回退到全局混杂先验。
  - 词表只从 train split 学习；test-only label 计为 OOV，不再泄漏到候选词表。
  - Pythia 适配器按 `run_id, position` 重建完整未来 suffix，覆盖率按
    candidate 是否为真实 suffix 前缀计算。
  - Retrieval 的并列排序固定为 `score, run_id, position`，避免文件顺序影响结果。
- `tracing/analysis/build_trace_enrichment.py`
  - 自动读取 `states/state_{position:04d}.json`。
  - prefix 增加 `state_features`：任务分类、视频元数据、观测进度、累计动作、
    coverage、帧数、模态/目标计数、OCR/时序证据、重试/错误和泄漏门禁。
  - 缺失 sidecar 会明确标记 `state_present=false`，不会伪造字段。
- `tracing/analysis/trace_predictor.py`
  - 新增自有的分层 count-interpolation predictor；上下文依次回退到
    prefix/task、prefix、stack+last、last、position、task+position、baseline
    和 global，并输出可审计的 context explanation。
  - 评测同时报告全体行和非终止行的 Top-1、Top-3、NLL、ECE、OOV，避免只靠
    大量 `__END__` 行把指标抬高。

## 输入/泄漏契约

允许：`baseline`、`model_stack_id`、prefix 中已经执行的 action、position、
任务 taxonomy、state_t 的累计证据和资源观测。

禁止：`video_id` 作为特征、未来 action、答案文本/答案标签、
`remaining_steps`、`remaining_runtime_ms`。后两者只用于离线标签或回归评测，
不能喂给 next-node predictor。

## 验收口径

正式评测固定为视频级 `48/8/8` split，先用 train 拟合，在 validation 调整，
最后只报告 locked test。目标为 Top-1 ≥ 0.80、Top-3 ≥ 0.95；如果非终止子集
达不到，会同时报告 all/nonterminal 数字和失败原因，不用合成样本或 test 信息补齐。

上述交付要求已在本文的“final collection 与错误审计”及后续 canonical/v0.4 小节中
补充；后续只追加新的版本记录，不覆盖这些已锁定结果。

## 2026-08-04 locked core 结果

修复后正式 core 输入为 4,441 个 prefix、640 个 run、64 个视频，固定 split
门禁通过。test 的 fine-grained 自有模型为 Top-1 0.7352、Top-3 0.9428，非终止
行 0.6931/0.9353；因此 exact next-tool 的全体 0.80/0.95 目标尚未达到，不能
把 scheduler 粗粒度结果冒充 exact 结果。按 position≥4 的 warm-prefix 子集，
fine-grained 已达到 0.8506/0.9544。

调度器实际 admission/routing 使用四类 `observe/summarize/answer/end` 时，test
为 Top-1 0.9589、Top-3 0.9946，非终止行 0.9645/0.9979，ECE 0.0216。这是当前
用于 H3/调度价值验证的预测对象；fine-grained tool 选择仍保留其独立报告。

state_t 覆盖为 3,801/4,441 prefix；缺失的 640 行均是采集器没有为终止后的
synthetic `__END__` 决策写 pre-action state，不是从未来事件倒填。修复后的报告、
enrichment 和 workload 都保留 source hash，可从 raw trace 重建。

正式 core 中仍有 288/640 个 run 的历史 `model_stack_id=unknown`。这不是把答案
泄漏进来，而是早期 manifest 没有写栈 ID；报告中保留这个事实，并把 stack 作为
部署条件而不是内容语义特征。后续若要比较模型栈增益，应补写 provenance 或做
stack-stratified split，不能把 unknown 直接重命名为 Stack A/B。

## 2026-08-04 final collection 与错误审计

资源补齐 cohort 已自然结束并由
`scripts/finalize_core_resource_fixed.sh` 收尾。最终主线集合为 **640 core + 128
resource = 768 条 run、64 个视频**；validator、重复槽位和 48/8/8 视频 split
门禁均通过。资源层的 128 条由既有 112 条 YOLO batch 候选和新 cohort 中 round-robin
选出的 16 条组成；这 16 条新候选为 STAR/ReAct 各 8 条，未用单一 baseline 填满
缺口。

事件级审计文件为
`results/processed/event_audit_core_resource_fixed_20260804.json`，口径如下：

| 范围 | run 状态 | 初始解析错误 | retry 成功 | retry 仍错误 | 严格 OOM | 正 queue/wait |
|---|---:|---:|---:|---:|---:|---:|
| core 640 | 640/640 success | 163 | 134 | 29 | 0 | 0 |
| resource 128 | 128/128 success | 14 | 10 | 4 | 0 | 0 |
| 合计 768 | 768/768 success | 177 | 144 | 33 | 0 | 0 |

因此“retry 成功率约 84%”应精确解释为初始 planner 解析错误的恢复率：
`144/177 = 81.36%`（core `134/163 = 82.21%`，resource `10/14 = 71.43%`）。
33 次 retry 仍解析失败没有被删除或伪装成成功；采集器的 fallback 使 run 仍能正常结束，
所以它们是“非致命解析失败”，不是 run-level failure。严格 OOM 只检查错误字段中的
`out of memory`/CUDA allocation 异常，避免把模型输出里的 `zoom`、`room` 等普通文本
误报为 OOM。所有事件的正 queue/wait 字段为 0，说明当前数据还没有真实等待/OOM
恢复样本；resource 数量门槛通过，但 robustness 子门槛没有通过，Phase 4 不能声称
已由真实 contention 校准。

预测器仍以 640-core 作为 locked content split，避免 112 条 YOLO batch 资源重复槽位
把同一任务过度加权；128 resource run 用于模板和 workload 的资源分层。最终 workload
已由 768 条模板重建：train/validation/test 为 20,000/1,000/1,000 episodes，
split isolation errors=0。若将 resource 重复槽位纳入预测训练，必须把 `yolo_batch`
和资源条件明确加入特征，并单独报告，不能与当前 exact next-tool 指标混在一起。

## 2026-08-04 0.5 概率专项复核与 v0.3

### 先验不是随机数，但旧输入确实混合了不同 planner

对 train-only 的 position=0 计数复核如下（只看 core，不看 test 标签）：

| 条件 | 样本数 | 最大分支 | 最大分支比例 | 熵（bit） |
|---|---:|---|---:|---:|
| `langgraph_react` | 192 | `sample_seek` | 0.510 | 1.675 |
| `star` | 192 | `sample_seek` | 0.510 | 1.799 |
| `langgraph_react + Qwen3-4B` | 96 | `sample_seek` | 0.896 | 0.574 |
| `langgraph_react + Qwen3-VL-8B` | 96 | `temporal_qa` | 0.479 | 1.667 |
| `star + Qwen3-4B` | 96 | `sample_seek` | 0.948 | 0.346 |
| `star + Qwen3-VL-8B` | 96 | `temporal_qa` | 0.490 | 1.802 |

因此旧模型在没有 planner 条件时给出约 0.5，是对两个不同策略族的混合经验先验，
不是代码在每次预测时抛硬币。真正的代码缺陷是 v0.2 的主干回退上下文没有显式使用
`planner_model_id`，把 Qwen3-4B 与 Qwen3-VL-8B 摊平了；同时旧的空前缀回退曾把不同
position/阶段混在一起，已在前一轮修复。

### 信息缺口

在 4,441 个 core prefix 中，`state_t` 覆盖 3,801 行；640 个缺失行是终止后的
synthetic `__END__`。更重要的可用信息缺口是：约 2,387 行没有
`official_task_type`，约 1,805 行的 `question_type` 仍为 `unknown`，288/640 个
core run 的历史 `model_stack_id` 是 `unknown`。采集器写入的 state 里虽然有
`action_histogram`、coverage、frames 和 modality，但 v0.2 只把它们压成了很少的
状态桶；原始问题文本也没有进入 prefix 输入。故“给的信息太少”成立，但不是未来
字段缺失：缺的是 planner provenance、任务 taxonomy 和更稳定的 pre-action route
state。

### v0.3 改动与锁定结果

新增 `PlannerAwareTracePredictor`，只使用 prefix 时已经可见的
`baseline`、`planner_model_id`、position、已执行 action 及 state 的
`last_actions`，并保留 baseline/position backoff；没有加入 video_id、答案、未来
事件、remaining steps/runtime。代码与报告分别为：

```text
tracing/analysis/trace_predictor.py
results/processed/trace_predictor_core_fixed_v03_20260804.json
```

作为独立的基线消融，`reproduce_prediction_baselines.py` 还新增了
`planner_markov1`：只给 Markov-1 增加 planner 条件，不改变其它适配器。locked test
为 Top-1/Top-3/NLL=`0.7657/0.9571/0.7544`，相对不带 planner 的 Markov-1
`0.6959/0.9428/0.8517`，说明主要收益来自正确拆分 planner 行为族，而不是把概率
硬推向某一个 action。对应报告为：

```text
results/processed/prediction_baselines_core_fixed_planner_20260804.json
```

## 2026-08-04 第二次审计：代码缺陷与信息缺口的分离

为避免把“缺字段”与“模型确实随机分支”混在一起，使用
`configs/phase3_videomme_32_source.jsonl`、`configs/phase3_videomme_expansion32_source.jsonl`
和 `configs/phase3_dvd_formal128.jsonl` 按 `video_id + question_id` 对 640 个 run 做了
离线元数据 join。该 join 只带入任务类型、domain、sub-category、选项数量等 intake
字段，不带入 answer、问题答案文本、video_id 或任何未来事件；640/640 run 均能匹配
到来源记录。由于部分旧 source 本身没有官方 taxonomy，join 后仍有 276 个 prefix
缺少 official/task taxonomy（不是凭空填充）。

### 概率是否“纯猜”

锁定 test 上按模型最大概率分桶的实际命中率如下，概率来自 train-only 计数和
Dirichlet 平滑，未读取 test 标签：

| 最大概率区间 | 行数 | 实际 Top-1 |
|---|---:|---:|
| 0.2–0.3 | 29 | 0.379 |
| 0.3–0.4 | 95 | 0.632 |
| 0.4–0.5 | 90 | 0.600 |
| 0.5–0.6 | 62 | 0.726 |
| 0.6–0.7 | 51 | 0.843 |
| 0.7–0.8 | 52 | 0.827 |
| 0.8–0.9 | 75 | 1.000 |
| 0.9–1.0 | 105 | 1.000 |

所以 0.5 不是每次抛硬币：它表示在当前可见上下文下多个合法下一节点的经验
混合概率；在 0.5–0.6 桶中仍有 0.726 的实际命中率。低置信度桶的命中率也随
置信度单调上升，说明概率至少具有可解释的排序/校准意义，不能为了好看直接改成
0.8 或 one-hot。

### 当前输入到底少了什么

首步 train 计数显示，去掉 planner 条件时 `langgraph_react`/`star` 的
`sample_seek` 比例都约为 0.510；拆开 planner 后，Qwen3-4B 的比例为 0.896/0.948，
而 Qwen3-VL-8B 的最大分支只有 0.479/0.490。这个差异确认旧代码的 planner
conditioning 缺陷。修复 v0.3 后仍未稳定达到 exact Top-1 0.80，原因是：

- 8B planner 在同一 baseline/position 内仍有 temporal/spatial/sample 等真实分支；
- 当前 locked test 只有 8 个视频，首步每个 planner 组合只有 16 行：例如 Qwen3-VL-8B
  的两个 test 组合都是 temporal 11/16，而 train 先验约 0.48–0.49，存在明显小样本
  分布波动；
- 原始 enrichment 中约 2,387 行没有 `official_task_type`、约 1,805 行
  `question_type=unknown`，288/640 个历史 run 的 stack provenance 为 unknown。

离线把官方 taxonomy 拼回后，作为低权重附加特征只能改善 NLL/Top-3，不能把 locked
Top-1 稳定推过 0.80；直接使用稀疏的完整 task signature 反而下降。因此当前瓶颈不是
一个未发现的概率计算 bug，而是 planner 行为本身的动态性、任务字段不完整和 test
样本太小共同造成的泛化上限。

### 结论与执行边界

1. v0.2 的空前缀/阶段混合回退和缺 planner 条件是已确认并已修复的代码问题；
   `planner_markov1` 的 test 改善（0.6959→0.7657）提供了独立复核。
2. v0.3 的 exact test 为 Top-1/Top-3/NLL=`0.7800/0.9517/0.7795`，不能宣称
   fine-grained 0.80 gate 已通过；coarse routing 的 `0.9589/0.9946` 才是当前
   Phase 4 admission 的合格预测对象。
3. 不需要为补 taxonomy 重跑已有视频：官方字段可以安全地离线 join；但要让 exact
   Top-1 稳定过 0.80，下一批采集必须固定 planner/stack provenance、把结构化 intake
   字段完整写入每个 run，并增加同 workflow family 的独立视频/重复，随后以视频级
   split 或交叉验证重新锁定指标。不得用 test 标签、答案或 future suffix 反推。

相对 v0.2，locked test exact 指标为：

| 版本 | Top-1 | Top-3 | 非终止 Top-1 | 非终止 Top-3 | NLL |
|---|---:|---:|---:|---:|---:|
| v0.2 | 0.7352 | 0.9428 | 0.6931 | 0.9353 | 0.8181 |
| v0.3 planner/state-aware | **0.7800** | **0.9517** | **0.7474** | 0.9457 | **0.7795** |

v0.3 已把全体 Top-3 推过 0.95，且 NLL/Top-1 明显改善；但 exact Top-1 仍比 0.80
低 0.0200，非终止 Top-3 仍低 0.0043，不能宣称 exact 目标完全通过。coarse
`observe/summarize/answer/end` routing 仍为 0.9589/0.9946，是当前调度 admission
使用的合格对象。若要让 fine-grained Top-1 稳定达到 0.80，需要在新增 trace 中固定
planner/stack provenance、补齐任务 taxonomy/问题字段，并增加同一 workflow family
的独立重复，而不是把 0.5 的先验改成硬编码的 0.8。

## 2026-08-04 第三次审计：canonical 修复、v0.4 与基线置信度

### 先修复真实的动作规范化缺陷

复查原始 `standard_tool_call` 后确认两个公开工具名被错误落入 `other`：

| 原始动作 | 实际工具语义 | 修正后的 canonical action |
|---|---|---|
| `image-grid-selector` / `ImageGridSelect` | 时间帧/图像网格采样 | `sample_seek` |
| `patch-zoomer` / `PatchZoomer` | 局部空间放大问答 | `spatial_qa` |

修复位于 `tracing/collectors/structured_state.py`，并用
`scripts/repair_canonical_activity_labels.py` 对已有 derived prefix 离线迁移，未重跑
视频。4,441 行中 1,154 行发生 canonical 变化，其中 prefix 级映射为
`image-grid-selector→sample_seek` 2,534 次、`patch-zoomer→spatial_qa` 56 次；原始
动作、source trace hash、future/ground-truth leakage marker 全部保留。该问题是数据/代码
bug，不是模型猜测，迁移后的输入为：

```text
results/processed/trace_enrichment_core_fixed_canonical_v02_20260804/prefix_samples_v0_2.jsonl
```

### 0.5 概率的正确解释

计数基线使用 train-only 频数和 Laplace 平滑；例如在不区分 planner 的
`langgraph_react + position=0` 中，`sample_seek` 为 102/192，平滑后约 0.518，
不是调用时随机抛硬币，而是“当前可见上下文下有多个合法分支”。修复后的置信度报告还
显示，test 上 `planner_markov1` 的 0.5–0.6 置信度桶 Top-1 为 0.533；该模型总体
Top-1/Top-3/NLL 为 0.8014/0.9767/0.6231。概率必须和分桶命中率一起看，不能直接把
0.5 改写成 one-hot。

`structured_prefix` 的低分则有明确实现原因：它要求完整任务键完全匹配，locked test
中 309/559 行只能回退到全局先验，Top-1/Top-3 为 0.4902/0.7370。新增的
`structured_prefix_backoff` 保留相同任务字段但按 task、last-action、position 分层
回退，Top-1/Top-3 提升到 0.7030/0.9678；它是可审计 adapter，不冒充论文私有代码。
所有基线的 confidence/ECE/分桶命中率已写入：

```text
results/processed/prediction_baselines_core_fixed_canonical_v03_20260804.json
```

### v0.4 的选择和 locked 结果

v0.4 只在 validation 上选择权重：保留 planner、position、observed action tail 和
baseline last-action contexts，去掉会重新混合 planner 家族的宽泛 `position` 与
`baseline_position` backoff，并将 `planner_last_actions` 权重设为 2。test 标签没有
参与选择；代码同时保留 v0.3 repaired-label 对照。

| 版本 | test Top-1 | test Top-3 | 非终止 Top-1 | 非终止 Top-3 | NLL |
|---|---:|---:|---:|---:|---:|
| v0.2（canonical 修复后） | 0.7764 | 0.9750 | 0.7411 | 0.9729 | 0.6910 |
| v0.3（canonical 修复后） | 0.7996 | 0.9696 | 0.7704 | 0.9666 | 0.6576 |
| v0.4 planner/state-aware | **0.8157** | **0.9803** | 0.7891 | **0.9791** | **0.5833** |

coarse scheduler routing 仍为 0.9589/0.9946。因非终止 Top-1 仍为 0.7891，严谨口径是
“全体 exact gate 已通过，非终止子集尚差 0.0109”；不能把总体指标冒充所有子集均通过。
正式报告：

```text
results/processed/trace_predictor_core_fixed_canonical_v04_20260804.json
```

剩余瓶颈仍是信息和真实动态性，而不是另一个已发现的概率计算 bug：Qwen3-VL-8B 在
同一 baseline/position 下存在 temporal/spatial/sample 多分支；约 2,387 行没有
`official_task_type`，约 1,805 行 `question_type=unknown`，并有 288/640 个历史 run
缺少明确 stack provenance。后续新增 trace 应固定 planner/stack provenance 并补齐
结构化 intake 字段；已有 768 条正式 collection/workload 无需重采。
