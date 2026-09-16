# Phase 3：简单预测基线验证

> 阅读口径：第 1–8 节记录早期 96 条小样本 cohort；第 9 节以后是 640-core 的固定
> 48/8/8 复验；第 13 节是当前最新的 canonical 修复、基线诊断和 v0.4 locked 结果。
> 不要把历史小样本数字与 `results/processed/trace_predictor_core_fixed_canonical_v04_20260804.json`
> 混为同一实验。

## 1. 目标

本阶段验证指导书中的 H2：任务/问题特征与已执行动作前缀，是否能比静态先验和
一/二阶 Markov 更好地预测下一动作及剩余执行成本。该阶段不验证 GPU 调度收益；
Oracle/Myopic/预测式调度属于 Phase 4。

## 2. 数据与防泄漏划分

- 输入：Phase 2 固定模型 cohort 的 96 条有效运行。
- 内容：8 个 Video-MME 视频、3 个 baseline（各 32 条）、统一
  `qwen3-vl-flash` planner/API model。
- 展开：601 个 prefix 样本，其中 505 个下一动作目标和 96 个 `__END__` 目标。
- 问题文本：96/96 条运行均可从正式 manifest 复原。
- 划分：leave-one-video-out；同一视频的所有重复运行始终在同一折，避免重复轨迹泄漏。

## 3. 基线

| 模型 | 输入 |
|---|---|
| `static_global` | 全局动作先验 |
| `static_baseline` | 按 baseline 的动作先验 |
| `markov1` | baseline + 最近一个动作 |
| `markov2` | baseline + 最近两个动作 |
| `prefix_only_nb` | baseline、前缀末端动作、前缀长度 |
| `prefix_task_nb` | `prefix_only_nb` 加问题词特征 |

同时报告剩余动作数和剩余 action-runtime 的静态均值、Markov 均值基线。

## 4. 结果

### 下一动作

| 模型 | Top-1 | Top-3 | NLL | 视频宏平均 Top-1 |
|---|---:|---:|---:|---:|
| `static_global` | 0.232945 | 0.519135 | 2.086955 | 0.228372 |
| `static_baseline` | 0.217970 | 0.627288 | 1.934316 | 0.212262 |
| `markov1` | 0.490849 | 0.851913 | 1.350541 | 0.491591 |
| `markov2` | 0.517471 | 0.845258 | 1.444317 | 0.519129 |
| `prefix_only_nb` | 0.599002 | 0.831947 | 1.342965 | 0.600345 |
| `prefix_task_nb` | 0.625624 | 0.811980 | 1.409401 | 0.626808 |

### 剩余执行成本

| 目标 | 静态均值 MAE | Markov 均值 MAE |
|---|---:|---:|
| 剩余动作数 | 1.860 | 1.438 |
| 剩余 action-runtime | 7420.8 ms | 5306.1 ms |

## 5. H2 判断

当前结论为 **`mixed_evidence`**，不是完整通过：

1. `prefix_only_nb` 相比 `markov2` 的 Top-1 提升 0.081531，NLL 降低 0.101352，
   说明执行前缀本身包含稳定的未来动作预测信号。
2. 加入问题特征后，`prefix_task_nb` 相比 `prefix_only_nb` 的 Top-1 再提升
   0.026622，视频宏平均 Top-1 提升 0.026462。
3. 但问题特征使 NLL 增加 0.066435、Top-3 下降 0.019967，尚未证明内容特征带来
   更好的概率预测或校准。

按 baseline 看，STAR 的问题特征 Top-1 从 0.5777 提升到 0.6534，但 NLL 从 1.1437
   变为 1.1901；LangGraph/ReAct 的 Top-1 从 0.4685 降至 0.4567、NLL 从 2.0221
   变为 2.1268；ST fixed 的路径固定，两个模型均为 1.0。因此总体增益不能直接解释
   为跨 baseline 的内容泛化。

## 6. 边界与下一步

- 当前可以确认：执行前缀对下一动作和剩余成本有预测价值。
- 当前不能确认：问题/视频内容特征在未见视频上带来稳定、校准良好的额外收益。
- 8 个独立视频仍然偏少；不能训练高容量预测器，也不能据此宣称 H2 已充分成立。
- 暂不进入 Phase 4 的预测式调度收益实验；先扩充独立公开视频，或补充可复查的视觉
  证据摘要后重跑 `prefix_task` 消融。

## 7. 可复查产物

评估器：`tracing/analysis/phase3_predictor_baselines.py`

远端报告：

```text
/root/autodl-tmp/scheduler/results/processed/phase3_h2_20260731/phase3_h2_report.json
/root/autodl-tmp/scheduler/results/processed/phase3_h2_20260731/phase3_h2_metrics.csv
```

## 8. 结构化 local_qwen 复验（2026-08-01）

在 32 个 Video-MME 视频上用本地 Qwen3-VL-8B/YOLO11n 生成 96 条动态 trace（3 个
baseline，各 1 次），按视频 leave-one-video-out 展开 413 个 prefix 样本。此批次只保留
结构化 task/video/prefix/evidence 特征，不使用答案、视频 ID 或未来 sidecar。

| 模型 | Top-1 | Top-3 | NLL |
|---|---:|---:|---:|
| `markov2` | 0.523002 | 0.849879 | 1.225948 |
| `prefix_structured_nb` | 0.634383 | 0.837772 | 1.319060 |
| `prefix_evidence_structured_nb` | 0.615012 | 0.847458 | 1.573601 |
| `task_prefix_structured_nb` | 0.641646 | 0.900727 | 1.436816 |
| `structured_state_nb` | 0.627119 | 0.895884 | 1.678642 |

H2 规则要求 prefix 同时改善 Markov 的 Top-1/NLL，并要求任务增量也同时改善 Top-1、
NLL 和 video-macro Top-1；实际 `prefix_structured_nb` 的 Top-1 +0.111380 但 NLL
+0.093111，任务增量 Top-1 +0.007264 但 NLL +0.117757，evidence 增量 Top-1
-0.014528 且 NLL +0.241826。因此本批次判定 `insufficient_evidence`，只能说明执行前缀
具有信号，不能说明结构化内容或证据已经带来稳定、校准良好的额外收益。

复查入口：

```text
results/processed/phase3_localqwen_h2_20260801/phase3_h2_report.json
results/processed/phase3_localqwen_h2_20260801/phase3_h2_metrics.csv
```

## 9. 2026-08-04 fixed core 复验

前述 96/32-video 结果保留为历史小样本记录；正式口径改为 640 条 core、64 个视频、
视频级 48/8/8 split。修复后的输入为：

```text
results/processed/trace_enrichment_core_fixed_20260804/prefix_samples_v0_1.jsonl
results/processed/prediction_baselines_core_fixed_20260804.json
results/processed/trace_predictor_core_fixed_20260804.json
```

主要修复是 position-conditioned fallback、train-only label vocabulary、完整未来
suffix coverage，以及 state_t enrichment。locked test 的 exact next-activity
基线/自有模型仍是高熵问题（自有 Top-1/Top-3=0.7352/0.9428）；position≥4 的
warm-prefix 达到 0.8506/0.9544。面向 Phase 4 admission 的四类
`observe/summarize/answer/end` routing 达到 0.9589/0.9946，故后续调度实验使用
coarse routing 作为准入预测，同时保留 exact tool 指标用于 H1/H3 审计，二者不混报。

## 10. 2026-08-04 final collection 口径与错误边界

主线 collection 已冻结为 640 core + 128 resource = 768 条 run。预测器的 locked
content 结果仍以 640-core 的 48/8/8 split 为准：resource 层包含 7 个 YOLO batch
条件的重复任务槽位，若不把资源条件作为显式特征，直接合并会改变样本权重。因此
resource 层进入 workload/template 和资源回归，不替换 core 的 next-tool 预测报告。

最终事件审计显示所有 768 个 run 都成功退出；177 次初始 planner JSON 解析错误中
144 次 retry 恢复，33 次 retry 仍错误但由 fallback 收尾；严格 OOM=0、正
queue/wait=0。故这些 33 条不能从数据中抹掉，也不能当作真实 run failure；它们说明
当前 parser/retry 路径仍是 Phase 4 前需要校准的工程风险。

主线预测结论不变：exact fine-grained next-tool 在 locked test 为
Top-1/Top-3=`0.7352/0.9428`，非终止行为为 `0.6931/0.9353`；warm-prefix
（position≥4）为 `0.8506/0.9544`。调度实际使用的四类 routing 在 test 为
`0.9589/0.9946`，因此只能宣称 coarse admission 预测达到目标，不能把它改写成
exact tool 预测达到 0.80/0.95。

## 11. 0.5 概率审计与 planner-aware v0.3（2026-08-04）

约 0.5 的首步概率首先是可复查的经验先验：在不区分 planner 时，train 中
`langgraph_react` 和 `star` 的 `sample_seek` 比例都约为 0.510。拆开 planner
后，Qwen3-4B 的首步 `sample_seek` 比例分别为 0.896/0.948，而
Qwen3-VL-8B 的首步最大分支只有约 0.479/0.490。这说明旧预测器把两个策略族混在
同一个 position prior 中；不是每次调用随机猜一半。旧实现的空前缀/阶段混合回退也
已修复，基线报告中的 `structured_prefix` 等名称仍是可审计 adapter，不是论文私有
预测器的原样复现。

正式 core prefix 还存在输入缺口：4,441 行中 3,801 行有 state sidecar，288/640
个 run 的历史 stack provenance 为 unknown，约 2,387 行没有 `official_task_type`，
约 1,805 行 `question_type=unknown`，原始问题文本没有进入 predictor input。该缺口
解释了为什么加入 planner 后仍有细粒度不可判定的分支；不能用 test 标签或 future
suffix 填充。

新增 `PlannerAwareTracePredictor` 后，在同一 48/8/8 locked split 上的结果为：

| 版本 | test Top-1 | test Top-3 | 非终止 Top-1 | 非终止 Top-3 | NLL |
|---|---:|---:|---:|---:|---:|
| v0.2 | 0.7352 | 0.9428 | 0.6931 | 0.9353 | 0.8181 |
| v0.3 | **0.7800** | **0.9517** | **0.7474** | 0.9457 | **0.7795** |

v0.3 达到了全体 Top-3 门槛并显著降低 NLL，但 exact Top-1 仍为
`0.7800<0.80`，非终止 Top-3 为 `0.9457<0.95`，因此 H2/exact gate 仍记为
`mixed_evidence`。四类 scheduler routing 仍为 Top-1/Top-3=`0.9589/0.9946`，
可用于 Phase 4 admission；fine-grained tool 结果单独保留，不能互相替代。

复查产物：

```text
tracing/analysis/trace_predictor.py
results/processed/trace_predictor_core_fixed_v03_20260804.json
```

为确认改进不是 v0.3 私有结构带来的偶然收益，另增加了一个最小改动的
`planner_markov1` 基线：只在 Markov-1 的上下文中加入 `planner_model_id`。test 为
Top-1/Top-3/NLL=`0.7657/0.9571/0.7544`，而原 `dyorc_markov1` 为
`0.6959/0.9428/0.8517`。因此“信息缺失”是主要问题之一；但即使 planner 字段补齐，
fine-grained Top-1 仍未达到 0.80，剩余误差来自同一 planner/position 下的真实分支和
当前 trace 数量/任务 taxonomy 不足。

新增基线报告：

```text
results/processed/prediction_baselines_core_fixed_planner_20260804.json
```

## 12. 第二次 0.5 概率审计：不是硬编码随机，也不是只补一个字段就能解决

对 640 个 core run 按公开的 Video-MME source/Dvd metadata 做了离线 join；只加入
任务 taxonomy、domain、sub-category 和选项数量，不加入答案、问题答案文本、video_id
或未来事件。640/640 run 均匹配到来源记录，但来源本身缺 taxonomy 的记录仍保留
unknown（join 后 276 个 prefix 没有 official/task taxonomy）。

在 locked test 中，v0.3 最大概率为 0.5–0.6 的 62 行实际 Top-1 为 0.726；0.6–0.7
为 0.843，0.8–0.9 和 0.9–1.0 均为 1.000。故 0.5 是当前可见状态下的混合经验
概率，不是每次抛硬币；概率越高，实际命中率也相应提高。

首步 train 计数仍是最直接的代码审计：不区分 planner 时，STAR/ReAct 的
`sample_seek` 约为 0.510；区分后，Qwen3-4B 为 0.896/0.948，Qwen3-VL-8B 的最大
分支为 0.479/0.490。前者验证 v0.2 确有 planner 条件缺失，后者说明即使修复后，
8B planner 在同一阶段仍会选择不同证据工具。locked test 的 8 个视频使每个 planner
组合首步只有 16 行，Qwen3-VL-8B 两个组合都出现 temporal 11/16，和 train 先验的
0.48–0.49 形成小样本分布偏移。

把官方 taxonomy 作为低权重附加特征的离线消融只改善 NLL/Top-3，未把 Top-1 稳定
推过 0.80；稀疏完整 task signature 反而下降。因此剩余问题是“真实 planner 分支 +
字段缺失 + 每组测试样本太少”，不是另一个尚未发现的概率计算错误。后续应补齐
planner/stack provenance 和结构化 intake 字段，并增加同 workflow family 的独立
视频/重复，再用视频级 split 或交叉验证锁定；不应把 0.5 强行改成 one-hot。

## 13. 2026-08-04 canonical 修复、v0.4 与基线置信度复验

原始工具审计发现 `ImageGridSelect/image-grid-selector`（帧采样）和
`PatchZoomer/patch-zoomer`（空间放大）曾被 `structured_state` 错归为 `other`。现已
在 `tracing/collectors/structured_state.py` 修正，并由
`scripts/repair_canonical_activity_labels.py` 离线迁移 4,441 条 prefix：1,154 行变化，
映射次数分别为 2,534 和 56；raw action、source hash、future/ground-truth marker 均
未改动，不需要重跑视频。

修复后的 canonical 输入：

```text
results/processed/trace_enrichment_core_fixed_canonical_v02_20260804/prefix_samples_v0_2.jsonl
```

计数模型的 0.5 左右概率来自 train-only 经验频数和 Laplace 平滑。例如不带 planner
时 position=0 的 `sample_seek` 约为 102/192，平滑后约 0.518；它表示多分支混合，不是
每次随机猜。新增置信度诊断会同时给出 mean/median top probability、ECE 和 confidence
bucket hit rate。test 上 `planner_markov1` 的总体 Top-1/Top-3/NLL 为
`0.8014/0.9767/0.6231`，其中 0.5–0.6 bucket 的 Top-1 为 0.533。

`structured_prefix` 之所以只有 `0.4902/0.7370`，是因为完整精确 task key 在 test
中 309/559 行无法匹配 train，只能回退到全局先验；新增分层
`structured_prefix_backoff` 后为 `0.7030/0.9678`。该 adapter 仍不声称复现论文私有
实现，正式报告为：

```text
results/processed/prediction_baselines_core_fixed_canonical_v03_20260804.json
```

在 validation 选择并锁定 test 的 v0.4 去掉 broad `position`/`baseline_position`
backoff，保留 planner、observed action tail、position 和 baseline last-action。结果：

| 版本 | test Top-1 | test Top-3 | 非终止 Top-1 | 非终止 Top-3 | NLL |
|---|---:|---:|---:|---:|---:|
| v0.2（canonical 修复后） | 0.7764 | 0.9750 | 0.7411 | 0.9729 | 0.6910 |
| v0.3（canonical 修复后） | 0.7996 | 0.9696 | 0.7704 | 0.9666 | 0.6576 |
| v0.4 | **0.8157** | **0.9803** | 0.7891 | **0.9791** | **0.5833** |

coarse admission routing 仍为 `0.9589/0.9946`。因此全体 exact 目标已过，非终止
Top-1 仍差 0.0109，不能把总体结果冒充子集结果。v0.4 报告：

```text
results/processed/trace_predictor_core_fixed_canonical_v04_20260804.json
```

剩余限制是 Qwen3-VL-8B 同阶段的真实工具分支、任务 taxonomy 缺失（约 2,387 行没有
`official_task_type`、约 1,805 行 `question_type=unknown`）以及约 288/640 个历史
run 的 stack provenance 不明确。已有 768 条 collection/workload 保持不变；新增 trace
应把 planner/stack provenance 和结构化 intake 字段作为必填，并用视频级 split 重新锁定。
