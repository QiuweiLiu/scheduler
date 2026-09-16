# P9d 拓扑标签契约 v3.1（bounded future execution-trace contract）

状态：**frozen（2026-09-10）**。按网页评审 4 项修改更新；builder v3 策略 + 12/12 本地测试通过；重建 `topology_predictor_p9d_v3` 全部 P0 gates 0 violations；登记 `data/manifests/topology_predictor_p9d_v3.json`；features 与 v2 逐字节一致（解压 SHA-256 四 split 相等）。
适用范围：P9d predictor block 的未来结构/内容标签（`topology_predictor_p9d_v3`）。
不改动：原始 trace、v1/v2 数据集、模型输入契约、`S_*/T_final`。

## 1. 证据基础

- 预测器池 1,360 条 run（`langgraph_react` 596 / `star` 596 / `st_fixed` 168）+ R7 调度池 648 条 run，共 2,008 条已采集 run 全量只读审计。
- 三类 workflow 在事件粒度均为**顶层串行控制流**：一次迭代最多一个工具；`parent_step_ids` 只出现 `()` 或 `[N−1]`；链上步号只出现 same 或 +1；answer 恒为最后事件。
- trace 内可恢复的确定性信号：`standard_tool_call.id = compat-{star|react}-step-N`（与本步号 100% 一致）；planner `parsed_decision.tool_name` 与同一步工具动作 100% 一致；`retry_of` 与失败事件同一步；`Summarizer` 内部调用答案模型形成嵌套事件（位于工具时间窗内）。
- v2 规则（父步骤→最后事件）把同一步 planner 与其工具变成同层兄弟：train 13,754 行中 5,856 行含该假并行（18,028 对），工具前驱 91.6% 指向上一步工具。

## 2. 重建原则（评审修改 1）

> 边 = 经源码语义与独立审计验证的**顶层串行控制流**中相邻可调度/终止操作的直接执行依赖。
> 文件顺序仅是该已验证执行序列的序列化表示，不是因果来源。

- 对每个 run 先做 **Seriality gate**；任何不满足串行不变量的 run 一律 **fail-closed**（拒绝产出该数据集，而不是强行链化）。
- 当前支持的 workflow：`star`、`langgraph_react`（planner 兼容循环）、`st_fixed`（固定序列）。未登记的新 workflow → fail-closed。

## 3. 节点本体（评审 P0-2；R0 前必须冻结）

| 节点 | 是否拓扑节点 | resource_applicable | 说明 |
|---|---|---|---|
| `run_control:baseline_start` | 否（root 锚点） | 否 | 仅用于 join 首个行为样本 |
| `planner.generate`（含解析失败与重试前身） | 是 | 是 | 一次真实模型计算；失败事件也占用资源 |
| 工具动作（`videotool_*`） | 是 | 是 | 单次工具执行 |
| 合并后的 Summarizer 节点 | 是 | 是 | `R_node = R_total(工具动作)`，已包含嵌套调用 |
| 嵌套 `generalist.generate`（Summarizer 内部） | 否（合并进父节点） | 否（不得单独核算） | 作为 `nested_calls` 元数据保留 |
| post-loop `generalist.generate` | 是 | 是 | 循环结束后的独立答案模型调用 |
| 最终 `answer` 事件 | 是（链尾） | **否**（terminal marker） | `resource_applicable=false`，不占用 H=5 计算槽位，不进入 runtime/load/memory 回归 |

## 4. 重建规则

1. **节点集** = 全部 supported 事件 − `run_control` − 嵌套答案调用（合并进其 Summarizer 父节点）。
2. **边** = 沿已验证执行序列的相邻节点直接依赖（单前驱、单后继）；每条 run = 单链、单 root、终点为 `answer`。
3. **嵌套合并**：同一 step 内 `generalist.generate` 之后存在 `summarization-tool` 动作 → 该 generate 为嵌套调用；父节点获得 `merged_nested_call=true` 与 `nested_calls` 元数据；嵌套事件的时间窗必须包含于父工具窗口（有窗时断言）。
4. **重试**：失败 planner 保留为链节点；重试紧随其后（`retry_of` 指向失败事件）；学习目标侧使用 `is_retry` / `status_class`，真实 `retry_of` ID 只作 audit 字段。
5. **终局**：post-loop 答案生成为计算节点；`answer` 为 terminal marker（不占 H 槽位）。

## 5. 未来标签（H=5）

- `future_layers`：从当前节点沿链向后的**可调度计算节点**，每层 1 个节点，最多 5 层；terminal marker 不进入层。
- `width`：恒为 1，标记为 **deterministic compatibility field**（保留 schema 兼容；不作为学习/评测任务，不再报告 width accuracy）。
- `termination_status`：剩余计算节点 ≤ H → `terminated`；> H → `censored`。
- 研究表述：**bounded future execution-trace forecasting**（预测未来长度/终止、节点类型/族、资源签名、工作量规模），不再声称 "future DAG branching prediction"。

## 6. 资源语义（评审修改 2）

- 每个节点带 `resource_signature`：`{exec_class, tool_family, model_class, planner_mode, merged_nested_call, nested_model_class?}`。
- 合并节点使用 **composite signature**（外层工具类别 + 内层模型类别），不得用 `effective_model_id` 覆盖节点语义。
- 资源目标：`R_node = R_total`（父工具动作，含嵌套调用）；`nested_calls` 只作 label 侧诊断分解，**禁止 R_outer + R_nested 重复核算**。

## 7. 验收 gates（评审修改 3/6）

P0（fail-closed，构建期强制）：

1. **Seriality gate**：一迭代多工具 = 0；`parent_step_ids` 仅 `()`/`[N−1]`；链上步号仅 same/+1；未知 baseline 拒绝。
2. **Signal consistency gate**：tool step id 与步骤一致；planner `tool_name` 与同一步工具一致（star/langgraph）；`retry_of` 目标与重试相邻；无解析成功 planner 的工具仅允许已登记的 yolo 预观察模式。
3. **Nested accounting gate**：每步 ≤1 个嵌套调用；嵌套窗 ⊆ 父工具窗（有窗时）；父节点携带 composite signature；资源合计语义写死为 total。
4. **Path gate**：单 root、单后继、answer 恒最后；`cycle_count=0`。
5. **Schedulable-node gate**：全部节点显式 `resource_applicable`；terminal marker / run_control / nested 为 false。
6. **Mass preservation**：split/行数/视频数/runs 与 v2 逐 split 一致；嵌套合并与节点数守恒公式成立（节点数 = 事件数 − run_control − 嵌套数）。
7. **Leakage gate**：模型输入仍不含未来字段/ID/边/执行资源真值；与调度器池零交集（沿用 v2 校验）。
8. **Diff report**：输出 v2→v3 逐节点前驱差异分类计数（工具 / post-loop 生成 / 重试等）与样例。
9. **Reproducibility**：新目录 + manifest（策略、计数、gate 结果、源哈希）；v1/v2 只读。

P1（后置，不阻塞冻结）：嵌套"合并 vs 单独节点"的敏感性审计（节点数、剩余 H 分布、资源合计是否重复、节点类别直方图）。

## 8. Holdout 表述（评审 P0-3）

全量审计已覆盖 holdout 视频的 trace 统计，因此统一表述为：

> P_holdout_diag: contract-integrity inspected; never used for fitting, model/hyperparameter selection, or calibration.

不再声称 "completely untouched"；契约冻结后不得再依据 holdout 分布修改 v3 契约。

## 9. Scheduler 侧边界

- 本轮只冻结 predictor 侧 v3 标签契约；R7/R8 的 `job_templates`/future artifacts 存在同源问题但**不同步修改**。
- 顺序：predictor v3 → R0 → J 系列 → predictor 冻结 → （独立 gate）scheduler-side topology contract audit → scheduler integration。
- 首次 scheduler 集成前，两侧契约语义必须一致，否则 oracle/PredOpt/future cost 会使用两个"真值世界"。

## 10. 与 v1/v2 的关系

- v1/v2 数据集、所有既有实验、原始 trace 保持只读；v3 使用新目录 `results/processed/topology_predictor_p9d_v3` + 新 manifest。
- v3 的 `future_unit` 与 v2 不同（`verified_serial_control_flow_chain` vs `event_level_dag_longest_path_layer`），下游消费方必须显式选择策略，不得混用。

## 11. 冻结条件

以下全部成立后，v3.1 标记 frozen：

1. builder v3 策略实现并通过本地单元测试；
2. 重建产出通过全部 P0 gates，manifest 记录 0 violations；
3. 行数/split/视频数与 v2 一致，nested 合并计数可解释；
4. v2→v3 diff 报告生成；
5. 控制面（DECISIONS/STATE/HANDOFF）更新。
