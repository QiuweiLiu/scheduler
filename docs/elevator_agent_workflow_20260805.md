# 电梯检测视频 Agent 流程(2026-08-05)

- 来源:用户提供的流程图(2026-08-05)
- 目的:作为 Phase 6 电梯领域迁移的资源预测/调度分析基础
- 说明:节点名为图中原样转录;图中未出现具体工具名(Detect/Track/OCR/Pose 等)与 Critic/Analyzer/Repairer,仅节点级流程

## 1. 流程图结构(原样转录)

### 1.1 主链路(初始化)

```text
__start__
  ↓
normalize_input_node          输入规范化
  ↓
probe_video_node              视频探查
  ↓
build_role_segments_node      角色片段构建
  ↓
initialize_requirement_queue_node   初始化需求队列
  ↓
evidence_planner_node         证据规划
```

### 1.2 工具执行循环(核心)

```text
evidence_planner_node
  ↓
tool_router_node ── execute_tool ──► tool_execution_node
  ↓ (finish_requirement)                ↓
requirement_router_node          tool_result_validator_node
  ↓ (continue ↺)                        ↓
  └──────────► evidence_planner_node    update_evidence_graph_node
                                            ↓
                                     requirement_router_node
                                       ├─ continue ↺ → evidence_planner_node
                                       └─ done → aggregate_decisions_node
```

### 1.3 汇总与结束

```text
aggregate_decisions_node
  ↓
generate_report_node
  ↓
__end__
```

另外:tool_router_node 有一条 `done` 虚线分支直接连到 aggregate_decisions_node。

### 1.4 节点清单

| 序号 | 节点 | 阶段 |
|---|---|---|
| 1 | __start__ | 开始 |
| 2 | normalize_input_node | 初始化 |
| 3 | probe_video_node | 初始化 |
| 4 | build_role_segments_node | 初始化 |
| 5 | initialize_requirement_queue_node | 初始化 |
| 6 | evidence_planner_node | 证据规划 |
| 7 | tool_router_node | 工具执行 |
| 8 | tool_execution_node | 工具执行 |
| 9 | tool_result_validator_node | 工具执行 |
| 10 | update_evidence_graph_node | 工具执行 |
| 11 | requirement_router_node | 路由 |
| 12 | aggregate_decisions_node | 汇总 |
| 13 | generate_report_node | 汇总 |
| 14 | __end__ | 结束 |

## 2. 对资源预测的意义

1. **节点类型是电梯场景的资源分组维度**:14 个节点即可作为 node_type 分组的替代(对应现有 VideoMME 场景的 planner/spatial/answer 等分组)。模型+节点是显存与 runtime 的第一决定因素。
2. **循环是剩余时长预测的关键**:`evidence_planner → tool_* → requirement_router --continue--> evidence_planner` 构成主循环。剩余时长 ≈ 当前节点成本 + 剩余循环数 × 单循环成本。单循环成本 = tool_execution + validator + evidence graph 更新(多为 CPU/轻量)+ 一次 planner。
3. **路径长度不确定**:`continue / done / finish_requirement` 三条退出路径 → 循环次数分布重尾 → 调度需分位(P90/P99)而非均值。
4. **输入规模节点**:probe_video(整段视频探查,帧数相关)、build_role_segments(角色片段构建,分段数相关)是输入规模特征的主要来源,直接喂 roofline 层。
5. **工具执行为黑盒**:tool_execution_node 内部是 Detect/Track/OCR/Pose 等工具(图中未展开),每个工具的资源曲线(显存/耗时)需要在电梯场景实测校准——对应"少量校准层"。

## 3. 对调度的意义

- 文档定义的 score 函数:`queue_time + predicted_runtime + model_load_cost + deadline_risk + future_memory_conflict_risk - reuse_gain`
- predicted_runtime 在此流程 = 当前节点 + 循环数 × 单循环成本(roofline 估计)
- model_load_cost:tool_router 在不同工具间切换(检测/OCR/姿态)的模型加载成本
- deadline_risk:循环退出路径不确定 → 用循环数分布的分位
- memory_conflict:工具模型 + 主 LLM 的显存组合约束

## 4. 与 Phase 6 指导文档的衔接

- 指导文档(CODEX_WORK_GUIDE.md Phase 6):电梯材料包 = 三段视频 + 4-8 条结构化检验声明;工作流为 Detect/Track/OCR/Pose/语义工具 → Critic(支持/矛盾/证据不足/工具失败)→ Analyzer → Repairer
- 本流程图是节点级通用骨架(evidence 循环 + requirement 路由);Critic/Analyzer/Repairer 语义节点未出现在图中,预计落在 evidence_planner/requirement_router 的分支语义里,接入时需进一步映射
- 迁移验证的核心:repair 分支、重试、工具失败 —— 对应本图的 tool_result_validator(验证失败触发重试/修复路径)
