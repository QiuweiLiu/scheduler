# 电梯流程与当前 Trace 流程对比及迁移困难分析(2026-08-05)

- 当前流程数据:`results/processed/compute_events_core_resource_fixed_20260804.jsonl`(10,134 事件 / 768 runs,VideoMME)
- 电梯流程:用户提供流程图(见 `docs/elevator_agent_workflow_20260805.md`,14 节点)
- 目的:为 Phase 6 领域迁移做差距分析

## 1. 流程结构对比

### 1.1 当前 trace 流程(实测)

- node_type 全集仅 6 类:`run_control / planner / videotool_temporal / videotool_spatial / videotool_generalist / answer_generation`
- 实际形态为 **planner 与 videotool 交替的线性/重复链**,结尾 answer_generation:

```text
run_control → planner → videotool_temporal → planner → videotool_temporal
  → planner → videotool_spatial → ... → answer_generation → answer_generation
```

- 结构特征(实测):
  - 768 个 run 中 173 个不同 node_type 序列形态(流程多样但都在 6 类内)
  - 长度以 8 步为主(79%),无显式循环/验证/路由节点
  - `retry_of` 仅 177/10,134(1.7%),均为初始解析错误恢复,非流程级工具失败重试
  - `queue_ms > 0` 事件为 **0**(无等待/拥塞真实样本)

### 1.2 电梯流程(14 节点图)

```text
__start__ → normalize_input → probe_video → build_role_segments
  → initialize_requirement_queue → evidence_planner
  → tool_router →(execute_tool) tool_execution → tool_result_validator
  → update_evidence_graph → requirement_router
      ├─ continue ↺ evidence_planner(显式循环)
      ├─ done → aggregate_decisions
      └─ finish_requirement / done(侧路)
  → aggregate_decisions → generate_report → __end__
```

### 1.3 关键差异

| 维度 | 当前 trace(VideoMME) | 电梯流程 | 迁移影响 |
|---|---|---|---|
| 节点类型 | 6 类(planner/3 种 videotool/answer/run_control) | 14 节点(含初始化 4、验证、证据图、需求路由、汇总报告) | node_type 映射无直接对应 |
| 流程形态 | 线性/重复链,8 步为主 | **显式循环**(evidence_planner ↔ 工具) + 多分支退出 | 剩余时长从"位置×固定步"变"循环数分布" |
| 验证/重试 | 无流程级验证;retry=解析错误恢复(1.7%) | tool_result_validator 失败 → 重试/修复路径 | 失败/重试**零历史样本**可学 |
| 状态量 | task_structure + evidence 统计 + compute 前缀 | 新增 **evidence_graph(证据图)**、**requirement_queue(需求队列)** | 特征契约需扩展 |
| 任务语义 | question_type/domain(多选问答) | 4-8 条结构化检验声明(三段视频) | question_type 特征失效,需声明类型 |
| 等待/拥塞 | queue=0(无样本) | 工具间切换、验证等待可能产生 | 无法从现有数据校准 contention |
| 汇聚输出 | answer(短答案) | aggregate_decisions → generate_report(长报告) | 输出长度分布完全不同 |

## 2. 迁移困难清单(按层)

### 2.1 数据层
1. **失败/重试样本为零**:当前 768 runs 无流程级工具失败;电梯的 validator 失败→重试分支无历史可学。缓解:合成失败注入仿真,或电梯场景先采 30-50 runs。
2. **无拥塞样本**:queue=0;电梯多任务并发时的等待/冲突无法从现有数据校准。Phase 4 的 robustness 子门槛本来就未过,电梯亦然。
3. **任务语义不对应**:question_type(56% 覆盖,VideoMME 官方)与"检验声明类型"无映射;电梯声明(支持/矛盾/证据不足/工具失败)是新的语义标签,需新采集。

### 2.2 特征层
4. **证据图状态缺失**:update_evidence_graph 引入"累计证据图"状态,现有 trace 只有 evidence 统计量(coverage/frames/OCR),没有图结构状态;预测器若要条件化"当前证据覆盖",需扩展特征契约。
5. **需求队列状态缺失**:initialize_requirement_queue/requirement_router 引入"待办需求列表",现有数据无对应量(当前单问题单任务,无队列)。

### 2.3 预测器层
6. **v0.4 冷启动**:层级计数(planner/position/action 条件)在电梯节点上无历史计数 → 全部回退 global 先验,精度未知。
7. **熵降 R 需重测**:电梯流程的可预测性(循环结构下"下一步"的熵降)未测量;若 R 低,预测式调度收益存疑——迁移第一步应重跑熵降诊断。
8. **coarse 4 类失效风险**:observe/summarize/answer/end 的映射基于 VideoMME 动作;电梯的 tool_execution/validator/update_evidence_graph 需要新的粗粒度定义(如 execute/validate/aggregate/end)。

### 2.4 资源预测层
9. **可迁移的部分(好消息)**:
   - roofline 层(模型 + token/帧数 → runtime):电梯工具和 LLM 同样遵循"token 越多越慢",模型每 token 时间是物理属性,跨场景有效;
   - 显存基表(模型 17G/8G/7G):由模型决定,跨场景稳定;
   - 输出长度 probe(ProD/EGTP):特征来自模型内部状态,与领域无关。
10. **不可迁移的部分**:node_type × model 资源分位表(电梯无历史);工具(tool_execution 内 Detect/Track/OCR/Pose)资源曲线未知,需电梯场景实测校准(3-5 样本起)。

### 2.5 评估层
11. **split 口径**:VideoMME 的 48/8/8 视频 split 不适用;电梯材料包(三段视频 + 4-8 声明)需要新的视频/材料包级 split。
12. **评估指标**:P95/P99 完成时间、deadline miss 等保留;但"材料包完成时间"取代"单问题完成时间"作为主指标。

## 3. 迁移策略(分层缓解)

| 层 | 策略 |
|---|---|
| 结构抽象 | 把两种流程统一为"**循环单元**":当前 = (planner → 工具)×N → answer;电梯 = (evidence_planner → 工具 → 验证 → 路由)×N → 报告。剩余时长都 = 循环数 × 单循环成本 → 预测器只学"循环计数"与"单循环成本",跨场景可迁移 |
| 资源预测 | 以 roofline + 显存基表 + probe 为主(可迁移);node_type 查表降级为当前场景校准基线 |
| 失败/重试 | 电梯接入时先合成注入 validator 失败(概率、代价),校准重试分支的代价与概率;真实样本后补 |
| 可预测性 | 迁移第一步重跑熵降诊断(电梯 prefix 构建后),用 R 决定预测器投入程度 |
| 冷启动 | 电梯场景先采 30-50 runs(覆盖 14 节点 + 失败/重试路径),再做 v0.4 式条件计数 |

## 4. 结论

- 核心差距不是"模型或硬件",而是**结构**(线性链 vs 循环+验证)与**样本**(失败/重试/拥塞为零)
- 可迁移资产:模型/硬件物理参数(roofline)、显存基表、输出长度 probe、循环抽象、调度 score 框架
- 必须重做:电梯 prefix 构建 + 熵降测量、coarse 粒度定义、材料包 split、30-50 runs 冷启动采集
- 风险点:若电梯流程熵降 R 显著低于 VideoMME(循环结构更自由),预测式调度的收益需要重新验证(H4 假设在电梯场景可能不成立)
