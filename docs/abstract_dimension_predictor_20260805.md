# 抽象维度预测器验证计划(2026-08-05)

- 文档状态:已完成(2026-08-05 执行完毕,结果见第 7 节)
- 执行地点:远端 /root/autodl-tmp/scheduler
- 原则:只读分析,不改任何训练数据;偏离本文档时在"偏离记录"如实补充。
- 验收阈值:放宽版——抽象角色熵降 R ≥ 0.42 为通过;0.38-0.42 为部分通过(资源意图为主,角色预测降级);<0.38 不通过。

## 1. 目的

验证"在抽象维度(功能角色 + 资源意图)上预测"的可行性,支撑跨场景迁移(电梯 Phase 6)中"预测器绑定特征名"问题的解决。

## 2. 抽象维度定义

### 2.1 功能角色映射(基于实测 activity→node_type 映射,2026-08-05 核对)

| 抽象角色 | compute node_type | canonical activity | 电梯节点 |
|---|---|---|---|
| init | run_control | (无) | normalize/probe/segments/requirement_queue |
| plan | planner | (无) | evidence_planner |
| execute | videotool_spatial / videotool_temporal / videotool_generalist | sample_seek / spatial_qa / temporal_qa / object_detection / summarize | tool_router / tool_execution |
| verify | (VideoMME 无) | (无) | tool_result_validator |
| aggregate | answer_generation | answer | aggregate_decisions / generate_report |
| terminate | (run_control 收尾) | __END__ | __end__ |

注意:canonical 7 类映射到角色后仅 3 类(execute/aggregate/terminate);完整 5 类角色序列需从 compute_events 的 node_type 重建。

### 2.2 资源意图(与动作名无关,调度直接消费)

计算量级(轻/中/重)、显存档(无/小/中/大)、模型切换概率、失败概率、收尾标志——用 compute_events 实测分离度。

## 3. 验证内容(全部只读)

### 验证 A:抽象角色熵降诊断(两层)
- A1:compute_events 层面——每 run 的 node_type 序列 → 角色序列(5 类),跑熵降(一阶、+model、+baseline)
- A2:prefix 层面——target_next_activity → 角色(3 类),与细粒度 7 类(0.4041)/ coarse 4 类(0.4706)对比
- 验收:R ≥ 0.42 通过;0.38-0.42 部分通过

### 验证 B:资源意图区分度
- 按 (角色 × model) 分组的 runtime/peak/load P50/P90/P99
- 组间 runtime 中位最大比值、peak 分层(8B≈17G / 4B/3B≈7-8G / 无)
- 验收:跨组 runtime 中位比 ≥2 的组对占比高

### 验证 C:抽象角色转移结构
- 角色序列的循环数分布(plan→execute→plan→...→aggregate),供剩余时长预测

## 4. 执行步骤

1. 本规划文档落远端 docs/abstract_dimension_predictor_20260805.md
2. 脚本 scripts/abstract_dimension_analysis.py(本地写+语法检查+上传)
3. 远端运行,产物 results/processed/abstract_dimension_analysis_20260805.json
4. 结果与结论写回本文档第 6/7 节

## 5. 结论判定

- 通过 → 落"抽象角色预测器"正式方案,进迁移路线
- 部分通过 → 资源意图为主,角色预测降级
- 不通过 → 回退:预测器按场景重建,只迁移框架

## 6. 偏离记录

(执行中发现与原计划不一致之处,如实记录于此)

**2026-08-05 执行后补充**:

1. 无执行偏离:映射表、验证 A/B/C、命令均按第 3/4 节执行;产物 `results/processed/abstract_dimension_analysis_20260805.json`。
2. 数据性发现(非偏离,记录):compute_events 层面角色分布实测 4 类(init 768 / plan 3753 / execute 3911 / aggregate 1702),**无 terminate 事件**(run_control 只出现在开头);terminate 角色由 prefix 层面(A2,__END__)覆盖。5 类设计在 compute 数据上实际验证 4 类。
3. A2 的 3 类角色(execute/aggregate/terminate)一阶 R=0.52,与 A1 的 4 类 R=0.53 一致,相互印证。

## 7. 结果

(执行后填写)

**执行时间**:2026-08-05(远端);输入:compute_events(10,134 事件 / 768 runs)+ prefix(8,268 行 / 1,144 runs)。

### 7.1 验证 A:抽象角色熵降(R)

| 层面 | 条件 | R | 对比 |
|---|---|---|---|
| A1 compute 角色序列(4 类) | 一阶 | **0.5278** | 细粒度 7 类 0.4041 / coarse 4 类 0.4706 |
| A1 | +model | **0.7525** | |
| A1 | +model+baseline | **0.8149** | |
| A2 prefix 角色序列(3 类) | 一阶 | **0.5200** | |
| A2 | +planner+baseline | 0.5312 | |

**判定:通过**(阈值 0.42;实测 0.52-0.81,远超)。抽象角色维度比细粒度动作更可预测——"预测什么"从 7 类动作抽象到 4 类角色后,R 提升 0.12;加入 model 条件后达 0.75-0.81。

### 7.2 验证 B:资源意图区分度

| 角色×模型 | n | runtime P50 | runtime P90 | peak P99 |
|---|---|---|---|---|
| execute × Qwen3-VL-8B | 2133 | 3.2s | 10.9s | 19.1G |
| plan × Qwen3-VL-8B | 2018 | 4.2s | 15.3s | 19.1G |
| execute × cpu-adapter | 1764 | **0.1ms** | 0.1ms | 0 |
| plan × Qwen3-4B | 1735 | 6.6s | 21.1s | 8.1G |
| aggregate × Qwen3-VL-8B | 902 | 0.25s | 58.1s | 17.0G |
| aggregate × Qwen2.5-VL-3B | 587 | 13.5s | 74.0s | 7.3G |
| execute × yolo11x | 6 | 7.3s | 8.0s | 0.4G |

组间 runtime P50 最大比值 **143,678x**(cpu 组 vs 3B 组);peak 按模型分层(8B≈19G / 4B≈8G / 3B≈7G / yolo≈0.4G / cpu=0)。

**判定:通过**。(角色×模型)是资源分布的强索引,资源意图(计算量级/显存档)分离清晰。

### 7.3 验证 C:循环结构

plan→execute 循环数(per run):0 次 136 / 1-5 次 221 / **6 次 411**(54%)。循环数分布有强结构 → 剩余时长 ≈ 循环数 × 单循环成本 可行。

### 7.4 结论

**通过,进入"抽象角色预测器"正式方案**:

1. 抽象角色维度(init/plan/execute/aggregate/terminate,verify 预留)可预测性 R=0.52-0.81,显著高于细粒度;
2. 资源意图由 (角色×模型) 分位表直接索引,跨场景(电梯)可复用——模型与角色词表共享;
3. 电梯接入时:14 节点 → 同一角色词表映射,预测器框架直接复用;verify 角色为电梯独有,用模板先验 + 冷启动兜底;
4. 迁移路线更新:预测器在抽象维度训练(VideoMME)→ 电梯直接复用;细粒度 7 类仅作场景内研究输出。

**后续步骤**(另文档推进):实现"抽象角色 + 资源意图"预测器(复用 v0.4 层级计数框架,换角色词表),在 VideoMME 上训练并锁 baseline,电梯接入时只换映射表。
