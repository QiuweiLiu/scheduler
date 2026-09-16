# 资源预测补齐规划(2026-08-06)

- 文档状态:已完成(2026-08-06 执行完毕,结果见第 10 节)
- 执行地点:远端 /root/autodl-tmp/scheduler
- 原则:统一数据契约、统一 split、统一评估;偏离本文档时在"偏离记录"如实补充。
- 数据规范:**expansion 数据必须进入所有训练/测试集**(role_dataset_v0_3,17,303 行)
- 行为侧已就绪:角色 LightGBM(0.9151)→ 动作族 XGBoost(0.8381)→ 执行者派生 → 本规划消费这些输出

## 1. 目的

补齐资源预测器,服务调度三大决策:
- 排序:单步 runtime 分位
- 预留:peak VRAM 分位(P99,防 OOM)
- 冷启动:load_ms(模型加载代价)

## 2. 数据集

- 基础:`role_dataset_v0_2.jsonl`(17,303 行,含 raw_action/动作族/冷热/question_type/runtime_ms)
- **需补齐字段**:peak_allocated_mb、load_ms、frame_count
  - core:compute_events 有(run_id+event_index 已验证 10,134/10,134 可 join)
  - expansion:build_role_dataset 提取时已拿到(resource.peak_allocated_mb/load_ms/frame_count),但未写入输出 → **重建 role_dataset_v0_3**
- split 复用(240/30/30)

## 3. 预测目标与输出形态

| 目标 | 输出形态 | 调度用途 |
|---|---|---|
| runtime_ms(主) | **分位三元组(P50/P90/P99)** | 排序 + 剩余时长 |
| peak VRAM | P99 | 显存预留/并发 |
| load_ms | 点值(冷/热条件) | 冷启动代价 |
| 剩余时长(聚合) | Σ(预测剩余步数 × 单步分位) | 调度 score 核心 |

## 4. 特征(行为侧真实标签,离线评估;级联评估可选)

- 行为侧:动作族(6)、执行者(派生)、角色
- 模型栈 model_id、baseline
- 冷热 model_resident_before、yolo_batch
- 位置 event_index、question_type
- 输入规模 frame_count(补齐后;仅 39% 覆盖,缺失标 0)
- 双通道设计:LLM 事件(visual_qa/summarize/plan/aggregate)走 token 通道(roofline);工具事件(select_frames/temporal_ops/detect)走规模通道

## 5. 预测器矩阵

| 族 | 预测器 | 说明 |
|---|---|---|
| 统计 | 全局中位数 / 动作族×模型分位表 / 分位表+backoff | 免训练基线 |
| ML | LightGBM / XGBoost(log1p 回归) | 主候选 |
| NN | MLP 回归 / **ProD 式分布输出**(32 bin,中位+分位解码) | 分布能力,对照 |
| 解析 | **roofline**(per model: log1p(runtime) ~ frame_count) | 零历史、跨场景(电梯)兜底 |
| 检索 | KNN(相似 run 的中位) | 对照 |

输出统一:**分位三元组**;评估点预测用 P50,预留用 P99。

## 6. 评估口径

- 整体 + 分桶(按动作族×模型):MAE / RMSE / P95
- **噪声半径参照**:同 (video, baseline, model) 重复 run 相对差中位 39.8%——预测器相对 MAE 接近它即到数据上限
- **分位覆盖**:P99 预留下真实值 ≤ P99 的比例(预留可靠性)
- 对比基线:早期 model_activity_median(runtime RMSE 6,534 / peak 3,522)

## 7. 执行步骤

1. 本规划文档落远端 docs/resource_predictor_v0_6_plan_20260806.md
2. 重建 role_dataset_v0_3(补 peak/load/frame_count)
3. 实现统一资源评估框架 + 预测器矩阵(统计/ML/NN/roofline/检索)
4. 跑实验,结果 + 偏离写回本文档,上传远端
5. 剩余时长聚合(循环数 × 单循环成本)作为调度输入原型

## 8. 风险与预案

- frame_count 覆盖仅 39% → 缺失标 0 + 记录;roofline 受影响则如实报告
- expansion 无真实排队样本(queue=0)→ 调度 contention 仍需仿真(Phase 4)
- 噪声半径 39.8% 较高 → 若所有预测器接近它,选最简单(分位表/roofline)
- peak 覆盖率 58.4%(仅 LLM 事件)→ 工具事件 peak=0 处理

## 9. 偏离记录

(执行中发现与原计划不一致之处,如实记录于此)

**2026-08-06 执行后补充**:

1. 数据集为 role_dataset_v0_3(补 peak/load/frame_count;peak 54.5% / load 85.2% / frame_count 22.6%)。
2. 噪声半径重算(含 expansion 全量):同 (video, baseline, model) 重复 run 同位置 runtime 相对差中位 **29.6%**(此前 core 版 39.8%)。
3. **KNN 复核**:排除同 run 邻居后 MAE 不变(2,041ms)→ 确认非"温和泄漏",真实有效。
4. MLP / ProD 分布输出显著差于 LightGBM(与行为预测结论一致:结构化表格树模型主导);roofline 中等(75.1%)但保留零历史跨场景(电梯)价值。
5. 剩余时长聚合原型使用**真实剩余步数**(行为侧预测接入后误差会叠加,需在调度集成时重测)。
6. peak/load 用分位表即足够(模型决定,误差极小),无需训练型。

## 10. 结果

(执行后填写)

**执行时间**:2026-08-06(远端);数据 role_dataset_v0_3(17,303 行),split 240/30/30;产物 `results/processed/benchmark_resource_v06_20260806.json`、`results/processed/resource_supplement_20260806.json`。

### 10.1 runtime 预测(test 1,412 行)

| 预测器 | MAE | RMSE | 相对 MAE | 备注 |
|---|---|---|---|---|
| **R8_KNN** | **2,041ms** | 7,072 | **22.4%** | 复核无泄漏,精度上限 |
| **R3_LightGBM** | **3,034ms** | 10,588 | **33.3%** | 在线主选(毫秒级推理) |
| R4_XGBoost | 3,135ms | 10,578 | 34.4% | |
| R7_roofline | 6,849ms | 15,441 | 75.1% | 零历史,电梯兜底 |
| R1/R2 分位表 | 6,188ms | 15,448 | 67.9% | 免训练 |
| R0 全局中位 | 8,556ms | 17,806 | 93.8% | |
| R5_MLP / R6_ProD | 7,552-7,951ms | ~17.7k | — | NN 弱 |

**噪声半径参照 29.6%**:KNN(22.4%)低于噪声中位(重复 run 内仍有可预测结构);LightGBM(33.3%)接近噪声上限。

### 10.2 peak VRAM / load_ms(test)

| 目标 | 预测器 | MAE | 说明 |
|---|---|---|---|
| peak VRAM | 族×模型分位表 | **110MB** | 显存由模型决定(8B≈17G/3B≈7G),分位表近完美 |
| load_ms | 族×模型分位表 | **159ms** | 冷启动分位即可 |

### 10.3 剩余时长聚合(真实剩余步数,test 1,304 位置)

- MAE 49,214ms,相对误差 **55.9%**(单步误差 33% 累加放大)

### 10.4 结论与选择

1. **单步 runtime 主预测器:LightGBM(3,034ms, rel 33.3%)**——在线毫秒级,接近噪声上限;KNN(2,041ms)为离线精度上限参考
2. **peak/load 直接用分位表**(模型决定,误差极小,免训练)
3. **roofline 保留为电梯零历史兜底**(rel 75.1%,可接受)
4. **预留口径**:1.5×P50 预留覆盖 85.2%(LightGBM)
5. **剩余时长**:单步 LightGBM + 行为侧剩余步数预测,rel ≈ 56%(真实步数下);调度集成时重测
6. **调度输入成型**:行为侧(角色/动作族/执行者)+ 资源侧(单步 P50/P90/P99 + peak P99 + load)+ 剩余时长聚合 → score 函数五要素齐备

**下一步**:调度 score 函数集成(HexAGenT 风险分 + HeraSys 长任务保底 + Pythia 分位预留),在 20k workload 仿真上对比 myopic/rule。
