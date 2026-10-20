# 桥接完成:最新预测器(histres F1)在调度工作负载上跑出预测(2026-10-19)

## 背景

调度工作负载(Burst64,r7_workload_v041)的预测产物此前由 **B05 模型(8 月)**生成(家族精度 66.7%);最新预测器(histres F1,家族 84~86%)从未接入。本轮修好这座桥。

## 桥的设计(已验证)

- 两边**节点结构同类、词汇兼容**(model_id/node_type 直接映射;activity→action_family 用小映射表;role 由 node_type 1:1 推导);
- node_id 不重叠(不同加工线),因此**不从旧数据 join**——直接从工作负载节点序列**现场构造**预测器输入;
- 关键约定(经 5 模板试点修正后确认):**`history[-1] == current_node`**(当前节点也在历史里,其资源通道为未观测);history_resource 的 z 值用冻结的 train 拟合 `numeric_scaler.json` 同式换算;词表用 histres train 词表(未知值→UNK)。

## 全量结果(640 模板,8126 个锚点,7486 个有真实下一节点)

| 指标 | 全量 |
|---|---|
| slot-0 节点类型预测准确率 | **78.0%** |
| slot-0 模型类别预测准确率 | **80.0%** |

按真实下一节点类型:

| 类型 | n | 节点类型 acc | 模型类别 acc |
|---|---|---|---|
| answer_generation | 640 | 91.7% | 96.4% |
| planner | 3239 | 80.5% | 88.5% |
| videotool_temporal | 2508 | 97.0% | 63.1% |
| videotool_spatial | 930 | **22.3%** | 92.5% |
| videotool_generalist | 169 | **0.0%** | 36.7% |

**已知弱点(诚实记录)**:spatial/generalist 的工具选择系统性偏差(工具步的区分可能依赖任务上下文,而工作负载的 task_context 多为 UNK);temporal 的模型类别偏弱。总体 78%/80% 仍显著优于旧 B05 的家族 66.7%。

## 产物

- `artifacts/workload_predictions_F1_seed11.jsonl`:8126 行逐节点预测(slot-0 预测 + 运行时长 q50/q90 + 终止概率 + 真实值对照字段);
- `artifacts/workload_predict_report_F1_seed11.json`:质量报告;
- 脚本 `scripts/workload_histres_predict.py`(可复现;远端运行)。

## 下一步(可选)

① 把逐节点预测组装成 future_h5 式产物(多槽位),供调度实验消费;② 补跑 F0/其他种子做稳健性;③ 针对 spatial/generalist 弱点做上下文特征补全。

## 追加分析(2026-10-19 晚):上下文 A/B、与 B05 同口径对比、多步联合

### 1. 上下文到底缺不缺?——不缺,但补进去反而更差

- 工作负载模板只有 `question_sha256`(问题元数据不入模板);但**原始 trace 的 `states/state_0000.json` 里有** heuristic v0.2 派生的 task 上下文(question_type/domain/answer_type/temporal_scope/sub_category/required_modalities),按 `run_id` 可直接取;
- 受控 A/B:补上下文 vs 不补 —— **补上下文更差**(depth1 模型准确率 76.1% → 72.4%(全量)/ A/B 20 模板 80.4% → 76.1%;联合全深度正确率 **43.7% → 28.1%**);
- 原因:工作负载的上下文标签是**启发式派生**(derivation=rule_based v0.2),且 `official_task_type` 缺失→UNK,与预测器训练时的上下文分布不兼容 → 误导;
- **结论:桥的最优配置 = 不带上下文**(已作为正式产物)。

### 2. 与旧 B05 在**同一工作负载、同一批 7486 个锚点**上的直接对比

| 模型 | 下一节点 模型类别准确率 | 下一节点 角色准确率 |
|---|---|---|
| **B05(旧产物,8 月)** | **87.1%** | **94.8%** |
| histres F1(无上下文) | 80.0% | 90.4% |
| histres F1(带上下文) | 72.4% | 86.6% |

**重要结论:在调度工作负载上,B05 的预测反而更准**——因为 B05 的产物是在 **r7 同源 lineage** 上构建的(域内),而 histres 训练用的是**另一批 videomme 视频**(域外;两边视频集完全不重叠:161 vs 300)。**用 histres 原样替换 B05 不会提升,反而会降级**;真正的升级需要**在工作负载 lineage 上重训/微调**预测器。

### 3. 多步联合预测(histres F1,无上下文,全量 8126 锚点)

| 深度 | n | 节点类型 | 角色 | 家族 | 模型类别 |
|---|---|---|---|---|---|
| 1 | 7486 | 77.9% | 90.4% | 76.6% | **80.0%** |
| 2 | 6846 | 71.2% | 82.4% | 75.3% | 79.6% |
| 3 | 6206 | 66.4% | 80.7% | 66.8% | 72.0% |
| 4 | 5572 | 67.4% | 80.2% | 70.2% | 75.9% |
| 5 | 4938 | 59.2% | 74.5% | 66.9% | 66.4% |

**联合指标**:5 层全对率 **43.7%**;未来模型集合完全一致率 **50.9%**。
解读:单步可用(80%),多步逐层衰减但平稳;跨 5 步精确联合约 44%——适合"未来若干步的模型集合"类消费(51%),不适合精确逐步规划。

### 产物更新

- `artifacts/workload_predictions_F1_seed11_noctx.jsonl` + `workload_predict_report_F1_seed11_noctx.json`(正式版,无上下文);
- `artifacts/workload_predict_report_F1_seed11_ctx.json`(上下文消融,仅对照)。
