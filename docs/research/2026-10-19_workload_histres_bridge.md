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
