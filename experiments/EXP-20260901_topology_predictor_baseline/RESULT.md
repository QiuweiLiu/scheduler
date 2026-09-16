# EXP-20260901 topology predictor baseline

## 结论

本实验完成了一个不依赖 checkpoint 的 conditional empirical topology baseline：训练目标来自
`r7_s_train` 模板中由 `predecessor_node_ids` 重建的 DAG 层，预测对象为 `r7_s_val` 模板，
输出使用 `future_h5_layers` sidecar。预测节点只保留 scheduler-visible prototype，不携带
未来节点身份、前后继边、执行真值或资源真值。

同一 aligned-H5 action audit、topology audit 和 scheduler smoke 均通过。多节点 layer schema
已被真实消费，且最大层宽不再固定为 unary；但预测层数和节点数仍低于 True DAG 的加权参考，
因此本结果只证明契约与基线可运行，不证明 topology model 已完成，也不提升任何正式策略。

## 同一时刻 Pred/True 复核（2026-09-01）

为隔离 predictor 差异，使用同一份当前 action audit，以相同的
`episode_id + decision_index + state_hash` 分别重跑旧 `aligned_predopt_h5` 和新
`aligned_predopt_h5_layer`，两次都与 `aligned_trueopt_h5` 对比。两次输入哈希、决策数和候选数
完全相同；旧版和新版的逐 candidate/decision 明细分别见
`artifacts/old_pred_true_decomposition/` 与 `artifacts/pred_true_layer_decomposition/`。

复核结论是：候选过滤、Pred/True scored action set、priority/tie-break 和
`total = current + future` 均不是差异来源；新旧预测的 current 项逐候选完全不变，变化只来自
future H5 项。多节点 layer baseline 改善了局部 action ranking，但仍低估未来拓扑规模，且不同
模型/角色上的校准方向不一致，因此仍保留为 diagnostic policy，不进入正式矩阵。

## 证据位置

根目录 [`metrics.json`](metrics.json) 是本实验的唯一汇总数值事实源；它引用并校验以下产物：

- `artifacts/prediction_artifacts/baseline_metrics.json`
- `artifacts/prediction_artifacts/b05_future_h5_layers.jsonl.gz`
- `artifacts/action_value_audit/metrics.json`
- `artifacts/layer_topology_audit.json`
- `artifacts/smoke_10/metrics.json`

可复现入口为 [`run.sh`](run.sh)，运行日志位于 `logs/`。旧 `future_h5`、旧策略、冻结输入、
R7 和 `T_final` 均未覆盖或读取。

## 边界与下一步

本实验不使用 B05 checkpoint，不读取未来事件或目标标签作为特征，不复制 successor identity，
不改变动作空间、单节点单 GPU 语义、WAIT/RESERVE、抢占或多 GPU 语义，也未连接远端。

`aligned_predopt_h5_layer` 保留为 opt-in diagnostic policy，不进入正式矩阵。下一步只有在
评审该基线后，才考虑 learned train-only topology predictor；在此之前继续保持正式矩阵和
`T_final` 封存。
