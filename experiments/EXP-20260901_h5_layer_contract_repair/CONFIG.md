# EXP-20260901 H5 layer contract repair

本实验只做本地 CPU 仿真，回答一个诊断问题：把 Pred 的未来表示改成
显式的 H=5 DAG-layer 结构后，评分契约是否能正常读取多节点层，并且不破坏旧路径。

## Scope

- 旧 `future_h5`、旧策略和旧实验输入只读保留。
- 新 sidecar 为 `future_h5_layers`：每个 scenario 有连续的
  `layer_offset=1..5`，每一层包含 `nodes` 列表；节点只允许 scheduler-visible
  prototype 字段，不允许 `node_id` 或 predecessor/successor 字段。
- 本地冻结 artifact 没有 B05 checkpoint/dataset，因此本轮只生成明确标记的
  `legacy_event_to_unary_layer_projection`，每层固定一个节点。它用于契约和回归验证，
  不是正式的多节点 topology predictor。
- 不连接远端，不做真实 GPU 测量，不读取 `T_final`，不改变动作空间、GPU 语义、WAIT/RESERVE、抢占或多 GPU。

## Commands

```text
PYTHONPATH=src:. python3 scripts/build_h5_layer_contract_artifacts.py \
  --input-root .scratch/action_value_audit_remote_inputs_20260825/prediction_artifacts \
  --output-root experiments/EXP-20260901_h5_layer_contract_repair/artifacts/prediction_artifacts

PYTHONPATH=src:. python3 scripts/r8_action_value_audit.py \
  --templates .scratch/action_value_audit_remote_inputs_20260825/job_templates_r7_v02.jsonl \
  --episodes .scratch/action_value_audit_remote_inputs_20260825/workload_validation_r7.jsonl \
  --future-artifacts experiments/EXP-20260901_h5_layer_contract_repair/artifacts/prediction_artifacts \
  --output-dir experiments/EXP-20260901_h5_layer_contract_repair/artifacts/action_value_audit \
  --horizon 5 --limit 100 --reference-policy myopic \
  --policies myopic,predopt_h5,trueopt_h5,aligned_predopt_h5,aligned_predopt_h5_layer,aligned_trueopt_h5 \
  --truth-contract aligned_h5
```

数值事实写入 `metrics.json`；解释写入 `RESULT.md`。正式的真实多节点拓扑预测
需要补齐 P_dev 训练输入或用户授权远端读取后另开阶段。
