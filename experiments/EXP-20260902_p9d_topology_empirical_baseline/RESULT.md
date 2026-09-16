# EXP-20260902 P9d topology empirical baseline

## 结论

P9d 的第一阶段条件经验拓扑基线已在远端完成。它只用 P_dev/train 建立条件频数表，
在 P_dev/validation 选择条件键，在 P_dev/test 做诊断，并在冻结选择后评估
P_holdout_diag/holdout。最终选择 current_full；精确指标、支持度和各 split 结果只以
metrics.json 为准。

这个结果说明当前 identity-free 的经验分布可以产出多节点、H=5 DAG-layer top-3 场景，
但不能作为合格的 learned topology predictor：预测的未来层数和节点数在 test/holdout
整体偏少，holdout 的条件键全部落入全局回退，未见条件下的精确拓扑覆盖明显不足。
history2_full 在部分规模误差上更有利，但按预注册的 NLL 选择规则没有胜出，说明只
增加短历史并不能解决拓扑支持稀疏和分布偏移。

## 方法与边界

- 目标是从标签中的真实 future_layers 提取 identity-free signature；节点 ID、父子边、
  runtime/load/memory/status 和 execution_lane 真值不进入条件键或预测产物。
- 候选条件键为 current_full、current_structural、history2_full。所有候选共享同一
  train split 和全局回退分布；validation 只按 NLL 主指标选择。
- 输出的 prediction artifacts 只保留 sample ID、数字摘要和 identity-free top-3 场景；
  不输出真实 successor ID 或真实边。
- 远端实际运行使用已核验的 P9d 数据集，原始 trace、旧 R7 artifacts、S_train/S_val
  和 T_final 均未用于拟合或覆盖；没有启动 scheduler integration。

## 验收与下一步

本实验的本地/远端单元测试、编译检查、输出行数、概率归一、禁止字段审计和 SHA-256
回收核验均通过。它只完成 empirical baseline gate，不提升调度策略，也不解封
T_final。

下一步按计划进入 learned comparison：先在同一 P9d 边界上定义并验证 tabular topology
模型，再比较 shared causal GRU 的 behavior/topology 分头；两者都必须与该 empirical
baseline 在冻结 holdout 上比较，之后才考虑 scheduler-side inference。
