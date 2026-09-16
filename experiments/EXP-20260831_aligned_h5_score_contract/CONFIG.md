# EXP-20260831 aligned H5 score contract

本实验只回答一个问题：在动作集合、优先级规则、缓存更新近似和 H=5
保持相同的情况下，把 `PredOpt-H5` 的可部署评分和 `TrueOpt-H5` 的真值评分
对齐后，剩余差距还是多少。

## Scope

- 仅本地 CPU 仿真；不连接远端、不做真实 GPU 测量。
- 旧 `myopic`、`predopt_h5`、`trueopt_h5` 保留；新增策略为 opt-in。
- 当前动作计入 H5 总分；后续 DAG 层按相同的缓存状态推进。
- Predicted 只读取 train-only p50 runtime/load 与 p95 workspace，以及冻结的未来身份预测。
- Truth 只在审计策略和审计 evaluator 中读取模板 execution truth。
- transition profile 暂不加入，先隔离评分契约因素。

## Gates

1. 统一评分单元测试与旧审计回归测试通过。
2. 100 集 common-state action-value audit 完成且无缺失 future artifact。
3. 100 个分层状态的 full-event forced-action rollout 完成且 reference consistency 通过。
4. 10 集 paired scheduler smoke 的所有策略无 failed jobs。

数值事实写入 `metrics.json`；解释写入 `RESULT.md`。
