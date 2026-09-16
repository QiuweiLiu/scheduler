# Result

状态：`passed_diagnostic`。本地编译、全量 58 项 unittest、action-value audit、
full-event rollout audit 和 10 集 paired smoke 均通过；本实验不解封
`T_final`，也不升级任何策略为正式候选。

## 做了什么

本实验的判定重点不是把 `aligned_trueopt_h5` 当成可部署策略，而是检查：

1. 统一契约是否让 Predicted/Truth 的分数只在信息源上不同；
2. `aligned_predopt_h5` 相对旧 `predopt_h5` 是否更接近共享真值排序；
3. 统一后仍然存在的差距来自预测误差、候选动作表达能力，还是 full-event
   rollout 与五层 surrogate 的目标差异。

统一契约固定了当前动作计入、H=5 后继层、resident-plus-workspace 缓存推进、
硬优先级和 tie-break；Predicted 只使用 train-only 统计与未来身份预测，Truth
只在审计 evaluator 中使用模板 execution truth。transition profile 关闭，
因此本轮只隔离评分契约。

## 结果解释

- 统一后的 `aligned_trueopt_h5` 与它所定义的 truth key 完全一致，说明评分实现和
  审计参考没有再发生“同名不同排序”的问题。
- `aligned_predopt_h5` 相比旧 `predopt_h5` 在 common-state H5 排名上只有很小改善，
  在 full-event forced-action 结果上没有实质改善；因此旧的“评分口径不一致”确实
  存在，但不是剩余收益的主要来源。
- full-event 结果与 H5 排名仍明显不同，说明主要剩余问题是五层 surrogate 与完整
  episode continuation/JCT 目标不一致，且之后仍由 Myopic 继续调度。
- 10 集 smoke 证明新 opt-in 策略可以执行且不破坏容量/失败门；它不是方法选择的
  统计依据。旧策略仍保持为兼容性基线，`aligned_predopt_h5` 不提升为正式候选。

## Gates

所有本轮 gate 均通过：契约/回归测试、100 集/15,364 decision action audit、
100 状态/774 分支 full-event rollout、10 集/50 行 smoke、0 failed jobs、0
future artifact 缺失，以及最大模拟峰值显存低于 32,760MB。详细数值事实见
本目录的 `metrics.json` 和各 artifact 子目录的 `metrics.json`。

## 下一步

保持 WAIT/RESERVE、prefetch、preemption、多 GPU 资源契约、PPO 和 `T_final` 封存。
如果继续主线，应优先重新定义可部署的 full-episode surrogate/terminal objective，
而不是继续调 H5 分数权重或扩大动作空间。
