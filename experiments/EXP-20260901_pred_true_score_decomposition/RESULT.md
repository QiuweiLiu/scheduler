# Result

状态：`passed_diagnostic`。本地只读审计已完成；没有重放 episode、连接远端或读取
`T_final`。

## 审计范围

输入是 `EXP-20260831_aligned_h5_score_contract` 的 100 集 common-state action
audit。比较键为 `episode_id + decision_index + state_hash`，这些状态来自原审计的
Myopic reference trajectory，因此 Pred/True 在同一个时间和同一个 scheduler state
上比较，避免了两条策略各自走到不同状态造成的混淆。

覆盖 15,364 个 decision、44,690 个 candidate action，平均每个 decision 有 2.909
个候选。每个 candidate 的 Pred/True raw score 都与输入记录逐项复核一致。

## 结果

1. 候选筛选不是问题。所有 44,690 行都通过 raw/model/strict 三层过滤；过滤计数和
   输入记录一致，Pred/True score map 在 15,364 个 state 上 action key 集合完全相同，
   没有一个 filter/set/presence mismatch。
2. priority 和 tie-break 不是问题。44,690 个 candidate 的 priority/tie 字段完全
   共享；Pred/True top-1 的 priority 一致率为 100%。`total_ms = current_ms + future_ms`
   对所有 candidate 成立，最大加和残差约 `2.91e-11 ms`。
3. 真正的差异在 score 数值和相对排序。Pred 与 True 的确定性 top-1 一致率为
   `65.1979%`，考虑真值 total 平分后的 tie-aware 一致率为 `66.0570%`；同优先级
   动作错选率为 `34.8021%`。候选两两排序翻转为 `29,683 / 75,964 = 39.0751%`，
   多候选 decision 的平均 Spearman 为 `0.2963`。
4. 误差主要来自 future，不是 current。candidate-level mean absolute error 为：
   current `7,650.2 ms`、future `38,531.9 ms`、total `35,520.9 ms`；其中
   `41,318 / 44,690 = 92.45%` 的 candidate 是 future 误差大于 current 误差。Pred
   减 True 的 signed bias 为 current `+2,729.1 ms`、future `-27,729.1 ms`、total
   `-25,000.0 ms`。即预测侧整体低估了后续代价。
5. 误差最集中在 Qwen3-4B planner。该模型的 14,397 个 candidate 的 future
   mean absolute error 为 `49,724.9 ms`，signed bias 为 `-49,724.9 ms`；以 Pred
   top-1 为分组的 5,199 个 decision 中，exact top-1 只有 `51.5291%`，Pred top-1
   的 future mean absolute error 为 `61,945.2 ms`。

## 找到的根因

当前所谓 aligned H5 仍没有真正对齐 H5 的未来拓扑单位：

- Pred 侧的 `future_h5` 对 1,546 个候选 node 全部给出 3 个 synthetic scenario，
  每个 scenario 固定 5 个 event step，且 `synthetic_rollout=true`。
- True 侧的 `aligned_h5_score` 是从当前 node 向后展开 5 个 DAG layer；每层可以有
  多个 successor。按 44,690 次 candidate occurrence 加权，真值平均包含 `5.3889`
  个 DAG node，而预测固定为 `5.0` 个 event step；真值 node 数大于预测 event 数的
  情况占 `20,561 / 44,690 = 46.0081%`。
- Qwen3-4B planner 的真值平均后继数为 `9.1858`，预测仍固定 5 个 event；其
  `11,205 / 14,397 = 77.8287%` 的 candidate occurrence 处于真值后继数更大的
  情况。典型错例是 `validation_000008`, decision `104`：Pred 把 Qwen3-4B
  planner 评为 `17,856.754 ms`，True 为 `139,686.955 ms`，future 单项低估
  `120,884.070 ms`，因此把它排到第一名。

所以现在不能把问题简单归结为“调度器不会利用预测”，也不能直接把剩余差距全归因于
完整 episode/JCT。第一优先级是先让 Pred 和 True 使用同一个未来单位：要么预测
真实的 5-layer DAG continuation，要么把两边都改成明确的 5-event continuation，
并重新运行本审计。修复后，才有资格继续判断 H5 surrogate 与 full-event/JCT 的剩余
差距。

## 验收

- 输入 SHA-256：`b26bf3bc4fc851fe6acdd2711127a78d54dfa0436c1bbfa26340f15384efde28`。
- 明细覆盖：15,364/15,364 decisions，44,690/44,690 candidates。
- 逐动作 raw score、state key、candidate count 独立复核通过。
- Pred/True 原 policy top-1 重现率均为 100%。
- `PYTHONPATH=src:. python3 -m unittest discover -s tests`：58/58 通过；两个审计
  脚本均通过 `py_compile`。

完整数值见 `metrics.json`、`artifacts/topology_metrics.json`；逐 decision 明细见
`artifacts/decision_comparisons.jsonl`，逐 candidate 明细见
`artifacts/candidate_score_comparisons.jsonl`。

下一步只做 H5 future-topology contract 修复后的同样 100 集重审计；不扩大到 300/1,000
集，不加入 WAIT/RESERVE、抢占、多 GPU，不解封 `T_final`。
