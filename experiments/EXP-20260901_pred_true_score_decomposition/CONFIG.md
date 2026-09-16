# EXP-20260901 Pred/True score decomposition

本实验回答一个具体问题：在同一时间、同一 scheduler state、同一候选动作集合
上，`aligned_predopt_h5` 与 `aligned_trueopt_h5` 的候选筛选和每个动作的分数
分别在哪里产生差异。

## Scope

- 只读解析已经完成的 100 集 aligned H5 action audit，不重新运行 episode。
- 比较键为 `episode_id + decision_index + state_hash`；每个明细行保留
  `time_ms`，每个 candidate action 都保留完整 action identity。
- 逐 candidate 输出 predicted/truth 的 `priority/current/future/total`、误差、
  rank、top-1/tie 标记；逐 decision 输出过滤阶段计数、top-1、pairwise order
  inversion 和 score additivity。
- Truth score 只作为事后 evaluator 输入；本实验不训练、不改 scheduler policy，
  不连接远端，不读 `T_final`。

## Outputs

- `artifacts/candidate_score_comparisons.jsonl`：每个 candidate action 一行，包含
  被过滤动作的 null score 和严格可行动作的 Pred/True 完整对比。
- `artifacts/decision_comparisons.jsonl`：每个 dispatch decision 一行，包含同一
  state 的候选集、top-1、排序和误差分解。
- `artifacts/top_decision_mismatches.jsonl`：按 top-1 真实代价差排序的错例。
- `artifacts/top_candidate_score_errors.jsonl`：按 total score 误差排序的候选错例。
- `artifacts/metrics.json`：唯一数值汇总；`run_manifest.json` 记录输入哈希和输出。

## Acceptance gates

1. 输入 SHA-256、decision 数和 candidate 数可复核。
2. raw/model/strict 过滤计数与原记录一致；Pred/True score map 的 action key 集合一致。
3. Pred/True 的 `total_ms = current_ms + future_ms` 加和残差通过 `1e-6 ms` 门槛。
4. 重新按 score tuple 排序后，能复现原记录中的 Pred/True policy top-1。
