# EXP-20260818 R7 validation paired statistics

## Scope

Compare the frozen R7 validation policies on exactly the same episodes. Use episode-level pairing by `episode_id` and stratified resampling within the 135 registered scenario cells. This experiment does not read `T_final`.

## Inputs

- Scheduler results: `/root/autodl-tmp/scheduler/results/processed/r7_scheduler_matrix_validation_1000_final/scheduler_results.jsonl`
- RL results: `/root/autodl-tmp/scheduler/results/processed/r7_rl_matrix_20260817_fix2/`
- Validation episodes: `/root/autodl-tmp/scheduler/results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl`
- Analysis runner: `scripts/analysis/r7_paired_policy_stats.py`, SHA-256 `5492ed70b39580be0a9b503fbfd42b984c396b13bef304770de2cadeba50fa70`

## Frozen statistics

- Bootstrap unit: episode within scenario cell.
- Bootstrap count: 10,000.
- Bootstrap seed: 20260818.
- H1/H3/H5 multiple comparisons: Holm correction, separately against Myopic and Optimizer-0.
- RL primary comparison: RL-H5 versus RL-0; each seed is also reported separately.
- `p95_completion_ms` is retained only as an episode-level diagnostic because the summary-only matrix has no job-level completion values.
