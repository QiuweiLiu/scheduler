# Result

## Status

Workload gate, frozen optimizer/finite-horizon validation matrix, and RL-0/RL-H5 training matrix passed. `T_final` remains sealed.

## Evidence

- Trace quality: 640/640 successful and validator-valid; source/planner answer leakage 0; quality errors 0.
- Workload audit: `gate=true`; 640 templates (`480 train`, `160 validation`), 20,000 train + 1,000 validation episodes, 777,840 jobs total; 135 scenario cells complete; template/episode errors 0; capacity/arrival/congestion checks passed.
- Simulator validation: 1,000 episodes, 3,000 result rows for RR/Myopic/Oracle, 0 failed jobs and 0 capacity violations.
- Full scheduler matrix: 1,000 episodes × 10 policies = 10,000 result rows; status `passed`, 0 failed jobs, 0 capacity violations.
- RL matrix: 2 policies × 3 seeds; each run has 1,000 train + 1,000 validation episodes, 334,721 train decisions, 1,000 validation rows, 0 failed jobs, 1,001 flushed heartbeat lines, empty stderr, and `torch_threads=1`/`torch_interop_threads=1`. The six runs are in the remote `r7_rl_matrix_20260817_fix2` directory; the local aggregate is `reports/rl_matrix_report.json`.
- Paired validation statistics: 1,000 episodes across all 135 scenario cells, 10,000 fixed-seed stratified bootstrap resamples, episode-level pairing, and Holm correction for H1/H3/H5. Report: `reports/paired_policy_stats.json`.

## Main validation aggregates

Values below are means over 1,000 validation episodes; completion and queue values are milliseconds.

| Policy | Mean completion | Deadline miss | Mean queue | Oracle relative gap |
|---|---:|---:|---:|---:|
| Round-robin | 213,001.5 | 0.11663 | 153,558.3 | 23.31% |
| Myopic | 215,116.4 | 0.10980 | 123,365.3 | 24.54% |
| Optimizer-0 | 215,646.8 | 0.10977 | 123,494.7 | 24.84% |
| PredOpt-H1 | 213,049.8 | 0.10675 | 122,116.3 | 23.34% |
| PredOpt-H3 | 212,387.9 | 0.10698 | 121,549.7 | 22.96% |
| PredOpt-H5 | 210,635.1 | 0.10469 | 120,369.8 | 21.94% |
| TrueOpt-H1 | 209,217.8 | 0.10384 | 222,444.3 | 21.12% |
| TrueOpt-H3 | 195,021.9 | 0.08561 | 126,670.0 | 12.90% |
| TrueOpt-H5 | 183,845.1 | 0.07300 | 110,661.4 | 6.43% |
| Oracle | 172,732.9 | 0.06084 | 78,938.1 | 0% |

### RL aggregates

Values below aggregate the 3,000 validation rows from seeds 11/22/33 for each learned policy. These are standalone RL results and are not yet merged into the 10-policy scheduler table.

| Policy | Mean completion | Std across seeds | Deadline miss | Mean queue | Mean GPU evictions |
|---|---:|---:|---:|---:|---:|
| RL-0 | 204,313.6 | 9,511.9 | 0.09778 | 117,769.1 | 51.13 |
| RL-H5 | 202,289.4 | 6,945.9 | 0.09661 | 117,618.4 | 65.44 |

RL-H5 is about 1.0% lower in mean completion than RL-0 on this validation split, but its eviction count is higher; this is a validation observation, not a final claim of superiority over Myopic or PredOpt-H until a registered paired comparison is run.

### Selection gate

- `PredOpt-H5` is the best H candidate on mean completion and has non-crossing 95% CIs against both Myopic and Optimizer-0 after Holm correction; it is the current `H*=5` candidate.
- RL-H5 fails the registered seed-consistency gate: seeds 11 and 22 are worse than RL-0 on completion, while seed 33 is better; deadline direction is also inconsistent. RL-0 is not consistently better than Myopic across its three seeds either.
- Therefore no RL configuration is frozen and `T_final` is not started. The next action is RL stability repair or an explicit decision to omit RL from the final deployable comparison; the current evidence is not sufficient to claim an RL advantage.

## Interpretation and caveat

PredOpt-H improves monotonically from H1 to H5 relative to Myopic on mean completion, deadline misses, queue time, and the Oracle gap, but it remains a reference validation result until the final `T_final` run. `Oracle`/`TrueOpt-H` use future truth and are not deployable policies. The 1,000-episode simulator summary notes that its compressed event log was interrupted during writing; full result rows are complete and event invariants are accepted from the independent 50-episode smoke. Keep this as an audit caveat rather than silently treating the event log as complete.

## Remote continuation

The completed RL outputs are in `/root/autodl-tmp/scheduler/results/processed/r7_rl_matrix_20260817_fix2/`; the aggregate report SHA-256 is `43a5680f18c08f4f1a4d06fa9ed1d1c6d1e8e2198bdc7695b5b69210248bdc69`. The old stalled run remains preserved under `r7_rl_matrix_20260817_stalled_audit_20260817/` and is not mixed with the accepted results.
The paired report is in `/root/autodl-tmp/scheduler/results/processed/r7_paired_policy_stats_20260818/`; its local copy SHA-256 is `078fb8025d7083f86f4a9e553db020048224e4b6e39219c9aa81c315010a7516`.
