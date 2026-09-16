# Result

## Status

Statistics gate passed; strategy-freeze gate is partial. `PredOpt-H5` is the current finite-horizon candidate, but the RL seed-consistency gate failed, so `T_final` remains sealed.

## Evidence

- 1,000 validation episodes and all 135 scenario cells were paired successfully; each cell contains 7–8 episodes.
- 10,000 fixed-seed episode bootstrap resamples completed with no missing policies or failed jobs.
- PredOpt-H1/H3/H5 versus both Myopic and Optimizer-0 have completion and deadline 95% CIs excluding zero; Holm-adjusted bootstrap p-values are below 0.001 for all six comparisons.
- PredOpt-H5 has the largest observed completion improvement: -4,481.3 ms versus Myopic and -5,011.7 ms versus Optimizer-0.
- RL-H5 versus RL-0 is not seed-consistent: completion deltas are +9,690.7 ms (seed 11), +9,327.5 ms (seed 22), and -25,090.9 ms (seed 33). Deadline deltas have the same sign inconsistency.
- RL-0 versus Myopic is also not consistent across seeds: seeds 11/22 improve completion, seed 33 worsens it.

## Decision

Freeze `H*=5` only as the current validation candidate. Do not freeze RL-H5 or start `T_final`. The next decision is either to repair/stabilize RL training and rerun its registered seed gate, or explicitly omit RL from the final deployable comparison while retaining RL as an exploratory result.

## Output

- Remote: `/root/autodl-tmp/scheduler/results/processed/r7_paired_policy_stats_20260818/paired_policy_stats.json`
- Local: `experiments/EXP-20260817_r7_workload_scheduler/reports/paired_policy_stats.json`
- Local report SHA-256: `078fb8025d7083f86f4a9e553db020048224e4b6e39219c9aa81c315010a7516`
