# Result

## Status

The diagnostic Action-Value Audit and its sampled full-event rollout follow-up
both passed execution and input-integrity gates. They are not final policy
results and do not unlock `T_final`.

## Local-Q audit

The audit replayed a fixed Myopic reference trajectory on the predictor-unseen
validation workload. It evaluated 15,364 common dispatch states, enumerated the
strict-feasible ready-node/free-GPU actions, and applied the existing
`Q_local_H5` semantics.

| policy | local-Q top-1 agreement | mean local-Q regret | mean predicted-vs-true Spearman |
|---|---:|---:|---:|
| Myopic | 62.89% | 18,249.4 ms | -0.1470 |
| PredOpt-H5 | 63.37% | 18,404.7 ms | -0.0208 |
| PredOpt-v2-H5 | 62.48% | 19,108.3 ms | -0.0295 |

PredOpt-H5 is only 0.47 percentage points closer to the local-Q top action than
Myopic, while its mean regret is slightly higher. The largest error stratum is
Qwen3-4B/planner; cache-hit and cache-miss strata are much closer than this
model/role split. Strict width alone is not a sufficient explanation; the
width-12+ local-Q group contains only 51 decisions.

## Sampled full-event rollout audit

To test whether the local-Q definition was misleading, 100 common states were
selected deterministically from the same audit records, stratified by strict
width and PredOpt-H5 error/top-1 status. Every strict-feasible action was then
forced once and the rest of the episode was simulated with Myopic continuation:

- 100/100 target states completed;
- 717 candidate branches completed; 0 failures;
- forced reference-action consistency passed for all 100 states;
- value was full-episode mean completion after forcing one action, not a
  five-layer horizon value.

| policy | full-event top-1 agreement | mean full-event regret |
|---|---:|---:|
| Myopic | 32% | 3,401.0 ms |
| PredOpt-H5 | 32% | 3,183.1 ms |
| PredOpt-v2-H5 | 33% | 3,253.5 ms |
| local TrueOpt-H5 | 40% | 2,024.6 ms |

The important result is that local `TrueOpt-H5` reaches only 40% agreement with
the full-event optimum in this bounded diagnostic sample. Therefore it is not a
valid full-event oracle proxy, and the local-Q ranking cannot be used to argue
that WAIT is the missing action.

## Decision

WAIT/RESERVE is deferred. The current evidence does not establish that the
action space is the bottleneck; it establishes that the value definition and
deployable surrogate are not yet aligned well enough for that conclusion.
The next experiment is an objective/Q-contract audit comparing policy scores
against full-event counterfactual value. Prefetch, PPO, preemption, multi-GPU
resource contracts, and `T_final` remain frozen.

## Limitations

- Both audits use Myopic reference states, not each policy's own trajectory.
- The 100-state rollout sample is diagnostic and stratified from the first
  audit slice; it is not the registered 1,000-episode validation matrix.
- Full-event rollout uses Myopic continuation after the forced first action.
- True future is used only inside the audit evaluator, never by a deployable
  policy.

Numeric facts are canonical in `metrics.json`, `metrics.csv`, and
`rollout_audit/metrics.json`; per-decision evidence is in
`decision_records.jsonl` and `rollout_audit/rollout_decision_records.jsonl`.
