# R8 Optimizer/RL Upgrade

## Current status

`P2_completed` — formal experiment opened; P0 first replay completed and the first
PredOpt-v2 weight choice was rejected on the frozen validation replay.

P0 execution passed its data/feasibility gate on 1,000 validation episodes and
produced a paired decision audit. The initial soft-priority weights are not an
acceptable candidate: the revised policy was slower and had a higher deadline
miss rate, despite changing actions in 94.4% of episodes. Details are in
`p0_predopt_metrics.json`, `p0_predopt_audit_report.json`, and
`p0_paired_decision_audit.jsonl`. `T_final` has not been read or executed.

The S_train calibration pilot tested three pre-registered weight settings on
300 episodes. Priority weights 1.0 and 2.0 reproduced the same behavior as the
legacy score within +124.38 ms mean completion; future weight 2.0 was +1,204.89
ms and slightly reduced deadline miss. No score-only candidate was promoted.
The compact summary is `p0_calibration_summary.json`.

## Acceptance record

- [x] P0 execution/data gate passed; initial objective candidate rejected.
- [ ] P0b S_train-only weight calibration and S_val confirmation passed.
- [x] P0b S_train-only calibration completed; no weighted-score candidate promoted.
- [x] P1 CP-RHO V1 execution/feasibility/latency audit passed.
- [x] P2 MILP sanity check passed.
- [ ] P3 expert/BC passed.
- [ ] P4 PPO/GAE five-seed gate passed.
- [ ] P5 final paired matrix passed.

## P1 bounded smoke (2026-08-18)

The fixed OR-Tools dependency and the first CP-RHO implementation passed a
5-episode remote smoke on `S_val` inputs. The run used one CP-RHO policy and
one PredOpt-v2 comparator (`H=5`), producing 10 result rows and zero failed
jobs. CP-RHO returned no timeout or fallback in this smoke; mean solver time
was 42.22 ms per decision and the maximum episode p95 solver time was 252.80
ms under the 250 ms per-call budget. The smoke is an interface/feasibility
gate only, not a performance claim: its mean completion was 179,616.41 ms for
CP-RHO versus 176,655.96 ms for PredOpt-v2 on five episodes.

Remote output (canonical):
`/root/autodl-tmp/scheduler/results/processed/r8_optimizer_rl_upgrade/p1_cp_rho_smoke_20260818/`

Local copies of the small audit files:
`cp_rho_matrix_report.json` and `scheduler_results.jsonl` in this experiment
directory. At the time of the smoke this was the pending full-matrix gate; the
completed 1,000-episode matrix is recorded below. `T_final` remains sealed.

## P1 full S_val matrix (2026-08-19)

The remote full matrix completed with `status=passed`: 1,000 validation
episodes, 6,000 policy rows, and zero failed jobs. It compared
`cp_rho_h1/h3/h5` against `predopt_v2_h1/h3/h5` under the frozen deployable
information boundary. The canonical remote output is
`/root/autodl-tmp/scheduler/results/processed/r8_optimizer_rl_upgrade/p1_cp_rho_validation_1000_20260818/`.

| policy | mean completion (ms) | deadline miss | mean GPU evictions | mean solve (ms) | timeout/fallback |
|---|---:|---:|---:|---:|---:|
| cp_rho_h1 | 223,501.77 | 0.118313 | 42.500 | 54.15 | 1,866 / 1,866 |
| cp_rho_h3 | 223,502.14 | 0.118094 | 42.564 | 140.76 | 1,760 / 1,760 |
| cp_rho_h5 | 223,464.12 | 0.118344 | 42.561 | 54.17 | 1,787 / 1,787 |
| predopt_v2_h1 | 221,468.88 | 0.116594 | 43.160 | — | — |
| predopt_v2_h3 | 219,470.25 | 0.116563 | 42.291 | — | — |
| predopt_v2_h5 | 217,524.48 | 0.112938 | 41.043 | — | — |

The P1 gate passes as an execution and audit result, but CP-RHO V1 is not a
performance candidate: all three CP-RHO variants are slower than their
PredOpt-v2 comparators, and 1,760–1,866 solver calls per policy fell back after
the time budget. This is recorded as a V1 limitation, not hidden or removed
from the result set. `T_final` remains sealed.

The final report and full result table were copied locally and verified:

- `cp_rho_matrix_report.json`: SHA-256
  `3baf61edf87e3624fc0ffb6e0047345d98722ac8945a292eed4c1fe5e4d0162e`
- `scheduler_results.jsonl`: SHA-256
  `5a47a1a608611427cc8fe85988d9bd38d6bef3fa28d8db776bf4ca8bfecc458d`
- Local result table parses as 6,000 JSONL rows across the six expected policies,
  with `failed_jobs=0`.

## P2 MILP sanity check (2026-08-19, fix2 accepted)

The short-window MILP audit used SciPy 1.15.3/HiGHS on the same candidate fields
as CP-RHO V1: current-ready node/GPU assignment, per-GPU non-overlap, estimated
runtime/load/memory, priority, and frozen future cost. It did not receive
execution runtime, memory truth, status, or hidden suffix truth. Twenty
deterministic cases with four nodes and two GPUs were solved by both models.

- CP-RHO: 20/20 `OPTIMAL`; mean solve time 39.51 ms.
- MILP/HiGHS: 20/20 `OPTIMAL`; mean solve time 40.81 ms.
- Node-level first-action agreement: 20/20 (1.0).
- Exact node+GPU agreement: 16/20 (0.8); the four disagreements selected the
  same node but a different GPU label. We retain this as an exact-action
  diagnostic and do not claim physical GPU symmetry from this small fixture.
- Maximum relative objective difference: `2.6319902025099073e-13` (gate threshold `1e-6`).
- The accepted fix2 report records `seed=0`, `time_limit_s=2.0`,
  `cwd=/root/autodl-tmp/scheduler`, `PYTHONPATH=src`, the full command, remote
  Python/SciPy/OR-Tools/HiGHS/platform metadata, and SHA-256 for the four
  result-determining source files (`cp_rho.py`, `milp_sanity.py`,
  `workload_v02_simulator.py`, and `r8_milp_sanity.py`). The JSONL retains all
  160 candidates and both selected schedules (80 per solver) for exactly-one,
  NoOverlap, first-action, and objective rechecks; all rechecks passed.

This passes the P2 feasibility/objective-agreement sanity gate for the narrow
V1 assignment/NoOverlap model and supports that implementation. It does not
validate explicit future-node DAG precedence, hard memory capacity, or full
cache residency lifecycle, which are outside V1. It does not make MILP an online method, and it does not
reverse the P1 performance result: CP-RHO V1 still has a high time-budget
fallback rate and remains below PredOpt-v2 on the full S_val matrix.

Canonical remote output:
`/root/autodl-tmp/scheduler/results/processed/r8_optimizer_rl_upgrade/p2_milp_sanity_20260819_fix2/`

Local audit files were hash-verified:

- `milp_sanity_report_fix2.json`: SHA-256
  `523fd2763ab6c2558b0adc0a5877c91d5534b3be562959e739f3ba799ebdb921`
- `milp_sanity_results_fix2.jsonl`: SHA-256
  `3d47ef0932504c1742c3899a28c57d5b673d32a65d2211222dbce1fc42083bad`

The earlier smoke and fix1 artifacts remain in this directory as historical
audit evidence; fix2 is the only canonical P2 result.
