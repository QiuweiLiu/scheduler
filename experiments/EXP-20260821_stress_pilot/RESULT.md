# EXP-20260821_stress_pilot — Result

## Gate

**PASSED for pilot calibration.** The remote summary matrix produced 150
episodes × 4 policies = 600 result rows, 5,120/5,120 jobs completed, zero failed
jobs, and zero capacity violations. R7 inputs/results were not overwritten and
`T_final` was not read.

Full numeric output is in [`metrics.json`](metrics.json); the remote audit report
is `results/processed/EXP-20260821_stress_pilot/pilot_report.json`.

## Per-cell scheduler result

Mean completion is in milliseconds; the gap is relative to the Oracle in the
same cell. Deadline miss is the fraction of jobs missing the transformed deadline.

| Cell | Policy | Mean completion | Mean queue | Deadline miss | Oracle gap |
|---|---|---:|---:|---:|---:|
| S0 | RR | 210,513 | 148,169 | 13.0% | 23.6% |
| S0 | Myopic | 214,668 | 121,443 | 11.4% | 26.0% |
| S0 | PredOpt-H5 | 214,625 | 121,369 | 11.4% | 26.0% |
| S0 | Oracle | 170,352 | 76,209 | 5.6% | 0 |
| S1 | RR | 1,138,057 | 1,302,032 | 81.1% | 42.2% |
| S1 | Myopic | 1,097,550 | 1,011,856 | 81.1% | 37.1% |
| S1 | PredOpt-H5 | 1,097,550 | 1,011,856 | 81.1% | 37.1% |
| S1 | Oracle | 800,610 | 708,169 | 64.2% | 0 |
| S2 | RR | 210,513 | 148,169 | 26.8% | 23.6% |
| S2 | Myopic | 214,668 | 121,443 | 23.6% | 26.0% |
| S2 | PredOpt-H5 | 214,625 | 121,369 | 23.5% | 26.0% |
| S2 | Oracle | 170,352 | 76,209 | 15.9% | 0 |
| S3 | RR | 711,398 | 829,360 | 61.3% | 51.4% |
| S3 | Myopic | 728,313 | 641,843 | 57.5% | 55.0% |
| S3 | PredOpt-H5 | 728,313 | 641,843 | 57.5% | 55.0% |
| S3 | Oracle | 470,001 | 380,408 | 33.2% | 0 |
| S4 | RR | 1,138,057 | 1,302,032 | 92.2% | 42.2% |
| S4 | Myopic | 1,097,550 | 1,011,856 | 90.6% | 37.1% |
| S4 | PredOpt-H5 | 1,097,550 | 1,011,856 | 90.6% | 37.1% |
| S4 | Oracle | 800,610 | 708,169 | 81.0% | 0 |

## What this shows

1. The workload was indeed too easy in its original form. Aligning arrivals
   (S1) raises event-audit action-width median from 2 to 12 for Myopic/PredOpt
   and from 3 to 15 for RR. The 35% arrival compression (S3) raises queue/wait
   pressure and the Oracle gap to about 55% for Myopic/PredOpt.
2. Tightening only the deadline (S2) changes miss rates but not completion or
   queue time, because the current four policies do not use deadline slack in
   their ranking objective. This is an expected diagnostic result, not evidence
   that the deadline transform was ignored by the simulator.
3. PredOpt-H5 is effectively tied with Myopic in this pilot (exactly tied in
   S1/S3/S4). The current finite-horizon score is not changing the selected
   actions under these cells; the pilot therefore exposes the need to improve
   the prediction-to-action coupling before claiming a scheduling gain.
4. This is a stress calibration, not a final statistical claim: 30 episodes per
   cell and two episodes per cell with event logging are enough to validate the
   pressure mechanism, but not enough for final confidence intervals.

## Event audit

The 10-episode × 4-policy audit was 1.84 MB compressed. GPU truth runtime tails
were stable (P50 6,365 ms, P90 24,629 ms); the stress came from arrival overlap
and queueing, not changed node runtime. S1/S4 action width and RR cache churn
were visibly higher than S0. The detailed event audit remains remote to avoid
copying unnecessary output into the control plane.

## Next decision

Do not run `T_final` yet. Use this pilot to freeze a formal stress workload
definition, then run a larger paired S_val matrix with a scheduler objective that
explicitly includes deadline slack (and, separately, a prediction-aware policy
whose finite-horizon score can actually change action order). Keep the current
pilot and all R7 artifacts immutable.

