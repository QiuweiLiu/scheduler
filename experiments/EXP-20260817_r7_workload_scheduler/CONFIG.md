# EXP-20260817 R7 workload and scheduler validation

## Scope

Validate the predictor-unseen `S_train/S_val` workload generated from the 640 accepted R7 traces, then run the frozen finite-horizon scheduler matrix on the 1,000-episode validation split. `T_final` remains sealed.

## Remote inputs

- Project root: `/root/autodl-tmp/scheduler`
- Trace quality report: `results/processed/r7_trace_full_20260817_quality/quality_report.json`
- Workload audit: `results/processed/r7_workload_20260817/workload_audit.json`
- Validation episodes: `results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl`
- Future artifacts: `results/processed/r7_scheduling_future_20260817/prediction_artifacts/`
- Scheduler matrix: `results/processed/r7_scheduler_matrix_validation_1000_final/`

## Frozen boundaries

- 300 `P_dev` videos are used only for behavior/resource prediction development.
- `S_train/S_val` use predictor-unseen videos; `T_final` is not read.
- Truth is engine-only. `myopic`/`optimizer_0` use current-ready information; `predopt_h1/h3/h5` use train-only finite-horizon artifacts; `oracle` and `trueopt_h*` are reference policies.

## Matrix

The validation matrix contains 1,000 episodes and the ten registered policies: `round_robin`, `myopic`, `optimizer_0`, `oracle`, `predopt_h1`, `predopt_h3`, `predopt_h5`, `trueopt_h1`, `trueopt_h3`, and `trueopt_h5`.

## RL extension

- Runner: `scripts/r7_rl_train_eval.py`, SHA-256 `ce70cf5c710ecc1a9b4015f120b11d1a02a4d8ad03b97db61a2264cb240bdf1d`.
- Policies and seeds: `rl_0`/`rl_h5` with seeds 11/22/33; train/eval limits 1000/1000.
- Runtime controls: `--torch-threads 1`, `--torch-interop-threads 1`, `--progress-every 1`; OMP/MKL/OPENBLAS/NUMEXPR are all set to 1 per process.
- Output: `/root/autodl-tmp/scheduler/results/processed/r7_rl_matrix_20260817_fix2/`; this directory is separate from the stalled audit and does not overwrite it.
