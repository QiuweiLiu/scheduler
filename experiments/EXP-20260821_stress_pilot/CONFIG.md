# EXP-20260821_stress_pilot

## Status

`completed` — five-cell stress pilot passed the summary-runner gate. This is a
calibration experiment, not the final `T_final` evaluation.

## Objective

Check whether the current workload is too easy for the scheduler by increasing
ready-node width, offered load, and deadline pressure in a controlled way. The
pilot uses common random numbers: every cell is derived from the same 30 source
episodes.

## Frozen inputs

- Templates: remote `results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl` (640 templates).
- Source episodes: remote `results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl` (1,000 validation episodes).
- Future artifacts: remote `results/processed/scheduling_future_v1_20260812/prediction_artifacts`.
- Selected episodes: 30 evenly spaced source rows, reused in every cell.
- Policies: `round_robin`, `myopic`, `predopt_h5`, `oracle`.
- `T_final`: sealed and not read or executed.

## Cells

| Cell | Transformation |
|---|---|
| S0 baseline | Source arrival and deadline timing unchanged. |
| S1 ready-width | All jobs in an episode released at the first arrival; relative deadlines preserved. |
| S2 tight-deadline | Arrivals unchanged; deadline is `arrival + 1.1 × predicted_job_runtime_p90_ms`. |
| S3 offered-load | Arrival span compressed to 35%; relative deadlines preserved. |
| S4 combined | S1 ready-width plus S2 tight-deadline. |

Only `arrival_ms`, `deadline_ms`, episode metadata, and generated job instance IDs
are changed. Template IDs, DAGs, scheduler-visible resource predictions, source
video/trace references, and model artifacts are unchanged.

## Execution

- Host: `root@connect.westc.seetacloud.com:12469`.
- Remote cwd: `/root/autodl-tmp/scheduler`.
- Interpreter: `/root/miniconda3/bin/python` with `PYTHONPATH=src`.
- Derivation script: `experiments/EXP-20260821_stress_pilot/build_stress_pilot.py`.
- Matrix command:

```text
PYTHONPATH=src /root/miniconda3/bin/python scripts/r7_scheduler_matrix.py \
  --templates results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl \
  --episodes results/processed/EXP-20260821_stress_pilot/episodes_all.jsonl \
  --future-artifacts results/processed/scheduling_future_v1_20260812/prediction_artifacts \
  --output-dir results/processed/EXP-20260821_stress_pilot/matrix \
  --limit 150 --policies round_robin,myopic,predopt_h5,oracle
```

The matrix is summary-only. A two-episode-per-cell event audit was run separately
and compressed to `event_audit_events.jsonl.gz` (not copied to the local control
plane because it is diagnostic output).

## Reproducibility hashes

The full source and output hashes are in `metrics.json` and the remote
`results/processed/EXP-20260821_stress_pilot/pilot_report.json`.

