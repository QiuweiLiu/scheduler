# EXP-20260821_gpu_topology_pilot

## Status

`completed` — 4/8-GPU and heterogeneous-GPU topology pilot passed.

## Objective

Measure the effect of GPU count and GPU capacity heterogeneity while holding the
same jobs, DAGs, arrivals, deadlines, templates, and predictor artifacts fixed.

## Frozen inputs

- Templates: remote `results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl` (640 templates).
- Source workload: the already generated stress-pilot S0 baseline and S4 combined-pressure episodes (30 each).
- Policies: `round_robin`, `myopic`, `predopt_h5`, `oracle`.
- Extra GPUs start cold; the first two source GPU residency hints are preserved.
- GPU p95 memory demand in the frozen templates is at most about 19.52 GB, so the 24,576 MB heterogeneous tier is not an automatic OOM condition.
- `T_final` remains sealed and was not read.

## Topology cells

| Cell family | GPU topology (MB) | Arrival condition |
|---|---|---|
| G4 | `[32760] × 4` | S0 or S4 |
| G8 | `[32760] × 8` | S0 or S4 |
| H4 | `[32760,24576,32760,24576]` | S0 or S4 |
| H8 | `[32760,24576] × 4` | S0 or S4 |

This yields 8 cells × 30 episodes = 240 episodes and 960 policy rows. Only
`gpu_topology_mb`, extra-GPU residency fields, and experiment metadata are
changed. No trace, video, template, predictor, or node truth is changed.

## Execution

- Host: `root@connect.westc.seetacloud.com:12469`.
- Remote cwd: `/root/autodl-tmp/scheduler`.
- Interpreter: `/root/miniconda3/bin/python`, `PYTHONPATH=src`.
- Workload derivation: `experiments/EXP-20260821_gpu_topology_pilot/build_gpu_topology_pilot.py`.
- Event audit: `experiments/EXP-20260821_gpu_topology_pilot/audit_gpu_topology_pilot.py`.
- Matrix runner: existing `scripts/r7_scheduler_matrix.py`, summary-only.

Full source/output hashes are recorded in `metrics.json` and the remote
`results/processed/EXP-20260821_gpu_topology_pilot/gpu_topology_report.json`.

