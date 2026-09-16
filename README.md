# Scheduler — Forecast-Aware GPU Scheduling for Multi-Model Video-Agent Workflows

This repository contains the code, processed datasets, and experiment records for a study on
**whether predicting a video-agent's future action chain improves GPU scheduling**.

The core question: a multi-model agent workflow (planner → video tools → QA → generation) unfolds
dynamically at runtime — a node-level scheduler only sees what is *currently ready* and cannot tell
whether a request is about to finish, what it will call next, or how long the remaining chain will
take. We build a **future-action predictor** (per-step model/role + runtime/load quantiles + a
length/termination distribution), pack its output into a scheduler-facing interface, and study
**how the predicted future should be consumed** by an online greedy list scheduler.

## Headline findings (1,000 validation episodes, paired bootstrap)

| Comparison | Result |
|---|---|
| Predicted future + static table vs static table only | **−17.1 s** mean completion |
| **Per-step runtime p95 sum (r95/q95) vs point-cost baseline** | **−17.7 s** (dev), **−18.8 s** on a frozen 300-episode confirm split |
| r95 vs scale-matched p50 (same total magnitude) | **−13.0 s** → the gain is *not* simply a larger weight |
| r95 vs tail-shuffled predictions (same marginals, shuffled pairing) | **−7.0 s** → the tail must align with the specific predicted step |
| r95 vs scenario-sampling CVaR (fair re-implementation, N=128, comonotone variant) | **−14 to −19 s** |
| r95 vs classic baselines (FCFS / SJF / state-aware / current-node greedy) | **−33 to −35 s** |
| r95 vs oracle-information + point-cost arms | −3.5 to −4.2 s |
| r95 vs full-truth oracle | +4.4 % (closes 82 % of the no-future → oracle gap) |

A negative result worth stating up front: **the "shared slowdown factor" hypothesis is rejected on
real traces** (640 runs: run-level ICC ≈ 0, mean within-run covariance −0.12, cross-type residual
correlation −0.18…−0.26, PIT tail-lift 0.93/0.70/0.31 for q = 0.8/0.9/0.95). The simulator replays
fixed measured node runtimes, so the tail-aware cost acts as a **forecast-error-aware ranking
surrogate**, not as an estimator of execution risk. See `docs/research/2026-09-16_fas_verification_design_gpt.md`
and `experiments/EXP-20260911_forecast_aware_scheduling/TRACE_DEPENDENCE_REPORT.md`.

## Repository layout

```
src/tracing/            simulator, predictors, scheduling components
  analysis/workload_v02_simulator.py     node-level event simulator + all consumption policies
  scheduling/event_engine.py             execution-truth / future providers
  scheduling/cp_rho.py                   CP-SAT rolling-horizon reference (units documented in-file)
scripts/                dataset builders, packers, audits, experiment runners
  r7_scheduler_matrix.py                 main experiment runner (resume-style + input/code fingerprint)
  pack_j_predictor_artifacts.py          packs the frozen predictor into scheduler artifacts
  perturb_future_artifacts.py            artifact perturbations (tail shrink / length noise / content / tail shuffle)
  verify_trace_dependence.py             T1/T2/T3 trace-dependence tests (>=2000 bootstrap, two-sided p)
  reproduce_main.sh                      one-command reproduction of the main comparison
  analysis/                              per-phase statistical analysis scripts (phase 6-13)
docs/                   design documents, research notes, and PROVENANCE.md (datasets, seeds, commands)
docs/PROVENANCE.md      datasets, frozen choices, seeds, environment, known fixes
experiments/            experiment records (reports and small metrics; large artifacts excluded)
data/manifests/         dataset registries (splits, dataset cards, dev/confirm split)
results/processed/      processed datasets (predictor datasets, workload episodes, enrichment)
outputs/                frozen predictor checkpoints + packed prediction artifacts + audits
.project/               project control plane (state, plan, decisions, experiment gate)
```

## Data availability

Included: processed datasets and workload definitions (predictor datasets `j_series_dataset_v1` /
`_h10`, topology datasets v3 H5/H10, R7 workload validation episodes + template definitions, trace
enrichment tables), frozen predictor checkpoints, and packed scheduler-facing artifacts.

**Not included, and not regenerable from this repository alone**: the raw VideoMME videos, the
pretrained LLM/VLM weights, the raw agent traces, and multi-hundred-MB simulation event logs. The
**main simulation and all statistical analyses are reproducible** from the processed datasets
above; re-training the predictor from scratch would additionally require the original trace
collection.

## Reproducing the main comparison

```bash
python -m pip install -r configs/requirements.txt      # python 3.10
export PYTHONPATH=src

# one command: main matrix + dev/confirm + T1-T3 + tail-shuffle control
bash scripts/reproduce_main.sh outputs/repro_main

# or step by step:
python scripts/r7_scheduler_matrix.py \
    --templates  results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl \
    --episodes   results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl \
    --future-artifacts outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts \
    --output-dir outputs/repro_matrix --limit 1000 \
    --policies predopt_h5,predopt_h5_q95,predopt_h5_r95,fcfs,sjf_pred,state_aware,myopic

python scripts/verify_trace_dependence.py \
    --templates results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl \
    --output experiments/EXP-20260911_forecast_aware_scheduling/trace_dependence_report.json
```

## Notes

- The runner is **resume-style** and writes `run_fingerprint.json` (hashes of templates, episodes,
  artifact manifest and the semantic code files). Resuming a directory whose fingerprint differs is
  refused unless `--allow-fingerprint-change` is passed, so stale rows cannot silently mix into a
  report.
- `T_final` (80 sealed videos) is intentionally untouched by every experiment here; the frozen
  confirm split is 300 validation episodes.
- The simulator is a *node-level event simulator*, not a physical multi-GPU benchmark: runtimes and
  memory in the workload templates are measured values replayed from real agent traces.
- Known issues fixed after the first public revision, plus reporting discipline (descriptive moment
  estimates, lognormal-reconstruction caveat for the scenario family), are listed in
  `docs/PROVENANCE.md` §5-6.
- License: MIT (see `LICENSE`).

## Contact

Qiuwei Liu — issues and questions are welcome via GitHub.
