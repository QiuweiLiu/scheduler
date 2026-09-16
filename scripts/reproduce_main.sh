#!/usr/bin/env bash
# One-command reproduction of the main scheduler comparison and the trace-dependence tests.
#
# Usage:  bash scripts/reproduce_main.sh [output_root]
# Requires: python3.10 with configs/requirements.txt, repository root as CWD.
set -euo pipefail

OUT="${1:-outputs/repro_main}"
TEMPLATES="results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl"
EPISODES="results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl"
ARTIFACTS="outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts"
POLICIES="predopt_h5,predopt_h5_q95,predopt_h5_r50,predopt_h5_r90,predopt_h5_r95,predopt_h5_r50k,fcfs,sjf_pred,state_aware,myopic"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

echo "== [1/4] main consumer matrix (1,000 validation episodes) =="
python scripts/r7_scheduler_matrix.py \
  --templates "$TEMPLATES" \
  --episodes "$EPISODES" \
  --future-artifacts "$ARTIFACTS" \
  --output-dir "$OUT" \
  --limit 1000 \
  --policies "$POLICIES"

echo "== [2/4] dev/confirm split run (700 dev then 300 frozen confirm) =="
python scripts/r7_scheduler_matrix.py \
  --templates "$TEMPLATES" --episodes "$EPISODES" --future-artifacts "$ARTIFACTS" \
  --episode-ids-file data/manifests/validation_split_dev700_ids.txt \
  --output-dir "${OUT}_dev" --limit 700 \
  --policies predopt_h5,predopt_h5_q95
python scripts/r7_scheduler_matrix.py \
  --templates "$TEMPLATES" --episodes "$EPISODES" --future-artifacts "$ARTIFACTS" \
  --episode-ids-file data/manifests/validation_split_confirm300_ids.txt \
  --output-dir "${OUT}_confirm" --limit 300 \
  --policies predopt_h5,predopt_h5_q95

echo "== [3/4] trace-dependence tests T1-T3 (>=2000 bootstrap / permutation, two-sided p) =="
python scripts/verify_trace_dependence.py \
  --templates "$TEMPLATES" \
  --output "experiments/EXP-20260911_forecast_aware_scheduling/trace_dependence_report.json"

echo "== [4/4] tail-shuffled artifact control =="
python scripts/perturb_future_artifacts.py \
  --source "$ARTIFACTS" \
  --target "outputs/perturb_tail_shuffle/prediction_artifacts" \
  --kind tail_shuffle --intensity 1.0 --seed 20260915
python scripts/r7_scheduler_matrix.py \
  --templates "$TEMPLATES" --episodes "$EPISODES" \
  --future-artifacts "outputs/perturb_tail_shuffle/prediction_artifacts" \
  --output-dir "${OUT}_shuffle" --limit 1000 \
  --policies predopt_h5_q95

echo "done; see $OUT/ and experiments/EXP-20260911_forecast_aware_scheduling/"
