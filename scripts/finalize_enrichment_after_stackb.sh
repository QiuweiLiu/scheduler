#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/scheduler"
PYTHON="/root/miniconda3/envs/finetooling/bin/python"
STACKB_PID="993705"
LOG="${PROJECT_ROOT}/results/processed/finalize_enrichment_after_stackb_20260803.log"

mkdir -p "${PROJECT_ROOT}/results/processed"
exec >>"${LOG}" 2>&1
echo "[$(date -Is)] finalization monitor started pid=${STACKB_PID}"

while kill -0 "${STACKB_PID}" 2>/dev/null; do
  if [ -r "/proc/${STACKB_PID}/cmdline" ] && ! tr '\0' ' ' <"/proc/${STACKB_PID}/cmdline" | grep -q 'videotool_phase2_batch.py'; then
    echo "[$(date -Is)] pid reused without Stack B collector; continue"
    break
  fi
  echo "[$(date -Is)] Stack B still active"
  sleep 60
done

cd "${PROJECT_ROOT}"
echo "[$(date -Is)] Stack B ended; rebuilding final derived artifacts"

"${PYTHON}" -m tracing.analysis.build_trace_enrichment \
  --root results/raw/phase3_localqwen_yolo11x_videomme_32_r01 \
  --root results/raw/phase3_localqwen_yolo11x_old32_q2_r01 \
  --root results/raw/phase3_localqwen_yolo11x_videomme_expansion32_r01 \
  --root results/raw/phase3_stackb_qwen3_4b_constrained_formal256_r01 \
  --split-manifest configs/phase3_video_split_48_8_8.jsonl \
  --output-dir results/processed/trace_enrichment_core_final_20260803

"${PYTHON}" -m tracing.analysis.reproduce_prediction_baselines \
  --prefix results/processed/trace_enrichment_core_final_20260803/prefix_samples_v0_1.jsonl \
  --split-manifest configs/phase3_video_split_48_8_8.jsonl \
  --output results/processed/prediction_baselines_core_final_20260803.json

"${PYTHON}" -m tracing.analysis.reproduce_resource_predictor \
  --snapshot results/processed/trace_enrichment_core_final_20260803/candidate_snapshot_v0_1.jsonl \
  --compute results/processed/trace_enrichment_core_final_20260803/compute_events_v0_1.jsonl \
  --split-manifest configs/phase3_video_split_48_8_8.jsonl \
  --output results/processed/resource_predictor_core_final_20260803.json

"${PYTHON}" -m tracing.analysis.build_collection_index \
  --core results/processed/trace_enrichment_core_final_20260803/candidate_snapshot_v0_1.jsonl \
  --output results/processed/collection_index_core_final_20260803.jsonl

"${PYTHON}" -m tracing.workloads.build_workload \
  --snapshot results/processed/trace_enrichment_core_final_20260803/candidate_snapshot_v0_1.jsonl \
  --compute results/processed/trace_enrichment_core_final_20260803/compute_events_v0_1.jsonl \
  --templates results/processed/job_templates_core_final_20260803.jsonl

"${PYTHON}" -m tracing.workloads.build_workload \
  --templates results/processed/job_templates_core_final_20260803.jsonl \
  --split train --episodes results/processed/workload_train_core_final_20260803.jsonl \
  --count 20000 --seed 20260803
"${PYTHON}" -m tracing.workloads.build_workload \
  --templates results/processed/job_templates_core_final_20260803.jsonl \
  --split validation --episodes results/processed/workload_validation_core_final_20260803.jsonl \
  --count 1000 --seed 20260804
"${PYTHON}" -m tracing.workloads.build_workload \
  --templates results/processed/job_templates_core_final_20260803.jsonl \
  --split test --episodes results/processed/workload_test_core_final_20260803.jsonl \
  --count 1000 --seed 20260805

echo "[$(date -Is)] final core processing completed"
