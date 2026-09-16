#!/usr/bin/env bash
set -euo pipefail

# Remote, serial, resumable trace producer. The same public-video manifest is
# run in separate cohort directories so no previous raw trace is overwritten.
# Keep one heavy local VLM resident per run; do not launch these cohorts in
# parallel on the single 32 GiB GPU.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/scheduler}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/finetooling/bin/python}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/configs/phase3_structured_videomme_32_localqwen_r01_remote.jsonl}"
VIDEO_TOOL_ROOT="${VIDEO_TOOL_ROOT:-$PROJECT_ROOT/third_party/videotool}"
QWEN3_MODEL="${QWEN3_MODEL:-/root/autodl-tmp/Qwen3-VL-8B-Instruct}"
YOLO11X_MODEL="${YOLO11X_MODEL:-/root/autodl-tmp/upload/models/yolo11x.pt}"
COHORT_PREFIX="${COHORT_PREFIX:-phase3_localqwen_yolo11x_videomme_32}"

export PYTHONPATH="$PROJECT_ROOT/src"
cd "$PROJECT_ROOT"

run_cohort() {
  local suffix="$1"
  local output_root="results/raw/${COHORT_PREFIX}_${suffix}"
  local log_path="results/processed/${COHORT_PREFIX}_${suffix}.log"
  mkdir -p "$(dirname "$log_path")"
  echo "[$(date -Is)] starting $output_root" | tee -a "$log_path"
  "$PYTHON_BIN" src/tracing/collectors/videotool_phase2_batch.py \
    --manifest "$MANIFEST" \
    --output-root "$output_root" \
    --videotool-root "$VIDEO_TOOL_ROOT" \
    --planner-mode local_qwen \
    --model-name Qwen3-VL-8B-Instruct \
    --max-iterations 6 \
    --yolo-model "$YOLO11X_MODEL" \
    --yolo-python "$PYTHON_BIN" \
    --yolo-batch 1 \
    --qwen-model "$QWEN3_MODEL" \
    --qwen-python "$PYTHON_BIN" \
    --qwen-max-new-tokens 256 \
    --resume >> "$log_path" 2>&1
  local rows
  rows=$(wc -l < "$output_root/phase2_batch_summary.jsonl")
  local successes
  successes=$(grep -c success "$output_root/phase2_batch_summary.jsonl" || true)
  echo "[$(date -Is)] completed $output_root rows=$rows successes=$successes" | tee -a "$log_path"
}

run_cohort r02
run_cohort r03

echo "[$(date -Is)] all requested Qwen3-VL repeat cohorts complete"
