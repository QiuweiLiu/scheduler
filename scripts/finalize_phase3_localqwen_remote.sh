#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/autodl-tmp/scheduler"
TRACE_ROOT="$PROJECT_ROOT/results/raw/phase3_localqwen_videomme_32_r01"
SUMMARY="$TRACE_ROOT/phase2_batch_summary.jsonl"
PROCESSED="$PROJECT_ROOT/results/processed"
PYTHON_BIN="/root/miniconda3/envs/finetooling/bin/python"
BATCH_PATTERN='^/root/miniconda3/envs/finetooling/bin/python src/tracing/collectors/videotool_phase2_batch.py'
export PYTHONPATH="$PROJECT_ROOT/src"

mkdir -p "$PROCESSED/phase3_localqwen_h2_20260801" \
  "$PROCESSED/phase4_localqwen_burst_20260801" \
  "$PROCESSED/phase4_localqwen_staggered_20260801"

LOG="$PROCESSED/phase3_localqwen_finalize.log"
echo "[$(date -Is)] waiting for 96 local_qwen batch rows" | tee "$LOG"
while true; do
  rows=0
  if [[ -f "$SUMMARY" ]]; then
    rows=$(wc -l < "$SUMMARY")
  fi
  if [[ "$rows" -ge 96 ]]; then
    break
  fi
  if ! pgrep -f "$BATCH_PATTERN" >/dev/null 2>&1; then
    echo "[$(date -Is)] batch stopped before 96 rows (rows=$rows)" | tee -a "$LOG"
    exit 3
  fi
  sleep 30
done

while pgrep -f "$BATCH_PATTERN" >/dev/null 2>&1; do
  sleep 5
done

success_rows=$(grep -hc '"status": "success"' "$SUMMARY" || true)
error_rows=$(grep -hc '"status": "error"' "$SUMMARY" || true)
if [[ "$success_rows" -ne 96 || "$error_rows" -ne 0 ]]; then
  echo "[$(date -Is)] status gate failed: success_rows=$success_rows error_rows=$error_rows" | tee -a "$LOG"
  exit 4
fi

VALIDATION="$PROCESSED/phase3_localqwen_trace_validation.txt"
total=0
invalid=0
while IFS= read -r -d '' trace_path; do
  total=$((total + 1))
  if ! "$PYTHON_BIN" "$PROJECT_ROOT/src/tracing/validators/validate_trace.py" "$trace_path" >/dev/null; then
    invalid=$((invalid + 1))
    echo "INVALID $trace_path" >> "$VALIDATION"
  fi
done < <(find "$TRACE_ROOT" -mindepth 2 -maxdepth 2 -type f -name trace.jsonl -print0 | sort -z)
{
  echo "trace_files=$total"
  echo "invalid_traces=$invalid"
  echo "summary_rows=$(wc -l < "$SUMMARY")"
  echo "success_rows=$(grep -hc '"status": "success"' "$SUMMARY" || true)"
  echo "error_rows=$(grep -hc '"status": "error"' "$SUMMARY" || true)"
} > "$VALIDATION"
if [[ "$total" -ne 96 || "$invalid" -ne 0 ]]; then
  echo "[$(date -Is)] validation failed: traces=$total invalid=$invalid" | tee -a "$LOG"
  exit 4
fi

echo "[$(date -Is)] running H1/H3 statistics" | tee -a "$LOG"
"$PYTHON_BIN" "$PROJECT_ROOT/src/tracing/analysis/phase2_trace_stats.py" \
  --root "$TRACE_ROOT" \
  --output "$PROCESSED/phase3_localqwen_videomme_32_r01_stats.json" >> "$LOG"

echo "[$(date -Is)] running structured H2" | tee -a "$LOG"
"$PYTHON_BIN" "$PROJECT_ROOT/src/tracing/analysis/phase3_structured_predictor.py" \
  --root "$TRACE_ROOT" \
  --output "$PROCESSED/phase3_localqwen_h2_20260801/phase3_h2_report.json" \
  --csv "$PROCESSED/phase3_localqwen_h2_20260801/phase3_h2_metrics.csv" >> "$LOG"

echo "[$(date -Is)] running Phase 4 burst replay" | tee -a "$LOG"
"$PYTHON_BIN" "$PROJECT_ROOT/src/tracing/analysis/phase4_trace_simulator.py" \
  --root "$TRACE_ROOT" \
  --output "$PROCESSED/phase4_localqwen_burst_20260801/phase4_report.json" \
  --csv "$PROCESSED/phase4_localqwen_burst_20260801/phase4_metrics.csv" \
  --capacities 32760,24576 --seeds 0,1,2,3,4 --arrival-interval-ms 0 >> "$LOG"

echo "[$(date -Is)] running Phase 4 staggered replay" | tee -a "$LOG"
"$PYTHON_BIN" "$PROJECT_ROOT/src/tracing/analysis/phase4_trace_simulator.py" \
  --root "$TRACE_ROOT" \
  --output "$PROCESSED/phase4_localqwen_staggered_20260801/phase4_report.json" \
  --csv "$PROCESSED/phase4_localqwen_staggered_20260801/phase4_metrics.csv" \
  --capacities 32760,24576 --seeds 0,1,2,3,4 --arrival-interval-ms 500 >> "$LOG"

echo "[$(date -Is)] local_qwen finalization complete" | tee -a "$LOG"
