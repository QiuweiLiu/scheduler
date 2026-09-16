#!/usr/bin/env bash
set -u

PROJECT_ROOT="/root/autodl-tmp/scheduler"
PYTHON_BIN="/root/miniconda3/envs/finetooling/bin/python"
PILOT_PID="${1:?pilot collector pid is required}"
PILOT_ROOT="$PROJECT_ROOT/results/raw/phase3_stackb_qwen3_4b_constrained_pilot8_r01"
FORMAL_ROOT="$PROJECT_ROOT/results/raw/phase3_stackb_qwen3_4b_constrained_formal256_r01"
SUMMARY="$PILOT_ROOT/phase2_batch_summary.jsonl"
VALIDATION="$PROJECT_ROOT/results/processed/phase3_stackb_qwen3_4b_constrained_pilot8_validation.txt"
LOG="$PROJECT_ROOT/logs/phase3_stackb_qwen3_4b_constrained_auto.log"

mkdir -p "$PROJECT_ROOT/results/processed" "$PROJECT_ROOT/logs"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] watcher started; pilot_pid=$PILOT_PID"

while kill -0 "$PILOT_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date -Is)] pilot collector exited"

if [[ ! -f "$SUMMARY" ]]; then
  echo "[$(date -Is)] pilot summary missing: $SUMMARY"
  exit 2
fi

success_rows=$(grep -c '"status": "success"' "$SUMMARY" || true)
error_rows=$(grep -c '"status": "error"' "$SUMMARY" || true)
summary_rows=$(wc -l <"$SUMMARY")
echo "[$(date -Is)] pilot summary rows=$summary_rows success=$success_rows error=$error_rows"
if [[ "$summary_rows" -ne 32 || "$success_rows" -ne 32 || "$error_rows" -ne 0 ]]; then
  echo "[$(date -Is)] pilot status gate failed; formal Stack B will not start"
  exit 3
fi

total=0
invalid=0
: >"$VALIDATION"
while IFS= read -r -d '' trace_path; do
  total=$((total + 1))
  if ! "$PYTHON_BIN" "$PROJECT_ROOT/src/tracing/validators/validate_trace.py" "$trace_path" >/dev/null; then
    invalid=$((invalid + 1))
    echo "INVALID $trace_path" >>"$VALIDATION"
  fi
done < <(find "$PILOT_ROOT" -mindepth 2 -maxdepth 2 -type f -name trace.jsonl -print0 | sort -z)
{
  echo "trace_files=$total"
  echo "invalid_traces=$invalid"
  echo "summary_rows=$summary_rows"
  echo "success_rows=$success_rows"
  echo "error_rows=$error_rows"
} >>"$VALIDATION"
echo "[$(date -Is)] pilot validation traces=$total invalid=$invalid"
if [[ "$total" -ne 32 || "$invalid" -ne 0 ]]; then
  echo "[$(date -Is)] pilot validation gate failed; formal Stack B will not start"
  exit 4
fi

if [[ ! -f "$PROJECT_ROOT/configs/phase3_stackb_64video_128tasks.jsonl" ]]; then
  echo "[$(date -Is)] formal manifest missing"
  exit 5
fi
formal_rows=$(wc -l <"$PROJECT_ROOT/configs/phase3_stackb_64video_128tasks.jsonl")
if [[ "$formal_rows" -ne 128 ]]; then
  echo "[$(date -Is)] formal manifest row count=$formal_rows, expected=128"
  exit 6
fi

echo "[$(date -Is)] pilot passed; starting formal Stack B (256 runs)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
exec "$PYTHON_BIN" src/tracing/collectors/videotool_phase2_batch.py \
  --manifest configs/phase3_stackb_64video_128tasks.jsonl \
  --output-root "$FORMAL_ROOT" \
  --videotool-root "$PROJECT_ROOT/third_party/videotool" \
  --planner-mode local_split \
  --model-name Qwen2.5-VL-3B-Instruct \
  --max-iterations 6 \
  --yolo-model /root/autodl-tmp/upload/models/yolo26n.pt \
  --yolo-python "$PYTHON_BIN" \
  --yolo-batch 1 \
  --qwen-model "$PROJECT_ROOT/Qwen2.5-VL-3B-Instruct" \
  --qwen-python "$PYTHON_BIN" \
  --qwen-max-new-tokens 96 \
  --planner-max-new-tokens 96 \
  --answer-max-new-tokens 96 \
  --model-stack-id stack_b_qwen3_4b_qwen25vl3b_yolo26n \
  --planner-constrained-json \
  --resume
