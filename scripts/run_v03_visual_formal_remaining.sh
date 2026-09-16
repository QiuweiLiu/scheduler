#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/scheduler
export PYTHONPATH=/root/autodl-tmp/scheduler/grounded-video-llm:\${PYTHONPATH:-}
PY=/root/miniconda3/envs/finetooling/bin/python
LOG=logs/multimodal_predictor_v03_visual_formal_remaining.log
for SEED in 13 17; do
  echo "[$(date -Is)] seed=$SEED start" >> "$LOG"
  "$PY" src/tracing/analysis/neural_multimodal_predictor_v03_visual.py \
    --model all --seed "$SEED" --epochs 25 --patience 5 --batch-size 128 \
    --output-dir "results/processed/multimodal_predictor_v0_3_visual_formal_seed_${SEED}" \
    --device auto >> "$LOG" 2>&1
  echo "[$(date -Is)] seed=$SEED exit=$?" >> "$LOG"
done
echo "[$(date -Is)] formal complete" >> "$LOG"
