#!/usr/bin/env bash
set -euo pipefail

PID="${1:?collector pid is required}"
LOG="${LOG:-results/processed/resource_finalizer_20260804.log}"
exec >>"${LOG}" 2>&1
echo "[$(date -Is)] monitor started pid=${PID}"
while kill -0 "${PID}" 2>/dev/null; do
  cmd="$(tr '\0' ' ' <"/proc/${PID}/cmdline" 2>/dev/null || true)"
  case "${cmd}" in
    *videotool_phase2_batch.py*|*videotool_phase2_batch*) ;;
    *) echo "[$(date -Is)] pid reused or command changed: ${cmd}"; exit 1 ;;
  esac
  echo "[$(date -Is)] collector active"
  sleep 60
done
echo "[$(date -Is)] collector ended; running fixed core+resource finalizer"
bash scripts/finalize_core_resource_fixed.sh
echo "[$(date -Is)] resource finalization completed"
