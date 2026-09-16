# EXP-20260817_r7_event_engine_parity

## Purpose

Verify the R7 scheduler-facing information boundary before collecting any
S_train/S_val trace or workload. This is a contract smoke, not a performance
experiment and not evidence for a scheduling improvement.

## Inputs

- `src/tracing/scheduling/event_engine.py`
- `scripts/r7_event_engine_parity_smoke.py`
- deterministic causal-chain fixtures with one ready GPU node and one GPU

The legacy `workload_v02_simulator.py` and `scheduling_future_smoke.py` remain
compatibility diagnostics; they are not extended by this experiment.

## Checks

1. `none` does not serialize future scenarios.
2. `pred_h` exposes only identity/probability rows, not execution truth.
3. Mutating current runtime/load/workspace/status does not change a state view.
4. Mutating the suffix after the requested horizon does not change `true_h`.
5. Replaying the same fixture is deterministic.

## Runtime

Remote `/root/autodl-tmp/scheduler`, finetooling Python, `PYTHONPATH=src`,
seedless deterministic fixture. The formal scheduler workload remains
unstarted by design.
