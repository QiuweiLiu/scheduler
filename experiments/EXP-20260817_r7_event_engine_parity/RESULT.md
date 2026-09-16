# Result

## Status

`passed_contract_and_legacy_loop_gate_formal_workload_pending`

## Remote evidence

- Report: `/root/autodl-tmp/scheduler/results/processed/r7_event_engine_parity_smoke_20260817/report.json`
- Remote interface tests: 4/4 passed.
- Remote full suite after sync: 30/30 passed.
- Remote `py_compile`: passed.
- 10-fixture run: every check 10/10.
- 50-fixture run: every check 50/50.
- Real legacy event-loop replay (Myopic only) also passed: 10 validation episodes produced 1,332 dispatch/state pairs; 50 produced 6,744 dispatch/state pairs. Summary: `legacy_loop_replay.json`.
- GPU memory during smoke: 1 MiB used on RTX 4080 SUPER; no model was loaded.
- Remote free space after smoke: approximately 4.3G; no large artifacts were created.

## Interpretation

The `SchedulerStateView`, `ExecutionTruthProvider`, and `FutureProvider` are
now separated at the contract level and are constructed inside the existing
event loop before each GPU dispatch. The contract smoke and real legacy loop
replay prove the intended leakage invariants and deterministic serialization.
They do **not** prove that a real S_train/S_val workload has useful pressure
or that any policy improves completion time. The next authorized step is to
collect/assemble S_train and S_val traces and only then generate the pressure
workload.
