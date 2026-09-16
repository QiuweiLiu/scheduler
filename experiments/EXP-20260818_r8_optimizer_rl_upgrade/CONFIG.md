# EXP-20260818_r8_optimizer_rl_upgrade

## Status

`P2_completed` — R8-P2 fix2 MILP/CP-RHO sanity gate passed; `T_final` remains sealed.

## Objective

Determine whether the current finite-lookahead gain is limited by the lexicographic
PredOpt objective, and then test a bounded rolling-horizon mathematical optimizer
before changing the RL architecture.

## Scope

- P0: PredOpt-v2 objective audit on frozen R7 workloads.
- P1/P2/P3/P4/P5: CP-RHO, MILP audit, expert/BC, PPO/GAE, and final test only after gates.
- No new video download, trace collection, predictor retraining, C2/elevator migration, or `T_final` execution in P0.

## Frozen inputs

- R7 predictor-unseen scheduler pool: `S_train`, `S_val`, `T_final`, `T_backup` from `data/manifests/video_split_registry_r7_v1.json`.
- Existing finite-horizon predictor artifacts and R7 workload manifest; exact paths and hashes are recorded in the P0 metrics manifest.
- Behavior and resource predictors remain frozen.

## Information boundary

Deployable policies receive current ready nodes, current DAG prefix, current GPU/cache
state, train-only resource estimates, and the frozen finite-horizon predictor output.
Execution runtime/load/memory/status and true suffixes remain engine-only.

## Planned gates

1. P0: no leakage, deterministic replay, objective decomposition and future rank-flip audit.
2. P1: CP-RHO feasibility/latency/status audit with explicit timeout/fallback accounting.
3. P2: small-instance MILP/CP-SAT feasibility and objective agreement.
4. P3: expert action regret and BC rollout gate.
5. P4: five-seed PPO/GAE stability and hard-feasibility gate.
6. P5: one-shot `T_final` paired evaluation after configuration freeze.

## Execution environment

Canonical execution target is remote `/root/autodl-tmp/scheduler`; local source and
control plane are canonical. Remote SSH authentication must be verified before remote
execution; no credentials are stored in this experiment record.

## Accepted P2 execution

- Remote output: `/root/autodl-tmp/scheduler/results/processed/r8_optimizer_rl_upgrade/p2_milp_sanity_20260819_fix2/`
- Command: `PYTHONPATH=src /root/miniconda3/envs/finetooling/bin/python scripts/r8_milp_sanity.py --templates results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl --episodes results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl --future-artifacts results/processed/r7_scheduling_future_20260817/prediction_artifacts --output-dir results/processed/r8_optimizer_rl_upgrade/p2_milp_sanity_20260819_fix2 --cases 20 --nodes-per-case 4 --time-limit-s 2.0 --seed 0`
- Gate: 20/20 CP-RHO and MILP `OPTIMAL`, node agreement `1.0`, objective gap `2.6319902025099073e-13 <= 1e-6`; exact node+GPU agreement `0.8`.
- The report records `cwd=/root/autodl-tmp/scheduler`, `PYTHONPATH=src`, runtime versions, hardware, and hashes for all four result-determining source files. `T_final` was not read.
