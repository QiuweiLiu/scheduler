# EXP-20260824_transition_profile_simulator

## Purpose

Run an opt-in simulator pilot that overlays measured model transition costs on
the existing node-level single-GPU-slot event engine. The default simulator,
R7 inputs, traces, models, and `T_final` remain unchanged.

## Profile semantics

- `cold_load_ms` replaces the demand/prefetch load for a non-resident model
  when no more-specific per-batch load profile exists.
- `evict_proxy_ms` is charged when the simulator actually evicts that model;
  it is the explicit unload plus CUDA-cache-release proxy measured by
  `EXP-20260824_transition_profile`, not an automatic cache-manager timing.
- `resident_operation_ms` and `first_operation_ms` remain audit-only fields;
  they do not replace node runtime.
- Checkpoint/restore is unsupported for all measured models. Preemption keeps
  the existing full-node recompute semantics.
- The profile is strict and applies only to the measured 32,760 MB GPU class.

## Paired conditions

- `transition_off`: existing extension runner with the empty baseline config.
- `transition_on`: the same episodes and policies with `transition_config.json`.
- The formal paired set is the 55 source episodes whose topology is exactly
  `[32760.0, 32760.0]` MB, selected in frozen source order; each condition is
  55 episodes × 4 policies.
- Event audit: the first 10 episodes of that same filtered set × 4 policies
  with event logging enabled.
- A diagnostic profile-off run over all 150 source episodes was completed before
  the topology mismatch was found; it is not the paired comparison.

## Frozen inputs

- Templates: `results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl`
- Episodes: `results/processed/EXP-20260821_stress_pilot/episodes_all.jsonl`
- Derived paired episodes: `results/processed/EXP-20260824_transition_profile_simulator/paired_rtx4080_super/inputs/episodes_rtx4080_super.jsonl`
- Filter metadata: `results/processed/EXP-20260824_transition_profile_simulator/paired_rtx4080_super/inputs/episodes_rtx4080_super.meta.json`
- Future artifacts: `results/processed/scheduling_future_v1_20260812/prediction_artifacts`
- Policies: `round_robin,myopic,predopt_h5,oracle`
- Transition source: `experiments/EXP-20260824_transition_profile/metrics.json`

## Out of scope

- Single-node multi-GPU execution, GPU-set allocation, cross-GPU communication,
  parallel runtime scaling, and multi-GPU preemption.
- New video downloads, trace recollection, predictor retraining, or changes to
  R7 and `T_final`.
