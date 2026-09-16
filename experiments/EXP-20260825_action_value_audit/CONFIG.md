# EXP-20260825 Action-Value Audit

## Scope

Diagnostic audit of action ranking on common scheduler states. This experiment
does not change Myopic, PredOpt-H5, PredOpt-v2-H5, or TrueOpt-H5, and does not
read `T_final`.

## Common-state protocol

- Reference trajectory: `myopic`.
- Each `node_dispatch` scheduler-safe state on that trajectory is evaluated
  offline with the same ready nodes, free GPUs, GPU residency, and completed
  prefix.
- Action identity: `(job_instance_id, node_id, gpu_index)`.
- Widths: raw candidate, model-feasible, and strict admission-feasible.
- Primary value: `Q_local_H5`, implemented by the existing
  `limited_future_truth_cost` semantics: the next five DAG layers for the
  candidate node. This is not a full event-level counterfactual rollout.
- True suffix values are audit-only and are never passed to a policy.

## Inputs

- Remote host: `root@connect.westc.seetacloud.com:12469`;
  remote execution root: `/root/autodl-tmp/scheduler`.
- Remote templates:
  `/root/autodl-tmp/scheduler/results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl`
- Remote validation episodes:
  `/root/autodl-tmp/scheduler/results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl`
- Remote future artifacts:
  `/root/autodl-tmp/scheduler/results/processed/r7_scheduling_future_20260817/prediction_artifacts/`
- Local read-only staging:
  `.scratch/action_value_audit_remote_inputs_20260825/`
- Local input hashes match the remote SHA-256 values recorded in the handoff.

## Run

- Reference policy: `myopic`.
- Horizon: `5`.
- First real smoke: `3` validation episodes; passed with 462 decisions,
  0 failed jobs, and no missing future-artifact node keys.
- This run: first 100 validation episodes after smoke; no stress transform,
  no WAIT, no prefetch, no optimizer change, and no remote write.

## Full-event follow-up

- Input: the completed 100-episode audit records above.
- Sampling: at most 100 common states, deterministically stratified by strict
  action width and PredOpt-H5 error/top-1 status.
- Counterfactual: force each strict-feasible first action, then continue the
  episode with the Myopic reference policy in the event simulator.
- Value: full-episode mean completion after forcing one action; this is a
  diagnostic full-event value, not the existing five-layer `Q_local_H5`.
- Output: `rollout_audit/` with 100 target states and 717 candidate branches;
  0 failures and 100/100 reference-action consistency checks passed.
- The simulator hook is audit-only; the default policy path is unchanged.

## Outputs

- `decision_records.jsonl`: per-decision action/value audit rows.
- `metrics.json` and `metrics.csv`: numeric aggregates.
- `reference_summaries.jsonl`: reference simulation summaries.
- `run_manifest.json`: input and run provenance.
- `rollout_audit/metrics.json`: sampled full-event audit aggregates.
- `rollout_audit/rollout_decision_records.jsonl`: per-branch full-event values.
