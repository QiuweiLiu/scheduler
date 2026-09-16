# EXP-20260817_r7_boundary_gate

## Purpose

Freeze the R7 video boundary before trace collection or formal scheduling runs.

## Inputs

- `data/manifests/videomme_600_source_v2.jsonl`
- `data/manifests/video_provenance_v2_expanded_available.jsonl`
- Remote predictor split artifacts named in `data/manifests/video_split_registry_r7_v1.json`.

## Frozen design

`P_dev=300`, `P_holdout_diag=40`, `S_train=120`, `S_val=40`, `T_final=80`,
`T_backup=20`. The behavior and resource predictors share `P_dev`. The
scheduler does not train or tune on `P_dev` or `P_holdout_diag`.

## Out of scope

No trace collection, model training, workload generation, deletion, or formal
scheduler evaluation is started by this gate.
