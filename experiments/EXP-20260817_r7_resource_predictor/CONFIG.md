# EXP-20260817_r7_resource_predictor

## Purpose

Align the resource predictor with the same 300-video `P_dev` used by the
frozen behavior predictor, while keeping the 40-video predictor holdout
unseen.

## Inputs

- `results/processed/r7_resource_input_20260817/compute_p_dev.jsonl`
- `results/processed/r7_resource_input_20260817/static_metadata_p_dev.jsonl`
- `data/manifests/predictor_development_split_v1_normalized.jsonl`
- Remote final-holdout compute, metadata, and split artifacts.

## Contract

Fit uses only P_dev train and validation rows. The old P_dev test split is
diagnostic only. Runtime, load, and peak memory are labels; answers, teacher
labels, future events, and video_id are excluded from features. Queue, strict
OOM, resident-weight separation, and stall-risk prediction remain unsupported.

## Runtime

Remote finetooling environment, `PYTHONPATH=src`, seed 42, existing
`tracing.analysis.resource_predictor_v1` implementation. Output is written to
the versioned remote directory recorded in RESULT.md.
