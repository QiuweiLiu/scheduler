# Result — EXP-20260903_p9f_predictor_acceptance_audit

## Status

`completed_diagnostic`: the frozen P9e `B` checkpoint was evaluated without
retraining. The structural comparison gate passed against the empirical
baseline, but the probability-calibration and future-resource gates did not.
This experiment therefore does not authorize scheduler integration.

## Design and boundary

- Checkpoints: P9e reduced future-node `B`, seeds `11/22/33`, frozen before
  this experiment.
- Fit: no model fit or epoch/variant selection. Scalar temperatures were fit
  only on `P_dev/train` logits for diagnostic calibration.
- Evaluation: `P_dev/validation` and `P_dev/test` are diagnostics;
  `P_holdout_diag/holdout` is the frozen acceptance split.
- Resource proxy: train-only `role|action_family -> P50 runtime_ms`, joined
  to the P9d labels and P_dev compute rows.
- No `S_train/S_val` labels, scheduler groups, raw-trace changes, future
  truth as model input, or `T_final` access.

The dataset is `topology_predictor_p9d_v1`, with
`13,754/2,029/1,520/1,380` rows and `240/30/30/40` videos across
train/validation/test/holdout. Stage 0 passed: the split total is 18,683
rows, all split intersections and causal-input/leakage checks are zero, and
termination/truncation flags are complete.

## Frozen holdout results

The selected P9e `B` checkpoint means across three seeds were:

| Metric | Mean | Std |
|---|---:|---:|
| Layer-count MAE | 0.14420 | 0.00313 |
| Width-vector MAE | 0.13565 | 0.00083 |
| Future node-count MAE | 0.38068 | 0.01154 |
| Exact structure coverage | 0.82923 | 0.00181 |
| Auxiliary next-role macro-F1 | 0.96501 | 0.00192 |
| Auxiliary next-family macro-F1 | 0.55282 | 0.00897 |

Against the frozen empirical baseline on the same holdout, the structural
metrics were:

| Metric | Empirical baseline | Learned B |
|---|---:|---:|
| Layer-count MAE | 1.90797 | 0.14420 |
| Width-vector MAE | 0.92739 | 0.13565 |
| Node-count MAE | 4.63696 | 0.38068 |

All three learned-model errors are lower. This closes the structural
diagnostic comparison, but it does not by itself make the output a
scheduler-ready future-cost estimator.

## Calibration audit

The train-only scalar temperature calibration was not promoted. On the
holdout, mean raw versus calibrated NLL/ECE was:

| Head | Raw NLL | Calibrated NLL | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|
| Layer count | 0.33361 | 0.33446 | 0.01644 | 0.01906 |
| Width | 0.42303 | 0.42390 | 0.03458 | 0.02836 |
| Next role | 0.13095 | 0.13171 | 0.00783 | 0.00812 |
| Next family | 0.80845 | 0.81524 | 0.04453 | 0.05073 |

Width ECE improved, but its NLL worsened; all four heads had worse NLL and
the other three also had worse ECE. The current recommendation is to retain
raw probabilities and not deploy this temperature fit.

## Future-cost/resource audit

The train-only profile had 7 role/family groups, 81,426 matched successful
future-node occurrences, 2,230 excluded non-success rows and zero missing
training resource rows. Holdout runtime coverage across seeds was only
`0.8449/0.8435/0.9152`, so the aggregate resource proxy gate is false.

| Seed | Coverage | Top-1 runtime MAE (ms) | Bias (ms) | Underprediction |
|---:|---:|---:|---:|---:|
| 11 | 0.84493 | 122,544 | -115,940 | 68.61% |
| 22 | 0.84348 | 118,973 | -110,582 | 68.90% |
| 33 | 0.91522 | 114,879 | -105,013 | 64.05% |

Across seeds, top-1 runtime MAE was `118,798.7 ms`; the probability-weighted
scenario estimate was worse at `125,347.2 ms`. Both proxies systematically
underestimate the long tail. Load and peak-memory prediction were not
identifiable from the reduced future-node contract because it omits
`model_id`, `node_type`, `input_scale` and `cold_warm`.

## Stratified findings

Seed 22 is used as the representative stratified audit; the other seeds show
the same main pattern.

- By current role, structure exact coverage was `0.9338` for aggregate,
  `0.8213` for execute and `0.7814` for plan. Plan also had role/family F1 of
  `0.4042/0.4130`.
- Samples with true first-layer width 4 (`n=36`) were a clear failure stratum:
  exact structure coverage `0`, layer MAE `1.1667`, width MAE `0.9833` and
  node-count MAE `2.1389`.
- Samples with 6 or more future nodes (`n=680`) had exact structure coverage
  `0.7618`, node-count MAE `0.4132`, future role F1 `0.6600` and future family
  F1 `0.5291`.
- Horizon-truncated samples had lower semantic scores than non-truncated
  samples: future role/family F1 `0.6641/0.3581` versus `0.9602/0.5571`.

These strata point to two follow-up needs: preserve enough conditional node
content to estimate resource identity, and address wide/long-tail DAG cases
explicitly. The current coarse role/family proxy should not be fed into the
scheduler as if it were a calibrated future cost.

## Reproducibility and verification

- Remote host: `autodl-container-41d846924e-d62d0601`; execution root:
  `/root/autodl-tmp/scheduler`.
- Command:
  `cd /root/autodl-tmp/scheduler && sh experiments/EXP-20260903_p9f_predictor_acceptance_audit/run.sh`
- Remote P9f unit tests: `4/4`; Python compile check: passed.
- Local output JSON finite-value, manifest-output and local/remote hash
  checks: `4/4` each.
- P9f source hash:
  `81f00c6ce73a891a3ec153ff7c5f2d48a206c96acae6523096c25c4d55cb0937`.
- Output hashes:
  - `stage0_audit.json`:
    `e257ef37b93a8ba8859f80ef9b4237b1e54bed453fd078731acbc3248375a1f9`
  - `cost_profiles.json`:
    `0e3faaf5412787eb98be1786b3d4f8e4cd9cfba296e88b8a1a3bf7962832ddbd`
  - `metrics.json`:
    `5eef27a61d37d0f5be81968e511e1d9dee31cbc2c5b320688b38ff3bd659e841`
  - `run_manifest.json`:
    `bd959f722b8ed9f5d64ae332e5ddb1062b21c657549a004fcb73ab48034b68e1`

P9e checkpoints and prior P9d/P9e experiment directories were not modified.
The next step is a separately bounded resource-aware future-content or
conditional-aggregation experiment; do not run `S_train/S_val` integration
until that contract is accepted.
