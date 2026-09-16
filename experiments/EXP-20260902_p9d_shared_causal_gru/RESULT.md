# Result — EXP-20260902_p9d_shared_causal_gru

## Status

`passed_diagnostic`: the remote 12-run shared causal GRU experiment completed and its output contract passed independent local audit. This does not close the final topology predictor gate and does not authorize scheduler integration.

## Question and design

The experiment tests whether one causal history encoder can support both the existing behavior targets and a new identity-free future-DAG topology target.

- Encoder: one unidirectional causal GRU over the visible prefix/current-node history, plus visible task/stack context.
- Behavior heads: next role and execute-gated family.
- Topology heads: future layer count, five layer-width heads, node prototype fields, and auxiliary topology role/family heads.
- Output: top-3 H=5 identity-free DAG-layer scenarios. Predictions contain no true node IDs, edges, future events, execution truth, resource truth, or video ID.
- Variants: `behavior_only`, `topology_only`, `shared_multitask`, `shared_first_layer_consistency`.
- Seeds: 11, 22, 33.

## Data and evaluation boundary

The run used the predictor-aligned P9d dataset `topology_predictor_p9d_v1` with manifest SHA-256 `c921f6ee4dbdee1b01af102ad6d02b29a843f91392357ce7fad78cf24df84630`:

| Split | Anchors | Videos | Runs |
|---|---:|---:|---:|
| P_dev/train | 13,754 | 240 | 988 |
| P_dev/validation | 2,029 | 30 | 144 |
| P_dev/test | 1,520 | 30 | 108 |
| P_holdout_diag/holdout | 1,380 | 40 | 120 |

All model parameters and vocabularies were fit on `P_dev/train`; validation selected the best epoch; test was diagnostic; holdout was read only after each run was frozen. Behavior role labels joined 100% of anchors. Execute-gated family coverage was 34.26%/30.31%/36.97%/38.12% for train/validation/test/holdout.

## Results

Values below are mean ± population standard deviation over the three seeds. Lower is better for MAE; higher is better for coverage and accuracy.

### Topology

| Variant | Split | Layer MAE | Node MAE | Width-vector MAE | Exact shape | Exact signature top-1 / top-3 |
|---|---|---:|---:|---:|---:|---:|
| `topology_only` | validation | 0.0455 ± 0.0013 | 0.3480 ± 0.0024 | 0.0829 ± 0.0004 | 0.7958 | 0.4657 / 0.4677 |
| `shared_multitask` | validation | 0.0483 ± 0.0018 | 0.3509 ± 0.0026 | 0.0831 ± 0.0007 | 0.7973 | 0.4643 / 0.4664 |
| `shared_first_layer_consistency` | validation | 0.0471 ± 0.0023 | 0.3540 ± 0.0079 | 0.0839 ± 0.0016 | 0.7935 | 0.4613 / 0.4638 |
| `topology_only` | holdout | 0.1522 ± 0.0036 | 0.4261 ± 0.0313 | 0.1414 ± 0.0046 | 0.8169 | 0.2966 / 0.2966 |
| `shared_multitask` | holdout | **0.1442 ± 0.0026** | **0.3995 ± 0.0181** | **0.1378 ± 0.0033** | **0.8217** | 0.2995 / 0.2995 |
| `shared_first_layer_consistency` | holdout | 0.1447 ± 0.0028 | 0.4051 ± 0.0082 | 0.1394 ± 0.0014 | 0.8186 | **0.3036 / 0.3036** |

The pre-registered validation structural score was layer MAE + width-vector MAE: `topology_only=0.1284`, `shared_first_layer_consistency=0.1311`, and `shared_multitask=0.1314`. Therefore the protocol-selected topology variant is `topology_only`; holdout results are not used to change that selection. On frozen holdout, `shared_multitask` has the best combined structure errors, while consistency has the highest exact-signature coverage by a small margin.

### Behavior preservation

| Variant | Split | Role accuracy | Family accuracy | Joint accuracy |
|---|---|---:|---:|---:|
| `behavior_only` | holdout | 0.9621 | 0.6679 | 0.8355 |
| `shared_multitask` | holdout | **0.9630** | **0.6901** | **0.8449** |
| `shared_first_layer_consistency` | holdout | 0.9609 | 0.6787 | 0.8384 |

On this holdout, shared multi-task learning did not degrade the behavior target and improved the three behavior metrics relative to the same-run behavior-only control. Validation behavior is close but not uniformly better: shared multitask joint accuracy is 0.8894 versus 0.8924 for behavior-only.

### Comparison with earlier topology baselines

| Model | Holdout layer MAE | Node MAE | Width MAE | Exact shape | Signature top-3 |
|---|---:|---:|---:|---:|---:|
| Empirical `current_full` | 1.9080 | 4.6370 | 0.9274 | not reported | 0.3297 |
| Tabular `lgbm_small` | 0.1696 | 0.5435 | 0.1655 | 0.7435 | 0.0870 |
| GRU `topology_only` | 0.1522 | 0.4261 | 0.1414 | 0.8169 | 0.2966 |
| GRU `shared_multitask` | **0.1442** | **0.3995** | **0.1378** | **0.8217** | 0.2995 |

The neural topology heads substantially improve structural prediction over the tabular comparator. The empirical baseline has higher exact signature top-3 than the neural variants despite much worse scale/shape errors; this indicates that exact full-signature coverage and structural calibration are different failure modes and must both remain reported.

## Independent verification

- Remote PyTorch forward/loss/backward contract test: 4/4 passed.
- Local contract tests: 3 passed, 1 skipped because local macOS Python has no PyTorch; the skipped test passed remotely.
- Local/remote source, config, root metrics and run-manifest hashes matched.
- Run manifest output hashes matched all recovered local artifacts.
- 36 prediction files were read independently: 27 topology files contained all expected rows (`2,029/1,520/1,380` per split), 9 behavior-only topology files were intentionally empty.
- Top-3 probability normalization violations: 0.
- Predicted layer/width shape violations: 0.
- Prediction identity/edge/resource-truth leakage violations: 0.
- Duplicate or missing prediction sample IDs: 0.

## Decision and boundary

This experiment closes the shared-GRU diagnostic stage, not the complete predictor or scheduler stage. The evidence supports keeping a shared causal encoder as a serious candidate. The first-layer consistency loss does not show a clear benefit over plain shared multitask learning, so it is not promoted as the default interaction mechanism.

No scheduler groups were used for fitting, no scheduler integration was started, `T_final` was not read, and raw traces/old R7 artifacts were not modified. Before any scheduler experiment, the selected frozen topology output still needs resource/future-cost calibration and an opt-in scheduler contract test on `S_train/S_val`; the final `T_final` gate remains closed.
