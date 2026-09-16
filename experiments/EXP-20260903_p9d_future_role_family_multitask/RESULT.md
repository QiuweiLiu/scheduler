# Result — EXP-20260903_p9d_future_role_family_multitask

## Status

`passed_diagnostic`: the reduced future-node decoder completed the registered
remote experiment and independent output-contract verification. The selected
variant is `B` by the pre-registered validation structure score. This is a
predictor diagnostic, not a scheduler input.

## Design

- `N0`: next-step behavior-only reference.
- `A`: future layer/width structure plus per-future-node `role` and
  `action_family`.
- `B`: `A` plus auxiliary `next_role` and `next_family_if_execute` losses.
- `D`: `B` plus soft layer/width probability conditioning into the future-node
  role/family decoder.
- All variants use one shared unidirectional causal GRU, H=5, maximum width 5,
  and one joint layer-wise assignment reused across the two future-node fields.

## Data and boundary

Fit used `P_dev/train`; validation selected epochs and the topology variant;
`P_dev/test` was diagnostic; `P_holdout_diag/holdout` was opened only after
selection froze. The dataset was the predictor-aligned P9d dataset
`topology_predictor_p9d_v1` with 13,754/2,029/1,520/1,380 rows and
240/30/30/40 videos across train/validation/test/holdout. No `S_train/S_val`
labels, scheduler groups, future events/edges, node IDs, resource truth or
`T_final` were used as model inputs.

## Results

Validation means across three seeds:

| Variant | Structure score (layer MAE + width MAE) | Future role+family F1 | Next-role F1 | Next-family F1 |
|---|---:|---:|---:|---:|
| A | 0.13218 | 0.72584 | 0.10294 | 0.10973 |
| B | **0.12847** | **0.72873** | 0.95479 | 0.78436 |
| D | 0.13103 | 0.72838 | 0.95532 | **0.78914** |
| N0 | reference only | — | 0.95506 | 0.80821 |

On the frozen holdout, selected `B` achieved:

- structure score `0.27986` (`layer_count_mae=0.14420`,
  `width_vector_mae=0.13565`), exact structure coverage `0.82923`;
- future-node role+family macro-F1 `0.65255`, split into role `0.90151` and
  action-family `0.40360`;
- auxiliary next-role/next-family macro-F1 `0.96501/0.55282`.

The reduced two-field content score is not directly comparable with the prior
four-field macro-F1 because the label set changed. On the common
`action_family` field, selected `B` was `0.50666/0.49099/0.40360` on
validation/test/holdout, versus `0.52043/0.50889/0.40982` in the prior
four-field `B` experiment. Thus the simplification clearly improves the
output contract and makes future `role` prediction strong, but it does not
yet improve action-family accuracy by itself.

Compared with the prior four-field `B`, the new `B` structure score is almost
unchanged on validation (`0.12847` vs `0.12739`) and improves on holdout
(`0.27986` vs `0.28527`). `B` is selected by validation; `D` is not promoted
because its validation structure score and holdout structure exact coverage
are worse than `B` (`0.81715` vs `0.82923`).

## Output contract and verification

Each non-reference prediction row contains `next_step_behavior` for the
one-step auxiliary output and up to three synthetic future scenarios. Each
scenario contains contiguous predicted DAG layers; each active node contains
only `role` and `action_family` plus synthetic slot metadata. Node IDs, edges,
raw actions, model IDs, resource truth and future execution truth are not
emitted as predicted node content.

Stage 0 passed with zero split/causal/leakage errors. All 12 runs completed;
remote P9e tests passed 4/4; the real 8-row forward/loss/backward smoke passed;
36 prediction files and 62 non-checkpoint output files matched the run
manifest hashes; the independent prediction-contract audit reported zero
errors. Twelve checkpoints remain on the remote execution copy.

The old four-field experiment remains unchanged. Scheduler integration,
`S_train/S_val` inference, raw trace changes and `T_final` remain out of scope.
