# Result — EXP-20260903_p9d_future_content_multitask

## Status

`passed_p1_future_content_experiment`: the remote formal run completed all 12
variant/seed fits (`N0/A/B/D × 11/22/33`). Variant `A` was selected by the
pre-registered validation structure score. This is a predictor diagnostic;
no variant is promoted to scheduler integration.

## Question

Can one shared causal GRU predict the structure and content of all future DAG
nodes, while a next-step behavior head remains useful as an auxiliary task?
Does soft topology conditioning improve the future-node content decoder?

## Registered design

- `N0`: next-step behavior-only reference.
- `A`: future layer/width structure plus all-future-node content.
- `B`: `A` plus next-step behavior auxiliary loss.
- `D`: `B` plus soft layer/width probability conditioning of the content head.
- One shared unidirectional causal GRU encoder with separate behavior,
  structure and content heads.
- Fixed layer-wise unordered slots; one joint bipartite assignment per active
  layer is reused for `node_type`, `raw_action`, `model_id` and
  `action_family`.
- Horizon `H=5`, maximum layer width `5`, maximum `3` identity-free scenarios.

The structure score is validation `layer_count_mae + width_vector_mae`. The
implementation computes width-vector MAE over the full five-position horizon,
with inactive positions derived as zero; the configuration label
`active_width_vector_mae` is therefore a naming imprecision, not a different
metric or a post-hoc selection rule.

## Data and boundary

- Fit: `P_dev/train` (13,754 anchors, 240 videos, 988 runs).
- Epoch/variant selection: `P_dev/validation` (2,029 anchors, 30 videos,
  144 runs).
- Diagnostic: `P_dev/test` (1,520 anchors, 30 videos, 108 runs).
- Frozen acceptance: `P_holdout_diag/holdout` (1,380 anchors, 40 videos,
  120 runs).
- Dataset manifest SHA-256:
  `c921f6ee4dbdee1b01af102ad6d02b29a843f91392357ce7fad78cf24df84630`.
- No scheduler groups, `S_train/S_val`, future events/edges, node IDs,
  execution truth, resource truth, raw traces or `T_final` were used as model
  inputs. Raw traces were not modified.

## Aggregate results

Values are mean over three seeds; higher is better for F1/exact coverage and
lower is better for MAE.

| Variant | Validation structure score | Validation predicted-content F1 | Validation next-family F1 | Holdout structure exact | Holdout layer MAE | Holdout width MAE | Holdout predicted-content F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `N0` | — | — | 0.8075 | — | — | — | — |
| `A` | **0.1272** | 0.6715 | — | 0.8150 | 0.1498 | 0.1426 | **0.4158** |
| `B` | 0.1274 | **0.6748** | 0.7904 | 0.8191 | 0.1444 | 0.1408 | 0.4117 |
| `D` | 0.1277 | 0.6738 | 0.7829 | **0.8244** | **0.1437** | **0.1371** | 0.4127 |

`A` is selected because the selection rule is validation structure score, not
holdout performance. `D` has the best frozen-holdout structure metrics, but
that result cannot be used to replace the pre-registered selection.

`N0` structure/content metrics are intentionally not compared: its structure
and content losses are disabled. Conversely, `A`'s next-step head is not
trained by design, so its next-step metrics are not valid behavior evidence.
On validation, `B` and `D` do not establish next-family non-inferiority to
`N0`; their family macro-F1 is lower than `N0` (`0.7904/0.7829` vs `0.8075`).

The content head generalizes with a clear domain gap: predicted content F1 is
about `0.67` on validation and `0.416/0.412` for `A/D` on holdout. The exact
top-1/top-3 complete content-signature coverage is only about `0.071` on
validation/test and `0.087` on holdout, so it is retained as a secondary
diagnostic rather than a deployment claim.

## Output contract

- Every prediction row carries `next_step_behavior` with role and
  execute-gated family predictions; `N0/B/D` train this head, while `A`
  retains it structurally but disables its loss.
- `A/B/D` expose `top_scenarios`: probability-normalized, identity-free future
  layer scenarios. Each active layer contains unordered node slots with the
  four approved content fields. The active mask is derived from predicted
  layer count and prefix widths.
- Outputs do not emit real node IDs, successor edges, future resources or
  execution truth. They are not yet a scheduler-ready future-cost artifact.

## Verification and recovery

- Stage 0 passed on all 18,663 rows: split video/trace intersections `0`,
  causal input violations `0`, direct identifier leakage `0`, invalid layer
  labels `0`, duplicate node IDs `0`, and complete truncation flags.
- Local relevant suite: `22` tests, `20` passed and `2` skipped because the
  local environment lacks PyTorch. Remote P9d suite: `17/17` passed in the
  `finetooling` environment. Actual-data forward/loss/backward smoke for `B`
  and `D` also passed.
- The first remote process stopped at `A/seed_11` due to an evaluation
  variable-reuse bug. The bug was fixed, the incomplete run was safely resumed,
  and the final root metrics were generated only after all 12 runs completed.
- The recovered local experiment contains 62 non-checkpoint outputs, including
  36 prediction files. All 62 hashes match `run_manifest.json`; the 12 model
  checkpoints remain remote-only at
  `/root/autodl-tmp/scheduler/experiments/EXP-20260903_p9d_future_content_multitask/artifacts/`.

## Conclusion and next gate

The shared causal GRU plus separate structure/content/behavior heads is
implementable and passes the data, leakage and output-contract gates. `A` is
the current validation-selected structural reference; `B` is the strongest
validation content variant; `D` is the strongest frozen-holdout structural
diagnostic. None is frozen for scheduling because content distribution shift
remains substantial and the auxiliary behavior head did not pass validation
non-inferiority.

The next in-scope step is train-only field/condition calibration and
future-cost/resource aggregation diagnostics. Only after that passes may the
frozen candidate be run as inference-only on `S_train/S_val`; `T_final`,
`WAIT/RESERVE`, preemption, multi-GPU semantics and scheduler promotion remain
closed.
