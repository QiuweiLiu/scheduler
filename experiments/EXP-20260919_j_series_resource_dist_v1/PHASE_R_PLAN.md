# Phase R — resource head: from three quantiles to a coherent runtime distribution

Status: **R0 executed and green; R1–R4 implemented or explicitly gated; awaiting review before the R1 training run.**
Config: `experiments/EXP-20260919_j_series_resource_dist_v1/config.json`
Code: `scripts/j_series_resource_dist.py` (library), `scripts/j_series_resource_dist_train_eval.py` (CLI),
`tests/test_j_series_resource_dist.py` (13 unit tests, all passing).

## 1. Why this line exists

The frozen `J3:seed11` resource head emits three independent quantiles (`τ = 0.50 / 0.90 / 0.95`),
trained with log-space pinball at equal weight, evaluated in raw ms. Measured behaviour:

| | J validation | R7 pilot |
|---|---|---|
| p50 / p90 / p95 coverage | 0.5555 / 0.9166 / 0.9680 | 0.3960 / 0.8443 / 0.9157 |
| Spearman(true, p50) | 0.590 | 0.493 |
| tail recall (true top-10 % by p50) | 0.417 | 0.349 |
| quantile crossing (p50>p90 or p90>p95) | 6.55 % | 2.99 % |
| median p95/p50 | 9.19x | 13.32x |

Marginal coverage is fine; **node-level discrimination is not**. By true-runtime bucket the p50 is
essentially blind exactly where it matters (`point/truth` = 0.083 in the 8–12 s band in-domain,
0.166 on R7) while over-predicting the cheap band by an order of magnitude. The upper quantiles
converge toward the truth as events get expensive (p95/truth 1.1–1.4x in the 8–30 s bands), i.e. the
head hedges by widening instead of identifying.

## 2. What changes

Replace the three-point quantile head with a **categorical distribution over 24 log-spaced runtime
bins**. From one distribution the code derives p50/p90/p95, the conditional mean and any coarse band,
so the outputs are coherent by construction (no crossing, and ΣE[Y] is finally an additive statistic).

Loss (`config.resource_dist.loss_weights`):

```
L = 1.0 · NLL + 0.5 · RPS + 0.25 · pseudo-Huber(point, raw ms) + 0.25 · coarse-band CE
```

* `NLL` — mass must sit on the true bin; cannot be satisfied by widening an interval.
* `RPS` — discrete ranked probability score, respects bin order (7 s vs 8 s must cost less than 0.1 ms vs 8 s).
* `point` — `δ²(√(1+(r/δ)²)−1)` with `r = (μ̂−y)/median_train(y)`, `δ = 2`, so a handful of 22 s events
  cannot dominate the gradient.
* `band` — CE over `ultra_cheap <1 ms / light 1 ms–1 s / medium 1–5 s / expensive ≥5 s`, obtained by
  aggregating the same distribution (no second conflicting head).

Bins are **train-only**: `[0, 1 ms)` plus `n_bins−2` log1p-spaced edges up to the train p99.9 plus one
overflow bin. Recorded in `bins.json` with fill counts; never re-fitted on val/test/R7.

## 3. Why R1 is an exact "swap only the resource head" experiment

`--stage cache` runs the frozen checkpoint once per split with a forward pre-hook on `head_runtime` and
stores the exact 128-dim feature `z` the old head consumed (plus runtime labels, slot mask, video codes).
R1 trains a small head on that cache. Consequences:

* the encoder, structure/content/behaviour/attribute heads, vocabulary and checkpoint are untouched;
* any measured change is attributable to the resource head, not to a re-trained predictor;
* the training loop is cheap (the full cache is ~10 s of frozen forward on one 3060).

## 4. Stages and gates

| stage | what it does | gate |
|---|---|---|
| **R0** | rebuild bins, cache features, recompute the J3 baseline **with the R1 metric code** | must reproduce the recorded acceptance numbers: coverage 0.5555/0.9166/0.9680 and RuntimeQScore 845.0 ms |
| **R1** | train `DiscreteRuntimeHead` on the cache | see gate block below; **3 of 4** improvement gates plus both integrity gates |
| R2 | add a future-canonical-activity head | **not enabled**: `j_series_dataset_v1` has no `slot_activity` label; adding it is a data-side change |
| R3 | joint fine-tune (encoder + attribute layers open) | only after R2 passes; must not regress the existing NI endpoints |
| R4 | winner × seeds {11,22,33} + one test look | paired video-cluster CI |

R1 gates (pre-registered, `config.gates`):

* `mean|CE_τ|` not worse than J3 by more than 0.02 absolute;
* quantile crossing rate exactly 0;
* `Spearman(point, true)` ≥ J3 + 0.08;
* `tail_recall_top10(point)` ≥ J3 + 0.08;
* `log-MAE` ≥ 10 % better than J3;
* the `8000–12000 ms` bucket must reach `point/truth ≥ 0.50`;
* at least 3 of the last 4 must hold.

## 5. Evaluated metrics (four separate layers)

The previous episode's error was reading an aggregate ratio (`Σp50/Σtruth`) as calibration. The report
therefore keeps layers apart and states which quantity may be read as what:

* **A distribution calibration** — `C_τ`, `CE_τ = C_τ − τ`, crossing rate.
* **B distribution accuracy** — pinball, `NPB_τ = PB_τ / E[Y]` (normalised, so it is comparable across
  workloads), discrete RPS.
* **C node-level discrimination** — Spearman, top-10 % tail recall, log-MAE, raw MAE, per-truth-bucket table.
* **D scheduler consumption** — `Σq50/Σy`, `Σq95/Σy`, `Σμ̂/Σy`. **Only the last one is an additive
  expectation diagnostic**; the first two are consumption-scale quantities and must never be read as
  calibration.

`evaluate_quantiles()` produces the identical schema for the J3 baseline, so R1 and J3 are scored by the
same code path — the reproduction gate in R0 is what proves that.

## 6. Budget

`config.resource_dist.budget`: at most **3 single-seed model-selection runs** (R1 seed 11, R2 seed 11,
optionally one ablation); only the winner gets seeds {11,22,33}. Operational stop conditions: peak VRAM
> 5.5 GB, or > 45 min for a 30-epoch run (profile first). The v1 J line's 5-variant × 3-seed matrix is
**not** repeated.

## 7. Non-goals

* No change to the frozen `J3:seed11` checkpoint, the v1 dataset, the v03 workload, or any published
  scheduler result. Resource-v2, if adopted, produces a **new** artifact directory and a re-run of only
  the policies that consume future runtime predictions.
* No post-hoc scaling of p50 into a "mean"; that is consumer semantics, not a predictor fix.
