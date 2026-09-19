# Phase R / R1 — discrete runtime distribution, seed 11 (validation only)

**Verdict: viability PASS, strong FAIL.** Test split was NOT touched.

Frozen J3:seed11 features, only the resource output head replaced (Linear 128 -> 24, softmax over
train-only log-spaced bins). 30 epochs, no early stopping, best epoch chosen by the pre-registered
rule (calibration-eligible -> argmin val log-MAE): **epoch 27**. Wall time 64 s on one RTX 3060.

| metric (point = q50, like-for-like with J3) | J3 baseline | R1 discrete | change |
|---|---|---|---|
| quantile crossing | 0.00514 | 0.00000 | fixed |
| mean |CE| | 0.0301 | 0.0313 | within +0.02 |
| Spearman (q50) | 0.5898 | 0.6069 | +0.0170 |
| tail recall top-10%% (q50) | 0.4167 | 0.5278 | +0.1111 |
| log-MAE | 1.8020 | 1.6295 | -9.6% |
| raw MAE (ms) | 2388.7 | 2249.1 | -5.8% |
| 8-12 s bucket pred/true | 0.0833 | 0.3694 | x4.4 |
| sum q50 / sum y | 0.6322 | 0.8040 | toward 1 |
| sum q95 / sum y | 3.1008 | 2.3240 | less over-conservative |
| sum mean / sum y | n/a (J3 has no mean) | 0.9411 | additive diagnostic |

## Gate outcome

* integrity: **PASS** (calibration not worse by more than 0.02; quantile crossing exactly 0.0)
* viability: **PASS** (log-MAE -9.6 % <= -5 %; 8-12 s bucket 0.369 >= 0.25; tail recall +0.111 >= +0.04)
* strong: **FAIL** (Spearman +0.017 < +0.08; log-MAE -9.6 % vs the -10 % bar; bucket 0.369 < 0.50)

## Reading

1. The failure mode this line was built to fix is fixed: the 8-12 s bucket moved from 0.083 to 0.369
   (x4.4) and the cheap-node over-prediction shrank, while marginal coverage stayed inside the
   integrity band. Quantile crossing is now impossible by construction rather than merely rare.
2. Node-level ranking improved in the tail: tail recall +0.111 is the largest single gain.
3. The conditional mean is now available and is the only additive statistic: sum(mean)/sum(y) = 0.941,
   i.e. essentially unbiased at the aggregate level, versus sum(q50)/sum(y) = 0.632 for J3.
4. Point-metric comparability: layer C uses q50 for both predictors. The mean's log error is much
   larger (2.93) purely because the mean of a right-skewed runtime sits far above the median; this is
   reported as `mean_point_diagnostics` rather than gated.

## Pre-registered next step

The R1->R1b trigger agreed in review: viability PASS, no severe train/val gap, and at least one
ranking strong gate unmet -> run R1b (128 -> 64 GELU -> 24) as the second single-seed
model-selection run. Budget allows it (2 of 3 used).
