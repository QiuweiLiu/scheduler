# Phase R — R1 / R1b on the mass-balanced 16-bin layout (validation only, seed 11)

Frozen J3:seed11 features, only the resource output head replaced. Test split NOT touched.
30 epochs each, no early stopping; best epoch by the pre-registered rule (calibration-eligible ->
argmin val log-MAE, tie-break Spearman). Wall time 96 s (R1) / 100 s (R1b) on one RTX 3060.

## What changed versus the previous R1 report

1. **Binning**: the log-uniform 24-bin layout is gone. The new `mass_balanced` layout gives one
   dedicated near-zero bin plus empirical-quantile log bins, with the coarse-band boundaries (1 s,
   5 s) forced onto exact edges, a minimum-support rule (max(200, 0.5 % of N) = 242 samples, capped
   at half the average bin mass), and the overflow edge at train p99.5 instead of p99.9.
   Result: **0 empty bins** (was 9/24 = 37.5 %), minimum non-metadata bin 242 (was 49), overflow
   0.501 % (was 2.08 %), and the metadata cluster keeps its own bin (34.9 % of the mass).
2. **Per-slot reporting**: every report now carries slot0..slot4 blocks plus an unweighted macro,
   including the global-tail-within-slot recall and a video-cluster CI on the 8-12 s ratio.
   The pooled micro numbers remain the primary gate (pre-registered); the per-slot numbers are a
   diagnostic secondary endpoint and were NOT used to move any threshold.

## Pooled (primary) results

| arm | Spearman | tail recall | log-MAE | raw MAE | 8-12 s ratio | crossing | cal err |
|---|---|---|---|---|---|---|---|
| J3 (3 quantiles, frozen) | 0.5898 | 0.4167 | 1.8020 | 2389 | 0.0833 | 0.0051 | 0.0301 |
| R1_linear_16bin | 0.6162 | 0.5222 | 1.6046 | 2141 | 0.4395 | 0.0000 | 0.0259 |
| R1b_mlp_16bin | 0.6354 | 0.5028 | 1.5185 | 1952 | 0.4395 | 0.0000 | 0.0245 |

Coverage and aggregate totals:

| arm | cov90 | cov95 | sum(mean)/sum(y) | sum(q50)/sum(y) | sum(q95)/sum(y) |
|---|---|---|---|---|---|
| J3 | 0.9166 | 0.9680 | n/a | 0.6322 | 3.1008 |
| R1_linear_16bin | 0.8848 | 0.9429 | 0.9867 | 0.8367 | 2.4900 |
| R1b_mlp_16bin | 0.8874 | 0.9469 | 0.9961 | 0.8431 | 2.3434 |

## Per-slot 8-12 s ratio (the failure mode this line targets)

| slot | J3 | R1b (MLP) | R1b n | R1b video-cluster CI |
|---|---|---|---|---|
| slot0 | 0.0887 | 0.4403 | 188 | [0.436, 1.019] |
| slot1 | 0.0306 | 0.2998 | 186 | [0.298, 1.015] |
| slot2 | 0.1335 | 0.4401 | 181 | [0.360, 1.017] |
| slot3 | 0.0233 | 0.3627 | 161 | [0.361, 1.014] |
| slot4 | 0.1579 | 0.4393 | 152 | [0.361, 1.014] |

The improvement is **uniform across all five slots** (J3: 0.089/0.031/0.134/0.023/0.158 ->
R1b: 0.440/0.300/0.440/0.363/0.439), so the pooled gain is not a single-slot artifact. The
per-slot CIs are wide because each slot only holds 152-188 of these nodes.

## Gate outcome (pooled, pre-registered)

| arm | integrity | viability | strong |
|---|---|---|---|
| R1_linear_16bin | True | True | False |
| R1b_mlp_16bin | True | True | False |

R1b against the four strong components:
- Spearman +0.0456 (needs +0.08): MISS
- tail recall +0.0861 (needs +0.08): PASS
- log-MAE -15.7% (needs -10%): PASS
- 8-12 s ratio 0.4395 (needs 0.50): MISS
-> 2 of 4, needs 3. Strong FAIL, but only on two narrow misses.

## What this answers

1. **The old three-quantile output form was part of the problem**: with the representation held
   fixed, swapping only the resource head moves the 8-12 s band by x5.3, the tail recall by
   +0.086, log-MAE by -15.7 %, and it removes quantile crossing entirely while keeping the
   calibration error **better** than J3 (0.0245 vs 0.0301).
2. **The conditional mean is now available and is essentially unbiased at the aggregate**:
   sum(mean)/sum(y) = 0.9961 (R1b) versus sum(q50)/sum(y) = 0.632 for J3. This is the first
   additive workload statistic the project has; it must still be described as an aggregate
   calibration ratio, not as proof of unbiasedness.
3. **Capacity helps, but not enough**: the MLP beats the linear head on every node-level metric
   (Spearman +0.029, log-MAE -0.086, raw MAE -189 ms), so the frozen feature does carry
   non-linearly recoverable information - but the ranking gap to the strong bar does not close.
   This is consistent with the review's INFERENCE that freezing a representation co-adapted to the
   OLD objective understates what a new objective could reach; it is not yet proof.
4. **Wording discipline (from the review)**: do NOT call R1/R1b a measurement of a
   'representation ceiling'. Call it `fixed-representation probe / frozen-feature recoverability`.

## Next pre-registered step

R3a partial-unfreeze diagnostic (the third and last single-seed model-selection run): two arms with
an identical head, F = fully frozen vs U = only the last shared module before the 128-d feature
unfrozen; paired video-level bootstrap; the frozen representation is judged limiting only if the U-F
Spearman gain has a 95 % CI lower bound > 0 AND closes at least half of the remaining strong
gap (+0.0315 Spearman or +0.065 on the 8-12 s ratio).
