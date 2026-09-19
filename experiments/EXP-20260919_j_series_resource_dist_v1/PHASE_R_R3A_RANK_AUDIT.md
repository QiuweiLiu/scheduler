# R3a rank audit (zero training) — did the quantised q50 hide the ranking gain?

Trigger: the review flagged that Spearman computed from a 16-bin quantised median has massive
ties, so two different predicted distributions that share a bin get the same score, and any
ordering information the head learned may be invisible. This audit answers that with the existing
predictions only - no training, 297 s of compute.

## Spearman deltas (U - F, mean over seeds 11/22/33, paired video-cluster bootstrap)

| ranking score | mean delta | CI95 | fraction <= 0 |
|---|---|---|---|
| q50 | +0.0143 | [-0.0092, +0.0416] | 0.173 |
| E[T] | -0.0237 | [-0.0498, +0.0054] | 0.938 |
| E[log1pT] | +0.0170 | [-0.0121, +0.0548] | 0.272 |

## Within-video pairwise concordance (the ranking-specific statistic)

| ranking score | mean delta | CI95 | fraction <= 0 |
|---|---|---|---|
| q50 | +0.00066 | [-0.00481, +0.00641] | 0.389 |
| E[T] | +0.00035 | [-0.00438, +0.00542] | 0.446 |
| E[log1pT] | -0.00480 | [-0.01131, +0.00284] | 0.926 |

## Verdict: **ranking gain is genuinely small**

1. **The ties hypothesis is refuted.** Replacing the quantised median with a continuous score
   (`E[log1p T]`) raises the delta only to +0.0170, still below the review's 0.02 threshold, and
   `E[T]` is actually **negative** (-0.0237). The 16-bin quantisation was not hiding a ranking gain.
2. **The most ranking-specific statistic says zero.** Within-video pairwise concordance moves by
   +0.0007 (q50) / +0.0004 (E[T]) / -0.0048 (E[log1p T]) - i.e. no improvement, with the
   log-scale score slightly negative in 93 % of bootstrap replicates.
3. Therefore the earlier finding stands on its own merits: **adapting the last shared module
   improves the magnitude/likelihood of the runtime prediction (8-12 s ratio 0.36 -> 0.54,
   log-MAE 1.544 -> 1.466, NLL 1.535 -> 1.293 in every seed) but does not improve the ordering**
   of the predictions.

## What the review said to do next

The review pre-registered the order: continuous-score rank audit -> oracle-q(A) -> q(A) error audit
-> one-shot [repr_ctx + last GRU + ranking loss] -> only then deeper unfreezing. Because the
continuous delta came out below 0.02, the pre-registered next step is the **oracle-q(A) probe**:
two fully paired arms that differ only in the attribute input (predicted distribution vs true
one-hot), answering how much the resource prediction would gain if attribute prediction were
perfect. That separates the *attribute-prediction* bottleneck from the *resource-mapping* one.

The review also warned that this probe must NOT be built as 'train with predicted q(A), evaluate
with true q(A)' - that is a distribution shift. Both training and validation must use the arm's own
input type.

## Budget note

The review recommends moving from a model-search phase to a diagnostic phase plus one locked
confirmatory experiment: no LR sweeps, no unfreeze-depth sweeps, no head-width sweeps; inherit all
R3a hyper-parameters and give any newly unfrozen layer a lower LR (e.g. 0.1x the head LR).
