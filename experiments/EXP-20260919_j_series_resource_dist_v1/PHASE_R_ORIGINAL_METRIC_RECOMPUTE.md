# 用 J3 原始验收口径重算（RuntimeQScore / PB / Coverage）— validation, same 7,195 pairs

Trigger: the project owner objected that ranking (Spearman) is not the objective - the resource head
exists to predict runtime accurately - and asked for relative-error-style evaluation. The review
agreed, self-corrected ('I made Spearman too strong a proxy'), and its first instruction was:
**restore J3's original acceptance metrics for every arm before anything else.**

## Results

| arm | PB50 | PB90 | PB95 | RuntimeQScore | cal err | Cov50 | Cov90 | Cov95 |
|---|---|---|---|---|---|---|---|---|
| J3 (3-quantile, frozen acceptance baseline) | 1194.3 | 797.9 | 542.9 | **845.0** | 0.0301 | 0.5555 | 0.9166 | 0.9680 |
| R1 linear + 16 bin | 1070.3 | 702.9 | 461.1 | **744.8** | 0.0259 | 0.4446 | 0.8848 | 0.9429 |
| R1b MLP + 16 bin | 975.9 | 612.5 | 413.3 | **667.3** | 0.0245 | 0.4421 | 0.8874 | 0.9469 |
| R3a-F frozen (3-seed mean) | 998.8 | 613.4 | 411.7 | **674.6** | 0.0433 | 0.4112 | 0.8690 | 0.9399 |
| R3a-U adapt (3-seed mean) | 902.5 | 619.0 | 414.3 | **645.2** | 0.0761 | 0.4048 | 0.8026 | 0.9142 |

## Verdict

Judged by J3's own frozen acceptance protocol:

* **R1b (MLP + 16 bins) improves BOTH original criteria** - RuntimeQScore 845.0 -> 667.3 (**-177.8 ms, -21.0 %**) and the
  calibration error improves too (0.0301 -> 0.0245, i.e. better than J3).
* R1 (linear) also improves both, by 11.9 % on RuntimeQScore.
* **R3a-U buys a little more pinball (23.6 % vs J3) but BREAKS calibration**: its p90 coverage falls to
  0.8026 against J3's 0.9166, which would **fail the original integrity gate** (calibration not worse by more
  than 0.02 absolute). So the extra unfreezing is not worth it under the original protocol.

**Conclusion: R1b is the winner by the project's own original criterion**, and the whole Spearman chase
was measuring something the scheduler does not consume.

## The metric-governance issue (review's instruction)

The review's ruling on the Spearman gate: it was a legitimate *diagnostic* (it found that J3 calibrates
marginally but discriminates instances poorly) but promoting it to a mandatory acceptance threshold
(+0.08 or 'strong FAIL') was a **metric misalignment**, because the scheduler does not rank all 7,195
slots - it aggregates per-job costs and compares candidates at a decision state.

Governance (must not delete or rewrite history):

1. keep *Evaluation Protocol v1* and its result permanently: **R3a strong gate: FAIL**;
2. issue a timestamped **Metric Amendment / Evaluation Protocol v2** recording: the change reason
   (Spearman is a Phase-R-introduced node-level proxy, not part of the J3 frozen acceptance protocol,
   and not what the scheduler consumes), the new tiers, and the fact that the validation results were
   already seen while test stayed untouched;
3. never claim retrospectively that R3a passed. Correct phrasing: "R3a failed the original Phase-R
   strong proxy gate; the proxy was subsequently judged misaligned with the downstream task and was
   not retained as an acceptance criterion in protocol v2."

## Metric tiers for protocol v2

| tier | metric | role |
|---|---|---|
| PRIMARY-P | RuntimeQScore = mean(PB50, PB90, PB95) | J3's original acceptance criterion; new head must not be worse |
| PRIMARY-P | PB95 + coverage95 | the scheduler consumes the upper quantile |
| PRIMARY-S | the scheduler benchmark's original primary objective | system-level judge; do not invent a new one |
| SECONDARY | log-MAE(q50); RPS/CRPS; sum(E[T])/sum(Y); raw MAE(q50); normalized pinball per tau | relative / distributional accuracy |
| DIAGNOSTIC | per-band median pred/true; per-slot / per-family / per-role; **Spearman / pairwise concordance / tail recall** | locating failure modes; no longer acceptance criteria |

Normalized pinball should use a FIXED training-set scale, `s = mean(Y_train, Y>0)`, so it stays a
proportional rescaling of pinball and remains comparable with J3. MAPE/sMAPE/MdAPE are explicitly NOT
suitable as primary here: the data has a large 0.1 ms mass plus 10 s+ tails, which is the textbook
failure case for ratio metrics.

## Next step (review's ordering, after this table)

1. lock protocol v2 (no Spearman pass/fail; test untouched);
2. **300 paired-episode scheduler smoke**: J3 vs R1b vs R3a-U with workload, arrival trace, deadlines,
   GPU state and seeds all locked; stratified from the existing benchmark's scenario space;
   read the ORIGINAL scheduler objective, plus decision disagreement rate, candidate-cost rank
   agreement and decision regret as diagnostics;
   pre-fix: R1b/R3a-U must not degrade the main objective point estimate and at least one system metric
   must improve, with no systematic per-stratum degradation;
3. only if prediction improved but scheduling did not, run a decision-state audit before resuming the
   representation / ranking-loss line.
