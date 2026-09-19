# R3a — partial-unfreeze diagnostic, 3 seeds (validation only, test untouched)

Two arms with an **identical** head (Linear 180 -> 64 GELU -> 16) on the same cached input:

* **F** (frozen): only the head trains; `res_hidden` keeps the frozen J3 weights
* **U** (adapt): the head **and** `res_hidden` train — `res_hidden` is the last shared module,
  i.e. the one that maps `[repr_vec, slot_vec, q(A)]` to the 128-d feature the runtime head reads

## Per-seed results

| seed | Spearman F | Spearman U | delta | 8-12 s F | 8-12 s U |
|---|---|---|---|---|---|
| 11 | 0.6276 | 0.6418 | **+0.0141** | 0.3631 | **0.5355** |
| 22 | 0.6278 | 0.6441 | **+0.0162** | 0.3000 | **0.5355** |
| 33 | 0.6253 | 0.6378 | **+0.0124** | 0.3631 | **0.5355** |

**Sign consistency: 3/3 seeds with a positive Spearman delta.**

## Paired video-cluster bootstrap (2000 draws, averaging over the 3 seeds)

| quantity | mean delta | CI95 | fraction <= 0 |
|---|---|---|---|
| Spearman | +0.0143 | [-0.0096, +0.0425] | 0.172 |
| 8-12 s ratio | +0.1724 | [0.0000, 0.2734] | 0.149 |

## Verdict

**The pre-registered rule returns "no evidence that the frozen representation is limiting"** - the
Spearman delta CI covers 0 and, with only 30 video clusters, more seeds cannot narrow it (the seed-11
CI [-0.0103, +0.0438] and the 3-seed CI [-0.0096, +0.0425] are essentially the same width: the
dominant variance is the video sampling, not the seed).

**What did come out of it, stated carefully:**

1. Every seed moves in the same direction on every metric (3/3 on Spearman, 3/3 on the 8-12 s band,
   plus log-MAE and NLL in all three) - consistent, but not statistically resolved.
2. The 8-12 s effect is a discrete shift, not a fitted number: the median predicted value for those
   nodes moves from bin 3 (408 ms) to bin 8 (5590 ms) in all three seeds, i.e. it crosses five bin
   boundaries. The metric is therefore quantised by the bin representatives and should be reported
   with that granularity in mind.
3. The size is modest: 5.6 s against a true 10.4 s is still 1.9x low; Spearman moves by about +0.014.

**Reading for the next decision:** the input information is demonstrably present (the coarse
`action_family` plus `node_type`/`role` already separate the 0.1 ms metadata ops from the 5-10 s LLM
calls, so adding a finer action label would only affect the ~10 % of nodes that share
`node_type=videotool_spatial, role=execute`). What is left is a *mapping* problem: the frozen layer
that turns type information into a runtime scale under-uses it, and adapting that layer helps in the
right direction but not enough to be conclusive at this sample size.

## Budget note

The pre-registered allowance of 3 single-seed model-selection runs is now spent (R1, R1b, R3a). The
R3a run was additionally repeated for seeds 22 and 33 as a confirmation, not as further selection.

## Artefacts

* `r3a/r3a_multiseed.json`, `r3a/r3a_diagnostic.json`, `r3a/history_{F,U}_seed*.json`,
  `r3a/preds_{F,U}_seed*.npz`, `r3a/head_{F,U}_seed*.pt`, `r3a/cache_summary.json`
