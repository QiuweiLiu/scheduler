# Phase 21-C — why does the frozen predictor look so bad? (raw-trace audit, 2026-09-18)

Triggered by the project owner's objection: *"my predictor cannot possibly be this bad"*.
Everything below is computed on the existing raw traces / templates; **no GPU, no predictor rerun**.

## A. GPT's P0 confirmed but too small to matter

`src/tracing/collectors/videotool_phase1.py` writes the summarization tool's full wall clock as its
`action` runtime, while `AnswerModel.generate()` separately logs an inner
`event_type="api_call", action="generalist.generate"`. Claim: summing both double-counts.

Measured over all 640 R7 runs (total runtime 69,150,094 ms):

| quantity | value |
|---|---|
| `generalist.generate` events | 809, 3,434,412 ms (**4.97 %** of total) |
| `summarization-tool` events | 169, 343,317 ms (0.50 %) |
| generates whose timestamp falls INSIDE a summarizer interval | **169 / 169 summarizers** |
| inner / outer runtime ratio | median **0.758**, p90 0.992, max 0.996 |
| **double-counted time as a share of ALL runtime** | **0.48 %** |

So the containment is real (a generate sits inside every summarizer and accounts for ~76–99 % of it),
but the affected time is **0.48 % of the total**. With R50 = R50/(1−D) and D = 0.0048, R50 moves
0.436 → **0.438**. **This cannot explain the gap.** The J-training chain does merge this nested call
(`build_p9d_topology_dataset.py:634–650`), so Phase 21's truth is inconsistent with the training
contract — a real but negligible-magnitude defect.

## B. `answer` is not a residual container — v03 removal was correct

Runtime by `event_type`: `run` 50.236 % (1280 events), `api_call` 40.643 % (4688), `action` 9.121 % (3607).
Runtime by `action`: `answer` **50.236 %** (640 events) — the identical share.

Per run, `answer`'s `runtime_ms` equals the `run` container's `runtime_ms` **exactly**
(**640/640 runs, ratio min = median = max = 1.000**).

Conclusion: `answer` is the *same physical node* as the `run` container (dual labelling), so the v03
workload's container removal already eliminated it. **There is no residual 50 % inflation node.**

## C. It is NOT a runtime-scale domain shift either

Comparing the in-domain J training slots against the R7 v03 template events:

| | J train (48,343 future slots) | R7 v03 non-container (8,295 nodes) |
|---|---|---|
| median runtime | **2223.0 ms** | **2213.0 ms** |
| mean runtime | 3639.8 ms | 4148.5 ms |

Per node type:

| J `node_type` | n | median | | R7 `event_type` | n | median |
|---|---|---|---|---|---|---|
| planner | 21606 | **4992.0 ms** | | api_call | 4688 | **4658.2 ms** |
| videotool_temporal | 16828 | **0.1 ms** | | action | 3607 | **0.1 ms** |
| videotool_spatial | 3779 | 6166.1 ms | | | | |
| answer_generation | 4582 | 10177.4 ms | | | | |
| videotool_generalist | 1548 | 228.2 ms | | | | |

**The two distributions are the same.** J `planner` median 4992 ms vs R7 `api_call` median 4658 ms;
both have a huge mass at 0.1 ms. So the checkpoint was trained on this exact scale.

## D. The contradiction this leaves

- J3:seed11's in-domain acceptance: p50 coverage **0.556** (slightly over-covering), p90 0.917, p95 0.968.
- On the R7 S_* anchors (same distribution as above, same checkpoint): p50 coverage **0.39–0.43**,
  R50 **0.40–0.44**.

Same checkpoint, same runtime scale, yet a much worse p50. That means the difference is **not** the
data distribution and **not** the model — it must be in the **S_* inference path**: how the anchors,
their history and `current_node` are constructed, versus how the J dataset rows were constructed.

Known structural differences (not yet quantified):
1. J rows are built from a **chain** (`build_p9d_topology_dataset`): `run_control` events removed,
   the nested `generalist.generate` merged into the summarizer, retries handled.
   The S_* anchors are built from the **raw event list**: one anchor per raw event, history = `events[:index]`.
   So S_* history / `current_node` can contain `event_type='run'` containers and nested generates
   that never appeared in the J chain.
2. The S_* anchor set therefore includes anchor positions that the J training chain does not contain.
3. `generalist.generate` nodes are present in the v03 templates (they are `api_call` events), so they
   enter both the anchor set and the truth.

## E. Next step (recommended, still no GPU)

Compare, on the same runs, the model input produced by
`build_p9d_topology_dataset.make_feature` (chain policy) against
`build_sstar_predictor_anchors` (raw-event policy) for the same event: history token sequences,
`current_node` fields, `stack_context`, and the resulting context ids. Then re-run the pilot with the
**chain-policy anchors** and see whether p50 coverage returns toward 0.55.
