# Result

## Status

`passed_id_boundary_pending_trace_collection`

## Completed

- The 600-video group counts and split rules are recorded in the canonical R7
  registry.
- The corrected predictor-unseen scheduler pool is recorded as 260 videos.
- The final 80-video test set is marked sealed until all models and policies are
  frozen.
- No unresolved video ID was guessed or reassigned locally.

## Gate evidence

- 600/600 source IDs assigned exactly once.
- `P_dev=300`, `P_holdout_diag=40`, scheduler pool=260.
- Predictor/scheduler overlap=0; development/holdout overlap=0.
- 21 repairs recorded: 19 unique source-prefix repairs and 2 remote run_id repairs.
- Provenance join covers all 600 assigned IDs.
- Registry generation is deterministic on repeat execution.

## Next gate

Collect `S_train`/`S_val` traces only after the resource-predictor contract and
unified event-engine interfaces are frozen. `T_final` remains sealed.

## Evidence

- `data/manifests/video_split_registry_r7_v1.json`
- `scripts/build_r7_video_registry.py`
- `data/manifests/predictor_development_split_v1.json`
- `data/manifests/predictor_holdout_split_v1.jsonl`
- `.project/PLAN.md`
