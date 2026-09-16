# EXP-20260816 — R6 causal-v2

- remote project: `/root/autodl-tmp/scheduler`
- remote output: `results/processed/r6_causal_v2_20260816_fix1/`
- source collection: `trace_collection_v1_legacy_core` (read-only)
- split: `configs/phase3_video_split_48_8_8.jsonl` (48/8/8 video-level)
- C2 source pool: `data/manifests/video_provenance_v2_expanded_available.jsonl`
- seed: `20260816` for workload generation
- workload counts: train 20,000; validation 1,000; test-retrospective 6,750
- policy boundary: `causal-state-v2`; measured truth is engine-only
