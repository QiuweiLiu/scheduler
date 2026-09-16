# EXP-20260816 — Result

## Status

Completed. The first R6 build was retained as audit evidence but not selected as
canonical after review found that old templates omitted an explicit execution
lane and the first workload draft kept source refs at the episode top level. The
`fix1` build infers `cpu/gpu/api` from measured node identity and moves source
refs to engine-audit metadata.

## Acceptance

- C1 split is diagonal and video-level: 48 train, 8 validation, 8 test-retrospective.
- C2 registry contains 32 pending candidates and 8 deterministic backups, with no
  C1 overlap and no overlap with 341 existing trace IDs / 349 existing hashes.
- Prefix input has 4,683 rows and zero forbidden-field violations; labels remain
  separate from `input`.
- H1/H3/H5 future artifacts each cover all 9,366 nodes and have normalized
  probabilities.
- Resource predictions cover all 9,366 nodes and fit only 566 train templates
  (6,916 train truth nodes); no truth field is copied to the prediction row.
- Workload generation completed 20,000/1,000/6,750 episodes and
  743,680/34,160/248,400 jobs. Scheduler-visible job fields contain no whole-job
  predicted runtime or source video refs.

The machine-readable gate and hashes are in the remote `r6_manifest.json` and
`r6_gate_report.json`. C2 is intentionally `pending_trace_collection`; R7 must
freeze the predictor/resource artifacts before collecting those new traces.
