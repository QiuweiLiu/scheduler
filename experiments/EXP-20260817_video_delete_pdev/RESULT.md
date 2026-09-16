# EXP-20260817 — Delete legacy P_dev videos

## Status

Completed on the remote runtime. This was a storage cleanup operation, not a model or scheduling experiment.

## Scope and authorization

- Remote root: `/root/autodl-tmp/scheduler`
- Target: only the 300 `P_dev` original MP4 files from `video_split_registry_r7_v1.json`
- Explicit user instruction: keep the current R7 collector running; delete the old development videos; retain source URLs and provenance
- Archive exception: local archive/restore testing was explicitly waived for this cleanup
- Untouched: `P_holdout_diag`, `S_train`, `S_val`, `T_final`, `T_backup`, all trace/model/experiment outputs, and the 768 legacy traces

## Pre-delete checks

- Target count: 300 unique IDs
- Provenance records: 300/300
- Source URLs: 300/300
- Official archive locators: 300/300
- Size/path check: 300/300 passed
- Successful raw trace coverage: 300/300 videos
- Overlap with active R7 manifest: 0
- Active collector: PID `187515` remained untouched

## Action and verification

- Deleted: 300 MP4 files
- Bytes removed: `28,436,477,904` (about 28.44 GB decimal)
- Remaining target files: 0
- Unexpected non-target deletions: 0
- P_dev trace evidence after deletion: 300/300 video IDs still represented by successful raw runs
- Provenance retention updated to `original_status=deleted_after_gate`, `deletion_eligible=true`; original SHA256, byte size, resolution, duration, official ZIP locator, and source URL remain in the canonical JSONL
- R7 collector remained alive after deletion; its summary had 545 successful rows and 0 failures at the post-delete check

## Canonical artifacts

- Remote audit: `/root/autodl-tmp/scheduler/results/processed/video_delete_pdev_20260817/`
- Remote provenance: `/root/autodl-tmp/scheduler/data/manifests/video_provenance_v2_expanded_available.jsonl`
- Local audit copy: this directory (`pre_delete_report.json`, `pre_delete_targets.jsonl`, `post_delete_report.json`)

## Limitation

The deleted MP4 bytes are not archived on the remote host. The retained source URL and official archive locator identify how to reacquire them, but a future vision rerun requires reacquisition and revalidation; the existing trace-level experiments remain usable from their preserved artifacts.
