# EXP-20260824 transition profile

## Status

Passed: 4 existing remote models × 2 repeats completed with 8/8 successful rows, 0 errors, and 0 missing models.

## What this establishes

The benchmark provides a measured separation between cold model materialization, the first operation, a resident repeated operation, and explicit unload/cache release. The full numeric record is in [metrics.json](metrics.json); raw per-repeat outputs and checkpoint capability results are under `artifacts/run_20260824T100219+0800/`.

No tested model exposed a resumable node API through this benchmark path. Checkpoint/restore costs therefore remain unsupported rather than being represented as zero.

## Interpretation boundary

`evict_ms` is an explicit Python-object unload plus CUDA cache release proxy, not automatic scheduler cache-manager eviction. The repeated operation is a recompute proxy, not a real mid-node preemption result. Qwen runs were text-only; YOLO11x used one synthetic 640×360 black image.

These measurements are sufficient to prepare a transition-profile input for a later simulator pilot. They do not by themselves authorize simulator changes or unlock `T_final`.

## Remote execution

- Host: `root@connect.westc.seetacloud.com:12469`
- Environment: `/root/miniconda3/envs/finetooling/bin/python`
- Remote output: `/root/autodl-tmp/scheduler/experiments/EXP-20260824_transition_profile/run_20260824T100219+0800/`
- Scope exclusions honored: no video download, no trace recollection, no R7 or `T_final` access.
