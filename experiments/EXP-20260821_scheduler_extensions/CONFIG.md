# EXP-20260821_scheduler_extensions

Purpose: measure whether enlarging the simulator action/resource contract with
per-node YOLO batch choices, explicitly paid model prefetch, and node-level
recompute preemption changes scheduling outcomes.

This is an opt-in simulator experiment. The sealed R7 workload, 640 templates,
and T_final are not overwritten. The existing 112-run YOLO pilot supplies
per-tool memory, runtime, and load measurements; the original 100-episode
pilot was run before runtime/load were added to the canonical profile, so those
results still use the frozen template runtime/load. This experiment does not
claim cross-job dynamic batching.

## Conditions

- `baseline`: unchanged R7 simulator (`round_robin`, `myopic`, `oracle`).
- `batch_fixed_1`, `batch_fixed_16`, `batch_fixed_64`: same workload and
  policies with a fixed measured YOLO11x per-tool batch profile.
- `batch_adaptive`: measured options `{1, 8, 16, 32, 64}` exposed as the
  action `(ready_node, GPU, batch_size)`; `batch_myopic` minimizes current
  completion cost, while `batch_throughput` uses the explicit per-tool
  throughput proxy `runtime / batch_size`.
- `prefetch`: fixed plan that loads Qwen3-VL-8B on GPU0 and Qwen3-4B plus
  Qwen2.5-VL-3B on GPU1 at time zero. Each load occupies the GPU load lane and
  consumes the measured resident memory; a later eviction causes a normal
  reload and is counted as wasted prefetch if the model was never used.
- `preempt`: `myopic_preempt` may preempt an active normal node when a higher
  priority or sufficiently old/tight ready node exists. Nodes have no
  checkpoint contract, so progress is discarded and the node recomputes on
  resume; there is no free pause/resume.
- `combined`: adaptive batch + paid prefetch + recompute preemption.

Priority remains the existing binary `priority/normal` field. Arrival times,
deadlines, episode seeds, GPU topology, and future artifacts are unchanged.

## Acceptance gates

1. Every condition has the same episode IDs and expected result rows.
2. Failed jobs and GPU capacity violations are zero.
3. Batch action width, selected batch, prefetch load/waste, and preemption /
   recompute counters are present in result/event schemas.
4. No execution-truth fields are added to `SchedulerStateView`.
5. A result is reported as a paired impact relative to the unchanged R7
   baseline; pilot results are not treated as final workload evidence.

Remote execution root: `/root/autodl-tmp/scheduler`.
