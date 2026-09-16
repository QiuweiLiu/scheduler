# R8 scheduler extensions: 100-episode paired pilot

## Outcome

All five conditions passed: 100 shared validation episodes per condition,
1,600 completed jobs per policy, zero failed jobs, and zero GPU capacity
violations. The new action/resource semantics are therefore executable, but
this is still a pilot and does not replace the sealed R7 or T_final results.

Relative to the unchanged Myopic baseline (mean completion 160,957.96 ms):

| condition | mean completion change | eviction change | additional signal |
|---|---:|---:|---|
| measured batch action, `batch_myopic` | +0.36% | +1.44/episode | batch-action width p50 mean 2.805 |
| measured batch action, `batch_throughput` | +0.29% | +1.35/episode | batch-action width p50 mean 2.800 |
| paid prefetch | +0.29% | +1.85/episode | 9,181.90 ms prefetch load; 1.78 wasted prefetches/episode |
| node recompute preemption | +3.00% | +0.50/episode | 31.73 preemptions/episode; 17,046.53 ms discarded work; miss rate +0.625 pp |
| all three combined | +3.47% | +3.37/episode | 9,181.90 ms prefetch load; 31.73 preemptions; 16,587.66 ms discarded work; width 2.770 |

The measured batch profile changes the GPU-memory/action contract. It does not
yet change per-node runtime/load: the validated pilot aggregate used here
provided peak-memory by batch, while the frozen template remains the runtime
source. This isolates the scheduler/cache effect and avoids inventing a
runtime curve. A future batch-runtime condition should be added only after the
raw per-batch runtime fields are aggregated and hashed.

## Post-pilot profile update (2026-08-22)

The raw pilot fields have now been aggregated into `batch_profiles.json`.
Selectable batch options remain `{1, 8, 16, 32, 64}`; batch 2 and 4 measurements
are retained in the profile but are not activated in this extension condition.
The profile uses the per-batch p50 `resource.runtime_ms` and `resource.load_ms`
for the deterministic measured condition, retains runtime p90 for the
scheduler estimate, and preserves the previous conservative memory values.
The 100-episode table above is historical: it was not rerun after this profile
update. R8-P7 therefore remains pending until the paired pilot/full matrix is
rerun with the new runtime/load profile.

## Semantics implemented

- Batch action: `(ready_node, free_gpu, batch_size)` for explicitly batchable
  YOLO nodes, using the measured `{1, 8, 16, 32, 64}` peak-memory profile.
- Prefetch: `prefetch_start`/`prefetch_end` occupies a per-GPU load lane and
  resident memory; eviction before first use is recorded as
  `prefetch_wasted`; priority semantics are unchanged.
- Preemption: `node_preempt` pauses no hidden state; elapsed work is discarded
  and the node recomputes on resume. `max_preemptions=32` bounds the pilot.

## Verification

- Local: `py_compile` and extension contract tests `3/3` pass; the historical
  full discovery was interrupted after 47 tests because one legacy import
  test hung under the local environment, so it is not reported as a pass.
- Remote: `py_compile` and extension contract tests `3/3` pass.
- Remote reports include the runner/simulator/config/source hashes and
  execution context. Canonical remote roots are listed in `metrics.json`.
