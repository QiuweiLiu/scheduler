# EXP-20260824_transition_profile

Purpose: measure the real model-transition costs needed by the scheduler
extension without re-collecting video traces.

Scope:

- cold local model load from existing weights;
- one first operation and one repeated operation on the resident model;
- explicit unload/release timing used as an eviction-cost proxy;
- repeated execution timing used as a recompute proxy;
- checkpoint/restore capability audit only; no synthetic checkpoint numbers.

The benchmark uses existing remote weights only. It does not download videos,
overwrite trace collections, or read/modify `T_final`.

Models:

- `/root/autodl-tmp/Qwen3-VL-8B-Instruct`
- `/root/autodl-tmp/scheduler/Qwen3-4B`
- `/root/autodl-tmp/scheduler/Qwen2.5-VL-3B-Instruct`
- `/root/autodl-tmp/upload/models/yolo11x.pt`

Runtime:

- remote root: `/root/autodl-tmp/scheduler`;
- environment: `/root/miniconda3/envs/finetooling/bin/python`;
- device: one visible CUDA GPU;
- repeats: 2 per model for this bounded pilot;
- `local_files_only=true` for Transformers models;
- no checkpoint/restore is claimed unless a real resumable API is found.

Interpretation:

- `cold_load_ms` includes weight materialization and framework initialization;
- `first_operation_ms` is measured separately from load;
- `resident_operation_ms` is a second operation while the model remains loaded;
- `evict_ms` is explicit object unload plus CUDA cache release, not automatic
  cache-manager eviction;
- `recompute_operation_ms` is a repeated operation proxy, not a real mid-node
  preemption result;
- unsupported checkpoint/restore is recorded as a capability result, not as
  zero cost.
