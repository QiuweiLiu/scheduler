# Experiments

| ID | Purpose | Status | Canonical location |
|---|---|---|---|
| `EXP-20260815_video_640x360_pilot` | Measure 640×360 derivative size and readability before R5 batch conversion | completed (6 valid samples) | `experiments/EXP-20260815_video_640x360_pilot/` |
| `EXP-20260816_r6_causal_v2` | Rebuild C1 causal-v2 prefix, train-only resource artifacts, scheduler-visible workload and C2 pending registry | completed; C2 pending trace collection | `experiments/EXP-20260816_r6_causal_v2/` |
| `EXP-20260817_r7_boundary_gate` | Freeze the 600-video R7 boundary and repair predictor split IDs | completed; trace collection pending | `experiments/EXP-20260817_r7_boundary_gate/` |
| `EXP-20260817_r7_resource_predictor` | Refit resource predictor on the shared 300-video P_dev pool | completed; event-engine/resource contract passed | `experiments/EXP-20260817_r7_resource_predictor/` |
| `EXP-20260817_r7_event_engine_parity` | Verify StateView/Truth/FutureProvider leakage and deterministic boundaries | passed; formal S_train/S_val workload pending | `experiments/EXP-20260817_r7_event_engine_parity/` |
| `EXP-20260817_r7_trace_pilot` | Validate the two frozen dynamic stacks/baselines on two predictor-unseen videos | passed; ready for 640-run expansion | `experiments/EXP-20260817_r7_trace_pilot/` |
| `EXP-20260817_video_delete_pdev` | Remove legacy P_dev MP4s after preserving trace/provenance/source URLs | completed; 300 files removed, 0 non-target deletions | `experiments/EXP-20260817_video_delete_pdev/` |
