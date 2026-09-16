# Result

## Status

`passed_pilot_gate_ready_for_640_expansion`

The 8-run predictor-unseen pilot completed on the remote runtime:

- 8/8 runs finished with `run_status.status=success`.
- 8/8 traces passed `src/tracing/validators/validate_trace.py`.
- Coverage is balanced across 2 stacks (`stack_a_qwen3_vl8b_yolo11x`,
  `stack_b_qwen3_4b_qwen25vl3b_yolo26n`) and 2 dynamic baselines (`star`,
  `langgraph_react`), with 4 runs per stack and 4 per baseline.
- The 2 parse errors were retained as trace events and each had a successful
  `retry_of` successor; they are not silently discarded. Final run status was
  successful for all 8 runs.
- The source pilot manifest has no `answer` field and planner inputs have no
  answer/gold fields. The generated answer appears only in the terminal answer
  event/run manifest, as an output artifact.
- Remote report: `results/processed/r7_trace_pilot_20260817/pilot_report.json`.
  Local copy: `remote_pilot_report.json` (SHA-256
  `3ce67c4f2d672a198ca58a426f35eecae00b29a94ee256b50e614fe5be27105a`).

The deterministic manifests used for this gate are:

- full: `data/manifests/r7_trace_manifest_v1.jsonl` (640 run rows, 160 videos)
- pilot: `data/manifests/r7_trace_pilot_manifest_v1.jsonl` (8 run rows, 2 videos)

The pilot gate is passed. The next bounded step is the 640-run expansion using
the same frozen manifest; no workload, optimizer, or RL run has started yet.
