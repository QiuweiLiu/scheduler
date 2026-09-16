# EXP-20260817_r7_trace_pilot

## Purpose

Run the smallest end-to-end trace-collection pilot after the R7 boundary and
event-engine gates. This validates the two already-used dynamic stacks and the
two model-deciding baselines before expanding to the frozen 640-run manifest.

## Frozen scope

- Videos: one deterministic public Video-MME video from `S_train` and one from
  `S_val` (both are predictor-unseen).
- Stacks: `stack_a_qwen3_vl8b_yolo11x` and
  `stack_b_qwen3_4b_qwen25vl3b_yolo26n`.
- Baselines: `star` and `langgraph_react`.
- Runs: 2 videos × 2 stacks × 2 baselines = 8.
- Planner: Stack A uses the validated local Qwen 8B path; Stack B uses the
  validated constrained `local_split` Qwen3-4B planner with Qwen2.5-VL-3B
  visual/answer workers.
- Detector: existing YOLO11x for Stack A and YOLO26n for Stack B; no download.

## Acceptance

Every run must have a non-empty final answer, `run_status=success`, pass the
trace validator, and record the actual model stack/detector IDs. Any failure
remains in the pilot directory and blocks expansion; it is not silently
replaced by a different video.

The pilot is not part of S_train/S_val formal counts until its gate passes and
the full manifest is explicitly started.
