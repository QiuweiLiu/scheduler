# EXP-20260821_model_load_stackb_yolo26n

## Purpose

Measure the scheduling impact of a real YOLO26n GPU node in the Stack B
workflow. This is an isolated pilot and does not alter the frozen 640-template
R7 collection or `T_final`.

## Scope

- Base workflow: Stack B (`Qwen3-4B` planner + `Qwen2.5-VL-3B-Instruct` visual/answer).
- Added node: explicit `yolo-tracker` pre-observation using `/root/autodl-tmp/upload/models/yolo26n.pt`.
- Pilot: first four public Video-MME videos in the current `S_train` trace
  manifest, one `star` or `langgraph_react` row per source trace (8 source rows,
  8 runs).
- Output: a new remote raw-trace root and derived templates/episodes only.

## Information boundary

The source videos are predictor-unseen scheduler videos. The pilot is used only
to measure the added detector node and to calibrate an isolated scheduler
comparison; it is not merged into predictor training or `T_final`.

## Acceptance gates

1. All 16 runs either succeed or retain an explicit failure record.
2. Valid traces contain a real `model_id=yolo26n.pt` GPU node with runtime,
   load, workspace and resident-memory fields.
3. No existing R7 template, trace or formal result is overwritten.
4. The treatment comparison reports completion, queue, deadline miss, GPU
   eviction, peak memory and added-load deltas under the same seeds/topology.

## Planned remote roots

- Raw traces: `results/raw/EXP-20260821_model_load_stackb_yolo26n/`
- Derived enrichment/templates/episodes/results:
  `results/processed/EXP-20260821_model_load_stackb_yolo26n/`
