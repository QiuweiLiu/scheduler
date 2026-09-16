# Result

## Status

`completed_predictor_gate_pending_event_engine`

## Canonical remote output

`/root/autodl-tmp/scheduler/results/processed/resource_predictor_r7_pdev300_20260817_retry2/`

The first attempt is retained separately as audit evidence because it used the
wrong `PYTHONPATH` and produced a zero-byte pickle. It is not a valid model.

## Data coverage

- P_dev: 300 videos, 1,240 runs, 17,303 compute rows.
- P_dev split: train 988 runs / validation 144 / diagnostic test 108.
- Final predictor holdout: 40 videos, 1,380 compute rows.
- Static metadata: 300 videos; 112 runs use the permitted video-level metadata
  fallback because no run-specific row exists.
- Error events are retained as labels; no error was converted to success or 0.

## Selected resource models

- Runtime: `lgb_point_group_q99_interval`
- Load: `group_quantile`
- Inclusive peak memory: `group_quantile`

The selected candidates satisfy the development validation coverage policy.
The final holdout is diagnostic only and does not change model selection.

## Leakage and limitations

The report records zero development/holdout video overlap and false flags for
future-event, answer/teacher, target-runtime/load/peak, video-id, and
non-prefix feature leakage. Queue prediction, strict OOM prediction,
resident-weight separation, and stall-risk prediction are unsupported. The
next gate is the unified event-engine `StateView/Truth/FutureProvider` parity
smoke; no scheduler workload has been started.

## Evidence

- Remote `resource_predictor_report.json`
- Remote `scheduler_resource_contract.json`
- `data/manifests/video_split_registry_r7_v1.json`
- `scripts/build_r7_resource_input.py`
