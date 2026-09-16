# R8-P3 Behavior Cloning from CP-RHO Expert

## Status

`P3_passed` — BC successfully distilled CP-RHO knowledge; gate passed.

## Acceptance record

- [x] P3a data collection completed: 300 S_train episodes, 246,567 decisions
- [x] P3b BC training completed: BcActor, 50 epochs, best epoch 50
- [x] P3c BC evaluation on S_val completed: 100 episodes, 0 failed jobs
- [x] P3 gate passed: action agreement 86.86% (>= 60%), completion gap +2.2% (<= +10%)

## Data collection (2026-08-19)

- Remote command: `scripts/r8_bc_collect.py` on S_train episodes
- Limit: 300 episodes, horizons 1/3/5
- Output: `results/processed/r8_bc/p3_bc_dataset_20260819/`
  - `bc_dataset.jsonl`: 205 MB, 246,567 decision rows
  - `bc_dataset_report.json`: status=passed

## Training (2026-08-19)

- Model: `BcActor(14→128→128→1)` (same architecture as CandidateActor)
- Loss: CrossEntropy with padding/mask for variable-length candidate pools
- Optimizer: Adam, lr=1e-4, batch_size=64, 50 epochs
- Train/val split: 221,911 / 24,656 decisions
- Best epoch: 50 (val_loss=0.3589, val_accuracy=86.86%)
- Majority baseline: 43.06%
- Output: `results/processed/r8_bc/p3_bc_training_20260819/`
  - `bc_model_best.pt`: best checkpoint
  - `bc_train_report.json`: full report with loss curve
  - `loss_curve.jsonl`: per-epoch loss

## Evaluation on S_val (100 episodes)

| Policy | Mean Completion (ms) | Failed Jobs |
|---|---|---|
| **BC H5** | **165,463.63** | 0 |
| BC H3 | 165,532.24 | 0 |
| BC H1 | 165,636.78 | 0 |
| CP-RHO H5 (expert) | 165,512.93 | 0 |
| PredOpt-v2 H5 (baseline) | 161,888.13 | 0 |

## Gate

| Condition | Threshold | Actual | Result |
|---|---|---|---|
| Action agreement | >= 60% | **86.86%** | PASS |
| Completion gap vs PredOpt-v2 | <= +10% | **+2.2%** | PASS |
| **Overall** | | | **PASSED** |

## Key findings

1. BC achieves 86.86% action agreement with CP-RHO expert, 2× above majority baseline.
2. BC H5 mean completion (165,464 ms) is within 0.03% of CP-RHO H5 (165,513 ms).
3. BC is only 2.2% slower than the best deployable method (PredOpt-v2 H5), well within the 10% gate.
4. BC inference cost is negligible (~0.5 ms/decision, same as PredOpt-v2), with no timeout/fallback.
5. CP-RHO's mathematical optimization knowledge has been successfully distilled into a lightweight neural policy.