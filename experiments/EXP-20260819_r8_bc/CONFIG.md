# EXP-20260819_r8_bc — R8-P3 Behavior Cloning from CP-RHO Expert

## Status

`P3_pending` — experiment scripts created; data collection and training not yet run.

## Objective

Determine whether a behavior-cloned policy (BC) trained on CP-RHO expert demonstrations
can match CP-RHO's decision quality while having inference cost similar to PredOpt-v2.

## Scope

- P3a: Collect per-decision (state, action) pairs from CP-RHO H1/H3/H5 on S_train episodes.
- P3b: Train a BC policy (BcActor, same architecture as CandidateActor) with CE loss.
- P3c: Evaluate on S_val and compare against PredOpt-v2 H5 and CP-RHO H5.
- No video download, trace collection, predictor retraining, or T_final access.

## Hypothesis

A BC policy trained on CP-RHO expert demonstrations can achieve action agreement >= 80%
with the CP-RHO expert and mean completion time within 5% of CP-RHO on S_val, while
incurring < 1 ms per decision (same as PredOpt-v2).

## Independent variable

Training method: BC supervised learning (CrossEntropy) vs. CP-RHO mathematical optimization.

## Dependent variables

- Action agreement rate with CP-RHO expert (on held-out S_val decisions)
- Mean completion time on S_val episodes
- Mean inference time per decision (ms)

## Control / baseline

- PredOpt-v2 H5 (current best deployable, ~0.5 ms/decision)
- CP-RHO H5 (expert oracle, ~54 ms/decision, 57% fallback rate)

## Data

- Train: S_train episodes (120 videos, CP-RHO H1/H3/H5 decisions)
- Validation: held-out 10% of collected decisions
- Evaluation: S_val episodes (40 videos, 100 episodes default)

## Falsification criterion

BC fails to pass gate if:
- Action agreement < 60% on S_train validation split
- Mean completion on S_val > PredOpt-v2 H5 + 10%
- Inference time > 10 ms per decision

## Scripts

### Data collection

```bash
PYTHONPATH=src python scripts/r8_bc_collect.py \
  --templates results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl \
  --episodes results/processed/r7_workload_20260817/episodes/workload_train_r7.jsonl \
  --future-artifacts results/processed/r7_scheduling_future_20260817/prediction_artifacts \
  --output-dir results/processed/r8_bc/p3_bc_dataset_20260819 \
  --limit 300 \
  --horizons 1,3,5
```

### Training

```bash
PYTHONPATH=src python scripts/r8_bc_train.py \
  --dataset results/processed/r8_bc/p3_bc_dataset_20260819/bc_dataset.jsonl \
  --output-dir results/processed/r8_bc/p3_bc_training_20260819 \
  --epochs 50 --batch-size 64 --lr 1e-4 --hidden 128 \
  --eval-templates results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl \
  --eval-episodes results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl \
  --eval-future-artifacts results/processed/r7_scheduling_future_20260817/prediction_artifacts \
  --eval-limit 100
```

## Execution environment

Canonical execution target is remote `/root/autodl-tmp/scheduler`.
Local source and control plane are canonical.