# Provenance and reproduction notes

This file records the datasets, seeds, environments, and commands needed to reproduce the headline
numbers. Everything below refers to artifacts that are included in this repository unless stated
otherwise.

## 1. Environments

Development and experiment machine: Windows 11, Intel i5-12450H (8C/12T), 16 GB RAM, RTX 3060
Laptop 6 GB. Simulator-only runs also complete on CPU.

```
python 3.10.21  (conda env `scheduler`)
torch==2.6.0+cu124   (predictor training and packing; CUDA 12.4 wheels)
numpy==2.2.6  pandas==2.3.3  scipy==1.15.3  statsmodels==0.15.0  ortools==9.9.3963
```

`configs/requirements.txt` pins the Python packages. The scheduler simulator itself
(`src/tracing/analysis/workload_v02_simulator.py`) only needs the standard library; numpy/pandas/
scipy/statsmodels are needed for the analysis scripts, and ortools only for the CP-SAT reference.

## 2. Datasets

| Artifact | Path | Rows / size | Registry |
|---|---|---|---|
| R7 workload episodes (validation) | `results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl` | 1,000 episodes / 17.5 MB | `data/manifests/video_split_registry_r7_v1.json` |
| R7 job templates | `results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl` | 640 templates / 8,935 nodes | `data/manifests/` |
| J-series dataset (H=5) | `results/processed/j_series_dataset_v1/` | 60,925 supervision slots | `data/manifests/j_series_dataset_v1.json` |
| J-series dataset (H=10) | `results/processed/j_series_dataset_h10/` | 93,536 slots | `data/manifests/j_series_dataset_h10.json` |
| Topology datasets v3 (H5/H10) | `results/processed/topology_predictor_p9d_v3{,_h10}/` | 27,166 rows | `data/manifests/topology_predictor_p9d_v3.json` |
| Dev/confirm split | `data/manifests/validation_split_dev700_confirm300.json` | 700 dev / 300 confirm | — |

Not included (and not regenerable from this repository alone): the raw VideoMME videos, the
pretrained LLM/VLM weights (Qwen3-4B, Qwen2.5-VL-3B, Grounded-VideoLLM), the raw agent traces, and
the multi-hundred-MB simulation event logs. The **processed** datasets above are sufficient to
re-run the simulator matrix and the statistical analyses; re-training the predictor from raw traces
would require the original trace collection.

## 3. Frozen choices (no tuning on confirm)

| Choice | Value | Where frozen |
|---|---|---|
| Predictor checkpoint | `J3:seed11` (H5) | validation RuntimeQScore argmin; `experiments/EXP-20260911_p9d_j_predictor_acceptance/CANDIDATE.md` |
| H10 predictor | `J3:seed22` (H10) | validation argmin among seeds 11/22/33 |
| Scale-matched p50 constant | `k = 6.2293` | dev700 nodes only, `scripts/analysis/freeze_r50_scale.py` |
| Dev/confirm split | seed `20260914` | `data/manifests/validation_split_dev700_confirm300.json` |
| Paired bootstrap seeds | `20260911/20260914/20260915/20260916` | analysis scripts in `scripts/analysis/` |
| Episode generation seed | `20260818` | `results/processed/r7_workload_20260817/` |

## 4. Main reproduction commands

```bash
export PYTHONPATH=src

# (a) headline consumer comparison on 1,000 validation episodes
python scripts/r7_scheduler_matrix.py \
  --templates results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl \
  --episodes  results/processed/r7_workload_20260817/episodes/workload_validation_r7.jsonl \
  --future-artifacts outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts \
  --output-dir outputs/repro_main --limit 1000 \
  --policies predopt_h5,predopt_h5_q95,predopt_h5_r50,predopt_h5_r90,predopt_h5_r95,predopt_h5_r50k,fcfs,sjf_pred,state_aware,myopic

# (b) dev/confirm discipline (prepare then confirm)
python scripts/r7_scheduler_matrix.py ... --episode-ids-file data/manifests/validation_split_dev700_ids.txt --policies ...
python scripts/r7_scheduler_matrix.py ... --episode-ids-file data/manifests/validation_split_confirm300_ids.txt --policies predopt_h5,predopt_h5_q95

# (c) trace-dependence tests T1-T3 (>=2000 bootstrap / permutation, two-sided p)
python scripts/verify_trace_dependence.py \
  --templates results/processed/r7_workload_20260817/job_templates_r7_v02.jsonl \
  --output experiments/EXP-20260911_forecast_aware_scheduling/trace_dependence_report.json

# (d) artifact perturbations (tail shrink / length noise / content / tail shuffle)
python scripts/perturb_future_artifacts.py \
  --source outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts \
  --target outputs/perturb_tail_shuffle/prediction_artifacts \
  --kind tail_shuffle --intensity 1.0 --seed 20260915
```

The runner is resume-style: it skips `(episode_id, policy)` pairs already present in the output
directory, and it **refuses** to resume a directory whose input/code fingerprint differs
(`run_fingerprint.json`, `--allow-fingerprint-change` to override).

## 5. Known fixes applied after the first public revision

| Issue | Fix |
|---|---|
| CP-RHO objective had the future term under-weighted by ~1000x (flow-time term used `OBJECTIVE_SCALE`, the future term did not) and an inconsistent fallback normalisation | all objective terms now share the same millisecond unit (`src/tracing/scheduling/cp_rho.py`); CP-SAT reference arms re-run |
| Runner resume could silently mix rows produced by different code/inputs | `run_fingerprint.json` + per-row `run_fingerprint`, mismatch refuses to resume |
| Full-slot normaliser copied the pre-rewrite manifest hashes | manifest hashes recomputed for the rewritten gzip payloads |
| Tail-shuffle generation script lived only in a scratch directory | published as `scripts/perturb_future_artifacts.py --kind tail_shuffle`; analysis scripts moved to `scripts/analysis/` |
| README referenced a non-existent requirements file | `configs/requirements.txt` added |

## 6. Reporting discipline

- The variance-component estimates in `scripts/verify_trace_dependence.py` are moment
  approximations for an unbalanced nested design: they are descriptive, not unbiased
  variance-component estimators. The direct within-run covariance statistic is reported alongside.
- Scenario/CVaR results are valid *under the lognormal quantile reconstruction* used by the
  sampler; they should not be generalised to arbitrary CVaR implementations.
- `T_final` (80 sealed videos) is not used by any experiment in this repository.
