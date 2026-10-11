"""Train-set historical prior for the isolation re-run (exp F).

Frozen design: docs/research/2026-10-10_exp_f_train_prior_isolation.md

The prior is computed EXCLUSIVELY from the predictor's own training split
(``results/processed/j_series_dataset_v1/j_train.jsonl.gz``, 13,754 rows, 240
videos) — ground-truth future targets, never from the workload prediction pack
and never from dev/confirm data.  Observed isolation (report artifact): the
j-series collection shares NO video, NO run and NO episode with the r7 workload
(160 videos / 640 runs); j_train ∩ confirm runs = 0.

Prior channels (per future step k = 1..5):
* model:      frequency of ``future[k-1].model_class`` over rows with >= k steps
* length:     normalised histogram of ``len(future)`` in {0..4}, index 5 = ">=5"
* runtime:    mean / p50 / p90 / p95 / cvar95 of ``targets.runtime_ms``
* load:       occurrence = fraction of steps with ``load_ms > 0``; p50/p90/p95 of
              positive ``load_ms``
* lane:       majority of cpu-metadata-adapter-v1-vs-rest at step k

Pack transform modes (frozen field list):
* ``partial``: step ``model_probabilities`` + ``model_probability`` (prior max),
  row ``length_probabilities`` + ``predicted_future_length`` (prior argmax)
* ``full``: partial + step ``execution_lane``, ``resource.runtime_mean_ms``,
  ``resource.runtime_ms_quantiles``, ``resource.cvar95_ms``,
  ``resource.load_occurrence_probability``, ``resource.load_duration_ms_quantiles``
Untouched fields stay byte-identical (disclosed; the untouched step fields have
no scheduler consumers).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

J_TRAIN = Path("results/processed/j_series_dataset_v1/j_train.jsonl.gz")
HORIZON = 5
CPU_MODEL = "cpu-metadata-adapter-v1"

PRIOR_VERSION = "train-prior-v1"


def load_j_split_rows(path: Path = J_TRAIN) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * float(q)
    low = int(math.floor(position))
    high = min(len(ordered) - 1, low + 1)
    fraction = position - low
    return float(ordered[low]) * (1.0 - fraction) + float(ordered[high]) * fraction


def _cvar95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(1, int(round(0.05 * len(ordered))))
    tail = ordered[-k:]
    return float(sum(tail) / len(tail))


def compute_train_prior(rows: Sequence[Mapping[str, Any]], horizon: int = HORIZON) -> Dict[str, Any]:
    """Per-step aggregate statistics from the predictor's training split."""

    model_counts: List[Counter] = [Counter() for _ in range(horizon)]
    runtime_values: List[List[float]] = [[] for _ in range(horizon)]
    load_values: List[List[float]] = [[] for _ in range(horizon)]
    load_present: List[int] = [0 for _ in range(horizon)]
    gpu_counts: List[int] = [0 for _ in range(horizon)]
    length_counts: Counter = Counter()
    for row in rows:
        future = row.get("future") or []
        length_counts[min(len(future), horizon)] += 1
        for index, step in enumerate(future[:horizon]):
            model = step.get("model_class")
            if model:
                model_counts[index][str(model)] += 1
            if str(model) != CPU_MODEL:
                gpu_counts[index] += 1
            targets = step.get("targets") or {}
            runtime = targets.get("runtime_ms")
            if isinstance(runtime, (int, float)):
                runtime_values[index].append(float(runtime))
            load = targets.get("load_ms")
            if isinstance(load, (int, float)) and float(load) > 0.0:
                load_values[index].append(float(load))
                load_present[index] += 1

    total = sum(length_counts.values())
    length_dist = [length_counts.get(k, 0) / total for k in range(horizon + 1)] if total else []
    steps_observed = [sum(counts.values()) for counts in model_counts]
    model_dist = []
    runtime_stats = []
    load_stats = []
    lane_majority = []
    for index in range(horizon):
        counts = model_counts[index]
        n = steps_observed[index]
        model_dist.append({model: count / n for model, count in sorted(counts.items())} if n else {})
        values = runtime_values[index]
        runtime_stats.append({
            "mean": (sum(values) / len(values)) if values else 0.0,
            "p50": _quantile(values, 0.50),
            "p90": _quantile(values, 0.90),
            "p95": _quantile(values, 0.95),
            "cvar95": _cvar95(values),
            "n": len(values),
        })
        loads = load_values[index]
        load_stats.append({
            "occurrence": (load_present[index] / n) if n else 0.0,
            "p50": _quantile(loads, 0.50),
            "p90": _quantile(loads, 0.90),
            "p95": _quantile(loads, 0.95),
            "n_positive": len(loads),
        })
        lane_majority.append("gpu" if n and gpu_counts[index] * 2 >= n else "cpu")
    return {
        "version": PRIOR_VERSION,
        "source_rows": len(rows),
        "horizon": horizon,
        "model_dist": model_dist,
        "length_dist": length_dist,
        "runtime": runtime_stats,
        "load": load_stats,
        "lane_majority": lane_majority,
        "steps_observed": steps_observed,
    }


def prior_sha256(prior: Mapping[str, Any]) -> str:
    blob = json.dumps(prior, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _prior_step_for(steps: Sequence[Any], index: int) -> Mapping[str, Any]:
    if index >= len(steps):
        raise ValueError(f"prior has no step {index + 1}")
    return steps[index]


def build_prior_pack(
    pack: Mapping[str, Mapping[str, Any]],
    prior: Mapping[str, Any],
    mode: str,
) -> Dict[str, Dict[str, Any]]:
    """Transform the frozen prediction pack into a prior-driven pack.

    ``mode`` is ``partial`` (model + length channels) or ``full`` (adds runtime /
    load / lane).  Rows without future steps are returned as-is (they carry no
    replaced channel).  The input pack is not mutated.
    """

    if mode not in {"partial", "full"}:
        raise ValueError(f"unknown prior mode: {mode!r}")
    model_dist = prior["model_dist"]
    runtime_stats = prior["runtime"]
    load_stats = prior["load"]
    length_dist = prior["length_dist"]
    lane_majority = prior["lane_majority"]
    length_argmax = max(range(len(length_dist)), key=lambda k: (length_dist[k], k)) if length_dist else 0

    result: Dict[str, Dict[str, Any]] = {}
    for key, row in pack.items():
        new_row = dict(row)
        scenarios = row.get("future_h5") or []
        if not scenarios:
            result[str(key)] = new_row
            continue
        steps = [dict(step) for step in (scenarios[0].get("steps") or [])]
        for index in range(len(steps)):
            step = steps[index]
            distribution = model_dist[index]
            step["model_probabilities"] = dict(distribution)
            step["model_probability"] = max(distribution.values()) if distribution else 0.0
            if mode == "full":
                resource = dict(step.get("resource") or {})
                stats = _prior_step_for(runtime_stats, index)
                load = _prior_step_for(load_stats, index)
                resource["runtime_mean_ms"] = stats["mean"]
                resource["runtime_ms_quantiles"] = {
                    "p50": stats["p50"], "p90": stats["p90"], "p95": stats["p95"],
                }
                resource["cvar95_ms"] = stats["cvar95"]
                resource["load_occurrence_probability"] = load["occurrence"]
                resource["load_duration_ms_quantiles"] = {
                    "p50": load["p50"], "p90": load["p90"], "p95": load["p95"],
                }
                step["resource"] = resource
                step["execution_lane"] = lane_majority[index]
            steps[index] = step
        new_row["future_h5"] = [{**scenarios[0], "steps": steps}]
        new_row["length_probabilities"] = list(length_dist)
        new_row["predicted_future_length"] = length_argmax
        result[str(key)] = new_row
    return result


def isolation_report() -> Dict[str, Any]:
    """Observed data-isolation audit for the frozen collections.

    Episode/template facts come from the frozen workload files; the j-series
    facts from the dataset splits themselves.  All counts are computed here so
    the artifact can be re-verified from the same frozen inputs.
    """

    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from tracing.analysis.workload_v02_simulator import load_templates, read_jsonl

    manifest = json.loads((root / "data/manifests/validation_split_dev700_confirm300.json").read_text(encoding="utf-8"))
    episodes = read_jsonl(root / "results/processed/r7_workload_v03_no_run_container/episodes/workload_validation_r7_v03.jsonl")
    by_id = {str(e.get("episode_id")): e for e in episodes}
    templates = load_templates(
        root / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl",
        topology_view="causal_v3",
    )
    confirm_ids = [str(e) for e in manifest["confirm"]]
    dev_ids = [str(e) for e in manifest["development"]]

    def runs_of(ids: Iterable[str]) -> set:
        out = set()
        for episode_id in ids:
            episode = by_id.get(episode_id)
            if not episode:
                continue
            for job in episode.get("jobs") or []:
                out.add(str(job.get("template_id")))
        return out

    confirm_runs = runs_of(confirm_ids)
    dev_runs = runs_of(dev_ids)
    workload_videos = {
        str(getattr(template, "video_id", "") or "")
        for template in templates.values()
    } - {""}

    j_facts: Dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        path = root / f"results/processed/j_series_dataset_v1/j_{split}.jsonl.gz"
        rows = load_j_split_rows(path)
        runs = {str(r.get("run_id")) for r in rows}
        videos = {str(r.get("video_id")) for r in rows if r.get("video_id")}
        j_facts[split] = {
            "rows": len(rows),
            "runs": len(runs),
            "videos": len(videos),
            "runs_overlap_with_workload": len(runs & set(templates)),
            "videos_overlap_with_workload": len(videos & workload_videos),
            "runs_overlap_with_confirm": len(runs & confirm_runs),
            "runs_overlap_with_dev": len(runs & dev_runs),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    template_split = {str(t_id): template.split for t_id, template in templates.items()}
    confirm_by_split = Counter(template_split.get(run, "?") for run in confirm_runs)
    return {
        "episode_level": {
            "confirm_n": len(confirm_ids),
            "dev_n": len(dev_ids),
            "confirm_dev_overlap": len(set(confirm_ids) & set(dev_ids)),
        },
        "template_level": {
            "workload_templates": len(templates),
            "workload_template_split_counts": dict(Counter(template_split.values())),
            "confirm_runs": len(confirm_runs),
            "confirm_runs_by_template_split": dict(confirm_by_split),
            "dev_confirm_run_overlap": len(confirm_runs & dev_runs),
        },
        "j_series": j_facts,
        "prior_pool": "j_train only (predictor fitting split; j_validation used for model selection, j_test untouched by training)",
    }
