"""Phase 18 — aggregate-optimism diagnostic (read-only; no scheduler run).

Question
--------
Schedulers consume a *sum* of per-step predicted runtime quantiles over a
predicted future chain. A well-calibrated single-step quantile does not imply a
calibrated sum. For every scheduler anchor ``v``:

    Shat_tau(v) = sum of predicted runtime quantile tau over the predicted chain
    S_true(v)   = the anchor's true remaining **step** workload, from the R7
                  job template that scheduled it
    B_tau(v)    = S_true(v) - Shat_tau(v)      (positive = prediction too low)
    r_tau(v)    = Shat_tau(v) / S_true(v)

Truth definition (data-driven, 2026-09-17)
-----------------------------------------
The R7 trace is *step -> events*: one agent step emits 1-4 events
(`api_call`, `action`). Three facts were measured on 120 raw traces before
fixing the definition:

* ``event_type == "run"`` nodes are **whole-run wall-clock containers**: for all
  120 sampled runs the terminal run event's ``runtime_ms`` equals the whole file
  span (ratio median 1.000). Summing them double counts the entire workflow, so
  they are excluded from every truth sum and reported separately.
* ``api_call`` events carry ``timestamp_end == timestamp_start`` (889/889), so
  their interval is unmeasurable from timestamps; only ``runtime_ms`` is usable.
* Within one step the events do not overlap, so a step's workload is the sum of
  its non-container events. **Built-in check**: the sum of all non-container
  events of a template must equal that template's wall clock; the run reports
  the ratio distribution and fails if the median leaves [0.8, 1.2].

Truth variants reported side by side (none silently preferred):
* ``next_k_events``  — the next K non-container events, K = len(predicted chain) (event-level slots)
* ``remaining_all``  — all non-container events after the anchor (rest of step + later steps)
* ``s_container_total`` — the container wall-clock total, reported separately to keep the
  double-counting trap visible

Populations
-----------
``primary_gpu_with_remaining`` / ``gpu_with_remaining`` / ``last_step_gpu``
(anchors with no remaining step, where any emitted prediction is pure
over-prediction) / ``branching`` (more than one event per remaining step) /
``all_anchors``. Every population reports both anchors and videos.

Fail-closed rules (all enforced on every run)
---------------------------------------------
1. ``video_id`` from the template record; distinct count must equal ``--expect-videos``.
2. Anchors that do not join a template node fail the run, except ``:run:1``
   workflow-root pseudo-events, which are excluded and counted.
3. Anchors with zero predicted steps are kept (Shat = 0); a ratio is undefined
   when S_true = 0 but the bias is still defined.
4. Missing / non-finite quantiles, gated load durations, and ``runtime_ms`` abort the run.
5. Bootstrap values and their video clusters are filtered as pairs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TAUS = ("p50", "p90", "p95")
DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20260917
LOAD_OCCURRENCE_GATE = 0.5
HORIZON = 5
EXPECTED_VIDEOS = 160
RUN_ROOT_SUFFIX = ":run:1"
CONTAINER_EVENT_TYPE = "run"
WALL_CLOCK_TOLERANCE = (0.8, 1.2)


def _finite(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite or missing value at {where}: {value!r}")
    return float(value)


# ------------------------------------------------------------------------- IO


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_artifact_rows(path: Path, horizon: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        node_id = row.get("node_id")
        if not node_id:
            continue
        scenarios = row.get(f"future_h{int(horizon)}") or []
        steps = (scenarios[0].get("steps") or []) if scenarios else []
        rows[str(node_id)] = {"row": row, "steps": steps}
    return rows


class TemplateSteps:
    """A job template as an ordered sequence of agent steps and their events."""

    def __init__(self, template: Mapping[str, Any]) -> None:
        nodes = template.get("nodes") or []
        self.nodes = {str(node["node_id"]): node for node in nodes}
        self.steps: dict[int, list[str]] = defaultdict(list)
        self.position: dict[str, tuple[int, int]] = {}
        containers: list[str] = []
        for node in sorted(nodes, key=lambda item: int(item.get("sequence_index") or 0)):
            node_id = str(node["node_id"])
            if not self._is_container(node):
                step = self._step_of(node)
                self.steps[step].append(node_id)
            else:
                containers.append(node_id)
        for step, node_ids in self.steps.items():
            for index, node_id in enumerate(node_ids):
                self.position[node_id] = (step, index)
        self.containers = containers
        self.wall_clock_ms = max(
            [self.runtime_ms(node_id) for node_id in containers] or [0.0]
        )

    @staticmethod
    def _is_container(node: Mapping[str, Any]) -> bool:
        return str(node.get("event_type") or "") == CONTAINER_EVENT_TYPE

    @staticmethod
    def _step_of(node: Mapping[str, Any]) -> int:
        steps = node.get("source_step_ids") or []
        for value in steps:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return int(node.get("sequence_index") or 0)

    def runtime_ms(self, node_id: str) -> float:
        raw = self.nodes[node_id].get("runtime_ms")
        if raw is None:
            raise ValueError(f"missing runtime_ms for template node {node_id}")
        return _finite(raw, f"runtime_ms of {node_id}")

    def has(self, node_id: str) -> bool:
        return node_id in self.position

    def next_k_events(self, anchor: str, k_events: int) -> list[str]:
        """The K non-container events that follow the anchor, in execution order.

        Slot semantics: the frozen artifact's future slots are *events*, not agent
        steps. Verified 2026-09-17 on the J-series supervision the runtime head was
        trained on: 7,195 future slots are all ``api_call``/``action`` and **none**
        are ``run`` containers (anchor ``:run:1``, slots ``api_call:2``, ``action:3``
        — i.e. two events of the same step).
        """

        return self.remaining_after(anchor)[: int(k_events)]

    def remaining_after(self, anchor: str) -> list[str]:
        """Rest of the anchor's own step plus every later step (non-container events)."""

        step, index = self.position[anchor]
        collected: list[str] = []
        for current in self.step_numbers():
            if current < step:
                continue
            for position, node_id in enumerate(self.steps[current]):
                if current == step and position <= index:
                    continue
                collected.append(node_id)
        return collected

    def step_numbers(self) -> list[int]:
        return sorted(self.steps)

    def remaining_step_count(self, anchor: str) -> int:
        step, _index = self.position[anchor]
        return sum(1 for value in self.step_numbers() if value > step)

    def events_per_remaining_step(self, anchor: str) -> list[int]:
        step, _index = self.position[anchor]
        return [len(self.steps[value]) for value in self.step_numbers() if value > step]

    def workload_total_ms(self) -> float:
        return float(sum(self.runtime_ms(node_id) for node_ids in self.steps.values() for node_id in node_ids))

    def container_total_ms(self) -> float:
        return float(sum(self.runtime_ms(node_id) for node_id in self.containers))


def load_templates(path: Path) -> tuple[dict[str, TemplateSteps], dict[str, dict[str, str]]]:
    templates: dict[str, TemplateSteps] = {}
    meta: dict[str, dict[str, str]] = {}
    for template in iter_jsonl(path):
        if not template.get("nodes"):
            continue
        built = TemplateSteps(template)
        info = {
            "template_id": str(template.get("template_id") or template.get("run_id")),
            "video_id": str(template.get("video_id") or ""),
            "baseline": str(template.get("baseline") or ""),
            "split": str(template.get("split") or ""),
        }
        for node_id in built.nodes:
            if node_id in meta:
                continue
            templates[node_id] = built
            meta[node_id] = info
    return templates, meta


# ------------------------------------------------------------------- per-anchor


def predicted_sums(steps: Sequence[Mapping[str, Any]], node_id: str) -> dict[str, Any]:
    totals = {tau: 0.0 for tau in TAUS}
    violations = 0
    gated_steps = 0
    extra_load_p95 = 0.0
    for index, step in enumerate(steps):
        resource = step.get("resource") or {}
        quantiles = resource.get("runtime_ms_quantiles")
        if not isinstance(quantiles, Mapping):
            raise ValueError(f"missing runtime_ms_quantiles at {node_id} step {index}")
        values = []
        for tau in TAUS:
            value = _finite(quantiles.get(tau), f"{node_id} step {index} {tau}")
            values.append(value)
            totals[tau] += value
        if not (values[0] <= values[1] <= values[2]):
            violations += 1
        occurrence = resource.get("load_occurrence_probability")
        if isinstance(occurrence, (int, float)) and float(occurrence) >= LOAD_OCCURRENCE_GATE:
            duration = (resource.get("load_duration_ms_quantiles") or {}).get("p95")
            extra_load_p95 += max(0.0, _finite(duration, f"{node_id} step {index} gated load p95"))
            gated_steps += 1
    return {
        "sums": totals,
        "n_steps": len(steps),
        "quantile_violations": violations,
        "gated_steps": gated_steps,
        "extra_load_p95": extra_load_p95,
    }


def truth_definitions(template: TemplateSteps, anchor: str, n_pred_steps: int) -> dict[str, Any]:
    next_k = template.next_k_events(anchor, n_pred_steps)
    remaining = template.remaining_after(anchor)
    per_step = template.events_per_remaining_step(anchor)
    return {
        "n_remaining_steps": template.remaining_step_count(anchor),
        "n_remaining_events": len(remaining),
        "events_per_remaining_step_max": max(per_step) if per_step else 0,
        "branching": bool(per_step and max(per_step) > 1),
        "s_true_next_k": float(sum(template.runtime_ms(node_id) for node_id in next_k)),
        "s_true_remaining_all": float(sum(template.runtime_ms(node_id) for node_id in remaining)),
        "s_container_total": float(sum(template.runtime_ms(node_id) for node_id in template.containers)),
        "workload_total_ms": template.workload_total_ms(),
        "wall_clock_ms": template.wall_clock_ms,
    }


def anchor_metrics(
    artifact: Mapping[str, Any],
    template: TemplateSteps,
    anchor: str,
    info: Mapping[str, str],
) -> dict[str, Any]:
    pred = predicted_sums(artifact.get("steps") or [], anchor)
    truth = truth_definitions(template, anchor, pred["n_steps"])
    record: dict[str, Any] = {
        "node_id": anchor,
        "template_id": info.get("template_id"),
        "video_id": info.get("video_id"),
        "baseline": info.get("baseline") or str(artifact["row"].get("baseline") or ""),
        "split": info.get("split"),
        "predicted_future_length": artifact["row"].get("predicted_future_length"),
        "n_pred_steps": pred["n_steps"],
        "anchor_lane": template.nodes[anchor].get("execution_lane"),
        "anchor_event_type": template.nodes[anchor].get("event_type"),
        "quantile_violations": pred["quantile_violations"],
        "gated_steps": pred["gated_steps"],
        "extra_load_p95": pred["extra_load_p95"],
    }
    record.update(truth)
    for name in ("next_k", "remaining_all"):
        s_true = float(record[f"s_true_{name}"])
        record[f"s_true_{name}"] = s_true
        for tau in TAUS:
            shat = float(pred["sums"][tau])
            record[f"shat_{name}_{tau}"] = shat
            record[f"b_{name}_{tau}"] = s_true - shat
            record[f"r_{name}_{tau}"] = (shat / s_true) if s_true > 0 else None
    record["r_next_k_p95_plus_load"] = (
        (float(pred["sums"]["p95"]) + float(pred["extra_load_p95"])) / record["s_true_next_k"]
        if record["s_true_next_k"] > 0
        else None
    )
    # --- aggregation-gap and tail-gap inputs (review 2026-09-17, P1 metrics) -----
    record["tail_gap_p95_p50"] = record["shat_next_k_p95"] - record["shat_next_k_p50"]
    record["abs_err_p50"] = abs(record["s_true_next_k"] - record["shat_next_k_p50"])
    record["abs_err_p95"] = abs(record["s_true_next_k"] - record["shat_next_k_p95"])
    return record


def aggregation_gap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum-of-quantiles vs the empirical quantile of the true sums.

    ``sum_h Q_tau(x_h)`` is the comonotonic aggregation the consumers use. Its
    reference is the empirical quantile of the *observed* sums, computed across
    anchors; a ratio above 1 means the conservative sum overshoots the workload's
    own tail, not merely its mean.
    """

    out: dict[str, Any] = {"anchors": len(rows)}
    for tau in TAUS:
        predicted = [row[f"shat_next_k_{tau}"] for row in rows]
        truths = [row["s_true_next_k"] for row in rows if row["s_true_next_k"] > 0]
        if not predicted or not truths:
            continue
        ordered = sorted(truths)
        true_p95 = ordered[int(0.95 * (len(ordered) - 1))]
        true_median = statistics.median(ordered)
        out[f"predicted_sum_{tau}_mean"] = statistics.fmean(predicted)
        out[f"true_sum_p95"] = true_p95
        out[f"true_sum_median"] = true_median
        out[f"aggregation_gap_vs_true_p95_{tau}"] = (
            statistics.fmean(predicted) / true_p95 if true_p95 > 0 else None
        )
    return out


def tail_gap_validity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Does the predicted tail gap track the actual forecast error?

    Buckets anchors by ``sum_h p95 - sum_h p50`` and reports the mean absolute
    forecast error of the p50 sum per bucket; a monotone increase supports the
    "uncertainty radius is informative" reading.
    """

    valid = [row for row in rows if row["tail_gap_p95_p50"] is not None]
    if not valid:
        return {"anchors": 0}
    ordered = sorted(valid, key=lambda row: row["tail_gap_p95_p50"])
    buckets = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    out: dict[str, Any] = {"anchors": len(ordered)}
    for index, (low, high) in enumerate(buckets):
        start = int(low * (len(ordered) - 1))
        end = max(start + 1, int(high * (len(ordered) - 1)))
        chunk = ordered[start:end]
        out[f"bucket_{index + 1}"] = {
            "n": len(chunk),
            "tail_gap_mean": statistics.fmean([row["tail_gap_p95_p50"] for row in chunk]),
            "abs_err_p50_mean": statistics.fmean([row["abs_err_p50"] for row in chunk]),
            "abs_err_p95_mean": statistics.fmean([row["abs_err_p95"] for row in chunk]),
        }
    gaps = [row["tail_gap_p95_p50"] for row in ordered]
    errors = [row["abs_err_p50"] for row in ordered]
    out["spearman_tail_gap_vs_abs_err_p50"] = spearman(gaps, errors)
    return out


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        for position, index in enumerate(order):
            result[index] = float(position)
        return result

    a = ranks(left)
    b = ranks(right)
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return (num / den) if den else None


# -------------------------------------------------------------------- summarise

TRUTH_DEFINITIONS = {
    "next_k": "non-container events of the next len(predicted_steps) steps",
    "remaining_all": "all non-container events after the anchor (rest of its step + later steps)",
}


def describe(values: Sequence[float]) -> dict[str, float]:
    kept = [float(value) for value in values if value is not None]
    if not kept:
        return {"n": 0}
    ordered = sorted(kept)
    return {
        "n": len(kept),
        "mean": statistics.fmean(kept),
        "median": statistics.median(kept),
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
    }


def bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str],
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> list[float] | None:
    if len(values) != len(clusters):
        raise ValueError("bootstrap values and clusters must be aligned")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[str(cluster)].append(float(value))
    keys = sorted(grouped)
    if len(keys) < 2:
        return None
    stats = {key: (sum(grouped[key]), len(grouped[key])) for key in keys}
    rng = random.Random(seed)
    means = []
    for _ in range(int(n_boot)):
        total = 0.0
        count = 0
        for _ in range(len(keys)):
            cluster_sum, cluster_count = stats[keys[rng.randrange(len(keys))]]
            total += cluster_sum
            count += cluster_count
        if count:
            means.append(total / count)
    means.sort()
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def paired_bootstrap_ci(
    pairs: Sequence[tuple[float | None, str]],
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> list[float] | None:
    filtered = [(float(value), str(cluster)) for value, cluster in pairs if value is not None]
    if not filtered:
        return None
    return bootstrap_ci([value for value, _ in filtered], [cluster for _, cluster in filtered], n_boot, seed)


def _stat_block(rows: Sequence[Mapping[str, Any]], bootstrap: int, seed: int) -> dict[str, Any]:
    pairs = [(str(row.get("video_id")), row) for row in rows]
    out: dict[str, Any] = {"anchors": len(rows), "videos": len({video for video, _ in pairs})}
    for name in TRUTH_DEFINITIONS:
        for tau in TAUS:
            key = f"{name}_{tau}"
            r_pairs = [(row[f"r_{key}"], video) for video, row in pairs]
            b_pairs = [(row[f"b_{key}"], video) for video, row in pairs]
            under_pairs = [
                (None if row[f"b_{key}"] is None else (1.0 if row[f"b_{key}"] > 0 else 0.0), video)
                for video, row in pairs
            ]
            out[f"r_{key}"] = describe([value for value, _ in r_pairs])
            out[f"r_{key}_ci"] = paired_bootstrap_ci(r_pairs, bootstrap, seed)
            out[f"b_{key}"] = describe([value for value, _ in b_pairs])
            out[f"b_{key}_ci"] = paired_bootstrap_ci(b_pairs, bootstrap, seed)
            out[f"underestimate_rate_{key}"] = describe([value for value, _ in under_pairs]).get("mean")
            out[f"underestimate_rate_{key}_ci"] = paired_bootstrap_ci(under_pairs, bootstrap, seed)
            valid = [
                (float(row[f"shat_{key}"]), float(row[f"s_true_{name}"]))
                for _, row in pairs
                if float(row[f"s_true_{name}"]) > 0
            ]
            denom = sum(truth for _, truth in valid)
            out[f"ratio_of_sums_{key}"] = (sum(shat for shat, _ in valid) / denom) if denom else None
    out["r_next_k_p95_plus_load"] = describe([row["r_next_k_p95_plus_load"] for _, row in pairs])
    out["r_next_k_p95_plus_load_ci"] = paired_bootstrap_ci(
        [(row["r_next_k_p95_plus_load"], video) for video, row in pairs], bootstrap, seed
    )
    out["mean_extra_load_p95_ms"] = describe([row["extra_load_p95"] for _, row in pairs]).get("mean")
    out["quantile_violation_steps"] = sum(int(row["quantile_violations"]) for _, row in pairs)
    out["zero_pred_step_anchors"] = sum(1 for _, row in pairs if row["n_pred_steps"] == 0)
    out["no_remaining_step_anchors"] = sum(1 for _, row in pairs if row["n_remaining_steps"] == 0)
    out["branching_rate"] = describe([1.0 if row["branching"] else 0.0 for _, row in pairs]).get("mean")
    out["mean_n_pred_steps"] = describe([row["n_pred_steps"] for _, row in pairs]).get("mean")
    out["mean_remaining_steps"] = describe([row["n_remaining_steps"] for _, row in pairs]).get("mean")
    out["mean_remaining_events"] = describe([row["n_remaining_events"] for _, row in pairs]).get("mean")
    out["anchor_event_types"] = dict(sorted(Counter(str(row["anchor_event_type"]) for _, row in pairs).items()))
    out["lane_anchors"] = dict(sorted(Counter(str(row["anchor_lane"]) for _, row in pairs).items()))
    out["baseline_anchors"] = dict(sorted(Counter(str(row["baseline"]) for _, row in pairs).items()))
    return out


def summarize(rows: Sequence[Mapping[str, Any]], bootstrap: int, seed: int, wall_clock_ratio: Mapping[str, Any]) -> dict[str, Any]:
    populations = {
        "primary_gpu_with_remaining": [
            row for row in rows if row["anchor_lane"] == "gpu" and row["n_remaining_steps"] > 0
        ],
        "gpu_with_remaining": [row for row in rows if row["anchor_lane"] == "gpu" and row["n_remaining_steps"] > 0],
        "last_step_gpu": [row for row in rows if row["anchor_lane"] == "gpu" and row["n_remaining_steps"] == 0],
        "branching": [row for row in rows if row["branching"]],
        "all_anchors": list(rows),
    }
    primary = populations["primary_gpu_with_remaining"] or populations["gpu_with_remaining"]
    return {
        "populations": {
            name: _stat_block(population, bootstrap, seed) for name, population in populations.items() if population
        },
        "primary_aggregation_gap": aggregation_gap(primary),
        "primary_tail_gap_validity": tail_gap_validity(primary),
        "wall_clock_check": dict(wall_clock_ratio),
        "counts": {
            "joined_anchors": len(rows),
            "distinct_videos": len({str(row.get("video_id")) for row in rows}),
            "distinct_templates": len({str(row.get("template_id")) for row in rows}),
            "zero_pred_step_anchors": sum(1 for row in rows if row["n_pred_steps"] == 0),
            "no_remaining_step_anchors": sum(1 for row in rows if row["n_remaining_steps"] == 0),
            "anchors_with_extra_load": sum(1 for row in rows if row["extra_load_p95"] > 0),
        },
        "truth_definitions": TRUTH_DEFINITIONS,
    }


# ------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--templates", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--expect-videos", type=int, default=EXPECTED_VIDEOS)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=0, help="cap anchors (0 = no cap); smoke only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.project_root
    artifacts = args.artifacts or (
        root
        / "outputs"
        / "sstar_predictor_artifacts_dist_sched"
        / "prediction_artifacts"
        / f"b05_future_h{int(args.horizon)}.jsonl.gz"
    )
    templates = args.templates or (
        root / "results" / "processed" / "r7_workload_20260817" / "job_templates_r7_v02.jsonl"
    )
    return artifacts, templates


def wall_clock_check(unique_templates: Mapping[str, TemplateSteps]) -> dict[str, Any]:
    """Every template's non-container workload must equal its wall clock."""

    ratios = []
    for template in unique_templates.values():
        wall = template.wall_clock_ms
        if wall > 0:
            ratios.append(template.workload_total_ms() / wall)
    if not ratios:
        return {"n": 0}
    ordered = sorted(ratios)
    return {
        "n": len(ratios),
        "median": statistics.median(ordered),
        "p10": ordered[int(0.10 * (len(ordered) - 1))],
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts_path, templates_path = resolve_inputs(args)
    for path in (artifacts_path, templates_path):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    artifacts = load_artifact_rows(artifacts_path, args.horizon)
    templates, meta = load_templates(templates_path)

    joined: list[dict[str, Any]] = []
    unjoined_run_root: list[str] = []
    unjoined_container: list[str] = []
    unjoined_other: list[str] = []
    unique_templates: dict[str, TemplateSteps] = {}
    for node_id, artifact in artifacts.items():
        template = templates.get(node_id)
        if template is None:
            (unjoined_run_root if node_id.endswith(RUN_ROOT_SUFFIX) else unjoined_other).append(node_id)
            continue
        if not template.has(node_id):
            # a container event is not a step, so it has no step-level truth
            (unjoined_container if node_id in template.nodes else unjoined_other).append(node_id)
            continue
        unique_templates[meta[node_id]["template_id"]] = template
        joined.append(anchor_metrics(artifact, template, node_id, meta[node_id]))

    distinct_videos = {str(row["video_id"]) for row in joined}
    if len(distinct_videos) != int(args.expect_videos):
        raise SystemExit(
            f"video integrity check failed: {len(distinct_videos)} distinct videos, expected {int(args.expect_videos)}"
        )
    if unjoined_other:
        raise SystemExit(f"unexpected unjoined anchors: {len(unjoined_other)} (sample {unjoined_other[:5]})")

    wc = wall_clock_check(unique_templates)
    low, high = WALL_CLOCK_TOLERANCE
    if wc.get("n") and not (low <= wc["median"] <= high):
        raise SystemExit(f"wall-clock integrity check failed: {wc}")

    if args.limit:
        joined = joined[: int(args.limit)]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "operation": "dry-run",
                    "artifact_file": str(artifacts_path),
                    "artifact_anchors": len(artifacts),
                    "templates_file": str(templates_path),
                    "template_nodes": len(templates),
                    "joined_anchors": len(joined),
                    "distinct_videos": len(distinct_videos),
                    "templates_seen": len(unique_templates),
                    "unjoined_run_root_anchors": len(unjoined_run_root),
                    "excluded_container_anchors": len(unjoined_container),
                    "unjoined_other": len(unjoined_other),
                    "wall_clock_check": wc,
                    "horizon": int(args.horizon),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    summary = summarize(joined, args.bootstrap, args.seed, wc)
    summary["meta"] = {
        "artifact_file": str(artifacts_path),
        "templates_file": str(templates_path),
        "horizon": int(args.horizon),
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "unjoined_run_root_anchors": len(unjoined_run_root),
        "excluded_container_anchors": len(unjoined_container),
        "primary_population": "primary_gpu_with_remaining (gpu anchors with at least one remaining step)",
        "definitions": {
            "shat_tau": "sum of predicted runtime_ms_quantiles[tau] over the artifact scenario-0 steps (0 allowed)",
            "s_true_next_k": "the next len(predicted_steps) non-container events (event-level slots), summed",
            "s_true_remaining_all": "all non-container events after the anchor, summed",
            "b": "s_true - shat (positive = prediction too low)",
            "r": "shat / s_true (undefined when s_true = 0)",
            "extra_load_p95": "load_duration_ms_quantiles.p95 over steps with load_occurrence_probability >= 0.5",
            "bootstrap_cluster": "video_id from the template record",
        },
        "excluded_from_truth": "event_type == 'run' whole-run wall-clock containers (their runtime equals the whole template span)",
        "data_facts": {
            "anchors_from_r7_templates": "640/640 run ids overlap",
            "j_series_dataset_disjoint": "0/640 run-id overlap and 0/160 video overlap with the R7 workload",
            "workload_equals_wall_clock": "sum of non-container event runtimes vs run wall clock",
        },
    }

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with (out / "per_anchor.jsonl").open("w", encoding="utf-8") as handle:
            for record in joined:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {out / 'metrics.json'} and {out / 'per_anchor.jsonl'} ({len(joined)} anchors)")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
