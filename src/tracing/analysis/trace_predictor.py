#!/usr/bin/env python3
"""Train and evaluate the scheduler's own hierarchical trace predictors.

These are intentionally small, auditable models rather than a claim that a
paper's private predictor has been reproduced.  They predict the next
canonical agent activity from information available at prefix time.  The v0.4
primary model separates planner identity and observed state-tail context from
the v0.2 legacy model and removes broad priors that mix planner families.
Count interpolation has deterministic backoff, so every prediction can be
inspected as a set of context counts.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tracing.analysis.reproduce_prediction_baselines import (
    END_LABEL,
    Prefix,
    _classification_metrics,
    _normalise,
    load_prefixes,
    load_split,
)


def _text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _bucket(value: Any, cuts: Sequence[float]) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    for index, cut in enumerate(cuts):
        if number < cut:
            return f"b{index}"
    return f"b{len(cuts)}"


def _task_signature(row: Prefix) -> tuple[Any, ...]:
    task = row.task_structure
    modalities = tuple(sorted(_text(item) for item in (task.get("required_modalities") or [])))
    return (
        _text(task.get("question_type")),
        _text(task.get("temporal_scope")),
        _text(task.get("answer_type")),
        _text(task.get("official_task_type")),
        modalities,
        int(task.get("option_count") or 0),
    )


def _state_signature(row: Prefix) -> tuple[Any, ...]:
    state = row.state_features
    prefix = state.get("prefix") if isinstance(state.get("prefix"), Mapping) else {}
    evidence = state.get("evidence") if isinstance(state.get("evidence"), Mapping) else {}
    modalities = evidence.get("modality_counts") if isinstance(evidence.get("modality_counts"), Mapping) else {}
    modality_names = tuple(sorted(_text(key) for key, value in modalities.items() if float(value or 0) > 0))
    return (
        _bucket(evidence.get("coverage_ratio"), (0.05, 0.25, 0.50, 0.75, 1.0)),
        _bucket(evidence.get("frames_seen"), (1, 4, 8, 16, 32)),
        modality_names,
        int(prefix.get("retry_count") or 0) > 0,
        int(prefix.get("error_count") or 0) > 0,
    )


def _state_last_actions(row: Prefix) -> tuple[str, ...]:
    """Return only the already-observed state action tail.

    ``state_t`` is a pre-action snapshot, so this field is safe for next-node
    prediction.  Missing sidecars intentionally map to an empty tuple rather
    than inventing a state.
    """
    state = row.state_features
    prefix = state.get("prefix") if isinstance(state.get("prefix"), Mapping) else {}
    values = prefix.get("last_actions") if isinstance(prefix, Mapping) else []
    return tuple(_text(value, "unknown") for value in (values or [])[-3:])


def scheduler_class(activity: str) -> str:
    """Map fine semantic nodes to the routing class used by the scheduler."""
    if activity == "summarize":
        return "summarize"
    if activity == "answer":
        return "answer"
    if activity == END_LABEL:
        return "end"
    if activity == "retry":
        return "retry"
    return "observe"


class HierarchicalTracePredictor:
    """Count-interpolated next-node predictor with explicit feature levels."""

    # Stronger contexts are useful only when observed more than once.  The
    # lower levels stabilize unseen task/video combinations.
    LEVEL_WEIGHTS: tuple[tuple[str, float, int], ...] = (
        ("prefix_task", 6.0, 2),
        ("prefix", 8.0, 2),
        ("stack_last", 3.5, 1),
        ("last", 5.0, 1),
        ("position_baseline", 4.0, 1),
        ("position", 1.5, 1),
        ("task_position", 1.5, 1),
        ("baseline", 0.5, 1),
        ("global", 0.25, 1),
    )

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        self.target_fn = lambda row: row.target
        self.counts: dict[str, dict[tuple[Any, ...], Counter[str]]] = {
            name: defaultdict(Counter) for name, _, _ in self.LEVEL_WEIGHTS
        }
        self.context_seen: Counter[str] = Counter()

    @staticmethod
    def _last(row: Prefix) -> str:
        return row.prefix[-1] if row.prefix else "__START__"

    def _keys(self, row: Prefix) -> dict[str, tuple[Any, ...]]:
        task = _task_signature(row)
        state = _state_signature(row)
        last = self._last(row)
        base = (row.baseline,)
        position = (row.position,)
        return {
            "prefix_task": base + (row.model_stack_id, position, row.prefix, task, state),
            "prefix": base + (position, row.prefix),
            "stack_last": base + (row.model_stack_id, position, last),
            "last": base + (position, last),
            "position_baseline": base + position,
            "position": position,
            "task_position": base + (position, task),
            "baseline": base,
            "global": tuple(),
        }

    def fit(self, rows: Sequence[Prefix], target_fn: Any | None = None) -> "HierarchicalTracePredictor":
        if target_fn is not None:
            self.target_fn = target_fn
        for row in rows:
            target = self.target_fn(row)
            keys = self._keys(row)
            for name, _, _ in self.LEVEL_WEIGHTS:
                key = keys.get(name)
                if key is None:
                    continue
                self.counts[name][key][target] += 1
                self.context_seen[name] += 1
        return self

    def predict(self, row: Prefix) -> dict[str, float]:
        scores: Counter[str] = Counter()
        keys = self._keys(row)
        used = 0
        for name, weight, minimum_count in self.LEVEL_WEIGHTS:
            if weight <= 0:
                continue
            key = keys.get(name)
            if key is None:
                continue
            counts = self.counts[name].get(key)
            if not counts or sum(counts.values()) < minimum_count:
                continue
            probability = _normalise(counts, self.labels, alpha=0.25)
            for label, value in probability.items():
                scores[label] += weight * value
            used += 1
        if not used:
            return _normalise({}, self.labels)
        total = sum(scores.values())
        if total <= 0:
            return _normalise({}, self.labels)
        return {label: scores.get(label, 0.0) / total for label in self.labels}

    def explanation(self, row: Prefix) -> dict[str, Any]:
        keys = self._keys(row)
        contexts = []
        for name, weight, minimum_count in self.LEVEL_WEIGHTS:
            if weight <= 0:
                continue
            key = keys.get(name)
            if key is None:
                continue
            counts = self.counts[name].get(key)
            if counts and sum(counts.values()) >= minimum_count:
                contexts.append({
                    "level": name,
                    "weight": weight,
                    "count": sum(counts.values()),
                    "distribution": dict(counts),
                })
        probabilities = self.predict(row)
        return {
            "run_id": row.run_id,
            "position": row.position,
            "prefix": list(row.prefix),
            "top": sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))[:3],
            "contexts": contexts,
        }


class PlannerAwareTracePredictor(HierarchicalTracePredictor):
    """Planner/state-aware interpolation selected for the v0.4 report.

    The v0.2 model had a real blind spot: planner model identity was only
    present inside one very sparse context, so the broad position prior mixed
    Qwen3-4B and Qwen3-VL-8B behavior.  These contexts condition the stable
    backoff path on ``planner_model_id`` and the observed state action tail.
    Broad position-only and baseline-position priors are deliberately omitted
    because they mix planner families after the planner-specific contexts are
    exhausted.  The model never uses video identity, target fields, or future
    events.
    """

    LEVEL_WEIGHTS: tuple[tuple[str, float, int], ...] = (
        ("planner_last_actions", 2.0, 2),
        ("planner_last", 1.0, 2),
        ("planner_actions", 1.0, 2),
        ("planner_position", 1.0, 2),
        ("baseline_last", 1.0, 2),
    )

    def _keys(self, row: Prefix) -> dict[str, tuple[Any, ...]]:
        base = (row.baseline,)
        planner = (row.planner_model_id,)
        position = (row.position,)
        last = (self._last(row),)
        last_actions = (_state_last_actions(row),)
        return {
            "planner_last_actions": base + planner + position + last + last_actions,
            "planner_last": base + planner + position + last,
            "planner_actions": base + planner + position + last_actions,
            "planner_position": base + planner + position,
            "baseline_last": base + position + last,
            "baseline_position": base + position,
            "position": position,
        }


class PlannerAwareTracePredictorV03(PlannerAwareTracePredictor):
    """Frozen v0.3 weighting retained for an apples-to-apples comparison."""

    LEVEL_WEIGHTS: tuple[tuple[str, float, int], ...] = (
        ("planner_last_actions", 1.0, 2),
        ("planner_last", 1.0, 2),
        ("planner_actions", 1.0, 2),
        ("planner_position", 1.0, 2),
        ("baseline_last", 1.0, 2),
        ("baseline_position", 1.0, 2),
        ("position", 1.0, 2),
    )


class SchedulerClassPredictor:
    """Stable coarse routing prior used by admission/scheduling decisions."""

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        self.by_baseline_position: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
        self.by_position: dict[int, Counter[str]] = defaultdict(Counter)

    def fit(self, rows: Sequence[Prefix]) -> "SchedulerClassPredictor":
        for row in rows:
            target = scheduler_class(row.target)
            self.by_baseline_position[(row.baseline, row.position)][target] += 1
            self.by_position[row.position][target] += 1
        return self

    def predict(self, row: Prefix) -> dict[str, float]:
        counts = self.by_baseline_position.get((row.baseline, row.position))
        if not counts:
            counts = self.by_position.get(row.position, Counter())
        return _normalise(counts, self.labels, alpha=0.25)


def _ece(rows: Sequence[tuple[str, Mapping[str, float]]], labels: Sequence[str], bins: int = 10) -> float:
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for target, probabilities in rows:
        confidence = max(probabilities.values()) if probabilities else 0.0
        prediction = max(probabilities, key=probabilities.get) if probabilities else None
        index = min(bins - 1, int(confidence * bins))
        buckets[index].append((confidence, int(prediction == target)))
    total = max(1, len(rows))
    return sum(
        len(bucket) / total * abs(sum(conf for conf, _ in bucket) / len(bucket) - sum(hit for _, hit in bucket) / len(bucket))
        for bucket in buckets
        if bucket
    )


def _evaluate(
    train: Sequence[Prefix],
    test: Sequence[Prefix],
    model_type: type[HierarchicalTracePredictor] = HierarchicalTracePredictor,
) -> dict[str, Any]:
    labels = sorted({row.target for row in train})
    model = model_type(labels).fit(train)
    scheduler_labels = sorted({scheduler_class(row.target) for row in train})
    scheduler_model = SchedulerClassPredictor(scheduler_labels).fit(train)
    all_rows: list[tuple[str, Mapping[str, float]]] = []
    nonterminal: list[tuple[str, Mapping[str, float]]] = []
    scheduler_rows: list[tuple[str, Mapping[str, float]]] = []
    scheduler_nonterminal: list[tuple[str, Mapping[str, float]]] = []
    for row in test:
        probabilities = model.predict(row)
        pair = (row.target, probabilities)
        all_rows.append(pair)
        if row.target != END_LABEL:
            nonterminal.append(pair)
        scheduler_pair = (scheduler_class(row.target), scheduler_model.predict(row))
        scheduler_rows.append(scheduler_pair)
        if row.target != END_LABEL:
            scheduler_nonterminal.append(scheduler_pair)
    warmup_metrics: dict[str, Any] = {}
    for minimum_position in range(0, 7):
        selected = [index for index, row in enumerate(test) if row.position >= minimum_position]
        fine_subset = [all_rows[index] for index in selected]
        scheduler_subset = [scheduler_rows[index] for index in selected]
        warmup_metrics[str(minimum_position)] = {
            "n": len(selected),
            "fine": _classification_metrics(fine_subset, labels),
            "scheduler": _classification_metrics(scheduler_subset, scheduler_labels),
        }
    result = {
        "model": model.__class__.__name__,
        "classification": _classification_metrics(all_rows, labels),
        "classification_nonterminal": _classification_metrics(nonterminal, labels),
        "classification_scheduler": _classification_metrics(scheduler_rows, scheduler_labels),
        "classification_scheduler_nonterminal": _classification_metrics(scheduler_nonterminal, scheduler_labels),
        "ece": _ece(all_rows, labels),
        "ece_nonterminal": _ece(nonterminal, labels),
        "ece_scheduler": _ece(scheduler_rows, scheduler_labels),
        "ece_scheduler_nonterminal": _ece(scheduler_nonterminal, scheduler_labels),
        "metrics_by_minimum_position": warmup_metrics,
        "labels_from_train_only": labels,
        "scheduler_labels_from_train_only": scheduler_labels,
        "context_counts": dict(model.context_seen),
    }
    if test:
        result["example_explanation"] = model.explanation(test[0])
    return result


def build_report(prefix_path: Path, split_path: Path) -> dict[str, Any]:
    split_map = load_split(split_path)
    rows = load_prefixes(prefix_path, split_map)
    by_split: dict[str, list[Prefix]] = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)
    expected = {"train": 48, "validation": 8, "test": 8}
    video_counts = {split: len({row.video_id for row in values}) for split, values in by_split.items()}
    gate = all(video_counts.get(split, 0) == count for split, count in expected.items())
    train = by_split.get("train", [])
    report: dict[str, Any] = {
        "schema_version": "own-trace-predictor-v0.4",
        "input": str(prefix_path),
        "rows": len(rows),
        "runs": len({row.run_id for row in rows}),
        "videos": len({row.video_id for row in rows}),
        "video_counts": video_counts,
        "formal_split_gate": gate,
        "model": "planner_state_count_interpolation_v04",
        "feature_contract": {
            "uses": ["baseline", "planner_model_id", "position", "observed_prefix", "state_features.last_actions"],
            "does_not_use": ["video_id", "target_next_activity", "remaining_steps", "remaining_runtime_ms", "future_events", "answer_text"],
        },
        "metrics": {},
        "comparison": {},
        "model_selection": {
            "selection_split": "validation",
            "locked_evaluation_split": "test",
            "criterion": "validation_top1_then_nll_with_top3_gate",
            "broad_position_priors_removed": True,
            "test_labels_used_for_selection": False,
        },
    }
    if train:
        for split in ("validation", "test"):
            if by_split.get(split):
                report["metrics"][split] = _evaluate(train, by_split[split], PlannerAwareTracePredictor)
                report["comparison"][split] = {
                    "hierarchical_v0_2": _evaluate(train, by_split[split], HierarchicalTracePredictor),
                    "planner_state_v0_3_on_repaired_labels": _evaluate(
                        train, by_split[split], PlannerAwareTracePredictorV03
                    ),
                }
    report["state_coverage"] = {
        "prefix_rows_with_state": sum(bool(row.state_features.get("state_present")) for row in rows),
        "prefix_rows": len(rows),
        "fraction": sum(bool(row.state_features.get("state_present")) for row in rows) / max(1, len(rows)),
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_report(args.prefix, args.split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"formal_split_gate": report["formal_split_gate"], "metrics": report["metrics"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
