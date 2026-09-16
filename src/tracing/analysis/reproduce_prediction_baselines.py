#!/usr/bin/env python3
"""Reproduce lightweight future-node predictors on enriched trace prefixes.

The implementation is deliberately small and dependency-free.  It consumes
``prefix_samples_v0_1.jsonl`` produced by ``build_trace_enrichment`` and keeps
the target fields separate from the prefix input.  The reported names are
adaptation labels, not claims that the original papers' complete systems have
been reimplemented:

* ``dyorc_markov1``/``dyorc_markov2``: local Markov reimplementation;
* ``autotool_next_tool``: raw-tool transition adapter;
* ``pythia_path_length``: prefix-matched path and length adapter;
* ``rag_next_activity``: train-only nearest-prefix retrieval adapter;
* ``structured_prefix``: categorical task + observed-prefix adapter.

Formal evaluation requires a video-level split manifest.  Without one, the
``--diagnostic-loo`` flag runs a leave-one-video-out diagnostic and marks the
result non-formal.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


END_LABEL = "__END__"
SPLIT_ALIASES = {"valid": "validation", "val": "validation", "dev": "validation"}


@dataclass(frozen=True)
class Prefix:
    run_id: str
    video_id: str
    split: str
    baseline: str
    prefix: tuple[str, ...]
    raw_prefix: tuple[str, ...]
    target: str
    raw_target: str
    remaining_steps: int
    remaining_runtime_ms: float
    task_structure: Mapping[str, Any]
    model_stack_id: str = "unknown"
    planner_model_id: str = "unknown"
    position: int = 0
    state_features: Mapping[str, Any] = field(default_factory=dict)


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _normalise_split(value: Any) -> str:
    split = _text(value, "unassigned").lower()
    return SPLIT_ALIASES.get(split, split)


def load_split(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in _read_jsonl(path):
        video_id = _text(row.get("video_id") or row.get("source_video_id"))
        split = _normalise_split(row.get("split"))
        if not video_id or split == "unassigned":
            continue
        previous = mapping.get(video_id)
        if previous and previous != split:
            raise ValueError(f"video {video_id} has conflicting split assignments")
        mapping[video_id] = split
    return mapping


def load_prefixes(path: Path, split_map: Mapping[str, str] | None = None) -> list[Prefix]:
    rows = _read_jsonl(path)
    prefixes: list[Prefix] = []
    for row in rows:
        if row.get("future_events_included_in_input") is not False:
            raise ValueError(f"future-event leakage marker is not false: {row.get('prefix_id')}")
        if row.get("ground_truth_included_in_input") is not False:
            raise ValueError(f"ground-truth leakage marker is not false: {row.get('prefix_id')}")
        state_features = row.get("state_features") if isinstance(row.get("state_features"), Mapping) else {}
        if state_features.get("state_present"):
            state_guard = state_features.get("leakage_guard") if isinstance(state_features.get("leakage_guard"), Mapping) else {}
            if state_features.get("future_events_excluded") is not True or state_features.get("ground_truth_excluded") is not True:
                raise ValueError(f"state future-event leakage marker is not false: {row.get('prefix_id')}")
            if state_guard.get("future_events_excluded") is not True or state_guard.get("ground_truth_excluded") is not True:
                raise ValueError(f"state leakage guard failed: {row.get('prefix_id')}")
            if state_guard.get("answer_text_used_as_feature") is True or state_guard.get("video_id_used_as_feature") is True:
                raise ValueError(f"state feature leakage guard failed: {row.get('prefix_id')}")
        prefix = tuple(_text(value, "other") for value in (row.get("prefix_activities") or []))
        raw_prefix = tuple(_text(value, "unknown") for value in (row.get("prefix_raw_actions") or []))
        if len(prefix) != len(raw_prefix):
            raise ValueError(f"canonical/raw prefix length mismatch: {row.get('prefix_id')}")
        video_id = _text(row.get("video_id"), "unknown")
        manifest_split = _normalise_split(row.get("split"))
        mapped_split = _normalise_split(split_map.get(video_id)) if split_map else manifest_split
        if split_map and manifest_split not in {"unassigned", mapped_split}:
            raise ValueError(f"prefix split conflicts with split manifest for {video_id}")
        target = _text(row.get("target_next_activity"), END_LABEL)
        raw_target = _text(row.get("target_next_raw_action"), END_LABEL)
        prefixes.append(
            Prefix(
                run_id=_text(row.get("run_id"), "unknown"),
                video_id=video_id,
                split=mapped_split,
                baseline=_text(row.get("baseline"), "unknown"),
                model_stack_id=_text(row.get("model_stack_id"), "unknown"),
                planner_model_id=_text(row.get("planner_model_id"), "unknown"),
                prefix=prefix,
                raw_prefix=raw_prefix,
                target=target,
                raw_target=raw_target,
                position=max(0, int(row.get("position") if row.get("position") is not None else len(prefix))),
                remaining_steps=max(0, int(row.get("remaining_steps") or 0)),
                remaining_runtime_ms=max(0.0, _number(row.get("remaining_runtime_ms"))),
                task_structure=dict(row.get("task_structure") or {}),
                state_features=dict(state_features),
            )
        )
    if not prefixes:
        raise ValueError(f"no prefix rows found in {path}")
    return prefixes


def _labels(rows: Sequence[Prefix], raw: bool = False) -> list[str]:
    return sorted({(row.raw_target if raw else row.target) for row in rows})


def _normalise(counts: Mapping[str, float], labels: Sequence[str], alpha: float = 1.0) -> dict[str, float]:
    if not labels:
        return {}
    values = {label: max(0.0, float(counts.get(label, 0.0))) + alpha for label in labels}
    total = sum(values.values())
    return {label: value / total for label, value in values.items()}


class TransitionModel:
    def __init__(self, labels: Sequence[str], order: int, use_baseline: bool = True) -> None:
        self.labels = list(labels)
        self.order = order
        self.use_baseline = use_baseline
        self.global_counts: Counter[str] = Counter()
        self.baseline_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.position_counts: dict[int, Counter[str]] = defaultdict(Counter)
        self.baseline_position_counts: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
        self.transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.position_transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    def fit(self, rows: Sequence[Prefix], *, raw: bool = False) -> None:
        for row in rows:
            target = row.raw_target if raw else row.target
            sequence = row.raw_prefix if raw else row.prefix
            self.global_counts[target] += 1
            self.baseline_counts[row.baseline][target] += 1
            self.position_counts[row.position][target] += 1
            self.baseline_position_counts[(row.baseline, row.position)][target] += 1
            if sequence:
                history = sequence[-self.order :]
                key = ((row.baseline,) if self.use_baseline else tuple()) + history
                self.transitions[key][target] += 1
                position_key = ((row.baseline,) if self.use_baseline else tuple()) + (str(row.position),) + history
                self.position_transitions[position_key][target] += 1

    def predict(self, row: Prefix, *, raw: bool = False) -> dict[str, float]:
        sequence = row.raw_prefix if raw else row.prefix
        if sequence:
            history = sequence[-self.order :]
            position_key = ((row.baseline,) if self.use_baseline else tuple()) + (str(row.position),) + history
            counts = self.position_transitions.get(position_key)
            if counts:
                return _normalise(counts, self.labels)
            key = ((row.baseline,) if self.use_baseline else tuple()) + history
            counts = self.transitions.get(key)
            if counts:
                return _normalise(counts, self.labels)
        counts = self.baseline_position_counts.get((row.baseline, row.position)) if self.use_baseline else None
        if counts:
            return _normalise(counts, self.labels)
        counts = self.position_counts.get(row.position)
        if counts:
            return _normalise(counts, self.labels)
        counts = self.baseline_counts.get(row.baseline) if self.use_baseline else None
        return _normalise(counts or self.global_counts, self.labels)


class PlannerTransitionModel(TransitionModel):
    """Markov adapter that keeps planner identity in the observed context."""

    def __init__(self, labels: Sequence[str], order: int, use_baseline: bool = True) -> None:
        super().__init__(labels, order, use_baseline)
        self.planner_position_counts: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
        self.planner_transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.planner_position_transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    def fit(self, rows: Sequence[Prefix], *, raw: bool = False) -> None:
        super().fit(rows, raw=raw)
        for row in rows:
            target = row.raw_target if raw else row.target
            sequence = row.raw_prefix if raw else row.prefix
            planner = row.planner_model_id
            self.planner_position_counts[(row.baseline, planner, row.position)][target] += 1
            if sequence:
                history = sequence[-self.order :]
                prefix = (row.baseline, planner)
                self.planner_transitions[prefix + history][target] += 1
                self.planner_position_transitions[prefix + (str(row.position),) + history][target] += 1

    def predict(self, row: Prefix, *, raw: bool = False) -> dict[str, float]:
        sequence = row.raw_prefix if raw else row.prefix
        planner = (row.baseline, row.planner_model_id)
        if sequence:
            history = sequence[-self.order :]
            counts = self.planner_position_transitions.get(planner + (str(row.position),) + history)
            if counts:
                return _normalise(counts, self.labels)
            counts = self.planner_transitions.get(planner + history)
            if counts:
                return _normalise(counts, self.labels)
        counts = self.planner_position_counts.get((row.baseline, row.planner_model_id, row.position))
        if counts:
            return _normalise(counts, self.labels)
        return super().predict(row, raw=raw)


def _task_key(row: Prefix, include_baseline: bool = True) -> tuple[Any, ...]:
    task = row.task_structure
    values: tuple[Any, ...] = (
        _text(task.get("question_type"), "unknown"),
        _text(task.get("official_task_type"), "unknown"),
        _text(task.get("domain"), "unknown"),
        _text(task.get("sub_category"), "unknown"),
        _text(task.get("answer_type"), "unknown"),
        tuple(str(value) for value in (task.get("required_modalities") or [])),
        int(task.get("option_count") or 0),
        min(len(row.prefix), 8),
        row.prefix[-1] if row.prefix else "__START__",
    )
    return ((row.baseline,) if include_baseline else tuple()) + values


class StructuredModel:
    def __init__(self, labels: Sequence[str], include_baseline: bool = True) -> None:
        self.labels = list(labels)
        self.include_baseline = include_baseline
        self.counts: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
        self.backoff: Counter[str] = Counter()

    def fit(self, rows: Sequence[Prefix]) -> None:
        for row in rows:
            self.counts[_task_key(row, self.include_baseline)][row.target] += 1
            self.backoff[row.target] += 1

    def predict(self, row: Prefix) -> dict[str, float]:
        return _normalise(self.counts.get(_task_key(row, self.include_baseline), self.backoff), self.labels)


class StructuredBackoffModel:
    """Hierarchical structured-input adapter with auditable backoff.

    ``StructuredModel`` intentionally represents an exact-key adapter, but
    its full key is sparse for unseen videos and therefore falls back to a
    global prior on most test rows.  This companion keeps the same task
    fields while interpolating coarser task, observed-last-action and
    baseline-position counts.  It is an adapter baseline, not a claim about a
    paper's private implementation.
    """

    LEVEL_WEIGHTS: tuple[tuple[str, float, int], ...] = (
        ("task_position", 2.0, 2),
        ("task", 1.0, 2),
        ("last_position", 1.0, 2),
        ("position", 1.0, 2),
        ("global", 0.25, 1),
    )

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        self.counts: dict[str, dict[tuple[Any, ...], Counter[str]]] = {
            name: defaultdict(Counter) for name, _, _ in self.LEVEL_WEIGHTS
        }

    @staticmethod
    def _keys(row: Prefix) -> dict[str, tuple[Any, ...]]:
        task = row.task_structure
        question_type = _text(task.get("question_type"), "unknown")
        official_task_type = _text(task.get("official_task_type"), "unknown")
        answer_type = _text(task.get("answer_type"), "unknown")
        temporal_scope = _text(task.get("temporal_scope"), "unknown")
        task_key = (question_type, official_task_type, answer_type, temporal_scope)
        last = row.prefix[-1] if row.prefix else "__START__"
        return {
            "task_position": (row.baseline, row.position) + task_key,
            "task": (row.baseline,) + task_key,
            "last_position": (row.baseline, row.position, last),
            "position": (row.baseline, row.position),
            "global": tuple(),
        }

    def fit(self, rows: Sequence[Prefix]) -> "StructuredBackoffModel":
        for row in rows:
            keys = self._keys(row)
            for name, _, _ in self.LEVEL_WEIGHTS:
                self.counts[name][keys[name]][row.target] += 1
        return self

    def predict(self, row: Prefix) -> dict[str, float]:
        scores: Counter[str] = Counter()
        keys = self._keys(row)
        used = 0
        for name, weight, minimum_count in self.LEVEL_WEIGHTS:
            counts = self.counts[name].get(keys[name])
            if not counts or sum(counts.values()) < minimum_count:
                continue
            probabilities = _normalise(counts, self.labels)
            for label, value in probabilities.items():
                scores[label] += weight * value
            used += 1
        if not used or sum(scores.values()) <= 0:
            return _normalise({}, self.labels)
        total = sum(scores.values())
        return {label: scores.get(label, 0.0) / total for label in self.labels}


class RetrievalModel:
    """Train-only exact/nearest prefix retrieval adapter."""

    def __init__(self, labels: Sequence[str]) -> None:
        self.labels = list(labels)
        self.rows: list[Prefix] = []

    def fit(self, rows: Sequence[Prefix]) -> None:
        self.rows = list(rows)

    @staticmethod
    def _score(query: Prefix, candidate: Prefix) -> float:
        score = 0.0
        if _task_key(query, include_baseline=False)[:-3] == _task_key(candidate, include_baseline=False)[:-3]:
            score += 5.0
        if query.baseline == candidate.baseline:
            score += 1.0
        if query.prefix == candidate.prefix:
            score += 8.0
        shared = 0
        for left, right in zip(reversed(query.prefix), reversed(candidate.prefix)):
            if left != right:
                break
            shared += 1
        score += shared * 2.0
        score -= abs(len(query.prefix) - len(candidate.prefix)) * 0.25
        return score

    def predict(self, row: Prefix) -> dict[str, float]:
        if not self.rows:
            return _normalise({}, self.labels)
        ranked = sorted(
            self.rows,
            key=lambda candidate: (-self._score(row, candidate), candidate.run_id, candidate.position),
        )[:16]
        counts = Counter(candidate.target for candidate in ranked)
        return _normalise(counts, self.labels)


class PathLengthModel:
    """Prefix-matched suffix adapter for Pythia-style path/length output."""

    def __init__(self, labels: Sequence[str], path_k: int = 3) -> None:
        self.labels = list(labels)
        self.path_k = path_k
        self.by_prefix: dict[tuple[str, str, tuple[str, ...]], list[Prefix]] = defaultdict(list)
        self.fallback = TransitionModel(labels, order=2)

    def fit(self, rows: Sequence[Prefix]) -> None:
        self.fallback.fit(rows)
        for row in rows:
            self.by_prefix[(row.baseline, row.video_id, row.prefix)].append(row)
        # A cross-video prefix index is the intended adaptation; do not use
        # video_id as a retrieval key at prediction time.
        grouped: dict[str, list[Prefix]] = defaultdict(list)
        for row in rows:
            grouped[row.run_id].append(row)
        suffix_by_key: dict[tuple[str, int], list[str]] = {}
        for run_rows in grouped.values():
            ordered = sorted(run_rows, key=lambda item: item.position)
            for index, item in enumerate(ordered):
                suffix_by_key[(item.run_id, item.position)] = [tail.target for tail in ordered[index:]]

        # Cross-video prefix index; every candidate keeps its complete observed
        # future suffix so path coverage is not computed from prefix + target.
        self.cross_prefix: dict[tuple[str, tuple[str, ...]], list[tuple[Prefix, list[str]]]] = defaultdict(list)
        for row in rows:
            self.cross_prefix[(row.baseline, row.prefix)].append((row, suffix_by_key[(row.run_id, row.position)]))

    def predict(self, row: Prefix) -> tuple[dict[str, float], list[list[str]], float, float]:
        matches = self.cross_prefix.get((row.baseline, row.prefix), [])
        if matches:
            counts = Counter(match.target for match, _ in matches)
            tail_counts = Counter(tuple(suffix) for _, suffix in matches)
            tails = [list(tail) for tail, _ in sorted(tail_counts.items(), key=lambda item: (-item[1], len(item[0]), item[0]))[: self.path_k]]
            lengths = [match.remaining_steps for match, _ in matches]
            mean = statistics.fmean(lengths)
            std = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
            return _normalise(counts, self.labels), tails, max(0.0, mean - std), mean + std
        probabilities = self.fallback.predict(row)
        next_label = max(probabilities, key=probabilities.get) if probabilities else END_LABEL
        return probabilities, [[next_label]], float(row.remaining_steps), float(row.remaining_steps)


def _classification_metrics(rows: Sequence[tuple[str, Mapping[str, float]]], labels: Sequence[str]) -> dict[str, float]:
    if not rows:
        return {"n": 0}
    top1 = top3 = oov = 0
    mrr = nll = 0.0
    for target, probabilities in rows:
        ordered = sorted(labels, key=lambda label: probabilities.get(label, 0.0), reverse=True)
        if target not in ordered:
            oov += 1
        rank = ordered.index(target) + 1 if target in ordered else len(labels) + 1
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
        mrr += 1.0 / rank
        nll -= math.log(max(probabilities.get(target, 0.0), 1e-12))
    n = len(rows)
    return {"n": n, "top1": top1 / n, "top3": top3 / n, "mrr": mrr / n, "nll": nll / n, "oov": oov, "oov_rate": oov / n}


def _confidence_metrics(
    rows: Sequence[tuple[str, Mapping[str, float]]],
    bins: int = 10,
) -> dict[str, Any]:
    """Summarize whether reported probabilities rank outcomes sensibly.

    This is a diagnostic only: the bin hit rates use the held-out target in
    the evaluation split, while the model itself is still fit on train only.
    A probability near 0.5 is not labelled random unless its empirical hit
    rate is also near the corresponding chance level.
    """
    if not rows:
        return {"n": 0}
    values: list[tuple[float, int]] = []
    bucket_values: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for target, probabilities in rows:
        confidence = max((float(value) for value in probabilities.values()), default=0.0)
        prediction = max(probabilities, key=probabilities.get) if probabilities else None
        hit = int(prediction == target)
        values.append((confidence, hit))
        bucket_values[min(bins - 1, int(confidence * bins))].append((confidence, hit))
    bucket_report: dict[str, dict[str, float]] = {}
    total = len(values)
    ece = 0.0
    for index, bucket in enumerate(bucket_values):
        if not bucket:
            continue
        mean_confidence = statistics.fmean(confidence for confidence, _ in bucket)
        hit_rate = statistics.fmean(hit for _, hit in bucket)
        ece += len(bucket) / total * abs(mean_confidence - hit_rate)
        bucket_report[str(index)] = {
            "n": len(bucket),
            "mean_top_probability": mean_confidence,
            "top1": hit_rate,
        }
    return {
        "n": total,
        "mean_top_probability": statistics.fmean(confidence for confidence, _ in values),
        "median_top_probability": statistics.median(confidence for confidence, _ in values),
        "ece": ece,
        "confidence_bins": bucket_report,
    }


def _regression_metrics(rows: Sequence[tuple[float, float]]) -> dict[str, float]:
    if not rows:
        return {"n": 0}
    errors = [prediction - target for target, prediction in rows]
    return {
        "n": len(rows),
        "mae": statistics.fmean(abs(error) for error in errors),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
    }


def _future_suffixes(rows: Sequence[Prefix]) -> dict[tuple[str, int], list[str]]:
    """Return the observed future suffix for each prefix row in a split."""
    grouped: dict[str, list[Prefix]] = defaultdict(list)
    for row in rows:
        grouped[row.run_id].append(row)
    result: dict[tuple[str, int], list[str]] = {}
    for run_rows in grouped.values():
        ordered = sorted(run_rows, key=lambda item: item.position)
        for index, row in enumerate(ordered):
            result[(row.run_id, row.position)] = [item.target for item in ordered[index:]]
    return result


def _evaluate_models(train: Sequence[Prefix], test: Sequence[Prefix]) -> dict[str, Any]:
    # The vocabulary is learned from train only.  A test-only target is OOV,
    # not an extra answer choice silently revealed to the predictor.
    canonical_labels = _labels(train)
    raw_labels = _labels(train, raw=True)
    models: dict[str, tuple[str, Any, bool]] = {
        "dyorc_markov1": ("local_reimplementation", TransitionModel(canonical_labels, 1), False),
        "dyorc_markov2": ("local_reimplementation", TransitionModel(canonical_labels, 2), False),
        "planner_markov1": ("planner_aware_markov_ablation", PlannerTransitionModel(canonical_labels, 1), False),
        "autotool_next_tool": ("adapted_next_tool_only", TransitionModel(raw_labels, 1), True),
        "rag_next_activity": ("adapted_retrieval", RetrievalModel(canonical_labels), False),
        "structured_prefix": ("structured_input_adapter", StructuredModel(canonical_labels), False),
        "structured_prefix_no_baseline": ("structured_input_ablation", StructuredModel(canonical_labels, include_baseline=False), False),
        "structured_prefix_backoff": ("structured_input_backoff_adapter", StructuredBackoffModel(canonical_labels), False),
        "pythia_path_length": ("adapted_path_length_only", PathLengthModel(canonical_labels), False),
    }
    for name, (_, model, raw) in models.items():
        if raw:
            model.fit(train, raw=True)
        else:
            model.fit(train)

    classification: dict[str, dict[str, float]] = {}
    confidence: dict[str, dict[str, Any]] = {}
    regression: dict[str, dict[str, float]] = {}
    path_stats = {"n": 0, "coverage_at_1": 0, "coverage_at_3": 0, "mean_interval_width": 0.0}
    test_suffixes = _future_suffixes(test)
    for name, (_, model, raw) in models.items():
        if name == "pythia_path_length":
            predictions: list[tuple[str, Mapping[str, float]]] = []
            remaining: list[tuple[float, float]] = []
            for row in test:
                probabilities, candidates, lower, upper = model.predict(row)
                predictions.append((row.target, probabilities))
                remaining.append((float(row.remaining_steps), (lower + upper) / 2.0))
                actual = test_suffixes.get((row.run_id, row.position), [row.target])
                path_stats["n"] += 1
                path_stats["coverage_at_1"] += int(bool(candidates) and candidates[0] == actual[: len(candidates[0])])
                path_stats["coverage_at_3"] += int(any(candidate == actual[: len(candidate)] for candidate in candidates[:3]))
                path_stats["mean_interval_width"] += max(0.0, upper - lower)
            classification[name] = _classification_metrics(predictions, canonical_labels)
            confidence[name] = _confidence_metrics(predictions)
            regression[name] = _regression_metrics(remaining)
            continue
        predictions = []
        for row in test:
            probabilities = model.predict(row, raw=raw) if isinstance(model, TransitionModel) else model.predict(row)
            target = row.raw_target if raw else row.target
            labels = raw_labels if raw else canonical_labels
            predictions.append((target, probabilities))
        classification[name] = _classification_metrics(predictions, raw_labels if raw else canonical_labels)
        confidence[name] = _confidence_metrics(predictions)
    if path_stats["n"]:
        path_stats["coverage_at_1"] /= path_stats["n"]
        path_stats["coverage_at_3"] /= path_stats["n"]
        path_stats["mean_interval_width"] /= path_stats["n"]
    return {
        "classification": classification,
        "confidence": confidence,
        "regression": regression,
        "path": path_stats,
    }


def _fixed_split_report(rows: Sequence[Prefix], split_map: Mapping[str, str]) -> dict[str, Any]:
    videos: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        videos[row.video_id].add(_normalise_split(split_map.get(row.video_id, row.split)))
    conflicts = {video: sorted(values) for video, values in videos.items() if len(values) != 1}
    assigned = {video: next(iter(values)) for video, values in videos.items() if len(values) == 1}
    counts = Counter(assigned.values())
    leakage: list[str] = []
    if conflicts:
        leakage.append("video_has_multiple_splits")
    if any(split == "unassigned" for split in assigned.values()):
        leakage.append("unassigned_video")
    expected = {"train": 48, "validation": 8, "test": 8}
    fixed_gate = all(counts.get(name, 0) == count for name, count in expected.items())
    reports: dict[str, Any] = {}
    train = [row for row in rows if assigned.get(row.video_id) == "train"]
    for split in ("validation", "test"):
        test = [row for row in rows if assigned.get(row.video_id) == split]
        if train and test:
            reports[split] = _evaluate_models(train, test)
    return {
        "strategy": "fixed_video_group_48_8_8",
        "video_counts": dict(sorted(counts.items())),
        "expected_video_counts": expected,
        "fixed_split_gate": fixed_gate,
        "leakage_flags": leakage,
        "conflicting_videos": conflicts,
        "metrics": reports,
    }


def _loo_report(rows: Sequence[Prefix]) -> dict[str, Any]:
    videos = sorted({row.video_id for row in rows})
    fold_reports: list[dict[str, Any]] = []
    for held_out in videos:
        train = [row for row in rows if row.video_id != held_out]
        test = [row for row in rows if row.video_id == held_out]
        fold_reports.append({"held_out_video": held_out, **_evaluate_models(train, test)})
    return {
        "strategy": "leave_one_video_out_diagnostic",
        "formal": False,
        "videos": len(videos),
        "folds": fold_reports,
    }


def build_report(prefix_path: Path, split_path: Path | None = None, diagnostic_loo: bool = False) -> dict[str, Any]:
    split_map = load_split(split_path) if split_path else None
    rows = load_prefixes(prefix_path, split_map)
    split_report = _fixed_split_report(rows, split_map or {}) if split_map else None
    if split_report is None and not diagnostic_loo:
        raise ValueError("formal evaluation requires --split-manifest; use --diagnostic-loo only for a non-formal diagnostic")
    report: dict[str, Any] = {
        "schema_version": "prediction-baselines-v0.2",
        "input": str(prefix_path),
        "rows": len(rows),
        "runs": len({row.run_id for row in rows}),
        "videos": len({row.video_id for row in rows}),
        "baselines": dict(sorted(Counter(row.baseline for row in rows).items())),
        "leakage_checks": {
            "future_events_included_in_input": False,
            "ground_truth_included_in_input": False,
            "target_in_prefix_fields": False,
            "source_hashes_present": all(bool(row.get("source_trace_sha256")) for row in _read_jsonl(prefix_path)),
        },
        "implementation_labels": {
            "dyorc_markov1": "local_reimplementation",
            "dyorc_markov2": "local_reimplementation",
            "planner_markov1": "planner_aware_markov_ablation",
            "autotool_next_tool": "adapted_next_tool_only",
            "pythia_path_length": "adapted_path_length_only",
            "rag_next_activity": "adapted_retrieval",
            "structured_prefix": "structured_input_adapter",
            "structured_prefix_no_baseline": "structured_input_ablation",
            "structured_prefix_backoff": "structured_input_backoff_adapter",
        },
        "fixed_split": split_report,
        "diagnostic_loo": _loo_report(rows) if diagnostic_loo else None,
        "limitations": [
            "These are auditable adapters; no claim is made of reproducing original private training or full systems.",
            "Raw tool labels are retained only for the AutoTool adapter; canonical semantic labels are used elsewhere.",
            "Resource prediction is a separate compute-event task and is not mixed into Future Predictor inputs.",
        ],
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--diagnostic-loo", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_report(args.prefix, args.split_manifest, args.diagnostic_loo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": report["rows"],
        "videos": report["videos"],
        "formal_split": bool(report["fixed_split"]),
        "fixed_split_gate": report["fixed_split"]["fixed_split_gate"] if report["fixed_split"] else None,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
