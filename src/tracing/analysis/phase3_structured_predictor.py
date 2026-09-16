#!/usr/bin/env python3
"""Evaluate H2 with auditable structured task/state features.

The older Phase 3 evaluator intentionally used a question token bag and a
``min(prefix_length, 6)`` feature.  This evaluator consumes the immutable
``state_t`` sidecars written by the collector instead.  It keeps the target
strictly to the next observed canonical tool family (or ``__END__``), uses
leave-one-video-out folds, and never uses answer labels, raw question text,
video IDs, or future sidecars as features.

The models are small categorical Naive Bayes baselines.  They are not meant
to be a production scheduler; they answer the narrower H2 question: does a
structured task/evidence state contain predictive signal beyond a baseline
and action-history prior, with a split that prevents repetitions of one video
from leaking into train and test?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracing.collectors.structured_state import canonical_action, validate_state_snapshot  # noqa: E402


END_LABEL = "__END__"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip()
    return text or default


def _bucket(value: Any, edges: Sequence[float], labels: Sequence[str] | None = None) -> str:
    number = _number(value)
    names = list(labels or [str(index) for index in range(len(edges) + 1)])
    for index, edge in enumerate(edges):
        if number < edge:
            return names[min(index, len(names) - 1)]
    return names[-1]


def _count_bucket(value: Any) -> str:
    number = max(0, _int(value))
    # Exact low counts are retained; large counts remain distinct up to 16.
    # This avoids the old arbitrary ``6+`` collapse while keeping sparse
    # evidence features bounded.
    return str(min(number, 16))


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    video_id: str
    baseline: str
    model_name: str
    task_id: str
    states: tuple[dict[str, Any], ...]
    actions: tuple[str, ...]
    runtimes_ms: tuple[float, ...]
    peak_vram_mb: tuple[float, ...]


@dataclass(frozen=True)
class Example:
    run_id: str
    video_id: str
    baseline: str
    state: Mapping[str, Any]
    target: str
    remaining_steps: int
    remaining_runtime_ms: float
    remaining_peak_vram_mb: float
    position: int


def _run_success(run_dir: Path) -> bool:
    for name in ("run_status.json", "run_manifest.json"):
        path = run_dir / name
        if path.is_file():
            payload = _read_json(path)
            return str(payload.get("status", "")) == "success"
    return False


def _state_files(run_dir: Path) -> dict[int, dict[str, Any]]:
    state_dir = run_dir / "states"
    result: dict[int, dict[str, Any]] = {}
    for path in sorted(state_dir.glob("state_*.json")):
        try:
            state = _read_json(path)
            step = _int(state.get("step_id"), -1)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if step < 0:
            continue
        errors = validate_state_snapshot(state)
        if errors:
            raise ValueError(f"invalid state sidecar {path}: {'; '.join(errors)}")
        result[step] = state
    return result


def load_runs(root: Path) -> tuple[list[RunRecord], dict[str, Any]]:
    """Load complete successful runs and reject unsafe/incomplete sidecars."""

    runs: list[RunRecord] = []
    skipped = Counter()
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or not _run_success(run_dir):
            continue
        trace_path = run_dir / "trace.jsonl"
        if not trace_path.is_file():
            skipped["missing_trace"] += 1
            continue
        events = _read_jsonl(trace_path)
        action_events = [event for event in events if event.get("event_type") == "action"]
        states_by_step = _state_files(run_dir)
        expected = len(action_events) + 1
        if any(step not in states_by_step for step in range(expected)):
            skipped["incomplete_states"] += 1
            continue
        state0 = states_by_step[0]
        actions = tuple(str(event.get("action", "")) for event in action_events)
        runtimes = tuple(_number((event.get("resource") or {}).get("runtime_ms")) for event in action_events)
        peaks = tuple(
            max(
                _number((event.get("resource") or {}).get("peak_reserved_mb")),
                _number((event.get("resource") or {}).get("peak_allocated_mb")),
            )
            for event in action_events
        )
        manifest: dict[str, Any] = {}
        for name in ("run_manifest.json", "run_status.json"):
            path = run_dir / name
            if path.is_file():
                manifest = _read_json(path)
                if name == "run_manifest.json":
                    break
        runs.append(
            RunRecord(
                run_id=_text(manifest.get("run_id"), run_dir.name),
                video_id=_text(state0.get("video_id"), "unknown_video"),
                baseline=_text(state0.get("baseline"), "unknown"),
                model_name=_text(manifest.get("model_name"), "unknown"),
                task_id=_text(state0.get("task_id"), "unknown_task"),
                states=tuple(states_by_step[index] for index in range(expected)),
                actions=actions,
                runtimes_ms=runtimes,
                peak_vram_mb=peaks,
            )
        )
    if not runs:
        raise ValueError(f"no complete successful structured runs found under {root}")
    quality = {
        "complete_runs": len(runs),
        "skipped_runs": dict(sorted(skipped.items())),
        "unsafe_state_count": 0,
    }
    return runs, quality


def build_examples(runs: Sequence[RunRecord]) -> list[Example]:
    examples: list[Example] = []
    for run in runs:
        for position, state in enumerate(run.states):
            target = canonical_action(run.actions[position]) if position < len(run.actions) else END_LABEL
            examples.append(
                Example(
                    run_id=run.run_id,
                    video_id=run.video_id,
                    baseline=run.baseline,
                    state=state,
                    target=target,
                    remaining_steps=len(run.actions) - position,
                    remaining_runtime_ms=sum(run.runtimes_ms[position:]),
                    remaining_peak_vram_mb=max(run.peak_vram_mb[position:], default=0.0),
                    position=position,
                )
            )
    return examples


def _add(features: set[str], name: str, value: Any) -> None:
    features.add(f"{name}={_text(value)}")


def _task_features(state: Mapping[str, Any], features: set[str]) -> None:
    task = state.get("task") or {}
    for key in (
        "question_type",
        "answer_type",
        "temporal_scope",
        "domain",
        "sub_category",
        "official_task_type",
    ):
        if key in task:
            _add(features, f"task.{key}", task.get(key))
    modalities = task.get("required_modalities") or []
    if isinstance(modalities, list):
        _add(features, "task.modalities", "+".join(sorted(_text(item) for item in modalities)))
    _add(features, "task.option_count", _count_bucket(task.get("option_count")))
    _add(features, "task.question_chars", _bucket(task.get("question_chars"), [40, 80, 120, 180, 260, 400]))
    _add(features, "task.question_tokens", _bucket(task.get("question_tokens"), [8, 12, 20, 32, 48, 80]))
    _add(features, "task.option_chars_mean", _bucket(task.get("option_chars_mean"), [8, 16, 32, 64, 128]))


def _video_features(state: Mapping[str, Any], features: set[str]) -> None:
    video = state.get("video") or {}
    _add(features, "video.duration", _bucket(video.get("duration_s"), [30, 60, 120, 300, 600]))
    _add(features, "video.fps", _bucket(video.get("fps"), [15, 24, 30, 60]))
    _add(features, "video.width", _bucket(video.get("width"), [480, 720, 1080, 1920]))
    _add(features, "video.height", _bucket(video.get("height"), [360, 480, 720, 1080]))
    _add(features, "video.subtitles", bool(video.get("subtitle_available")))
    _add(features, "video.shots", _bucket(video.get("shot_count"), [4, 12, 32, 96]))


def _prefix_features(state: Mapping[str, Any], features: set[str]) -> None:
    prefix = state.get("prefix") or {}
    # Every observed step remains its own category.  No arbitrary 6+ cap is
    # used anywhere in this representation.
    _add(features, "prefix.steps", _int(prefix.get("observed_step_count")))
    _add(features, "prefix.max_steps", _int(prefix.get("max_steps")))
    _add(features, "prefix.progress", round(max(0.0, min(1.0, _number(prefix.get("progress_ratio")))) * 8) / 8)
    last = prefix.get("last_actions") or []
    if isinstance(last, list):
        _add(features, "prefix.last1", last[-1] if last else "NONE")
        _add(features, "prefix.last2", last[-2] if len(last) >= 2 else "NONE")
    histogram = prefix.get("action_histogram") or {}
    if isinstance(histogram, Mapping):
        for action, count in sorted(histogram.items()):
            _add(features, f"prefix.count.{action}", _count_bucket(count))
    _add(features, "prefix.runtime", _bucket(prefix.get("local_runtime_ms"), [100, 1000, 5000, 15000, 30000, 60000]))
    _add(features, "prefix.api_wait", _bucket(prefix.get("api_wait_ms"), [1, 100, 1000, 10000]))
    _add(features, "prefix.retries", _count_bucket(prefix.get("retry_count")))
    _add(features, "prefix.errors", _count_bucket(prefix.get("error_count")))


def _evidence_features(state: Mapping[str, Any], features: set[str]) -> None:
    evidence = state.get("evidence") or {}
    _add(features, "evidence.coverage", round(max(0.0, min(1.0, _number(evidence.get("coverage_ratio")))) * 8) / 8)
    _add(features, "evidence.frames", _bucket(evidence.get("frames_seen"), [1, 2, 4, 8, 16, 32]))
    _add(features, "evidence.ocr_chars", _bucket(evidence.get("ocr_chars"), [1, 16, 64, 256, 1024]))
    _add(features, "evidence.temporal_relations", _count_bucket(evidence.get("temporal_relation_count")))
    confidence = evidence.get("confidence_mean")
    _add(features, "evidence.confidence", "none" if confidence is None else round(max(0.0, min(1.0, _number(confidence))) * 10) / 10)
    modalities = evidence.get("modality_counts") or {}
    if isinstance(modalities, Mapping):
        for modality, count in sorted(modalities.items()):
            _add(features, f"evidence.modality.{modality}", _count_bucket(count))
    objects = evidence.get("object_counts") or {}
    if isinstance(objects, Mapping):
        for name, count in sorted(objects.items()):
            _add(features, f"evidence.object.{name}", _count_bucket(count))
    coverage_bins = evidence.get("coverage_bins") or []
    if isinstance(coverage_bins, list) and coverage_bins:
        n = len(coverage_bins)
        quarters = [coverage_bins[: n // 4], coverage_bins[n // 4 : n // 2], coverage_bins[n // 2 : (3 * n) // 4], coverage_bins[(3 * n) // 4 :]]
        for index, quarter in enumerate(quarters):
            _add(features, f"evidence.coverage_q{index + 1}", any(_number(value) > 0 for value in quarter))
        _add(features, "evidence.covered_bins", sum(_number(value) > 0 for value in coverage_bins))


def state_features(example: Example, mode: str) -> set[str]:
    """Return categorical features for one state without answer/video ID."""

    features: set[str] = set()
    _add(features, "baseline", example.baseline)
    flags = {
        "task": mode in {"task", "task_prefix", "full"},
        "video": mode in {"task", "task_prefix", "full"},
        "prefix": mode in {"prefix", "prefix_evidence", "task_prefix", "full"},
        "evidence": mode in {"prefix_evidence", "full"},
    }
    if flags["task"]:
        _task_features(example.state, features)
    if flags["video"]:
        _video_features(example.state, features)
    if flags["prefix"]:
        _prefix_features(example.state, features)
    if flags["evidence"]:
        _evidence_features(example.state, features)
    return features


def _labels(examples: Sequence[Example]) -> list[str]:
    return sorted({example.target for example in examples})


def _normalize(values: Mapping[str, float], labels: Sequence[str]) -> dict[str, float]:
    total = sum(max(0.0, float(values.get(label, 0.0))) for label in labels)
    if total <= 0.0:
        return {label: 1.0 / len(labels) for label in labels}
    return {label: max(0.0, float(values.get(label, 0.0))) / total for label in labels}


class CategoricalNB:
    def __init__(self, labels: Sequence[str], mode: str, alpha: float = 0.5) -> None:
        self.labels = list(labels)
        self.mode = mode
        self.alpha = alpha
        self.class_counts: Counter[str] = Counter()
        self.feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.feature_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, examples: Sequence[Example]) -> None:
        for example in examples:
            self.class_counts[example.target] += 1
            features = state_features(example, self.mode)
            self.vocabulary.update(features)
            for feature in features:
                self.feature_counts[example.target][feature] += 1
                self.feature_totals[example.target] += 1

    def predict_proba(self, example: Example) -> dict[str, float]:
        if not self.labels:
            return {}
        total_examples = sum(self.class_counts.values())
        vocab_size = max(1, len(self.vocabulary))
        features = state_features(example, self.mode)
        log_scores: dict[str, float] = {}
        for label in self.labels:
            class_count = self.class_counts.get(label, 0)
            score = math.log((class_count + self.alpha) / (total_examples + self.alpha * len(self.labels)))
            denominator = self.feature_totals.get(label, 0) + self.alpha * vocab_size
            for feature in features:
                if feature not in self.vocabulary:
                    continue
                numerator = self.feature_counts[label].get(feature, 0) + self.alpha
                score += math.log(numerator / denominator)
            log_scores[label] = score
        maximum = max(log_scores.values())
        return _normalize({label: math.exp(score - maximum) for label, score in log_scores.items()}, self.labels)


class StaticModel:
    def __init__(self, labels: Sequence[str], by_baseline: bool = False, alpha: float = 1.0) -> None:
        self.labels = list(labels)
        self.by_baseline = by_baseline
        self.alpha = alpha
        self.global_counts: Counter[str] = Counter()
        self.baseline_counts: dict[str, Counter[str]] = defaultdict(Counter)

    def fit(self, examples: Sequence[Example]) -> None:
        for example in examples:
            self.global_counts[example.target] += 1
            self.baseline_counts[example.baseline][example.target] += 1

    def predict_proba(self, example: Example) -> dict[str, float]:
        counts = self.baseline_counts.get(example.baseline) if self.by_baseline else self.global_counts
        counts = counts or self.global_counts
        return _normalize({label: counts.get(label, 0) + self.alpha for label in self.labels}, self.labels)


class MarkovModel:
    def __init__(self, labels: Sequence[str], order: int, alpha: float = 1.0) -> None:
        self.labels = list(labels)
        self.order = order
        self.alpha = alpha
        self.global_counts: Counter[str] = Counter()
        self.baseline_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    def fit(self, examples: Sequence[Example]) -> None:
        for example in examples:
            self.global_counts[example.target] += 1
            self.baseline_counts[example.baseline][example.target] += 1
            history = tuple((example.state.get("prefix") or {}).get("canonical_actions") or [])
            if history:
                self.transitions[(example.baseline,) + tuple(history[-self.order :])][example.target] += 1

    def predict_proba(self, example: Example) -> dict[str, float]:
        history = tuple((example.state.get("prefix") or {}).get("canonical_actions") or [])
        counts: Counter[str] | None = None
        for size in range(min(self.order, len(history)), 0, -1):
            counts = self.transitions.get((example.baseline,) + tuple(history[-size:]))
            if counts:
                break
        counts = counts or self.baseline_counts.get(example.baseline) or self.global_counts
        return _normalize({label: counts.get(label, 0) + self.alpha for label in self.labels}, self.labels)


class MeanModel:
    def __init__(self, target: str, by_last_action: bool = False) -> None:
        self.target = target
        self.by_last_action = by_last_action
        self.global_mean = 0.0
        self.baseline_mean: dict[str, float] = {}
        self.context_mean: dict[tuple[str, str], float] = {}

    def fit(self, examples: Sequence[Example]) -> None:
        values = [float(getattr(example, self.target)) for example in examples]
        self.global_mean = statistics.fmean(values) if values else 0.0
        by_baseline: dict[str, list[float]] = defaultdict(list)
        by_context: dict[tuple[str, str], list[float]] = defaultdict(list)
        for example in examples:
            value = float(getattr(example, self.target))
            by_baseline[example.baseline].append(value)
            if self.by_last_action:
                history = (example.state.get("prefix") or {}).get("last_actions") or []
                last = str(history[-1]) if history else "NONE"
                by_context[(example.baseline, last)].append(value)
        self.baseline_mean = {key: statistics.fmean(values) for key, values in by_baseline.items()}
        self.context_mean = {key: statistics.fmean(values) for key, values in by_context.items()}

    def predict(self, example: Example) -> float:
        if self.by_last_action:
            history = (example.state.get("prefix") or {}).get("last_actions") or []
            last = str(history[-1]) if history else "NONE"
            value = self.context_mean.get((example.baseline, last))
            if value is not None:
                return value
        return self.baseline_mean.get(example.baseline, self.global_mean)


def _classification_metrics(rows: Sequence[tuple[Example, Mapping[str, float]]], labels: Sequence[str]) -> dict[str, float]:
    if not rows:
        return {"n": 0}
    top1 = top3 = 0
    reciprocal = nll = brier = 0.0
    calibration: list[list[tuple[float, float]]] = [[] for _ in range(10)]
    for example, probabilities in rows:
        ordered = sorted(labels, key=lambda label: probabilities.get(label, 0.0), reverse=True)
        rank = ordered.index(example.target) + 1 if example.target in ordered else len(labels) + 1
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
        reciprocal += 1.0 / rank
        confidence = probabilities.get(ordered[0], 0.0) if ordered else 0.0
        calibration[min(9, max(0, int(confidence * 10)))].append((confidence, float(rank == 1)))
        probability = max(probabilities.get(example.target, 0.0), 1e-12)
        nll -= math.log(probability)
        brier += sum((probabilities.get(label, 0.0) - float(label == example.target)) ** 2 for label in labels)
    ece = 0.0
    for bucket in calibration:
        if bucket:
            mean_confidence = statistics.fmean(item[0] for item in bucket)
            mean_accuracy = statistics.fmean(item[1] for item in bucket)
            ece += len(bucket) / len(rows) * abs(mean_confidence - mean_accuracy)
    return {
        "n": len(rows),
        "top1": top1 / len(rows),
        "top3": top3 / len(rows),
        "mrr": reciprocal / len(rows),
        "nll": nll / len(rows),
        "brier": brier / len(rows),
        "ece": ece,
    }


def _macro_by_video(rows: Sequence[tuple[Example, Mapping[str, float]]], labels: Sequence[str]) -> dict[str, float]:
    groups: dict[str, list[tuple[Example, Mapping[str, float]]]] = defaultdict(list)
    for row in rows:
        groups[row[0].video_id].append(row)
    values = [_classification_metrics(group, labels) for group in groups.values()]
    return {
        "videos": len(values),
        "top1": statistics.fmean(value["top1"] for value in values) if values else 0.0,
        "top3": statistics.fmean(value["top3"] for value in values) if values else 0.0,
        "nll": statistics.fmean(value["nll"] for value in values) if values else 0.0,
    }


def _by_baseline(rows: Sequence[tuple[Example, Mapping[str, float]]], labels: Sequence[str]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[tuple[Example, Mapping[str, float]]]] = defaultdict(list)
    for row in rows:
        groups[row[0].baseline].append(row)
    return {baseline: _classification_metrics(group, labels) for baseline, group in sorted(groups.items())}


def _regression_metrics(rows: Sequence[tuple[Example, float]], target: str) -> dict[str, float]:
    if not rows:
        return {"n": 0}
    errors = [prediction - float(getattr(example, target)) for example, prediction in rows]
    return {
        "n": len(rows),
        "mae": statistics.fmean(abs(error) for error in errors),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
    }


def evaluate(runs: Sequence[RunRecord], examples: Sequence[Example], quality: Mapping[str, Any]) -> dict[str, Any]:
    labels = _labels(examples)
    videos = sorted({example.video_id for example in examples})
    classifiers: dict[str, list[tuple[Example, Mapping[str, float]]]] = defaultdict(list)
    regressions: dict[str, list[tuple[Example, float]]] = defaultdict(list)
    folds: list[dict[str, Any]] = []

    for held_out in videos:
        train = [example for example in examples if example.video_id != held_out]
        test = [example for example in examples if example.video_id == held_out]
        models: dict[str, Any] = {
            "static_global": StaticModel(labels),
            "static_baseline": StaticModel(labels, by_baseline=True),
            "markov1": MarkovModel(labels, order=1),
            "markov2": MarkovModel(labels, order=2),
            "task_only_structured_nb": CategoricalNB(labels, mode="task"),
            "prefix_structured_nb": CategoricalNB(labels, mode="prefix"),
            "prefix_evidence_structured_nb": CategoricalNB(labels, mode="prefix_evidence"),
            "task_prefix_structured_nb": CategoricalNB(labels, mode="task_prefix"),
            "structured_state_nb": CategoricalNB(labels, mode="full"),
        }
        for model in models.values():
            model.fit(train)
        fold: dict[str, Any] = {"held_out_video": held_out, "n_examples": len(test)}
        for name, model in models.items():
            predictions = [(example, model.predict_proba(example)) for example in test]
            classifiers[name].extend(predictions)
            fold[name] = _classification_metrics(predictions, labels)

        regression_models = {
            "remaining_steps_static": MeanModel("remaining_steps"),
            "remaining_steps_markov1": MeanModel("remaining_steps", by_last_action=True),
            "remaining_runtime_static": MeanModel("remaining_runtime_ms"),
            "remaining_runtime_markov1": MeanModel("remaining_runtime_ms", by_last_action=True),
            "remaining_peak_vram_static": MeanModel("remaining_peak_vram_mb"),
            "remaining_peak_vram_markov1": MeanModel("remaining_peak_vram_mb", by_last_action=True),
        }
        for name, model in regression_models.items():
            model.fit(train)
            target = name.removeprefix("remaining_steps_").removeprefix("remaining_runtime_").removeprefix("remaining_peak_vram_")
            target = {
                "static": "remaining_steps" if "steps" in name else "remaining_runtime_ms" if "runtime" in name else "remaining_peak_vram_mb",
                "markov1": "remaining_steps" if "steps" in name else "remaining_runtime_ms" if "runtime" in name else "remaining_peak_vram_mb",
            }[target]
            predictions = [(example, model.predict(example)) for example in test]
            regressions[name].extend(predictions)
            fold[name] = _regression_metrics(predictions, target)
        folds.append(fold)

    classification: dict[str, Any] = {}
    for name, rows in classifiers.items():
        classification[name] = {
            "overall": _classification_metrics(rows, labels),
            "video_macro": _macro_by_video(rows, labels),
            "by_baseline": _by_baseline(rows, labels),
        }
    regression: dict[str, Any] = {}
    for name, rows in regressions.items():
        target = "remaining_steps" if "steps" in name else "remaining_runtime_ms" if "runtime" in name else "remaining_peak_vram_mb"
        regression[name] = _regression_metrics(rows, target)

    markov = classification["markov2"]
    prefix = classification["prefix_structured_nb"]
    task_prefix = classification["task_prefix_structured_nb"]
    full = classification["structured_state_nb"]
    prefix_signal = (
        prefix["overall"]["top1"] > markov["overall"]["top1"]
        and prefix["overall"]["nll"] < markov["overall"]["nll"]
        and prefix["video_macro"]["top1"] > markov["video_macro"]["top1"]
    )
    task_increment = (
        task_prefix["overall"]["top1"] > prefix["overall"]["top1"]
        and task_prefix["overall"]["nll"] < prefix["overall"]["nll"]
        and task_prefix["video_macro"]["top1"] > prefix["video_macro"]["top1"]
    )
    evidence_increment = (
        full["overall"]["top1"] > task_prefix["overall"]["top1"]
        and full["overall"]["nll"] < task_prefix["overall"]["nll"]
        and full["video_macro"]["top1"] > task_prefix["video_macro"]["top1"]
    )
    h2_observation = "preliminary_supported" if prefix_signal and task_increment else "mixed_evidence" if prefix_signal else "insufficient_evidence"
    return {
        "schema_version": "phase3-structured-0.1",
        "task": "H2 structured next-action and remaining-cost prediction",
        "data": {
            "runs": len(runs),
            "videos": len(videos),
            "baselines": dict(sorted(Counter(run.baseline for run in runs).items())),
            "models": dict(sorted(Counter(run.model_name for run in runs).items())),
            "examples": len(examples),
            "action_examples": sum(example.target != END_LABEL for example in examples),
            "terminal_examples": sum(example.target == END_LABEL for example in examples),
            "canonical_labels": labels,
        },
        "quality": dict(quality),
        "split": {
            "strategy": "leave_one_video_out",
            "groups": videos,
            "repetitions_stay_together": True,
            "video_id_is_group_only": True,
        },
        "feature_contract": {
            "task": "fixed taxonomy, modality requirements, option/length buckets; no raw question text",
            "video": "duration/fps/resolution/subtitle/shot metadata when available; no video_id",
            "prefix": "exact observed step count, progress, canonical history, action counts, runtime/retry/error state",
            "evidence": "observed coverage, frames, modality/object counts, OCR/temporal/confidence summaries",
            "forbidden": ["answer label/text", "future sidecar", "video_id as a feature", "raw question token bag"],
        },
        "labels": labels,
        "classification": classification,
        "regression": regression,
        "folds": folds,
        "h2_observation": h2_observation,
        "h2_evidence": {
            "prefix_signal_vs_markov2": {
                "supported": prefix_signal,
                "top1_delta": prefix["overall"]["top1"] - markov["overall"]["top1"],
                "nll_delta": prefix["overall"]["nll"] - markov["overall"]["nll"],
                "video_macro_top1_delta": prefix["video_macro"]["top1"] - markov["video_macro"]["top1"],
            },
            "task_increment_vs_prefix": {
                "supported": task_increment,
                "top1_delta": task_prefix["overall"]["top1"] - prefix["overall"]["top1"],
                "nll_delta": task_prefix["overall"]["nll"] - prefix["overall"]["nll"],
                "video_macro_top1_delta": task_prefix["video_macro"]["top1"] - prefix["video_macro"]["top1"],
            },
            "evidence_increment_vs_task_prefix": {
                "supported": evidence_increment,
                "top1_delta": full["overall"]["top1"] - task_prefix["overall"]["top1"],
                "nll_delta": full["overall"]["nll"] - task_prefix["overall"]["nll"],
                "video_macro_top1_delta": full["video_macro"]["top1"] - task_prefix["video_macro"]["top1"],
            },
        },
        "h2_rule": "prefix_structured_nb must beat markov2; task_prefix_structured_nb must improve top1, NLL, and video-macro top1 over prefix_structured_nb for a preliminary H2 pass",
        "limitations": [
            "Repeated runs are grouped by video; the 32-video cohort still does not establish broad domain generalization.",
            "The collector records structured evidence summaries, not a learned visual embedding or ground-truth evidence quality.",
            "A predictive signal is not a scheduling gain; Phase 4 must compare policies using held-out resource traces.",
        ],
    }


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for name, payload in report["classification"].items():
        row = {"task": "next_action", "model": name, "scope": "overall"}
        row.update(payload["overall"])
        rows.append(row)
        row = {"task": "next_action", "model": name, "scope": "video_macro"}
        row.update(payload["video_macro"])
        rows.append(row)
    for name, payload in report["regression"].items():
        row = {"task": "remaining_cost", "model": name, "scope": "overall"}
        row.update(payload)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    runs, quality = load_runs(args.root.expanduser().resolve())
    examples = build_examples(runs)
    report = evaluate(runs, examples, quality)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.csv, report)
    print(json.dumps({
        "runs": report["data"]["runs"],
        "videos": report["data"]["videos"],
        "examples": report["data"]["examples"],
        "h2_observation": report["h2_observation"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
