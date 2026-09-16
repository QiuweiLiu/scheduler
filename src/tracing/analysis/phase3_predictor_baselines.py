#!/usr/bin/env python3
"""Dependency-free Phase 3 predictor baselines.

The script evaluates H2 on saved trace files without calling an API.  A run is
expanded into prefix examples and evaluated with leave-one-video-out splits so
repeated runs of the same video cannot leak into both train and test.  The
primary target is the next action (including an explicit terminal label); two
secondary targets are remaining action count and remaining action runtime.

This is intentionally a small, auditable baseline suite.  It does not train a
large model and it does not use GPU state, future events, or the current model
decision as semantic input to the predictor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


END_LABEL = "__END__"
_RUN_RE = re.compile(r"_r\d+$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "and",
    "are",
    "based",
    "does",
    "from",
    "how",
    "into",
    "that",
    "the",
    "their",
    "this",
    "what",
    "when",
    "which",
    "with",
}


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    video_id: str
    baseline: str
    model_name: str
    question: str
    actions: tuple[str, ...]
    runtimes_ms: tuple[float, ...]


@dataclass(frozen=True)
class PrefixExample:
    run_id: str
    video_id: str
    baseline: str
    question: str
    prefix: tuple[str, ...]
    target: str
    remaining_steps: int
    remaining_runtime_ms: float
    position: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _question_map(manifest_path: Path | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if manifest_path is None:
        return mapping
    for record in _read_jsonl(manifest_path):
        task_id = str(record.get("task_id", ""))
        question = str(record.get("question", ""))
        video_path = str(record.get("video_path", ""))
        if task_id and question:
            mapping[task_id] = question
        if video_path and question:
            mapping[Path(video_path).stem] = question
    return mapping


def _base_task_id(task_id: str) -> str:
    return _RUN_RE.sub("", task_id)


def load_runs(root: Path, manifest_path: Path | None = None) -> list[RunRecord]:
    """Load successful run directories and retain only action-level events."""

    questions = _question_map(manifest_path)
    runs: list[RunRecord] = []
    for run_dir in sorted(root.iterdir()):
        trace_path = run_dir / "trace.jsonl"
        manifest_file = run_dir / "run_manifest.json"
        if not trace_path.is_file() or not manifest_file.is_file():
            continue
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("status") != "success":
            continue
        events = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        action_events = [event for event in events if event.get("event_type") == "action"]
        actions = tuple(str(event.get("action")) for event in action_events)
        runtimes = tuple(
            float((event.get("resource") or {}).get("runtime_ms") or 0.0)
            for event in action_events
        )
        task_id = str(manifest.get("task_id", ""))
        video_id = Path(str(manifest.get("video_path", ""))).stem or task_id
        question = questions.get(task_id) or questions.get(_base_task_id(task_id)) or questions.get(video_id, "")
        runs.append(
            RunRecord(
                run_id=str(manifest.get("run_id", run_dir.name)),
                video_id=video_id,
                baseline=str(manifest.get("baseline", "unknown")),
                model_name=str(manifest.get("model_name", "unknown")),
                question=question,
                actions=actions,
                runtimes_ms=runtimes,
            )
        )
    if not runs:
        raise ValueError(f"no successful trace runs found under {root}")
    return runs


def build_examples(runs: Sequence[RunRecord]) -> list[PrefixExample]:
    examples: list[PrefixExample] = []
    for run in runs:
        for position in range(len(run.actions) + 1):
            target = run.actions[position] if position < len(run.actions) else END_LABEL
            examples.append(
                PrefixExample(
                    run_id=run.run_id,
                    video_id=run.video_id,
                    baseline=run.baseline,
                    question=run.question,
                    prefix=run.actions[:position],
                    target=target,
                    remaining_steps=len(run.actions) - position,
                    remaining_runtime_ms=sum(run.runtimes_ms[position:]),
                    position=position,
                )
            )
    return examples


def _labels(examples: Sequence[PrefixExample]) -> list[str]:
    return sorted({example.target for example in examples})


def _normalize(values: Mapping[str, float], labels: Sequence[str]) -> dict[str, float]:
    total = sum(max(0.0, float(values.get(label, 0.0))) for label in labels)
    if total <= 0.0:
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}
    return {label: max(0.0, float(values.get(label, 0.0))) / total for label in labels}


class StaticModel:
    def __init__(self, labels: Sequence[str], by_baseline: bool = False, alpha: float = 1.0) -> None:
        self.labels = list(labels)
        self.by_baseline = by_baseline
        self.alpha = alpha
        self.global_counts: Counter[str] = Counter()
        self.baseline_counts: dict[str, Counter[str]] = defaultdict(Counter)

    def fit(self, examples: Sequence[PrefixExample]) -> None:
        for example in examples:
            self.global_counts[example.target] += 1
            self.baseline_counts[example.baseline][example.target] += 1

    def predict_proba(self, example: PrefixExample) -> dict[str, float]:
        counts = self.baseline_counts.get(example.baseline) if self.by_baseline else self.global_counts
        counts = counts or self.global_counts
        return _normalize({label: counts.get(label, 0) + self.alpha for label in self.labels}, self.labels)


class MarkovModel:
    def __init__(self, labels: Sequence[str], order: int, alpha: float = 1.0) -> None:
        if order not in {1, 2}:
            raise ValueError("Markov order must be 1 or 2")
        self.labels = list(labels)
        self.order = order
        self.alpha = alpha
        self.global_counts: Counter[str] = Counter()
        self.baseline_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.order1_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.order2_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    def fit(self, examples: Sequence[PrefixExample]) -> None:
        for example in examples:
            self.global_counts[example.target] += 1
            self.baseline_counts[example.baseline][example.target] += 1
            if example.prefix:
                self.order1_counts[(example.baseline, example.prefix[-1])][example.target] += 1
            if len(example.prefix) >= 2:
                self.order2_counts[(example.baseline, example.prefix[-2], example.prefix[-1])][example.target] += 1

    def _fallback(self, example: PrefixExample) -> Counter[str]:
        if self.order == 2 and len(example.prefix) >= 2:
            counts = self.order2_counts.get((example.baseline, example.prefix[-2], example.prefix[-1]))
            if counts:
                return counts
        if example.prefix:
            counts = self.order1_counts.get((example.baseline, example.prefix[-1]))
            if counts:
                return counts
        return self.baseline_counts.get(example.baseline) or self.global_counts

    def predict_proba(self, example: PrefixExample) -> dict[str, float]:
        counts = self._fallback(example)
        return _normalize({label: counts.get(label, 0) + self.alpha for label in self.labels}, self.labels)


def _question_tokens(question: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(question.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    ]


def _nb_features(example: PrefixExample, include_question: bool = True) -> set[str]:
    features = {f"baseline={example.baseline}", f"length_bin={min(example.position, 6)}"}
    if example.prefix:
        features.add(f"last={example.prefix[-1]}")
    if len(example.prefix) >= 2:
        features.add(f"prev={example.prefix[-2]}")
    if include_question:
        features.update(f"question={token}" for token in _question_tokens(example.question))
    return features


class PrefixTaskNB:
    """Small multinomial Naive Bayes model over prefix and optional task features."""

    def __init__(self, labels: Sequence[str], alpha: float = 0.5, include_question: bool = True) -> None:
        self.labels = list(labels)
        self.alpha = alpha
        self.include_question = include_question
        self.class_counts: Counter[str] = Counter()
        self.feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.feature_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, examples: Sequence[PrefixExample]) -> None:
        for example in examples:
            self.class_counts[example.target] += 1
            features = _nb_features(example, include_question=self.include_question)
            self.vocabulary.update(features)
            for feature in features:
                self.feature_counts[example.target][feature] += 1
                self.feature_totals[example.target] += 1

    def predict_proba(self, example: PrefixExample) -> dict[str, float]:
        if not self.vocabulary:
            return _normalize(self.class_counts, self.labels)
        total_examples = sum(self.class_counts.values())
        vocab_size = len(self.vocabulary)
        log_scores: dict[str, float] = {}
        for label in self.labels:
            class_count = self.class_counts.get(label, 0)
            log_score = math.log((class_count + self.alpha) / (total_examples + self.alpha * len(self.labels)))
            denominator = self.feature_totals.get(label, 0) + self.alpha * vocab_size
            for feature in _nb_features(example, include_question=self.include_question):
                if feature in self.vocabulary:
                    numerator = self.feature_counts[label].get(feature, 0) + self.alpha
                    log_score += math.log(numerator / denominator)
            log_scores[label] = log_score
        max_score = max(log_scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in log_scores.items()}
        return _normalize(exp_scores, self.labels)


class RegressionModel:
    def __init__(self, target: str, by_last_action: bool = False) -> None:
        self.target = target
        self.by_last_action = by_last_action
        self.global_mean = 0.0
        self.baseline_mean: dict[str, float] = {}
        self.context_mean: dict[tuple[str, str], float] = {}

    def fit(self, examples: Sequence[PrefixExample]) -> None:
        values = [float(getattr(example, self.target)) for example in examples]
        self.global_mean = statistics.fmean(values) if values else 0.0
        grouped: dict[str, list[float]] = defaultdict(list)
        contexts: dict[tuple[str, str], list[float]] = defaultdict(list)
        for example in examples:
            value = float(getattr(example, self.target))
            grouped[example.baseline].append(value)
            if example.prefix:
                contexts[(example.baseline, example.prefix[-1])].append(value)
        self.baseline_mean = {key: statistics.fmean(values) for key, values in grouped.items()}
        self.context_mean = {key: statistics.fmean(values) for key, values in contexts.items()}

    def predict(self, example: PrefixExample) -> float:
        if self.by_last_action and example.prefix:
            value = self.context_mean.get((example.baseline, example.prefix[-1]))
            if value is not None:
                return value
        return self.baseline_mean.get(example.baseline, self.global_mean)


def _classification_metrics(rows: Sequence[tuple[PrefixExample, Mapping[str, float]]], labels: Sequence[str]) -> dict[str, float]:
    if not rows:
        return {"n": 0}
    top1 = 0
    top3 = 0
    reciprocal = 0.0
    nll = 0.0
    brier = 0.0
    confidence_bins: list[list[float]] = [[] for _ in range(10)]
    for example, probabilities in rows:
        ordered = sorted(labels, key=lambda label: probabilities.get(label, 0.0), reverse=True)
        rank = ordered.index(example.target) + 1 if example.target in ordered else len(labels) + 1
        confidence = probabilities.get(ordered[0], 0.0)
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
        reciprocal += 1.0 / rank
        probability = max(probabilities.get(example.target, 0.0), 1e-12)
        nll -= math.log(probability)
        brier += sum((probabilities.get(label, 0.0) - float(label == example.target)) ** 2 for label in labels)
        confidence_bins[min(9, int(confidence * 10))].append(float(rank == 1))
    ece = 0.0
    for index, values in enumerate(confidence_bins):
        if values:
            confidence = (index + 0.5) / 10.0
            ece += len(values) / len(rows) * abs(confidence - statistics.fmean(values))
    return {
        "n": len(rows),
        "top1": top1 / len(rows),
        "top3": top3 / len(rows),
        "mrr": reciprocal / len(rows),
        "nll": nll / len(rows),
        "brier": brier / len(rows),
        "ece": ece,
    }


def _regression_metrics(rows: Sequence[tuple[PrefixExample, float]], target: str) -> dict[str, float]:
    if not rows:
        return {"n": 0}
    errors = [prediction - float(getattr(example, target)) for example, prediction in rows]
    return {
        "n": len(rows),
        "mae": statistics.fmean(abs(error) for error in errors),
        "rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
    }


def _macro_by_video(
    rows: Sequence[tuple[PrefixExample, Mapping[str, float]]],
    labels: Sequence[str],
) -> dict[str, float]:
    groups: dict[str, list[tuple[PrefixExample, Mapping[str, float]]]] = defaultdict(list)
    for row in rows:
        groups[row[0].video_id].append(row)
    values = [_classification_metrics(group, labels) for group in groups.values()]
    return {
        "videos": len(values),
        "top1": statistics.fmean(value["top1"] for value in values) if values else 0.0,
        "top3": statistics.fmean(value["top3"] for value in values) if values else 0.0,
        "nll": statistics.fmean(value["nll"] for value in values) if values else 0.0,
    }


def _by_baseline(
    rows: Sequence[tuple[PrefixExample, Mapping[str, float]]],
    labels: Sequence[str],
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[tuple[PrefixExample, Mapping[str, float]]]] = defaultdict(list)
    for row in rows:
        groups[row[0].baseline].append(row)
    return {
        baseline: _classification_metrics(group, labels)
        for baseline, group in sorted(groups.items())
    }


def evaluate(runs: Sequence[RunRecord], examples: Sequence[PrefixExample]) -> dict[str, Any]:
    labels = _labels(examples)
    videos = sorted({example.video_id for example in examples})
    classification_rows: dict[str, list[tuple[PrefixExample, Mapping[str, float]]]] = defaultdict(list)
    regression_rows: dict[str, list[tuple[PrefixExample, float]]] = defaultdict(list)
    fold_rows: list[dict[str, Any]] = []

    for held_out in videos:
        train = [example for example in examples if example.video_id != held_out]
        test = [example for example in examples if example.video_id == held_out]
        classifiers = {
            "static_global": StaticModel(labels),
            "static_baseline": StaticModel(labels, by_baseline=True),
            "markov1": MarkovModel(labels, order=1),
            "markov2": MarkovModel(labels, order=2),
            "prefix_only_nb": PrefixTaskNB(labels, include_question=False),
            "prefix_task_nb": PrefixTaskNB(labels),
        }
        regressors = {
            "remaining_steps_static": RegressionModel("remaining_steps"),
            "remaining_steps_markov1": RegressionModel("remaining_steps", by_last_action=True),
            "remaining_runtime_static": RegressionModel("remaining_runtime_ms"),
            "remaining_runtime_markov1": RegressionModel("remaining_runtime_ms", by_last_action=True),
        }
        for model in classifiers.values():
            model.fit(train)
        for model in regressors.values():
            model.fit(train)
        fold_summary: dict[str, Any] = {"held_out_video": held_out, "n_examples": len(test)}
        for name, model in classifiers.items():
            predictions = [(example, model.predict_proba(example)) for example in test]
            classification_rows[name].extend(predictions)
            fold_summary[name] = _classification_metrics(predictions, labels)
        for name, model in regressors.items():
            target = "remaining_steps" if "steps" in name else "remaining_runtime_ms"
            predictions = [(example, model.predict(example)) for example in test]
            regression_rows[name].extend(predictions)
            fold_summary[name] = _regression_metrics(predictions, target)
        fold_rows.append(fold_summary)

    classification_report: dict[str, Any] = {}
    for name, rows in classification_rows.items():
        classification_report[name] = {
            "overall": _classification_metrics(rows, labels),
            "video_macro": _macro_by_video(rows, labels),
            "by_baseline": _by_baseline(rows, labels),
        }
    regression_report: dict[str, Any] = {}
    for name, rows in regression_rows.items():
        target = "remaining_steps" if "steps" in name else "remaining_runtime_ms"
        regression_report[name] = _regression_metrics(rows, target)

    prefix_model = classification_report["prefix_task_nb"]
    prefix_only_model = classification_report["prefix_only_nb"]
    markov_model = classification_report["markov2"]
    static_model = classification_report["static_baseline"]
    prefix_only_beats_markov = (
        prefix_only_model["overall"]["top1"] > markov_model["overall"]["top1"]
        and prefix_only_model["overall"]["nll"] < markov_model["overall"]["nll"]
        and prefix_only_model["video_macro"]["top1"] > markov_model["video_macro"]["top1"]
    )
    task_beats_prefix_on_top1 = (
        prefix_model["overall"]["top1"] > prefix_only_model["overall"]["top1"]
        and prefix_model["video_macro"]["top1"] > prefix_only_model["video_macro"]["top1"]
    )
    task_beats_prefix_calibrated = (
        task_beats_prefix_on_top1
        and prefix_model["overall"]["nll"] < prefix_only_model["overall"]["nll"]
    )
    h2_supported = (
        prefix_only_beats_markov
        and prefix_model["overall"]["top1"] > markov_model["overall"]["top1"]
        and prefix_model["overall"]["nll"] < markov_model["overall"]["nll"]
        and prefix_model["video_macro"]["top1"] > markov_model["video_macro"]["top1"]
        and task_beats_prefix_calibrated
        and prefix_model["overall"]["nll"] < static_model["overall"]["nll"]
    )
    if h2_supported:
        h2_observation = "preliminary_supported"
    elif prefix_only_beats_markov and task_beats_prefix_on_top1:
        h2_observation = "mixed_evidence"
    else:
        h2_observation = "insufficient_evidence"
    question_count = sum(bool(run.question) for run in runs)
    return {
        "schema_version": "phase3-0.1",
        "task": "H2 next-action and remaining-cost prediction",
        "data": {
            "runs": len(runs),
            "videos": len({run.video_id for run in runs}),
            "baselines": dict(sorted(Counter(run.baseline for run in runs).items())),
            "models": dict(sorted(Counter(run.model_name for run in runs).items())),
            "examples": len(examples),
            "action_examples": sum(example.target != END_LABEL for example in examples),
            "terminal_examples": sum(example.target == END_LABEL for example in examples),
            "questions_available": question_count,
        },
        "split": {
            "strategy": "leave_one_video_out",
            "groups": videos,
            "leakage_control": "all repetitions of a video stay in the same fold",
        },
        "labels": labels,
        "classification": classification_report,
        "regression": regression_report,
        "folds": fold_rows,
        "h2_observation": h2_observation,
        "h2_evidence": {
            "prefix_only_vs_markov2": {
                "top1_delta": prefix_only_model["overall"]["top1"] - markov_model["overall"]["top1"],
                "nll_delta": prefix_only_model["overall"]["nll"] - markov_model["overall"]["nll"],
                "video_macro_top1_delta": prefix_only_model["video_macro"]["top1"] - markov_model["video_macro"]["top1"],
            },
            "prefix_task_vs_prefix_only": {
                "top1_delta": prefix_model["overall"]["top1"] - prefix_only_model["overall"]["top1"],
                "top3_delta": prefix_model["overall"]["top3"] - prefix_only_model["overall"]["top3"],
                "nll_delta": prefix_model["overall"]["nll"] - prefix_only_model["overall"]["nll"],
                "video_macro_top1_delta": prefix_model["video_macro"]["top1"] - prefix_only_model["video_macro"]["top1"],
            },
        },
        "h2_rule": "prefix_only_nb must beat markov2; prefix_task_nb must then improve both discrimination and calibrated NLL over prefix_only_nb for a preliminary H2 pass",
        "limitations": [
            "Only video-level grouped validation is valid because repetitions share content.",
            "The prefix+task model uses question text and structured prefix features; no raw visual embedding is inferred from absent trace output.",
            "This is a predictive-signal test, not a scheduling-gain test; H4 remains for Phase 4.",
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
        row = {"task": "regression", "model": name, "scope": "overall"}
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
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    runs = load_runs(args.root.expanduser().resolve(), args.manifest.expanduser().resolve() if args.manifest else None)
    examples = build_examples(runs)
    report = evaluate(runs, examples)
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
