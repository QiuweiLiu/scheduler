#!/usr/bin/env python3
"""Leakage-audited multi-step and graph-constrained trace predictors.

This module is an intentionally small extension of the existing one-step
baselines.  It predicts a short future path from the observed canonical
prefix, while fitting only on the training videos in the supplied split
manifest.  The graph model is a train-only legal-transition graph; it is not
an oracle and it never reads validation/test suffixes during fitting.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tracing.analysis.reproduce_prediction_baselines import (
    END_LABEL,
    Prefix,
    TransitionModel,
    load_prefixes,
    load_split,
)
from tracing.analysis.trace_predictor import PlannerAwareTracePredictor


START_LABEL = "__START__"


@dataclass(frozen=True)
class BeamPath:
    path: tuple[str, ...]
    logprob: float

    @property
    def probability(self) -> float:
        return math.exp(self.logprob)


def _ordered_runs(rows: Iterable[Prefix]) -> dict[str, list[Prefix]]:
    grouped: dict[str, list[Prefix]] = defaultdict(list)
    for row in rows:
        grouped[row.run_id].append(row)
    ordered: dict[str, list[Prefix]] = {}
    for run_id, run_rows in grouped.items():
        values = sorted(run_rows, key=lambda row: (row.position, row.prefix))
        positions = [row.position for row in values]
        if len(positions) != len(set(positions)):
            raise ValueError(f"duplicate prefix positions in run {run_id}")
        ordered[run_id] = values
    return ordered


def future_suffixes(rows: Sequence[Prefix]) -> dict[tuple[str, int], tuple[str, ...]]:
    """Return held-out scoring suffixes without exposing them to a model."""
    result: dict[tuple[str, int], tuple[str, ...]] = {}
    for run_rows in _ordered_runs(rows).values():
        for index, row in enumerate(run_rows):
            result[(row.run_id, row.position)] = tuple(item.target for item in run_rows[index:])
    return result


def _deduplicate_paths(paths: Iterable[BeamPath], beam_size: int) -> list[BeamPath]:
    best: dict[tuple[str, ...], BeamPath] = {}
    for candidate in paths:
        previous = best.get(candidate.path)
        if previous is None or candidate.logprob > previous.logprob:
            best[candidate.path] = candidate
    return sorted(best.values(), key=lambda item: (-item.logprob, item.path))[:beam_size]


def _top_labels(probabilities: Mapping[str, float], limit: int) -> list[tuple[str, float]]:
    return sorted(probabilities.items(), key=lambda item: (-float(item[1]), item[0]))[:limit]


class TransitionBeamPredictor:
    """Recursive Markov beam search, with no graph legality constraint."""

    def __init__(self, labels: Sequence[str], order: int = 2) -> None:
        self.labels = list(labels)
        self.order = order
        self.model = TransitionModel(self.labels, order=order)

    def fit(self, rows: Sequence[Prefix]) -> "TransitionBeamPredictor":
        self.model.fit(rows)
        return self

    def predict(self, row: Prefix, horizon: int, beam_size: int = 5) -> list[BeamPath]:
        if horizon <= 0:
            return [BeamPath((), 0.0)]
        beam = [BeamPath((), 0.0)]
        for _ in range(horizon):
            expanded: list[BeamPath] = []
            for state in beam:
                if state.path and state.path[-1] == END_LABEL:
                    expanded.append(state)
                    continue
                probe = replace(
                    row,
                    prefix=row.prefix + state.path,
                    raw_prefix=row.raw_prefix + state.path,
                    position=row.position + len(state.path),
                )
                for label, probability in _top_labels(self.model.predict(probe), beam_size):
                    expanded.append(
                        BeamPath(state.path + (label,), state.logprob + math.log(max(probability, 1e-12)))
                    )
            beam = _deduplicate_paths(expanded, beam_size)
            if not beam:
                break
            if all(path.path and path.path[-1] == END_LABEL for path in beam):
                break
        return beam


class TransitionGraph:
    """Training-only directed graph of canonical action transitions."""

    def __init__(self) -> None:
        self.edges: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    def fit(self, rows: Sequence[Prefix]) -> "TransitionGraph":
        for run_rows in _ordered_runs(rows).values():
            previous = START_LABEL
            baseline = run_rows[0].baseline if run_rows else "unknown"
            for row in run_rows:
                if row.baseline != baseline:
                    raise ValueError(f"baseline changes inside run {row.run_id}")
                self.edges[(baseline, previous)][row.target] += 1
                previous = row.target
        return self

    def probabilities(self, baseline: str, source: str, alpha: float = 0.0) -> dict[str, float]:
        counts = self.edges.get((baseline, source), Counter())
        if not counts:
            return {}
        values = {label: float(count) + alpha for label, count in counts.items()}
        total = sum(values.values())
        return {label: value / total for label, value in values.items()}

    def is_legal(self, baseline: str, observed_prefix: Sequence[str], path: Sequence[str]) -> bool:
        source = observed_prefix[-1] if observed_prefix else START_LABEL
        for label in path:
            if label not in self.edges.get((baseline, source), {}):
                return False
            source = label
        return True

    @property
    def node_count(self) -> int:
        nodes: set[tuple[str, str]] = set()
        for (baseline, source), targets in self.edges.items():
            nodes.add((baseline, source))
            nodes.update((baseline, target) for target in targets)
        return len(nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(targets) for targets in self.edges.values())


class GraphBeamPredictor:
    """Context-scored beam search restricted to train-observed graph edges.

    The first implementation ranked legal edges by their marginal graph
    frequency.  That was safe but discarded the observed prefix, position and
    planner identity at every branch.  The graph remains a hard legality
    constraint, while the existing planner/state-aware train-only predictor
    supplies the conditional score.  ``graph_prior_weight`` is retained as an
    explicit knob, but validation selection for the current core split uses
    zero: the graph constrains the support and the context model ranks it.
    """

    def __init__(self, graph_prior_weight: float = 0.0) -> None:
        self.graph = TransitionGraph()
        self.graph_prior_weight = max(0.0, float(graph_prior_weight))
        self.context_model: PlannerAwareTracePredictor | None = None

    def fit(self, rows: Sequence[Prefix]) -> "GraphBeamPredictor":
        self.graph.fit(rows)
        labels = sorted({row.target for row in rows})
        self.context_model = PlannerAwareTracePredictor(labels)
        self.context_model.fit(rows)
        return self

    def _probabilities(self, row: Prefix) -> dict[str, float]:
        source = row.prefix[-1] if row.prefix else START_LABEL
        legal = self.graph.probabilities(row.baseline, source)
        if not legal:
            return {}
        context = self.context_model.predict(row) if self.context_model is not None else {}
        scores: dict[str, float] = {}
        for label, graph_probability in legal.items():
            context_probability = max(float(context.get(label, 0.0)), 1e-12)
            prior = max(float(graph_probability), 1e-12) ** self.graph_prior_weight
            scores[label] = context_probability * prior
        total = sum(scores.values())
        return {label: value / total for label, value in scores.items()} if total > 0 else legal

    def predict(self, row: Prefix, horizon: int, beam_size: int = 5) -> list[BeamPath]:
        if horizon <= 0:
            return [BeamPath((), 0.0)]
        beam = [BeamPath((), 0.0)]
        for _ in range(horizon):
            expanded: list[BeamPath] = []
            for state in beam:
                if state.path and state.path[-1] == END_LABEL:
                    expanded.append(state)
                    continue
                probe = replace(
                    row,
                    prefix=row.prefix + state.path,
                    raw_prefix=row.raw_prefix + state.path,
                    position=row.position + len(state.path),
                )
                for label, probability in _top_labels(self._probabilities(probe), beam_size):
                    expanded.append(
                        BeamPath(state.path + (label,), state.logprob + math.log(max(probability, 1e-12)))
                    )
            beam = _deduplicate_paths(expanded, beam_size)
            if not beam:
                break
            if all(path.path and path.path[-1] == END_LABEL for path in beam):
                break
        return beam


class SuffixIndexPredictor:
    """Train-prefix to complete-suffix index with graph fallback."""

    def __init__(self, graph_predictor: GraphBeamPredictor) -> None:
        self.graph_predictor = graph_predictor
        self.index: dict[tuple[str, tuple[str, ...]], Counter[tuple[str, ...]]] = defaultdict(Counter)

    def fit(self, rows: Sequence[Prefix]) -> "SuffixIndexPredictor":
        for run_rows in _ordered_runs(rows).values():
            for index, row in enumerate(run_rows):
                suffix = tuple(item.target for item in run_rows[index:])
                self.index[(row.baseline, row.prefix)][suffix] += 1
        return self

    def predict(self, row: Prefix, horizon: int, beam_size: int = 5) -> tuple[list[BeamPath], bool]:
        matches = self.index.get((row.baseline, row.prefix))
        if matches:
            total = sum(matches.values())
            paths = [
                BeamPath(suffix[:horizon], math.log(count / total))
                for suffix, count in matches.items()
            ]
            return _deduplicate_paths(paths, beam_size), False
        return self.graph_predictor.predict(row, horizon, beam_size), True


def _prefix_match(candidate: Sequence[str], actual: Sequence[str]) -> bool:
    return len(candidate) >= len(actual) and tuple(candidate[: len(actual)]) == tuple(actual)


def _first_labels(paths: Sequence[BeamPath]) -> list[str]:
    labels: list[str] = []
    for path in paths:
        if path.path and path.path[0] not in labels:
            labels.append(path.path[0])
    return labels


def _evaluate_model(
    model: Any,
    rows: Sequence[Prefix],
    suffix_map: Mapping[tuple[str, int], Sequence[str]],
    horizons: Sequence[int],
    beam_size: int,
    graph: TransitionGraph | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for horizon in horizons:
        next_hits = top3_hits = prefix_hits = prefix_hits_k = exact_hits = exact_hits_k = legal = 0
        fallback = 0
        lengths: list[tuple[int, int]] = []
        n = 0
        for row in rows:
            actual = tuple(suffix_map[(row.run_id, row.position)])
            if isinstance(model, SuffixIndexPredictor):
                paths, used_fallback = model.predict(row, horizon, beam_size)
                fallback += int(used_fallback)
            else:
                paths = model.predict(row, horizon, beam_size)
            if not paths:
                continue
            n += 1
            first = _first_labels(paths)
            next_hits += int(bool(first) and first[0] == actual[0])
            top3_hits += int(actual[0] in first[:3])
            prefix_hits += int(_prefix_match(paths[0].path, actual[:horizon]))
            prefix_hits_k += int(any(_prefix_match(path.path, actual[:horizon]) for path in paths))
            exact_hits += int(tuple(paths[0].path) == tuple(actual[:horizon]))
            exact_hits_k += int(any(tuple(path.path) == tuple(actual[:horizon]) for path in paths))
            lengths.append((len(paths[0].path), min(horizon, len(actual))))
            if graph is not None:
                legal += int(graph.is_legal(row.baseline, row.prefix, paths[0].path))
        report[str(horizon)] = {
            "n": n,
            "next_top1": next_hits / n if n else 0.0,
            "next_top3": top3_hits / n if n else 0.0,
            "prefix_hit_at_1": prefix_hits / n if n else 0.0,
            "prefix_hit_at_k": prefix_hits_k / n if n else 0.0,
            "exact_suffix_hit_at_1": exact_hits / n if n else 0.0,
            "exact_suffix_hit_at_k": exact_hits_k / n if n else 0.0,
            "mean_predicted_length": sum(value for value, _ in lengths) / n if n else 0.0,
            "mean_actual_length": sum(value for _, value in lengths) / n if n else 0.0,
            "fallback_rate": fallback / n if n else 0.0,
            "graph_legal_top1": legal / n if n and graph is not None else None,
        }
    return report


def _sample_predictions(model: Any, rows: Sequence[Prefix], horizon: int, beam_size: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows[:3]:
        if isinstance(model, SuffixIndexPredictor):
            paths, fallback = model.predict(row, horizon, beam_size)
        else:
            paths, fallback = model.predict(row, horizon, beam_size), False
        samples.append(
            {
                "run_id": row.run_id,
                "video_id": row.video_id,
                "position": row.position,
                "prefix": list(row.prefix),
                "predictions": [{"path": list(path.path), "probability": path.probability} for path in paths],
                "used_graph_fallback": fallback,
            }
        )
    return samples


def build_report(
    prefix_path: Path,
    split_manifest: Path,
    *,
    horizons: Sequence[int] = (2, 3, 5),
    beam_size: int = 5,
) -> dict[str, Any]:
    split_map = load_split(split_manifest)
    rows = load_prefixes(prefix_path, split_map)
    split_names = {row.split for row in rows}
    required = {"train", "validation", "test"}
    if not required.issubset(split_names):
        raise ValueError(f"formal split requires train/validation/test, got {sorted(split_names)}")
    train = [row for row in rows if row.split == "train"]
    validation = [row for row in rows if row.split == "validation"]
    test = [row for row in rows if row.split == "test"]
    labels = sorted({row.target for row in train})
    if END_LABEL not in labels:
        labels.append(END_LABEL)
    transition = TransitionBeamPredictor(labels).fit(train)
    graph = GraphBeamPredictor().fit(train)
    suffix = SuffixIndexPredictor(graph).fit(train)
    models: dict[str, Any] = {
        "markov_beam": transition,
        "graph_beam": graph,
        "suffix_index": suffix,
    }
    suffix_maps = {name: future_suffixes(group) for name, group in (("validation", validation), ("test", test))}
    metrics: dict[str, Any] = {}
    samples: dict[str, Any] = {}
    for model_name, model in models.items():
        metrics[model_name] = {
            split: _evaluate_model(
                model,
                group,
                suffix_maps[split],
                horizons,
                beam_size,
                graph=graph.graph if model_name == "graph_beam" else None,
            )
            for split, group in (("validation", validation), ("test", test))
        }
        samples[model_name] = _sample_predictions(model, test, max(horizons), beam_size)
    return {
        "schema_version": "multistep-graph-predictor-v0.2",
        "input": {
            "prefix_path": str(prefix_path),
            "split_manifest": str(split_manifest),
            "fit_split": "train",
            "horizons": list(horizons),
            "beam_size": beam_size,
        },
        "data": {
            "rows": len(rows),
            "runs": len({row.run_id for row in rows}),
            "videos": len({row.video_id for row in rows}),
            "rows_by_split": {name: sum(row.split == name for row in rows) for name in ("train", "validation", "test")},
        },
        "train_graph": {"nodes": graph.graph.node_count, "edges": graph.graph.edge_count},
        "graph_model": {
            "legality_source": "train_observed_edges",
            "ranker": "PlannerAwareTracePredictor",
            "graph_prior_weight": graph.graph_prior_weight,
            "selection_rule": "validation_only_context_score_with_hard_graph_mask",
        },
        "metrics": metrics,
        "samples": samples,
        "leakage_checks": {
            "fit_rows_only_train": True,
            "future_events_included_in_input": False,
            "ground_truth_included_in_input": False,
            "remaining_steps_used_as_feature": False,
            "remaining_runtime_used_as_feature": False,
            "validation_suffix_used_for_fit": False,
            "test_suffix_used_for_fit": False,
            "video_id_used_as_feature": False,
        },
        "limitations": [
            "The graph and suffix index are lightweight adapters, not claims of reproducing private paper implementations.",
            "Exact multi-step hits remain conservative when an unseen video has a novel prefix or transition.",
            "Resource-aware scheduling still needs a separate runtime/VRAM head and simulator evaluation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", default="2,3,5")
    parser.add_argument("--beam-size", type=int, default=5)
    args = parser.parse_args()
    horizons = tuple(sorted({int(value) for value in args.horizons.split(",") if value.strip()}))
    if not horizons or any(value <= 0 for value in horizons):
        raise SystemExit("--horizons must contain positive integers")
    report = build_report(args.prefix, args.split_manifest, horizons=horizons, beam_size=max(1, args.beam_size))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "schema_version": report["schema_version"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
