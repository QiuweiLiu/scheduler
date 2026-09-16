#!/usr/bin/env python3
"""Summarize Phase 2 traces into path and resource-dynamicity evidence."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def _entropy(values: Iterable[str]) -> float:
    counts = collections.Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return round(-sum((count / total) * math.log2(count / total) for count in counts.values()), 6)


def _cv(values: list[float]) -> float | None:
    if len(values) < 2 or not values:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return 0.0
    return round(statistics.pstdev(values) / mean, 6)


def _conditional_next_entropy(pairs: list[tuple[str, str]]) -> float:
    by_previous: dict[str, list[str]] = collections.defaultdict(list)
    for previous, current in pairs:
        by_previous[previous].append(current)
    total = sum(len(values) for values in by_previous.values())
    if total == 0:
        return 0.0
    weighted = 0.0
    for values in by_previous.values():
        weighted += (len(values) / total) * _entropy(values)
    return round(weighted, 6)


def _trace_paths(root: Path) -> list[Path]:
    direct = root / "trace.jsonl"
    if direct.is_file():
        return [direct]
    return sorted(root.rglob("trace.jsonl"))


def summarize(root: Path) -> dict[str, Any]:
    traces = _trace_paths(root)
    rows: list[dict[str, Any]] = []
    transition_counter: collections.Counter[str] = collections.Counter()
    tool_counter: collections.Counter[str] = collections.Counter()
    path_counter: collections.Counter[str] = collections.Counter()
    transition_pairs: list[tuple[str, str]] = []
    lengths: list[float] = []
    runtimes: list[float] = []
    local_runtimes: list[float] = []
    api_waits: list[float] = []
    loads: list[float] = []
    peaks: list[float] = []
    retry_count = 0
    error_count = 0
    for trace_path in traces:
        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        actions = [event for event in events if event.get("event_type") == "action"]
        action_names = [str(event.get("action")) for event in actions]
        path_key = " -> ".join(action_names)
        path_counter[path_key] += 1
        for left, right in zip(action_names, action_names[1:]):
            transition_counter[f"{left} -> {right}"] += 1
            transition_pairs.append((left, right))
        tool_counter.update(action_names)
        lengths.append(float(len(action_names)))
        for event in events:
            if event.get("status") == "error":
                error_count += 1
            if event.get("retry_of") is not None:
                retry_count += 1
            resource = event.get("resource", {}) or {}
            runtimes.append(float(resource.get("runtime_ms") or 0.0))
            local_runtimes.append(float(resource.get("local_runtime_ms") or 0.0))
            api_waits.append(float(resource.get("api_wait_ms") or 0.0))
            if resource.get("load_ms") is not None:
                loads.append(float(resource["load_ms"]))
            if resource.get("peak_allocated_mb") is not None:
                peaks.append(float(resource["peak_allocated_mb"]))
        manifest_path = trace_path.with_name("run_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        rows.append(
            {
                "trace_path": str(trace_path),
                "run_id": manifest.get("run_id"),
                "dataset": manifest.get("dataset"),
                "task_id": manifest.get("task_id"),
                "baseline": manifest.get("baseline"),
                "status": manifest.get("status"),
                "action_count": len(action_names),
                "path": action_names,
                "path_key": path_key,
                "elapsed_ms": manifest.get("elapsed_ms"),
            }
        )
    unique_paths = len(path_counter)
    run_count = len(rows)
    path_ratio = round(unique_paths / run_count, 6) if run_count else 0.0
    # H1 is reported as an observation, not a claim of causal content
    # dependence.  Repeated paths are still useful controls for the predictor.
    h1_observation = "pass" if unique_paths >= 2 and path_ratio >= 0.2 else "insufficient_evidence"
    # H3 uses measured local runtime and model load/peak samples.  A high CV is
    # a resource-heterogeneity signal, not proof that scheduling will help.
    h3_observation = (
        "pass"
        if (_cv(local_runtimes) or 0.0) >= 0.2 or (_cv(loads) or 0.0) >= 0.2 or (_cv(peaks) or 0.0) >= 0.2
        else "insufficient_evidence"
    )
    return {
        "root": str(root),
        "runs": run_count,
        "success_runs": sum(row.get("status") == "success" for row in rows),
        "unique_paths": unique_paths,
        "path_ratio": path_ratio,
        "path_length": {
            "mean": round(statistics.fmean(lengths), 6) if lengths else 0.0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "path_distribution": path_counter,
        "tool_frequency": tool_counter,
        "transition_frequency": transition_counter,
        "next_action_entropy_bits": _conditional_next_entropy(transition_pairs),
        "retry_count": retry_count,
        "error_count": error_count,
        "resource": {
            "runtime_ms_cv": _cv(runtimes),
            "local_runtime_ms_cv": _cv(local_runtimes),
            "api_wait_ms_cv": _cv(api_waits),
            "load_ms_cv": _cv(loads),
            "peak_allocated_mb_cv": _cv(peaks),
            "runtime_samples": len(runtimes),
            "load_samples": len(loads),
            "peak_samples": len(peaks),
        },
        "gates": {"H1_path_dynamicity_observation": h1_observation, "H3_resource_heterogeneity_observation": h3_observation},
        "runs_detail": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    payload = summarize(args.root.expanduser().resolve())
    output = args.output or (args.root / "phase2_trace_stats.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")
    print(json.dumps({"runs": payload["runs"], "unique_paths": payload["unique_paths"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
