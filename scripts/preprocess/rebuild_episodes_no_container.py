"""Phase 19 step B — regenerate R7 episodes on the container-excluded templates.

The episode builder derives every template-dependent field from the templates:

    total_p50, total_p90, gpu_p50 = sum of per-node estimates over the template
    deadline_ms                  = arrival_ms + deadline_multiplier * total_p90
    predicted_total_gpu_runtime_ms = sum of gpu_p50 over the episode's jobs
    window_ms                    = total_gpu_runtime / (pressure * n_gpu)
    arrivals                     = normalized_position * window_ms

so removing nodes changes all of them. This script recomputes those fields on the
v03 templates while holding everything else (episode ids, template ids, arrival
order, seeds, pressures, topologies, initial residency) exactly as in v02, which
makes v02/v03 a paired comparison.

`--mode self-check` is the trust anchor: it re-derives the v02 fields from the v02
templates and requires them to match the stored v02 episodes field by field. If
that check passes, the same code applied to v03 is a faithful rebuild.

Resource stats are deliberately taken from the **v02** train templates and held
fixed, so the only change between v02 and v03 is the removal of container nodes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tracing.workloads import build_workload_v02 as v02  # noqa: E402

TEMPLATE_FIELDS = ("total_p50", "total_p90", "gpu_p50")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def job_estimates(template: dict[str, Any], stats: dict[str, Any]) -> tuple[float, float, float]:
    return v02._job_estimates(template, stats)


def rebuild_episode(
    episode: dict[str, Any],
    templates_by_id: dict[str, dict[str, Any]],
    stats: dict[str, Any],
) -> dict[str, Any]:
    jobs = episode.get("jobs") or []
    pressure = float(episode.get("target_offered_compute_load") or 0.0)
    n_gpu = max(1, len(episode.get("gpu_topology_mb") or []))
    old_window = float(episode.get("episode_window_ms") or 0.0)

    # 1. recover the normalized arrival positions from the stored v02 episode
    positions = []
    for job in jobs:
        arrival = float(job.get("arrival_ms") or 0.0)
        positions.append(arrival / old_window if old_window > 0 else 0.0)

    # 2. per-job estimates on the target templates
    estimates = []
    for job in jobs:
        template = templates_by_id[str(job["template_id"])]
        estimates.append(job_estimates(template, stats))

    total_gpu = sum(gpu_p50 for _p50, _p90, gpu_p50 in estimates)
    total_gpu = float(round(total_gpu, 3))
    window = max(1.0, total_gpu / (pressure * n_gpu)) if pressure > 0 else max(1.0, old_window)

    rebuilt = dict(episode)
    rebuilt_jobs = []
    for job, (total_p50, total_p90, gpu_p50), position in zip(jobs, estimates, positions):
        new_job = dict(job)
        arrival = round(position * window, 3)
        new_job["arrival_ms"] = arrival
        new_job["deadline_ms"] = round(arrival + float(job.get("deadline_multiplier") or 0.0) * total_p90, 3)
        new_job["scheduler_visible"] = {
            "predicted_job_runtime_p50_ms": round(total_p50, 3),
            "predicted_job_runtime_p90_ms": round(total_p90, 3),
            "predicted_gpu_runtime_p50_ms": round(gpu_p50, 3),
        }
        rebuilt_jobs.append(new_job)

    arrivals = [float(job["arrival_ms"]) for job in rebuilt_jobs]
    rebuilt["jobs"] = rebuilt_jobs
    rebuilt["episode_window_ms"] = round(window, 3)
    rebuilt["arrival_span_ms"] = round(max(arrivals) - min(arrivals), 3) if arrivals else 0.0
    rebuilt["predicted_total_gpu_runtime_ms"] = total_gpu
    rebuilt["realized_offered_compute_load"] = round(total_gpu / (window * n_gpu), 6)
    return rebuilt


TOL_MS = 0.01  # 1e-2 ms: stored values carry sub-microsecond float rounding only


def close(left: Any, right: Any, tol: float = TOL_MS) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=tol)
    except (TypeError, ValueError):
        return left == right


def compare(original: dict[str, Any], recomputed: dict[str, Any]) -> dict[str, Any]:
    """Field-by-field diff of a self-check rebuild against the stored episode."""

    problems: list[str] = []
    if original["episode_id"] != recomputed["episode_id"]:
        problems.append("episode_id")
    for key in ("episode_window_ms", "arrival_span_ms", "predicted_total_gpu_runtime_ms"):
        if not close(original[key], recomputed[key]):
            problems.append(f"{key}: {original[key]} != {recomputed[key]}")
    if not close(original["realized_offered_compute_load"], recomputed["realized_offered_compute_load"], 1e-6):
        problems.append("realized_offered_compute_load")
    for left, right in zip(original.get("jobs") or [], recomputed.get("jobs") or []):
        if left["job_instance_id"] != right["job_instance_id"] or left["template_id"] != right["template_id"]:
            problems.append(f"job identity {left['job_instance_id']}")
            break
        if not close(left["arrival_ms"], right["arrival_ms"]):
            problems.append(f"arrival_ms {left['job_instance_id']}: {left['arrival_ms']} != {right['arrival_ms']}")
        if not close(left["deadline_ms"], right["deadline_ms"]):
            problems.append(f"deadline_ms {left['job_instance_id']}: {left['deadline_ms']} != {right['deadline_ms']}")
        left_visible = left.get("scheduler_visible") or {}
        right_visible = right.get("scheduler_visible") or {}
        if set(left_visible) != set(right_visible) or any(
            not close(left_visible[key], right_visible[key]) for key in left_visible
        ):
            problems.append(f"scheduler_visible {left['job_instance_id']}")
        if len(problems) >= 5:
            break
    return {"episode_id": original["episode_id"], "problems": problems}


def run(mode: str, root: Path, source_templates: Path, target_templates: Path, episodes: list[Path], out_dir: Path | None) -> int:
    source = read_jsonl(source_templates)
    target = read_jsonl(target_templates)
    stats = v02._resource_stats([row for row in source if v02._text(row.get("split")) == "train"])
    by_id = {str(row["template_id"]): row for row in target}

    mismatched = 0
    checked = 0
    for path in episodes:
        for episode in read_jsonl(path):
            rebuilt = rebuild_episode(episode, by_id, stats)
            if mode == "self-check":
                checked += 1
                diff = compare(episode, rebuilt)
                if diff["problems"]:
                    mismatched += 1
                    if mismatched <= 5:
                        print(json.dumps(diff, ensure_ascii=False))
    if mode == "self-check":
        print(
            json.dumps(
                {
                    "mode": "self-check",
                    "episodes_checked": checked,
                    "episodes_with_mismatch": mismatched,
                    "verdict": "PASS" if mismatched == 0 else "FAIL",
                },
                ensure_ascii=False,
            )
        )
        return 0 if mismatched == 0 else 1

    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in episodes:
        rows = [rebuild_episode(episode, by_id, stats) for episode in read_jsonl(path)]
        name = path.name.replace("_r7.jsonl", "_r7_v03.jsonl")
        write_jsonl(out_dir / name, rows)
        print(f"wrote {out_dir / name} ({len(rows)} episodes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("self-check", "apply"), required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-templates", type=Path, default=None)
    parser.add_argument("--target-templates", type=Path, default=None)
    parser.add_argument("--episodes", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.project_root
    legacy = root / "results" / "processed" / "r7_workload_20260817"
    source_templates = args.source_templates or legacy / "job_templates_r7_v02.jsonl"
    target_templates = args.target_templates or (
        root / "results" / "processed" / "r7_workload_v03_no_run_container" / "job_templates_r7_v03.jsonl"
    )
    episodes = args.episodes or [
        legacy / "episodes" / "workload_train_r7.jsonl",
        legacy / "episodes" / "workload_validation_r7.jsonl",
    ]
    if args.mode == "self-check":
        # self-check compares a rebuild from the SOURCE templates with the stored episodes
        return run("self-check", root, source_templates, source_templates, episodes, None)
    out_dir = args.out_dir or (root / "results" / "processed" / "r7_workload_v03_no_run_container" / "episodes")
    return run("apply", root, source_templates, target_templates, episodes, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
