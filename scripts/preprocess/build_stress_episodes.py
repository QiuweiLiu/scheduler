#!/usr/bin/env python
"""Build the High-Contention Stress Regime v1 from the frozen Natural episodes.

Design (fixed before running, per the stress protocol review):

  * templates / graphs / job count / job order: UNCHANGED
  * episode membership: exactly paired with the Natural episodes (same ids + template
    assignment); only a deterministic transform is applied
  * arrival realization: the SAME normalized realization, scaled by ``rho_old / rho`` so the
    burst synchronisation, gap ratios and ordering are preserved
  * service class: ``all_normal`` (removes the builder's synthetic ``job_index % 4``
    priority barrier) -- optional, default on
  * deadline: the SAME absolute post-arrival budget, ``deadline' = arrival' + (deadline - arrival)``
  * window / span / load metadata: RECOMPUTED, never left stale

The transform is rejected if it is not exactly invertible: recomputing the realized load from
the written file must equal the target rho.

Usage::

    python scripts/preprocess/build_stress_episodes.py \
        --episodes <natural.jsonl> --rho 1.20 --out-dir <dir> [--keep-service-class]

Writes ``<out-dir>/workload_validation_<tag>.jsonl`` plus a ``stress_manifest.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

TRANSFORM_VERSION = "stress-arrival-compression-all-normal-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def transform_episode(episode: Dict[str, Any], rho_target: float, all_normal: bool) -> Dict[str, Any]:
    rho_old = float(episode.get("target_offered_compute_load") or 0.0)
    if rho_old <= 0.0:
        raise ValueError("episode %r has no target_offered_compute_load" % episode.get("episode_id"))
    n_gpu = len(episode.get("gpu_topology_mb") or [])
    total_gpu = float(episode.get("predicted_total_gpu_runtime_ms") or 0.0)
    window_old = float(episode.get("episode_window_ms") or 0.0)
    scale = rho_old / float(rho_target)

    new = json.loads(json.dumps(episode))          # deep copy
    new_jobs = []
    for job in episode.get("jobs") or []:
        arrival_old = float(job.get("arrival_ms") or 0.0)
        deadline_old = float(job.get("deadline_ms") or 0.0)
        budget = deadline_old - arrival_old        # absolute post-arrival deadline budget
        arrival_new = round(arrival_old * scale, 3)
        j = dict(job)
        j["arrival_ms"] = arrival_new
        j["deadline_ms"] = round(arrival_new + budget, 3)
        if all_normal:
            j["service_class"] = "normal"
        new_jobs.append(j)

    window_new = window_old * scale
    arrivals = [float(j["arrival_ms"]) for j in new_jobs]
    new["jobs"] = new_jobs
    new["episode_window_ms"] = round(window_new, 3)
    new["arrival_span_ms"] = round(max(arrivals) - min(arrivals), 3) if arrivals else 0.0
    new["target_offered_compute_load"] = float(rho_target)
    new["realized_offered_compute_load"] = (
        round(total_gpu / (window_new * n_gpu), 6) if window_new > 0 and n_gpu else None
    )
    new["parent_episode_id"] = episode.get("episode_id")
    new["parent_episode_sha256"] = None                 # filled by the writer (per-episode)
    new["stress_transform_version"] = TRANSFORM_VERSION
    new["stress_rho"] = float(rho_target)
    new["stress_arrival_scale"] = scale
    new["stress_service_class_transform"] = "all_normal" if all_normal else "unchanged"
    new["stress_deadline_transform"] = "preserve_absolute_budget"
    new["stress_template_assignment_changed"] = False
    tag = "rho%03d" % round(float(rho_target) * 100)
    new["episode_id"] = "%s__stress_%s%s" % (
        episode.get("episode_id"), tag, "_allnormal" if all_normal else "")
    new["scenario_cell"] = "%s|stress" % (episode.get("scenario_cell"),)
    return new


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--keep-service-class", action="store_true",
                        help="do NOT force all-normal (QoS sensitivity only)")
    args = parser.parse_args()

    src = Path(args.episodes)
    src_sha = sha256_file(src)
    rows = read_jsonl(src)
    all_normal = not args.keep_service_class
    out_rows = [transform_episode(ep, args.rho, all_normal) for ep in rows]

    # self-check: recompute the realized load from the written fields and confirm the target
    bad = []
    for ep in out_rows:
        n_gpu = len(ep.get("gpu_topology_mb") or [])
        total = float(ep.get("predicted_total_gpu_runtime_ms") or 0.0)
        window = float(ep.get("episode_window_ms") or 0.0)
        if not n_gpu or window <= 0:
            bad.append((ep["episode_id"], "degenerate window"))
            continue
        realized = total / (window * n_gpu)
        if abs(realized - args.rho) > 1e-6:
            bad.append((ep["episode_id"], "realized %r != rho %r" % (realized, args.rho)))
    if bad:
        raise AssertionError("stress transform is not invertible: %r" % bad[:3])

    tag = "rho%03d" % round(args.rho * 100)
    out_dir = Path(args.out_dir)
    out_file = out_dir / ("workload_validation_%s.jsonl" % tag)
    write_jsonl(out_file, out_rows)
    manifest = {
        "schema_version": "stress-episode-manifest-v1",
        "transform_version": TRANSFORM_VERSION,
        "source_episodes": str(src),
        "source_episodes_sha256": src_sha,
        "rho": args.rho,
        "all_normal": all_normal,
        "episodes": len(out_rows),
        "output": str(out_file),
        "output_sha256": sha256_file(out_file),
        "rules": {
            "templates_changed": False,
            "job_count_changed": False,
            "arrival_realization": "same normalized realization scaled by rho_old/rho",
            "service_class": "all_normal" if all_normal else "unchanged",
            "deadline": "preserve_absolute_budget",
        },
    }
    (out_dir / ("stress_manifest_%s.json" % tag)).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote %s (%d episodes, rho=%.2f, all_normal=%s)" % (out_file, len(out_rows), args.rho, all_normal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
