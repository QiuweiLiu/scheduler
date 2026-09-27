#!/usr/bin/env python
"""Build the High-Contention Stress Regime v2 from the frozen Natural episodes.

v2 replaces the v1 absolute-target transform (``arrival x rho_old/rho``) with a
single *multiplicative arrival scale* ``alpha`` applied identically to every
inter-arrival gap of every episode.  v1 was found to be construct-invalid: because
the Natural episodes carry heterogeneous ``rho_old`` in [0.50, 1.05], scaling by
``rho_old/rho`` EXPANDS the episodes whose ``rho_old`` exceeds the target,
homogenising the load instead of uniformly pressurising it.

Design (frozen 2026-09-27, after the v1 construct-validity audit):

  * templates / graphs / job membership / service semantics: UNCHANGED
  * episode membership: exactly paired with the Natural episodes (same ids + template
    assignment); only a deterministic transform is applied
  * arrival: ``arrival' = arrival * alpha`` for every job; every inter-arrival gap is
    scaled by the SAME ``alpha`` in (0, 1], so ordering, burst shape and the natural
    high/low-load heterogeneity are all preserved and every episode is strictly
    compressed (``alpha < 1``)
  * service class: ``all_normal`` (removes the builder's synthetic ``job_index % 4``
    priority barrier) -- default on
  * deadline: the SAME absolute post-arrival budget, ``deadline' = arrival' + (deadline - arrival)``
  * window / span / load metadata: RECOMPUTED, never left stale

The transform is rejected if it is not exactly invertible: recomputing the realized
load from the written file must equal ``rho_before / alpha`` for every episode.

Usage::

    python scripts/preprocess/build_stress_v2_episodes.py \
        --episodes <natural.jsonl> --alpha 0.80 --out-dir <dir> [--keep-service-class]

``--alpha 1.0`` produces the required all-normal *control* (B-only: service-class
removal, no arrival compression).  Writes ``<out-dir>/workload_validation_<tag>.jsonl``
plus ``<out-dir>/stress_manifest_<tag>.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

TRANSFORM_VERSION = "stress-arrival-scale-alpha-v2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_sha256(episode: Dict[str, Any]) -> str:
    blob = json.dumps(episode, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def transform_episode(episode: Dict[str, Any], alpha: float, all_normal: bool) -> Dict[str, Any]:
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1], got %r" % alpha)
    rho_before = float(episode.get("target_offered_compute_load") or 0.0)
    if rho_before <= 0.0:
        raise ValueError("episode %r has no target_offered_compute_load" % episode.get("episode_id"))
    n_gpu = len(episode.get("gpu_topology_mb") or [])
    total_gpu = float(episode.get("predicted_total_gpu_runtime_ms") or 0.0)
    window_old = float(episode.get("episode_window_ms") or 0.0)

    new = json.loads(json.dumps(episode))          # deep copy
    new_jobs = []
    for job in episode.get("jobs") or []:
        arrival_old = float(job.get("arrival_ms") or 0.0)
        deadline_old = float(job.get("deadline_ms") or 0.0)
        budget = deadline_old - arrival_old        # absolute post-arrival deadline budget
        arrival_new = round(arrival_old * alpha, 3)
        j = dict(job)
        j["arrival_ms"] = arrival_new
        j["deadline_ms"] = round(arrival_new + budget, 3)
        if all_normal:
            j["service_class"] = "normal"
        new_jobs.append(j)

    window_new = window_old * alpha
    arrivals = [float(j["arrival_ms"]) for j in new_jobs]
    new["jobs"] = new_jobs
    new["episode_window_ms"] = round(window_new, 3)
    new["arrival_span_ms"] = round(max(arrivals) - min(arrivals), 3) if arrivals else 0.0
    new["target_offered_compute_load"] = round(rho_before / alpha, 6)
    new["realized_offered_compute_load"] = (
        round(total_gpu / (window_new * n_gpu), 6) if window_new > 0 and n_gpu else None
    )
    new["parent_episode_id"] = episode.get("episode_id")
    new["parent_episode_sha256"] = None                 # filled by the writer (per-episode)
    new["stress_transform_version"] = TRANSFORM_VERSION
    new["stress_alpha"] = float(alpha)
    new["stress_load_before"] = round(rho_before, 6)
    new["stress_load_after"] = round(rho_before / alpha, 6)
    new["stress_arrival_scale"] = float(alpha)
    new["stress_service_class_transform"] = "all_normal" if all_normal else "unchanged"
    new["stress_deadline_transform"] = "preserve_absolute_budget"
    new["stress_template_assignment_changed"] = False
    tag = "alpha%03d" % round(float(alpha) * 100)
    new["episode_id"] = "%s__stressv2_%s%s" % (
        episode.get("episode_id"), tag, "_allnormal" if all_normal else "")
    new["scenario_cell"] = "%s|stressv2" % (episode.get("scenario_cell"),)
    return new


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--keep-service-class", action="store_true",
                        help="do NOT force all-normal (QoS sensitivity only)")
    args = parser.parse_args()

    src = Path(args.episodes)
    src_sha = sha256_file(src)
    rows = read_jsonl(src)
    all_normal = not args.keep_service_class
    out_rows = [transform_episode(ep, args.alpha, all_normal) for ep in rows]
    for src_ep, out_ep in zip(rows, out_rows):
        out_ep["parent_episode_sha256"] = episode_sha256(src_ep)

    # ---- source -> output invariant checks -------------------------------------
    # Verifies the arrival transform itself, not just the realized-load identity.
    bad = []
    tol = 1e-6
    for src_ep, ep in zip(rows, out_rows):
        eid = ep["episode_id"]
        if "jobs" not in src_ep or "jobs" not in ep:
            bad.append((eid, "missing 'jobs' list"))
            continue
        src_jobs = src_ep["jobs"]
        out_jobs = ep["jobs"]
        if len(src_jobs) != len(out_jobs):
            bad.append((eid, "job count changed %d -> %d" % (len(src_jobs), len(out_jobs))))
            continue
        prev_new = None
        for s, o in zip(src_jobs, out_jobs):
            for field in ("job_instance_id", "template_id"):
                if s.get(field) is None or o.get(field) is None:
                    bad.append((eid, "job missing %s" % field))
                elif s[field] != o[field]:
                    bad.append((eid, "%s changed %r -> %r" % (field, s[field], o[field])))
            for field in ("arrival_ms", "deadline_ms"):
                if s.get(field) is None:
                    bad.append((eid, "source job missing %s" % field))
                if o.get(field) is None:
                    bad.append((eid, "output job missing %s" % field))
            if s.get("arrival_ms") is None or o.get("arrival_ms") is None:
                continue
            # every arrival scaled by the SAME alpha (rounding to 3dp allowed)
            a_old, a_new = float(s["arrival_ms"]), float(o["arrival_ms"])
            if abs(a_new - a_old * args.alpha) > 1e-3 + tol:
                bad.append((eid, "arrival %r*alpha != %r" % (a_old, a_new)))
            # order preserved (non-decreasing input stays non-decreasing)
            if prev_new is not None and a_new < prev_new - 1e-9:
                bad.append((eid, "arrival order not preserved"))
            prev_new = a_new
            # absolute post-arrival deadline budget preserved
            budget_old = float(s["deadline_ms"]) - a_old
            budget_new = float(o["deadline_ms"]) - a_new
            if abs(budget_old - budget_new) > 1e-3 + tol:
                bad.append((eid, "deadline budget changed %r -> %r" % (budget_old, budget_new)))
        # window / span recomputed from the transformed arrivals must stay consistent
        if "episode_window_ms" not in src_ep or "episode_window_ms" not in ep:
            bad.append((eid, "missing episode_window_ms"))
            continue
        old_window = float(src_ep["episode_window_ms"])
        new_window = float(ep["episode_window_ms"])
        if abs(new_window - old_window * args.alpha) > 1e-3 + tol:
            bad.append((eid, "episode_window_ms %r != %r*alpha" % (new_window, old_window)))
        if out_jobs:
            span = max(float(j["arrival_ms"]) for j in out_jobs) - min(float(j["arrival_ms"]) for j in out_jobs)
            if abs(float(ep.get("arrival_span_ms") or 0.0) - span) > 1e-3 + tol:
                bad.append((eid, "arrival_span_ms %r != recomputed %r" % (ep.get("arrival_span_ms"), span)))
            src_span = max(float(j["arrival_ms"]) for j in src_jobs) - min(float(j["arrival_ms"]) for j in src_jobs)
            # only assert window == span when the SOURCE satisfied it (else the identity does not apply)
            if abs(old_window - src_span) <= 1e-3 + tol and abs(new_window - span) > 1e-3 + tol:
                bad.append((eid, "source had window==span but output window %r != span %r" % (new_window, span)))
        # realized-load identity (invertibility)
        n_gpu = len(ep.get("gpu_topology_mb") or [])
        total = float(ep.get("predicted_total_gpu_runtime_ms") or 0.0)
        window = float(ep.get("episode_window_ms") or 0.0)
        expect = float(ep.get("stress_load_before") or 0.0) / args.alpha
        if not n_gpu or window <= 0:
            bad.append((eid, "degenerate window"))
            continue
        realized = total / (window * n_gpu)
        if abs(realized - expect) > 1e-6:
            bad.append((eid, "realized %r != expected %r" % (realized, expect)))
    if bad:
        raise AssertionError("stress-v2 invariants violated (%d): %r" % (len(bad), bad[:5]))

    tag = "alpha%03d" % round(args.alpha * 100)
    out_dir = Path(args.out_dir)
    out_file = out_dir / ("workload_validation_%s.jsonl" % tag)
    write_jsonl(out_file, out_rows)
    manifest = {
        "schema_version": "stress-episode-manifest-v2",
        "transform_version": TRANSFORM_VERSION,
        "source_episodes": str(src),
        "source_episodes_sha256": src_sha,
        "alpha": args.alpha,
        "all_normal": all_normal,
        "episodes": len(out_rows),
        "output": str(out_file),
        "output_sha256": sha256_file(out_file),
        "rules": {
            "templates_changed": False,
            "job_count_changed": False,
            "arrival": "every inter-arrival gap scaled by the same alpha (arrival' = arrival * alpha)",
            "service_class": "all_normal" if all_normal else "unchanged",
            "deadline": "preserve_absolute_budget",
        },
    }
    (out_dir / ("stress_manifest_%s.json" % tag)).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote %s (%d episodes, alpha=%.2f, all_normal=%s)" % (out_file, len(out_rows), args.alpha, all_normal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
