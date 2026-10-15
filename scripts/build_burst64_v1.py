#!/usr/bin/env python3
"""Burst64-v1 high-contention workload builder (STRUCTURAL DIAGNOSTICS ONLY).

Construction (per the 2026-10-14 directive):
- 640 frozen VideoSeek templates -> 10 episodes x 64 unique jobs, stratified
  balanced by (source split, model stack, GPU-cost tercile); provenance kept.
- One contiguous 63-gap Alibaba PAI v2020 segment per episode (reused loader,
  sampled without replacement); the SAME segment is used by all 4 conditions.
- Arrivals per condition: dt_i' = T * dt_i^beta / sum(dt_j^beta), T = W/(2*rho),
  W = episode GPU workload (sum of GPU-lane runtime_p50; the repo's
  total_gpu_runtime convention).  beta=1.5 is a SYNTHETIC burst enhancement,
  NOT production trace data.
- Deadlines preserve the natural post-arrival budget:
  deadline = arrival + (1.5|2.0|3.0)[episode % 3] * total_p90.
- Conditions: (rho,beta) = (0.85,1.0) (1.10,1.0) (0.85,1.5) (1.10,1.5).
- Shared across conditions: membership, identities, segment, GPU config,
  initial residency (cold), seed.  Only arrivals/deadlines change.
- Main conditions use all-normal priority; the legacy synthetic priority can be
  emitted separately via --priority-variant (sensitivity control only).

This mixed-full-set workload is for structural diagnostics ONLY; it must not be
used for predictor-generalization conclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracing.workloads.build_workload_v02 import _load_arrival_gaps  # noqa: E402

TEMPLATES = ROOT / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl"
ALIBABA_CSV = ROOT / "data/public/arrivals/alibaba_v2020/pai_job_table.csv"
OUT_DIR = ROOT / "experiments/EXP-20261014_burst64_v1/artifacts"
EPISODES = 10
JOBS_PER_EPISODE = 64
GAP_SPLIT = "train"          # Alibaba split used for segments (documented)
GPU_TOPOLOGY = [32760.0, 32760.0]  # natural episode convention
BASE_SEED = 20261014
CONDITIONS = (
    ("r085_b10", 0.85, 1.0),
    ("r110_b10", 1.10, 1.0),
    ("r085_b15", 0.85, 1.5),
    ("r110_b15", 1.10, 1.5),
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_templates():
    from tracing.analysis.workload_v02_simulator import load_templates as _lt, train_resource_stats

    templates = _lt(TEMPLATES, topology_view="causal_v3")
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    return templates, stats


def template_estimates(templates, stats):
    from tracing.analysis.workload_v02_simulator import estimate

    out = {}
    for tid, template in templates.items():
        total_p50 = total_p90 = gpu_p50 = 0.0
        for node in template.nodes:
            row = estimate(node, stats)
            total_p50 += float(row["runtime_p50_ms"])
            total_p90 += float(row["runtime_p90_ms"])
            if node.lane == "gpu":
                gpu_p50 += float(row["runtime_p50_ms"])
        out[str(tid)] = {"total_p50": total_p50, "total_p90": total_p90, "gpu_p50": gpu_p50}
    return out


def load_metadata():
    """template_id -> raw provenance (model_stack_id / video_id / split)."""
    meta = {}
    for line in TEMPLATES.read_text(encoding="utf-8").strip().split("\n"):
        row = json.loads(line)
        meta[str(row["template_id"])] = {
            "model_stack_id": str(row.get("model_stack_id") or ""),
            "video_id": str(row.get("video_id") or ""),
            "split": str(row.get("split") or ""),
        }
    return meta


def stratify(templates, estimates, meta):
    """Group templates by (split, model stack, GPU-cost tercile)."""
    costs = sorted(estimates[str(tid)]["gpu_p50"] for tid in templates)
    t1, t2 = costs[len(costs) // 3], costs[2 * len(costs) // 3]

    def tercile(value):
        return 0 if value <= t1 else (1 if value <= t2 else 2)

    groups = defaultdict(list)
    for tid in templates:
        key = (meta[str(tid)]["split"], meta[str(tid)]["model_stack_id"], tercile(estimates[str(tid)]["gpu_p50"]))
        groups[key].append(str(tid))
    return groups, (t1, t2)


def build_assignment(groups, rng):
    """Round-robin each stratum across the 10 episodes (rotating cursor for
    even remainders), then a deterministic residual rebalance to exactly 64."""
    episodes = [[] for _ in range(EPISODES)]
    cursor = 0
    for key in sorted(groups):
        members = sorted(groups[key])
        rng.shuffle(members)
        for tid in members:
            episodes[cursor % EPISODES].append(tid)
            cursor += 1
    # small source pools: top up episodes that are under 64 by re-sampling the
    # pool (reuse across episodes allowed; uniqueness within an episode kept)
    all_tids = sorted({t for members in groups.values() for t in members})
    counts = [len(ep) for ep in episodes]
    for i, count in enumerate(counts):
        while len(episodes[i]) < JOBS_PER_EPISODE:
            pick = all_tids[rng.randrange(len(all_tids))]
            if pick not in episodes[i]:
                episodes[i].append(pick)
    counts = [len(ep) for ep in episodes]
    over = [i for i, c in enumerate(counts) if c > JOBS_PER_EPISODE]
    under = [i for i, c in enumerate(counts) if c < JOBS_PER_EPISODE]
    for i in over:
        while len(episodes[i]) > JOBS_PER_EPISODE and under:
            j = under[0]
            episodes[j].append(episodes[i].pop())
            if len(episodes[j]) == JOBS_PER_EPISODE:
                under.pop(0)
    for ep in episodes:
        rng.shuffle(ep)
    assert all(len(ep) == JOBS_PER_EPISODE for ep in episodes), [len(ep) for ep in episodes]
    return episodes


def select_segments(gaps, rng):
    """One contiguous 63-gap segment per episode, without replacement."""
    need = JOBS_PER_EPISODE - 1
    used = []
    segments = []
    for _ in range(EPISODES):
        for _attempt in range(10_000):
            start = rng.randrange(0, len(gaps) - need + 1)
            end = start + need
            if all(end <= a or start >= b for a, b in used):
                used.append((start, end))
                segments.append((start, end))
                break
        else:
            raise RuntimeError("could not find a non-overlapping segment")
    return segments


def transform_gaps(chosen, rho, beta, W):
    T = W / (2.0 * rho)
    powered = [max(0.0, float(g)) ** beta for g in chosen]
    total = sum(powered) or 1.0
    return [T * value / total for value in powered], T


def build_episode(ep_index, tids, estimates, gaps, segment, rho, beta, tag, templates, priority_variant, meta):
    start, end = segment
    chosen = [float(g) for g in gaps[start:end]]
    W = sum(estimates[t]["gpu_p50"] for t in tids)
    deltas, T = transform_gaps(chosen, rho, beta, W)
    arrivals = [0.0]
    for delta in deltas:
        arrivals.append(round(arrivals[-1] + delta, 3))
    multiplier = (1.5, 2.0, 3.0)[ep_index % 3]
    rho_realized = W / (T * len(GPU_TOPOLOGY)) if T else None
    jobs = []
    for job_index, tid in enumerate(tids):
        row = estimates[tid]
        template = templates[tid]
        jobs.append({
            "arrival_ms": arrivals[job_index],
            "deadline_ms": round(arrivals[job_index] + multiplier * row["total_p90"], 3),
            "deadline_multiplier": multiplier,
            "job_instance_id": f"episode_burst64_{tag}_ep{ep_index:02d}_job_{job_index:03d}",
            "scheduler_visible": {
                "predicted_job_runtime_p50_ms": round(row["total_p50"], 3),
                "predicted_job_runtime_p90_ms": round(row["total_p90"], 3),
                "predicted_gpu_runtime_p50_ms": round(row["gpu_p50"], 3),
            },
            "service_class": "priority" if priority_variant else "normal",
            "template_id": tid,
        })
    source_videos = sorted({meta[t]["video_id"] for t in tids if meta[t]["video_id"]})
    return {
        "schema_version": "workload-episode-v0.2",
        "derivation_version": "burst64-v1",
        "episode_id": f"burst64_{tag}_ep{ep_index:02d}",
        "split": "structural_mixed",
        "scenario_cell": f"alibaba_replay_burst64|{rho:.2f}|{int(GPU_TOPOLOGY[0])},{int(GPU_TOPOLOGY[1])}|cold",
        "value_source": "frozen_640_templates_stratified_plus_alibaba_gaps_burst_transform",
        "arrival_source": "alibaba_v2020_pai_job_table",
        "arrival_pattern": "alibaba_contiguous_segment_burst_transform",
        "arrival_metadata": {
            "gap_count": len(chosen),
            "gap_start_index": start,
            "gap_end_index": end,
            "gap_split": GAP_SPLIT,
            "segment_without_replacement": True,
            "transform": "dt_prime = T * dt^beta / sum(dt^beta)",
            "rho_target": rho,
            "beta": beta,
            "beta_is_synthetic_burst_enhancement": bool(beta != 1.0),
            "T_ms": round(T, 3),
            "W_gpu_runtime_p50_ms": round(W, 3),
        },
        "arrival_span_ms": round(arrivals[-1] - arrivals[0], 3),
        "episode_window_ms": round(T, 3),
        "deadline_multiplier": multiplier,
        "deadline_transform": "preserve_absolute_post_arrival_budget",
        "gpu_topology_mb": list(GPU_TOPOLOGY),
        "initial_state": "cold",
        "initial_residency_hint": [[], []],
        "initial_residency_memory_mb": [[], []],
        "predicted_total_gpu_runtime_ms": round(W, 3),
        "realized_offered_compute_load": round(rho_realized, 6) if rho_realized else None,
        "priority_mode": "all_normal" if not priority_variant else "legacy_priority_sensitivity",
        "source_template_ids": sorted(tids),
        "source_video_ids": source_videos,
        "seed": BASE_SEED,
        "jobs": jobs,
    }


def validate(episodes_by_tag, templates, estimates, stats):
    """Structural invariants; raises on failure. Returns a report dict."""
    report = {}
    all_tids = set()
    for ep in episodes_by_tag[CONDITIONS[0][0]]:
        ids = [j["template_id"] for j in ep["jobs"]]
        assert len(ids) == JOBS_PER_EPISODE, f"episode {ep['episode_id']} has {len(ids)} jobs"
        assert len(set(ids)) == len(ids), f"duplicate jobs in {ep['episode_id']}"
        all_tids.update(ids)
    assert len(all_tids) == len(templates), f"template coverage {len(all_tids)} != pool {len(templates)}"
    report["template_uniqueness"] = f"{len(all_tids)} unique templates across {EPISODES} episodes"

    # per-condition checks
    for tag, rho, beta in CONDITIONS:
        for ep in episodes_by_tag[tag]:
            arrivals = [float(j["arrival_ms"]) for j in ep["jobs"]]
            assert all(b >= a for a, b in zip(arrivals, arrivals[1:])), f"arrivals not monotone in {ep['episode_id']}"
            assert abs(arrivals[0]) < 1e-9, "first arrival must be 0"
            W = float(ep["arrival_metadata"]["W_gpu_runtime_p50_ms"])
            span = arrivals[-1]
            rho_real = W / (span * len(GPU_TOPOLOGY))
            assert abs(rho_real - rho) < 0.02, f"realized rho {rho_real:.3f} != target {rho} in {ep['episode_id']}"
            for job in ep["jobs"]:
                budget = float(job["deadline_ms"]) - float(job["arrival_ms"])
                expected = float(job["deadline_multiplier"]) * estimates[job["template_id"]]["total_p90"]
                assert abs(budget - expected) < 2e-3, "deadline budget drift"
            assert all(j["service_class"] == "normal" for j in ep["jobs"]), "main conditions must be all-normal"
    report["monotone_arrivals_and_rho"] = "ok (tolerance 0.02)"
    report["deadline_budget_preserved"] = "ok"

    # cross-condition invariance: identity sets + budgets equal
    base = episodes_by_tag[CONDITIONS[0][0]]
    for tag, _rho, _beta in CONDITIONS[1:]:
        other = episodes_by_tag[tag]
        for ep_a, ep_b in zip(base, other):
            ids_a = [j["template_id"] for j in ep_a["jobs"]]
            ids_b = [j["template_id"] for j in ep_b["jobs"]]
            assert ids_a == ids_b, "membership/order drifted across conditions"
            budgets_a = [float(j["deadline_ms"]) - float(j["arrival_ms"]) for j in ep_a["jobs"]]
            budgets_b = [float(j["deadline_ms"]) - float(j["arrival_ms"]) for j in ep_b["jobs"]]
            assert all(abs(a - b) < 2e-3 for a, b in zip(budgets_a, budgets_b)), \
                "deadline budgets drifted across conditions"
            seg_a = (ep_a["arrival_metadata"]["gap_start_index"], ep_a["arrival_metadata"]["gap_end_index"])
            seg_b = (ep_b["arrival_metadata"]["gap_start_index"], ep_b["arrival_metadata"]["gap_end_index"])
            assert seg_a == seg_b, "segment drifted across conditions"
    report["cross_condition_invariance"] = "ok (identities, budgets, segments)"

    # memory legality: every GPU node's estimated memory fits the largest GPU
    from tracing.analysis.workload_v02_simulator import estimate as _est
    max_cap = max(GPU_TOPOLOGY)
    worst = 0.0
    for tid, template in templates.items():
        for node in template.nodes:
            if node.lane != "gpu":
                continue
            row = _est(node, stats)
            need = float(row["memory_p95_mb"] or 0.0)
            worst = max(worst, need)
            assert need <= max_cap + 1e-6, f"node memory {need} exceeds capacity {max_cap}"
    report["memory_legality"] = f"ok (max node memory {round(worst, 1)} <= {int(max_cap)})"
    report["ground_truth_leakage"] = "none (only natural episode/job fields + burst metadata)"
    return report


def main() -> int:
    global EPISODES

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--priority-variant", action="store_true",
                    help="emit the legacy synthetic-priority sensitivity control instead of all-normal")
    ap.add_argument("--source", choices=("mixed", "train", "validation"), default="mixed")
    ap.add_argument("--episodes-per-source", type=int, default=EPISODES)
    args = ap.parse_args()
    args.out_dir = args.out_dir.resolve()

    templates, stats = load_templates()
    meta = load_metadata()
    if args.source != "mixed":
        templates = {k: v for k, v in templates.items() if meta[k]["split"] == args.source}
        print(f"source-isolated: {args.source} pool = {len(templates)} templates")
    EPISODES = args.episodes_per_source
    estimates = template_estimates(templates, stats)
    gaps = _load_arrival_gaps(ALIBABA_CSV)[GAP_SPLIT]
    groups, terciles = stratify(templates, estimates, meta)
    rng = random.Random(BASE_SEED)
    assignment = build_assignment(groups, rng)
    segments = select_segments(gaps, rng)

    episodes_by_tag = {}
    for tag, rho, beta in CONDITIONS:
        rows = []
        for ep_index in range(EPISODES):
            rows.append(build_episode(ep_index, assignment[ep_index], estimates, gaps,
                                      segments[ep_index], rho, beta, tag, templates,
                                      args.priority_variant, meta))
        episodes_by_tag[tag] = rows

    checks = validate(episodes_by_tag, templates, estimates, stats)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for tag, rho, beta in CONDITIONS:
        suffix = ("_priority" if args.priority_variant else "") + ("" if args.source == "mixed" else f"_{args.source}")
        path = args.out_dir / f"workload_burst64_{tag}{suffix}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in episodes_by_tag[tag]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs[tag] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                        "rho": rho, "beta": beta}
    manifest = {
        "schema": "burst64-v1-manifest",
        "purpose": "structural diagnostics only; not for predictor generalization",
        "templates": {"path": str(TEMPLATES.relative_to(ROOT)), "sha256": sha256_file(TEMPLATES), "count": len(templates)},
        "alibaba": {"path": str(ALIBABA_CSV.relative_to(ROOT)), "sha256": sha256_file(ALIBABA_CSV),
                    "gap_split": GAP_SPLIT, "gap_count": len(gaps)},
        "episodes": EPISODES, "jobs_per_episode": JOBS_PER_EPISODE,
        "stratification": {"keys": ["split", "model_stack_id", "gpu_cost_tercile"], "terciles_ms": list(terciles)},
        "segments": [{"episode": i, "gap_start": s, "gap_end": e} for i, (s, e) in enumerate(segments)],
        "conditions": [{"tag": t, "rho": r, "beta": b,
                        "beta_synthetic_burst": bool(b != 1.0)} for t, r, b in CONDITIONS],
        "priority_mode": "legacy_priority_sensitivity" if args.priority_variant else "all_normal",
        "checks": checks,
        "outputs": outputs,
    }
    suffix = ("_priority" if args.priority_variant else "") + ("" if args.source == "mixed" else f"_{args.source}")
    manifest_path = args.out_dir / f"burst64_v1_manifest{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))
    print("manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
