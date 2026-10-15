#!/usr/bin/env python3
"""JCT-MILP-v1 Tail estimator feasibility: remaining work beyond H5 from TRAIN data.

For every train template and every possible anchor position p (a GPU node), the
"remaining beyond the H5 boundary" is the template's realized work after p+5:
- remaining GPU compute + load (sum over GPU-lane nodes after the boundary),
- remaining CPU/API time (sum over CPU-lane nodes),
- termination probability (chain ends within +5 steps).

Aggregated by (model_stack_id, anchor position bucket) -> the Tail lookup table.
Train split only; no validation/hidden truth.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEMPLATES = ROOT / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl"


def main() -> int:
    from tracing.analysis.workload_v02_simulator import load_templates, train_resource_stats, estimate

    templates = load_templates(TEMPLATES, topology_view="causal_v3")
    stats = train_resource_stats({k: v for k, v in templates.items() if v.split == "train"})
    meta = {}
    for line in TEMPLATES.read_text(encoding="utf-8").strip().split("\n"):
        row = json.loads(line)
        meta[str(row["template_id"])] = str(row.get("model_stack_id") or "")

    # per (stack, position bucket): remaining beyond p+5 (legacy) plus
    # remaining-after-p+d for d in 0..6 (truncation-consistent tail lookup).
    buckets: dict = defaultdict(lambda: {
        "gpu_ms": [], "load_ms": [], "cpu_ms": [], "terminated": 0, "total": 0, "nodes_left": [],
        "by_depth": {d: {"gpu_ms": [], "cpu_ms": []} for d in range(0, 7)},
    })
    for tid, template in templates.items():
        if template.split != "train":
            continue
        nodes = sorted(template.nodes, key=lambda n: n.sequence_index)
        rows = []
        for node in nodes:
            row = estimate(node, stats)
            rows.append((node.sequence_index, node.lane, float(row["runtime_p50_ms"]), float(row["load_p50_ms"] or 0.0)))
        for anchor in nodes:
            if anchor.lane != "gpu":
                continue
            p = anchor.sequence_index
            key = (meta[str(tid)], p // 5 * 5)
            b = buckets[key]
            b["total"] += 1
            # truncation-consistent slices: remaining after the last explicit node
            for d in range(0, 7):
                rem_d = [r for r in rows if r[0] > p + d]
                if not rem_d:
                    continue
                b["by_depth"][d]["gpu_ms"].append(
                    sum(max(0.1, r[2] - r[3]) for r in rem_d if r[1] == "gpu"))
                b["by_depth"][d]["cpu_ms"].append(
                    sum(r[2] for r in rem_d if r[1] != "gpu"))
            remaining = [r for r in rows if r[0] > p + 5]
            if not remaining:
                b["terminated"] += 1
                continue
            # simulator convention: runtime_p50 already includes the load;
            # pure compute = runtime - load; conditional load reported separately
            gpu_compute = sum(max(0.1, r[2] - r[3]) for r in remaining if r[1] == "gpu")
            gpu_load = sum(r[3] for r in remaining if r[1] == "gpu")
            cpu_ms = sum(r[2] for r in remaining if r[1] != "gpu")
            b["gpu_ms"].append(gpu_compute)
            b["load_ms"].append(gpu_load)
            b["cpu_ms"].append(cpu_ms)
            b["nodes_left"].append(len(remaining))

    table = {}
    for key, b in sorted(buckets.items()):
        stack, pos = key
        n_cont = len(b["gpu_ms"])
        MIN_CONTINUE = 30  # advisor P0-C: minimum continuation samples per bucket
        reliable = n_cont >= MIN_CONTINUE
        table[f"{stack}|pos{pos}"] = {
            "anchors": b["total"],
            "terminated_within_h5_frac": round(b["terminated"] / max(1, b["total"]), 3),
            "continued": n_cont,
            "reliable": reliable,
            "remaining_compute_ms_mean": round(statistics.fmean(b["gpu_ms"]), 1) if n_cont else None,
            "remaining_compute_ms_p50": round(statistics.median(b["gpu_ms"]), 1) if n_cont else None,
            "remaining_compute_ms_p90": round(sorted(b["gpu_ms"])[int(0.9 * (n_cont - 1))], 1) if n_cont else None,
            "conditional_load_ms_mean": round(statistics.fmean(b["load_ms"]), 1) if n_cont else None,
            "remaining_cpu_ms_mean": round(statistics.fmean(b["cpu_ms"]), 1) if n_cont else None,
            "nodes_left_mean": round(statistics.fmean(b["nodes_left"]), 2) if n_cont else None,
            "remaining_after_d": {
                str(d): (round(statistics.fmean(b["by_depth"][d]["gpu_ms"]), 1)
                         if b["by_depth"][d]["gpu_ms"] else None)
                for d in range(0, 7)
            },
            "continued_after_d": {str(d): len(b["by_depth"][d]["gpu_ms"]) for d in range(0, 7)},
        }
    out = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/tail_estimator_train_table.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "jct-milp-v1-tail-table-v1", "source": "train split only",
                               "table": table}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # compact print
    for key, row in list(table.items())[:18]:
        print(f"{key:28s} anchors={row['anchors']:4d} cont={row['continued']:4d} reliable={row['reliable']} "
              f"compute_mean={row['remaining_compute_ms_mean']} p90={row['remaining_compute_ms_p90']} "
              f"load_mean={row['conditional_load_ms_mean']} cpu_mean={row['remaining_cpu_ms_mean']}")
    print(f"... total rows: {len(table)} | evidence: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
