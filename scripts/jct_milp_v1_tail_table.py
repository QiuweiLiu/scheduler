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

ROOT = Path("/Volumes/Lenovo/scheduler")
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

    # per (stack, position bucket): remaining beyond p+5
    buckets: dict = defaultdict(lambda: {"gpu_ms": [], "cpu_ms": [], "terminated": 0, "total": 0, "nodes_left": []})
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
            remaining = [r for r in rows if r[0] > p + 5]
            key = (meta[str(tid)], p // 5 * 5)
            b = buckets[key]
            b["total"] += 1
            if not remaining:
                b["terminated"] += 1
                continue
            gpu_ms = sum(r[2] + r[3] for r in remaining if r[1] == "gpu")
            cpu_ms = sum(r[2] for r in remaining if r[1] != "gpu")
            b["gpu_ms"].append(gpu_ms)
            b["cpu_ms"].append(cpu_ms)
            b["nodes_left"].append(len(remaining))

    table = {}
    for key, b in sorted(buckets.items()):
        stack, pos = key
        n_cont = len(b["gpu_ms"])
        table[f"{stack}|pos{pos}"] = {
            "anchors": b["total"],
            "terminated_within_h5_frac": round(b["terminated"] / max(1, b["total"]), 3),
            "continued": n_cont,
            "remaining_gpu_ms_mean": round(statistics.fmean(b["gpu_ms"]), 1) if n_cont else None,
            "remaining_gpu_ms_p50": round(statistics.median(b["gpu_ms"]), 1) if n_cont else None,
            "remaining_gpu_ms_p90": round(sorted(b["gpu_ms"])[int(0.9 * (n_cont - 1))], 1) if n_cont else None,
            "remaining_cpu_ms_mean": round(statistics.fmean(b["cpu_ms"]), 1) if n_cont else None,
            "nodes_left_mean": round(statistics.fmean(b["nodes_left"]), 2) if n_cont else None,
        }
    out = ROOT / "experiments/EXP-20261015_jct_milp_v1/artifacts/tail_estimator_train_table.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "jct-milp-v1-tail-table-v1", "source": "train split only",
                               "table": table}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # compact print
    for key, row in list(table.items())[:18]:
        print(f"{key:28s} anchors={row['anchors']:4d} terminated={row['terminated_within_h5_frac']:.2f} "
              f"gpu_mean={row['remaining_gpu_ms_mean']} p90={row['remaining_gpu_ms_p90']} cpu_mean={row['remaining_cpu_ms_mean']}")
    print(f"... total rows: {len(table)} | evidence: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
