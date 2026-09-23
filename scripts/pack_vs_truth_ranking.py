"""Rank each predictor against the TRUTH, not against the incumbent.

The earlier table measured agreement with A0, which is only a disruption proxy. What
matters for quality is whether a pack's consumed future-cost ordering is closer to the
truth-informed ordering.

For every S* anchor (an R7 template node) we compute
  * the truth-informed H=5 successor cost via limited_future_truth_cost
  * each pack's legacy consumed score (sum of per-step p95 + the legacy load rule)
and then report, against the truth ordering:
  * Spearman over all anchors
  * pairwise concordance within each template
  * top-1 agreement within each template

Residency caveat: the truth walk needs a GPU for its residency bookkeeping. The anchors
are a static list with no live scheduler state, so an empty-residency GPU at the v03
topology capacity is used for every anchor. That is a fixed proxy, not a per-decision
state, and it is stated here rather than hidden.
"""
from __future__ import annotations

import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(r"F:\scheduler")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

import j_series_common as common  # noqa: E402
from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    GPU,
    Job,
    limited_future_truth_cost,
    load_templates,
    train_resource_stats,
)

TEMPLATES = PROJECT_ROOT / "results/processed/r7_workload_v03_no_run_container/job_templates_r7_v03.jsonl"
ANCHORS = PROJECT_ROOT / "outputs/sstar_predictor_anchors/features_sstar.jsonl.gz"
PACKS = {
    "J3 (A0)": PROJECT_ROOT / "outputs/sstar_predictor_artifacts_dist_sched/prediction_artifacts/b05_future_h5.jsonl.gz",
    "R1b (A1)": PROJECT_ROOT / "outputs/resource_v2_artifacts/r1b/b05_future_h5.jsonl.gz",
    "R3a-U (A2)": PROJECT_ROOT / "outputs/resource_v2_artifacts/r3a_u/b05_future_h5.jsonl.gz",
    "F0_seed11": PROJECT_ROOT / "outputs/resource_v2_artifacts/f0_seed11/b05_future_h5.jsonl.gz",
}
HORIZON = 5
CAPACITY_MB = 32760.0


def load_pack(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {str(json.loads(line)["node_id"]): json.loads(line) for line in handle if line.strip()}


def legacy_score(row):
    scenarios = row.get("future_h%d" % HORIZON) or []
    if not scenarios:
        return None
    total = 0.0
    steps = (scenarios[0].get("steps") or [])[:HORIZON]
    if not steps:
        return None
    for step in steps:
        resource = step.get("resource") or {}
        p95 = (resource.get("runtime_ms_quantiles") or {}).get("p95")
        if not (isinstance(p95, (int, float)) and float(p95) > 0):
            return None
        load_ms = 0.0
        if step.get("execution_lane") == "gpu":
            occ = resource.get("load_occurrence_probability")
            dur = (resource.get("load_duration_ms_quantiles") or {}).get("p95")
            if isinstance(occ, (int, float)) and isinstance(dur, (int, float)) and float(occ) >= 0.5:
                load_ms = max(0.0, float(dur))
        total += float(p95) + load_ms
    return total


def spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for pos, i in enumerate(order):
            out[i] = float(pos)
        return out
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    templates = load_templates(TEMPLATES)
    stats = train_resource_stats(templates)
    print("templates:", len(templates))

    anchors = []
    with gzip.open(ANCHORS, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                anchors.append(json.loads(line))
    print("anchors:", len(anchors))

    packs = {name: load_pack(path) for name, path in PACKS.items() if path.is_file()}
    print("packs:", {k: len(v) for k, v in packs.items()})

    # truth-informed H=5 successor cost per anchor, with a fixed empty-residency GPU
    truth = {}
    unresolved = 0
    for row in anchors:
        node_id = str(row["current_node_id"])
        template_id = node_id.split(":", 1)[0]
        template = templates.get(template_id)
        if template is None or node_id not in template.by_id:
            unresolved += 1
            continue
        gpu = GPU(index=0, capacity_mb=CAPACITY_MB)
        job = Job(job_instance_id="anchor", template=template, arrival_ms=0.0,
                  deadline_ms=None, service_class="normal")
        truth[node_id] = float(limited_future_truth_cost(job, node_id, gpu, stats, HORIZON))
    print("truth computed: %d | anchor not resolvable to a template node: %d" % (len(truth), unresolved))
    print()

    scores = {}
    for name, pack in packs.items():
        values = {}
        for node_id in truth:
            row = pack.get(node_id)
            if row is None:
                continue
            value = legacy_score(row)
            if value is not None:
                values[node_id] = value
        scores[name] = values

    shared = sorted(set(truth) & set.intersection(*(set(v) for v in scores.values())))
    print("nodes with truth AND every pack's score:", len(shared))
    print()
    print("=== agreement with the TRUTH-informed ordering (higher is better) ===")
    print("%-14s %10s %14s %14s" % ("pack", "spearman", "within-template", "top-1 agree"))
    truth_vec = [truth[k] for k in shared]
    for name, values in scores.items():
        rho = spearman(truth_vec, [values[k] for k in shared])

        by_template = defaultdict(list)
        for k in shared:
            by_template[k.split(":", 1)[0]].append(k)
        concordant = pairs = 0
        top1 = 0
        groups = 0
        for _t, keys in by_template.items():
            if len(keys) < 2:
                continue
            groups += 1
            tv = [truth[k] for k in keys]
            pv = [values[k] for k in keys]
            best = keys[int(np.argmin(tv))]
            top1 += int(keys[int(np.argmin(pv))] == best)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    pairs += 1
                    concordant += int((tv[i] - tv[j]) * (pv[i] - pv[j]) > 0)
        print("%-14s %10.4f %14.4f %14.3f"
              % (name, rho, concordant / max(1, pairs), top1 / max(1, groups)))

    print()
    print("=== scale of the consumed score (for context) ===")
    for name, values in scores.items():
        ordered = sorted(values[k] for k in shared)
        print("  %-14s median=%9.1f  mean=%9.1f" % (name, statistics.median(ordered), sum(ordered) / len(ordered)))

    out = PROJECT_ROOT / "experiments/EXP-20260921_histres_causal_input_v1/artifacts/pack_vs_truth_ranking.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "nodes_compared": len(shared),
        "anchors_not_resolvable": unresolved,
        "residency_caveat": "empty-residency GPU at the v03 topology capacity for every anchor",
        "agreement_with_truth": {name: {"spearman": spearman(truth_vec, [values[k] for k in shared])}
                                 for name, values in scores.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
