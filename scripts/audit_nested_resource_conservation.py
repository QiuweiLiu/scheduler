"""P0: preserve the resource attribution of merged nested calls.

The review found that all 169 merged nested calls are GPU work while all 169 parents
are CPU nodes, so deleting the nested node and keeping the CPU parent makes real GPU
demand vanish from GPU contention.

v3.1 fixes the node's DURATION at R_total(parent) and forbids counting the nested
resource again, but its composite signature still records the inner model class so the
resource can be attributed.  This patch carries that information forward:

    nested_model_class   the inner GPU model
    nested_model_mb      its measured peak allocation
    nested_load_ms       its measured load cost
    nested_runtime_ms    its own interval, kept for audit only

and then audits conservation: the GPU memory demand present before the merge must still
be present after it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

ROOT = Path(r"F:\scheduler")
RAW = ROOT / "results/raw/r7_trace_full_20260817"
V03 = ROOT / "results/processed/r7_workload_v03_no_run_container/job_templates_r7_v03.jsonl"
V041 = ROOT / "results/processed/r7_workload_v041_ontology_no_run_container/job_templates_r7_v041.jsonl"


def load_trace(run_id: str) -> Dict[str, Mapping[str, Any]]:
    p = RAW / run_id / "trace.jsonl"
    if not p.is_file():
        return {}
    return {str(json.loads(l).get("event_id")): json.loads(l)
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def nested_resources(event: Mapping[str, Any]) -> Dict[str, Any]:
    res = event.get("resource") or {}
    return {
        "nested_model_class": str(event.get("model_id") or ""),
        "nested_model_mb": res.get("peak_allocated_mb"),
        "nested_reserved_mb": res.get("peak_reserved_mb"),
        "nested_load_ms": res.get("load_ms"),
        "nested_runtime_ms": res.get("runtime_ms"),
    }


def main() -> int:
    with open(V041, "rt", encoding="utf-8") as h:
        wl = [json.loads(l) for l in h if l.strip()]
    with open(V03, "rt", encoding="utf-8") as h:
        old = {str(json.loads(l)["template_id"]): json.loads(l)
               for l in h if l.strip()}

    # ---- 1. carry the nested resource forward -------------------------------- #
    enriched = 0
    missing = 0
    per_template = []
    for r in wl:
        run_id = str(r.get("run_id") or "")
        trace = load_trace(run_id)
        for n in r["nodes"]:
            if not n.get("nested_calls"):
                continue
            for m in n["nested_calls"]:
                ev = trace.get(str(m))
                if ev is None:
                    missing += 1
                    continue
                n.update(nested_resources(ev))
                enriched += 1
        per_template.append(r)

    # ---- 2. conservation audit: GPU memory demand before vs after ----------- #
    before_gpu_mb = 0.0
    after_gpu_mb = 0.0
    for r in wl:
        o = old.get(str(r["template_id"]))
        if o is None:
            continue
        for n in o["nodes"]:
            if str(n.get("execution_lane")) != "gpu":
                continue
            mb = n.get("workspace_peak_mb")
            if isinstance(mb, (int, float)):
                before_gpu_mb += float(mb)
        for n in r["nodes"]:
            if str(n.get("execution_lane")) == "gpu":
                mb = n.get("workspace_peak_mb")
                if isinstance(mb, (int, float)):
                    after_gpu_mb += float(mb)
            nmb = n.get("nested_model_mb")
            if isinstance(nmb, (int, float)):
                after_gpu_mb += float(nmb)

    print("=" * 88)
    print("nested resource attribution")
    print("=" * 88)
    print("  parents enriched with nested resources : %d" % enriched)
    print("  nested events missing from raw trace   : %d" % missing)
    print()
    print("  GPU-side memory demand (sum over the corpus)")
    print("    before the merge (v03)               : %14.1f MB" % before_gpu_mb)
    print("    after  the merge (v04.1 + nested)    : %14.1f MB" % after_gpu_mb)
    print("    difference                           : %14.1f MB (%.4f%%)"
          % (after_gpu_mb - before_gpu_mb,
             100.0 * (after_gpu_mb - before_gpu_mb) / max(1.0, before_gpu_mb)))

    lanes = Counter()
    for r in wl:
        for n in r["nodes"]:
            if n.get("nested_model_class"):
                lanes[(str(n.get("execution_lane")), str(n.get("nested_model_class")))] += 1
    print()
    print("  (parent lane, nested model) pairs:")
    for k, v in lanes.most_common():
        print("     %-64s %d" % (str(k), v))

    out = ROOT / "experiments/EXP-20260921_scheduler_replication_v1/artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "nested_resource_conservation.json").write_text(json.dumps({
        "parents_enriched": enriched,
        "nested_missing_from_raw": missing,
        "gpu_memory_mb_before_merge": before_gpu_mb,
        "gpu_memory_mb_after_merge": after_gpu_mb,
        "delta_mb": after_gpu_mb - before_gpu_mb,
        "note": "the merged parent keeps the CPU lane and the R_total duration, and now also "
                "carries the inner model class and its measured allocation so the GPU demand is "
                "not lost from contention accounting",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote nested_resource_conservation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
