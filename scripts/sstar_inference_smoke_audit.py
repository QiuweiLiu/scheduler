#!/usr/bin/env python3
"""S_* inference smoke: compare frozen J3 predictions against S_* execution truth.

Inference/audit only: no fitting, no selection, no tuning. Uses the packed
artifacts (j_future_h5) and the raw S_* traces for truth.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

builder_spec = importlib.util.spec_from_file_location("p9d_builder", ROOT / "scripts" / "build_p9d_topology_dataset.py")
builder = importlib.util.module_from_spec(builder_spec)
assert builder_spec and builder_spec.loader
builder_spec.loader.exec_module(builder)  # type: ignore[union-attr]

TAUS = (0.50, 0.90, 0.95)


def pinball(y: float, q: float, tau: float) -> float:
    diff = y - q
    return max(tau * diff, (tau - 1.0) * diff)


def main() -> int:
    runs_root = ROOT / "results" / "raw" / "r7_trace_full_20260817"
    preds_path = ROOT / "outputs" / "sstar_predictor_artifacts" / "j_future_h5.jsonl.gz"
    out_path = ROOT / "outputs" / "sstar_predictor_artifacts" / "sstar_audit.json"

    preds: Dict[str, Dict[str, Any]] = {}
    with gzip.open(preds_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            preds[str(row["node_id"])] = row

    stats = defaultdict(list)
    by_group = defaultdict(lambda: defaultdict(list))
    by_baseline = defaultdict(lambda: defaultdict(list))
    n_anchors = 0
    for run in sorted(p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("._")):
        trace = run / "trace.jsonl"
        if not trace.is_file():
            continue
        events: List[Dict[str, Any]] = []
        with trace.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        run_manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        baseline = str(run_manifest.get("baseline") or "unknown")
        group = "s_val" if "s_val" in str(run_manifest.get("task_id") or "") else "s_train"
        for index, event in enumerate(events[:-1]):
            key = str(event.get("event_id"))
            pred = preds.get(key)
            if not pred:
                continue
            truth = events[index + 1: index + 6]
            steps = pred["future_h5"][0]["steps"]
            n_anchors += 1
            pred_len = len(steps)
            truth_len = len(truth)

            def add(name: str, value: float) -> None:
                stats[name].append(value)
                by_group[group][name].append(value)
                by_baseline[baseline][name].append(value)

            add("length_abs_error", abs(pred_len - truth_len))
            if steps and truth:
                role_ok = int(steps[0]["role"] == builder.role_for_event(truth[0]))
                fam_ok = int(steps[0]["action_family"] == builder.family_for_event(truth[0]))
                add("next_role_acc", float(role_ok))
                add("next_family_acc", float(fam_ok))
            for t, truth_event in enumerate(truth):
                resource = truth_event.get("resource") or {}
                runtime = resource.get("runtime_ms")
                load = resource.get("load_ms")
                if t >= pred_len:
                    continue
                step = steps[t]
                if runtime is not None and float(runtime) > 0:
                    q = step["resource"]["runtime_ms_quantiles"]
                    preds_q = [q["p50"], q["p90"], q["p95"]]
                    add("runtime_pinball_mean", float(np.mean([pinball(float(runtime), preds_q[k], TAUS[k]) for k in range(3)])))
                    add("runtime_p50_abs_error", abs(preds_q[0] - float(runtime)))
                if load is not None:
                    occ_pred = float(step["resource"]["load_occurrence_probability"])
                    add("load_occ_brier", (occ_pred - (1.0 if float(load) > 0 else 0.0)) ** 2)
                    if float(load) > 0:
                        qd = step["resource"]["load_duration_ms_quantiles"]
                        add("load_dur_p50_abs_error", abs(qd["p50"] - float(load)))

    def summarize(container: Mapping[str, List[float]]) -> Dict[str, Any]:
        out = {}
        for key, values in sorted(container.items()):
            arr = np.asarray(values, dtype=float)
            out[key] = {"n": int(arr.size), "mean": float(arr.mean()) if arr.size else None}
        return out

    report = {
        "schema_version": "sstar-inference-smoke-v1",
        "anchors_evaluated": n_anchors,
        "overall": summarize(stats),
        "by_group": {group: summarize(values) for group, values in sorted(by_group.items())},
        "by_baseline": {baseline: summarize(values) for baseline, values in sorted(by_baseline.items())},
        "notes": [
            "inference/audit only; no fitting, no selection",
            "truth = next events in trace file order (cap 5); roles/families derived with the P9d builder functions",
            "load_dur error only on steps with load>0; runtime pinball on steps with runtime>0",
        ],
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"anchors": n_anchors, "overall": report["overall"]}, ensure_ascii=False, indent=1)[:1800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
