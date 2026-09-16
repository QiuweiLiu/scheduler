#!/usr/bin/env python3
"""Build C1/C2 consumption-channel artifacts from frozen J predictor outputs.

Variants (all single-scenario, sched-compatible b05_* names):
- j_lenonly:        steps keep lane but model_id="unknown" (lane-average cost only)
- j_expected:       per-step expected_cost = Σ_m P(model) * table_cost(model, lane(m))
- j_oracle_content: truth identities (next true events), count = predicted length
- j_surv:           fixed-5 steps + length_probabilities (survival weighting in policy)

Inference-only; no fitting. Truth is used solely to build the oracle-content control.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tracing.analysis.workload_v02_simulator import (  # noqa: E402
    load_templates,
    train_resource_stats,
    _step_estimate_cost,
)

builder_spec = importlib.util.spec_from_file_location("p9d_builder", ROOT / "scripts" / "build_p9d_topology_dataset.py")
builder = importlib.util.module_from_spec(builder_spec)
assert builder_spec and builder_spec.loader
builder_spec.loader.exec_module(builder)  # type: ignore[union-attr]

BASE = ROOT / "outputs" / "sstar_predictor_artifacts"  # variable length, no distributions
DIST = ROOT / "outputs" / "sstar_predictor_artifacts_dist"  # fixed5 + distributions
RUNS = ROOT / "results" / "raw" / "r7_trace_full_20260817"
TEMPLATES = ROOT / "results" / "processed" / "r7_workload_20260817" / "job_templates_r7_v02.jsonl"
OUT = ROOT / "outputs" / "c1c2"


def lane_for(model: str) -> str:
    return "cpu" if model.startswith("cpu") or model == "finish_argument" else "gpu"


def read_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_rows(path: Path, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def emit_dir(name: str, rows_h5: List[Dict[str, Any]], extra: Dict[str, Any] | None = None) -> None:
    dst = OUT / name / "prediction_artifacts"
    dst.mkdir(parents=True, exist_ok=True)
    # h1/h3 empty-safety: reuse h5 content (the classic policy reads h{horizon}; predopt_h5 reads future_h5)
    write_rows(dst / "b05_future_h5.jsonl.gz", rows_h5)
    write_rows(dst / "b05_future_h3.jsonl.gz", rows_h5)
    write_rows(dst / "b05_node_h1.jsonl.gz", rows_h5)
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in dst.glob("b05_*")}
    manifest = {"schema_version": f"c1c2-{name}-manifest-v1", "variant": name, "files": hashes}
    if extra:
        manifest.update(extra)
    (dst / "b05_artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("emitted", dst, "rows", len(rows_h5))


def main() -> int:
    templates = load_templates(TEMPLATES)
    train_stats = train_resource_stats(templates)

    # ---------- j_lenonly ----------
    lenonly = []
    for row in read_rows(BASE / "j_future_h5.jsonl.gz"):
        row = dict(row)
        scenario = dict((row.get("future_h5") or [{}])[0])
        steps = []
        for step in scenario.get("steps") or []:
            step = dict(step)
            step["model_id"] = "unknown"
            steps.append(step)
        scenario["steps"] = steps
        row["future_h5"] = [scenario]
        lenonly.append(row)
    emit_dir("j_lenonly", lenonly)

    # ---------- j_expected ----------
    expected = []
    for row in read_rows(DIST / "j_future_h5.jsonl.gz"):
        row = dict(row)
        scenario = dict((row.get("future_h5") or [{}])[0])
        steps = []
        for step in scenario.get("steps") or []:
            step = dict(step)
            probs = dict(step.get("model_probabilities") or {})
            top = str(step.get("model_id") or "unknown")
            if not probs:
                probs = {top: 1.0}
            total = sum(probs.values())
            if total < 1.0:
                probs[top] = probs.get(top, 0.0) + (1.0 - total)
            expected_cost = 0.0
            for model, probability in probs.items():
                probe = {"model_id": model, "execution_lane": lane_for(model)}
                expected_cost += float(probability) * _step_estimate_cost(probe, train_stats)
            step["expected_cost"] = expected_cost
            steps.append(step)
        scenario["steps"] = steps
        row["future_h5"] = [scenario]
        expected.append(row)
    emit_dir("j_expected", expected, {"transform": "per-step expected_cost = sum_m P(model) * table_cost(model, lane(model))"})

    # ---------- j_surv ----------
    surv = []
    for row in read_rows(DIST / "j_future_h5.jsonl.gz"):
        row = dict(row)
        probabilities = row.get("length_probabilities") or []
        scenario = dict((row.get("future_h5") or [{}])[0])
        steps = []
        for h, step in enumerate(scenario.get("steps") or [], start=1):
            step = dict(step)
            step["survival_probability"] = sum(float(p) for p in probabilities[h:]) if probabilities else 1.0
            steps.append(step)
        scenario["steps"] = steps
        row["future_h5"] = [scenario]
        surv.append(row)
    emit_dir("j_surv", surv, {"transform": "fixed-5 steps with per-step survival_probability = P(T>=h)"})

    # ---------- j_oracle_content ----------
    # truth map: event_id -> (events, index)
    truth_steps: Dict[str, List[Dict[str, Any]]] = {}
    for run in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("._")):
        trace = run / "trace.jsonl"
        if not trace.is_file():
            continue
        events = []
        with trace.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        for index, event in enumerate(events):
            event_id = str(event.get("event_id"))
            nxt = []
            for follow in events[index + 1: index + 6]:
                model = str(follow.get("model_id") or "unknown")
                nxt.append(
                    {
                        "role": builder.role_for_event(follow),
                        "action_family": builder.family_for_event(follow),
                        "model_id": model,
                        "role_probability": 1.0,
                        "family_probability": 1.0,
                        "model_probability": 1.0,
                        "execution_lane": lane_for(model),
                        "prototype_source": "oracle_content",
                    }
                )
            truth_steps[event_id] = nxt
    oracle = []
    for row in read_rows(BASE / "j_future_h5.jsonl.gz"):
        row = dict(row)
        node_id = str(row["node_id"])
        scenario = dict((row.get("future_h5") or [{}])[0])
        predicted = scenario.get("steps") or []
        truth = truth_steps.get(node_id) or []
        use = min(len(predicted), len(truth))
        steps = []
        for t in range(use):
            step = dict(truth[t])
            step["step_offset"] = t + 1
            steps.append(step)
        scenario["steps"] = steps
        scenario["topology_source"] = "oracle_content_same_length"
        row["future_h5"] = [scenario]
        oracle.append(row)
    emit_dir("j_oracle_content", oracle, {"transform": "truth identities, same per-anchor step count as predicted length"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
