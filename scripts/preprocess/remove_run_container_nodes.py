"""Phase 19 step A — build the container-excluded R7 workload templates (v03).

`event_type == "run"` nodes are whole-run wall-clock containers: for all 640
templates the container's runtime equals the template's own workload total
(ratio median 0.982), and the simulator dispatches them as ordinary GPU nodes
(640/8935 nodes but 50.2% of total node runtime). They are terminal leaves with
no incoming edges, so removing them is a pure node deletion with no rewiring.

Read-only with respect to the legacy workload: writes only into the new v03
directory and records provenance in its summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CONTAINER_EVENT_TYPE = "run"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_containers(template: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    nodes = template.get("nodes") or []
    containers = [str(node["node_id"]) for node in nodes if node.get("event_type") == CONTAINER_EVENT_TYPE]
    remaining = [node for node in nodes if node.get("event_type") != CONTAINER_EVENT_TYPE]
    trimmed = dict(template)
    trimmed["nodes"] = remaining
    trimmed["node_count"] = len(remaining)
    return trimmed, containers


def verify(original: list[dict[str, Any]], trimmed: list[dict[str, Any]]) -> dict[str, Any]:
    removed = 0
    for source, target in zip(original, trimmed):
        nodes = target.get("nodes") or []
        if any(node.get("event_type") == CONTAINER_EVENT_TYPE for node in nodes):
            raise SystemExit(f"container survived in template {target.get('template_id')}")
        ids = {str(node["node_id"]) for node in nodes}
        for node in nodes:
            for predecessor in node.get("predecessor_node_ids") or []:
                if predecessor not in ids:
                    raise SystemExit(f"dangling predecessor {predecessor} in {target.get('template_id')}")
        if not [node for node in nodes if not (node.get("predecessor_node_ids") or [])]:
            raise SystemExit(f"template lost its root: {target.get('template_id')}")
        removed += len(source.get("nodes") or []) - len(nodes)

    remaining_nodes = sum(len(row.get("nodes") or []) for row in trimmed)
    runtime_all = sum(float(node.get("runtime_ms") or 0.0) for row in original for node in row.get("nodes") or [])
    runtime_removed = sum(
        float(node.get("runtime_ms") or 0.0)
        for row in original
        for node in row.get("nodes") or []
        if node.get("event_type") == CONTAINER_EVENT_TYPE
    )
    ratios = []
    for row in original:
        nodes = row.get("nodes") or []
        wall = max(
            [float(node.get("runtime_ms") or 0.0) for node in nodes if node.get("event_type") == CONTAINER_EVENT_TYPE]
            or [0.0]
        )
        if wall > 0:
            ratios.append(sum(float(node.get("runtime_ms") or 0.0) for node in nodes if node.get("event_type") != CONTAINER_EVENT_TYPE) / wall)
    return {
        "removed_event_type": CONTAINER_EVENT_TYPE,
        "removed_nodes": removed,
        "remaining_nodes": remaining_nodes,
        "templates": len(trimmed),
        "roots": sum(1 for row in trimmed for node in row.get("nodes") or [] if not (node.get("predecessor_node_ids") or [])),
        "dangling_predecessors": 0,
        "runtime_removed_ratio": (runtime_removed / runtime_all) if runtime_all else None,
        "workload_over_container_wall_clock_median": statistics.median(ratios) if ratios else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--templates", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.project_root
    templates = args.templates or root / "results" / "processed" / "r7_workload_20260817" / "job_templates_r7_v02.jsonl"
    out = args.out or (
        root
        / "results"
        / "processed"
        / "r7_workload_v03_no_run_container"
        / "job_templates_r7_v03.jsonl"
    )
    if not templates.exists():
        raise SystemExit(f"missing input: {templates}")
    out.parent.mkdir(parents=True, exist_ok=True)

    original = read_jsonl(templates)
    trimmed: list[dict[str, Any]] = []
    for row in original:
        template, _ = strip_containers(row)
        trimmed.append(template)
    report = verify(original, trimmed)
    write_jsonl(out, trimmed)
    summary = {
        "schema_version": "job-template-v0.2",
        "source": str(templates),
        "output": str(out),
        "derivation": "phase19 A: remove event_type=='run' whole-run containers (terminal leaves, no rewiring)",
        **report,
    }
    (out.parent / "job_templates_r7_v03.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report["removed_nodes"] == 0 or report["remaining_nodes"] == 0:
        raise SystemExit("template transform produced an empty result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
