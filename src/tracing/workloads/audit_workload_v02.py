#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_CELLS = 3 * 5 * 3 * 3
EXPECTED_JOB_COUNTS = {16, 32, 64}
ALLOWED_EDGE_PROVENANCE = {"root", "raw_parent_step_ids", "same_step_event_order", "sequence_fallback_no_parent_data"}

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def finite_nonnegative(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= 0.0
    except (TypeError, ValueError):
        return False

def audit_templates(rows: list[dict[str, Any]]):
    errors = []
    edge_counts = Counter()
    split_counts = Counter()
    ids = [str(row.get("template_id")) for row in rows]
    for template_id, count in Counter(ids).items():
        if count != 1:
            errors.append(f"duplicate_template_id:{template_id}")
    for template in rows:
        template_id = str(template.get("template_id"))
        split_counts[str(template.get("split"))] += 1
        nodes = template.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            errors.append(f"{template_id}:empty_nodes")
            continue
        node_ids = [str(node.get("node_id")) for node in nodes]
        node_map = {node_id: node for node_id, node in zip(node_ids, nodes)}
        if len(node_ids) != len(set(node_ids)):
            errors.append(f"{template_id}:duplicate_node_id")
        for index, node in enumerate(nodes):
            provenance = str(node.get("edge_provenance"))
            edge_counts[provenance] += 1
            if provenance not in ALLOWED_EDGE_PROVENANCE:
                errors.append(f"{template_id}:{node.get('node_id')}:forbidden_edge_provenance:{provenance}")
            for predecessor in node.get("predecessor_node_ids") or []:
                predecessor = str(predecessor)
                if predecessor not in node_map:
                    errors.append(f"{template_id}:{node.get('node_id')}:missing_predecessor:{predecessor}")
                elif int(node_map[predecessor].get("sequence_index", -1)) >= int(node.get("sequence_index", index)):
                    errors.append(f"{template_id}:{node.get('node_id')}:non_forward_predecessor:{predecessor}")
        roots = [node for node in nodes if not (node.get("predecessor_node_ids") or [])]
        if not roots:
            errors.append(f"{template_id}:no_root")
        reachable = {str(node.get("node_id")) for node in roots}
        changed = True
        while changed:
            changed = False
            for node in nodes:
                node_id = str(node.get("node_id"))
                if node_id in reachable:
                    continue
                if all(str(parent) in reachable for parent in node.get("predecessor_node_ids") or []):
                    reachable.add(node_id)
                    changed = True
        if len(reachable) != len(nodes):
            errors.append(f"{template_id}:unreachable_nodes:{len(nodes)-len(reachable)}")
    return errors, edge_counts, dict(sorted(split_counts.items()))

def audit_episodes(episode_paths, template_map, expected_contract, allow_incomplete_matrix):
    errors = []
    warnings = []
    all_cells = Counter()
    split_counts = Counter()
    job_count_counts = Counter()
    total_jobs = 0
    for path in episode_paths:
        rows = read_jsonl(path)
        if not rows:
            errors.append(f"{path}:empty")
            continue
        for episode in rows:
            episode_id = str(episode.get("episode_id"))
            split = str(episode.get("split"))
            split_counts[split] += 1
            jobs = episode.get("jobs")
            if not isinstance(jobs, list) or len(jobs) not in EXPECTED_JOB_COUNTS:
                errors.append(f"{episode_id}:job_count:{len(jobs) if isinstance(jobs, list) else 'invalid'}")
                continue
            total_jobs += len(jobs)
            job_count_counts[len(jobs)] += 1
            all_cells[str(episode.get("scenario_cell"))] += 1
            if path.stem.split("_")[0] in {"train", "validation", "test"} and split != path.stem.split("_")[0]:
                errors.append(f"{episode_id}:file_split_mismatch")
            if not finite_nonnegative(episode.get("episode_window_ms")):
                errors.append(f"{episode_id}:invalid_episode_window")
            target = episode.get("target_offered_compute_load")
            realized = episode.get("realized_offered_compute_load")
            if not finite_nonnegative(target) or not finite_nonnegative(realized):
                errors.append(f"{episode_id}:missing_realized_load")
            elif abs(float(target) - float(realized)) > 1e-4:
                errors.append(f"{episode_id}:realized_load_mismatch:{target}:{realized}")
            arrivals = []
            seen_templates = set()
            for job in jobs:
                template_id = str(job.get("template_id"))
                if template_id in seen_templates:
                    errors.append(f"{episode_id}:template_reuse:{template_id}")
                seen_templates.add(template_id)
                template = template_map.get(template_id)
                if template is None:
                    errors.append(f"{episode_id}:unknown_template:{template_id}")
                elif str(template.get("split")) != split:
                    errors.append(f"{episode_id}:{template_id}:split_leak:{template.get('split')}->{split}")
                arrival = job.get("arrival_ms")
                if not finite_nonnegative(arrival):
                    errors.append(f"{episode_id}:{template_id}:invalid_arrival")
                else:
                    arrivals.append(float(arrival))
                if not finite_nonnegative(job.get("deadline_ms")) or float(job.get("deadline_ms")) < float(arrival or 0.0):
                    errors.append(f"{episode_id}:{template_id}:invalid_deadline")
                visible = job.get("scheduler_visible") or {}
                for field in ("predicted_job_runtime_p50_ms", "predicted_job_runtime_p90_ms", "predicted_gpu_runtime_p50_ms"):
                    if not finite_nonnegative(visible.get(field)):
                        errors.append(f"{episode_id}:{template_id}:missing_scheduler_visible:{field}")
            if arrivals != sorted(arrivals):
                errors.append(f"{episode_id}:arrivals_not_monotone")
            topology = episode.get("gpu_topology_mb") or []
            residency = episode.get("initial_residency_memory_mb") or []
            if len(topology) != len(residency):
                errors.append(f"{episode_id}:residency_topology_length")
            for gpu_index, (capacity, memory) in enumerate(zip(topology, residency)):
                if not finite_nonnegative(capacity) or not isinstance(memory, list):
                    errors.append(f"{episode_id}:gpu_{gpu_index}:invalid_residency")
                elif sum(float(value) for value in memory if finite_nonnegative(value)) > float(capacity) * 0.70 + 1e-6:
                    errors.append(f"{episode_id}:gpu_{gpu_index}:residency_capacity")
            metadata = episode.get("arrival_metadata") or {}
            if not metadata.get("scale_to_episode_window_ms"):
                errors.append(f"{episode_id}:arrival_not_scaled_with_metadata")
            if expected_contract is not None:
                binding = episode.get("resource_contract") or {}
                if binding.get("schema_version") != expected_contract["schema_version"] or binding.get("sha256") != expected_contract["sha256"]:
                    errors.append(f"{episode_id}:resource_contract_mismatch")
    expected_complete = len(all_cells) == EXPECTED_CELLS and all(value > 0 for value in all_cells.values())
    if not expected_complete:
        message = f"scenario_matrix_incomplete:{len(all_cells)}/{EXPECTED_CELLS}"
        if allow_incomplete_matrix:
            warnings.append(message)
        else:
            errors.append(message)
    details = {
        "episodes": sum(split_counts.values()),
        "jobs": total_jobs,
        "split_counts": dict(sorted(split_counts.items())),
        "job_count_counts": dict(sorted(job_count_counts.items())),
        "scenario_cells": len(all_cells),
        "scenario_cell_counts": dict(sorted(all_cells.items())),
        "scenario_matrix_expected_cells": EXPECTED_CELLS,
        "scenario_matrix_complete": expected_complete,
    }
    return errors, warnings, details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--resource-contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete-matrix", action="store_true")
    args = parser.parse_args()
    templates = read_jsonl(args.templates)
    template_errors, edge_counts, split_counts = audit_templates(templates)
    template_map = {str(row.get("template_id")): row for row in templates}
    expected_contract = None
    if args.resource_contract:
        contract = json.loads(args.resource_contract.read_text(encoding="utf-8"))
        expected_contract = {"schema_version": contract.get("schema_version"), "sha256": sha256(args.resource_contract)}
    episode_errors, warnings, episode_details = audit_episodes(
        args.episodes, template_map, expected_contract, args.allow_incomplete_matrix
    )
    summary = {
        "schema_version": "workload-audit-v0.1",
        "templates": len(templates),
        "template_split_counts": split_counts,
        "template_edge_provenance_counts": dict(sorted(edge_counts.items())),
        "template_errors": template_errors,
        "episode_errors": episode_errors,
        "warnings": warnings,
        "episode_details": episode_details,
        "gate": not template_errors and not episode_errors,
        "incomplete_matrix_allowed": args.allow_incomplete_matrix,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["gate"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
