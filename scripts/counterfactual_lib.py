"""Future-information counterfactual transforms (exp D).

Frozen design: docs/research/2026-10-10_exp_d_counterfactual.md

Two counterfactual cells over the SAME confirm300 episodes / arrivals /
capacities, to test whether the main line's residency-action gain depends on
model-switching structure:

* ALL-RESIDENT: capacity pinned large enough that every model fits everywhere
  -> loads happen once, evictions/prefetch go idle.
* SINGLE-MODEL: all LLM GPU nodes relabeled to one model (Qwen3-VL-8B, whose
  role coverage is a superset of 3B/4B), memory/load normalized to the 8B
  records, pack future-model distributions collapsed onto the same model
  (CPU/finish mass preserved) -> model identity carries no switching structure.

Runtime predictions and chain-length distributions are deliberately LEFT AS
RECORDED (disclosure: the counterfactuals remove identity/switching structure,
not the recorded work amounts).  yolo11x.pt nodes keep their identity (outside
the covered global-model set; a 512 MB, non-switching deployment).
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from pressure_grid_lib import build_pressure_cell  # noqa: E402

SINGLE_MODEL_ID = "Qwen3-VL-8B-Instruct"
LLM_MODELS = ("Qwen2.5-VL-3B-Instruct", "Qwen3-4B", "Qwen3-VL-8B-Instruct")
ALL_RESIDENT_CAPACITY_MB = (131072.0, 131072.0)


def build_all_resident_cell(episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """natural alpha=1.0 cell (cap3232) with capacity raised to 128 GB/device."""

    out = build_pressure_cell(episodes, 1.0, "cap3232")
    for episode in out:
        episode["gpu_topology_mb"] = [float(value) for value in ALL_RESIDENT_CAPACITY_MB]
        episode["pressure_capacity_key"] = "cap131072"
        episode["pressure_capacity_mb"] = [float(value) for value in ALL_RESIDENT_CAPACITY_MB]
        episode["counterfactual"] = "all_resident_v1"
    return out


def single_model_transform(
    templates: Mapping[str, Any],
    episodes: Sequence[Dict[str, Any]],
    pack: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Relabel every LLM GPU node (templates + episode hints + pack) to one model."""

    memories = [
        float(node.resident_model_mb)
        for template in templates.values()
        for node in template.nodes
        if node.model_id == SINGLE_MODEL_ID and node.resident_model_mb is not None
    ]
    loads = [
        float(node.load_ms)
        for template in templates.values()
        for node in template.nodes
        if node.model_id == SINGLE_MODEL_ID and node.load_ms is not None and float(node.load_ms) > 0.0
    ]
    if not memories:
        raise ValueError("single-model transform found no records for %s" % SINGLE_MODEL_ID)
    target_memory = statistics.median(memories)
    target_load = statistics.median(loads) if loads else 0.0

    new_templates: Dict[str, Any] = {}
    for template_id, template in templates.items():
        changed = False
        nodes = []
        for node in template.nodes:
            if node.lane == "gpu" and node.model_id in LLM_MODELS and node.model_id != SINGLE_MODEL_ID:
                nodes.append(replace(
                    node,
                    model_id=SINGLE_MODEL_ID,
                    resident_model_mb=target_memory,
                    load_ms=target_load,
                ))
                changed = True
            else:
                nodes.append(node)
        if changed:
            new_nodes = tuple(nodes)
            new_templates[template_id] = replace(
                template, nodes=new_nodes, by_id={node.node_id: node for node in new_nodes}
            )
        else:
            new_templates[template_id] = template

    new_episodes: List[Dict[str, Any]] = []
    for episode in episodes:
        new_episode = dict(episode)
        hints = episode.get("initial_residency_hint")
        if hints:
            rebuilt_hints: List[List[str]] = []
            rebuilt_memory: List[List[float]] = []
            memory_rows = episode.get("initial_residency_memory_mb") or []
            for gpu_index, row in enumerate(hints):
                mem_row = memory_rows[gpu_index] if gpu_index < len(memory_rows) else []
                merged: List[str] = []
                rebuilt_row: List[float] = []
                for position, model in enumerate(row):
                    mapped = SINGLE_MODEL_ID if model in LLM_MODELS else str(model)
                    if mapped in merged:
                        continue
                    merged.append(mapped)
                    if mapped == SINGLE_MODEL_ID:
                        rebuilt_row.append(target_memory)
                    else:
                        value = float(mem_row[position]) if position < len(mem_row) else 0.0
                        rebuilt_row.append(value)
                rebuilt_hints.append(merged)
                rebuilt_memory.append(rebuilt_row)
            new_episode["initial_residency_hint"] = rebuilt_hints
            new_episode["initial_residency_memory_mb"] = rebuilt_memory
        new_episode["counterfactual"] = "single_model_v1"
        new_episodes.append(new_episode)

    new_pack: Dict[str, Dict[str, Any]] = {}
    for key, row in pack.items():
        new_row = dict(row)
        scenarios = row.get("future_h5") or []
        if scenarios:
            steps = []
            for step in scenarios[0].get("steps") or []:
                distribution = step.get("model_probabilities") or {}
                if distribution:
                    merged: Dict[str, float] = {}
                    for model, probability in distribution.items():
                        mapped = SINGLE_MODEL_ID if model in LLM_MODELS else str(model)
                        merged[mapped] = merged.get(mapped, 0.0) + float(probability)
                    step = {**step, "model_probabilities": merged}
                steps.append(step)
            new_row["future_h5"] = [{**scenarios[0], "steps": steps}]
        new_pack[str(key)] = new_row
    return new_templates, new_episodes, new_pack


def build_single_model_cell(
    templates: Mapping[str, Any],
    episodes: Sequence[Dict[str, Any]],
    pack: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """natural alpha=1.0 / cap3232 cell + single-model relabel."""

    cell_episodes = build_pressure_cell(episodes, 1.0, "cap3232")
    for episode in cell_episodes:
        episode["counterfactual"] = "single_model_v1"
    new_templates, new_episodes, new_pack = single_model_transform(templates, cell_episodes, pack)
    return new_templates, new_episodes, new_pack
