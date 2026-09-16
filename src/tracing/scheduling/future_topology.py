"""Schema helpers for predicted finite-horizon DAG-layer artifacts.

The legacy future artifact represents one predicted dispatch event per step.
This module defines the opt-in layer representation used by the repaired H5
contract: one scenario contains ordered layers and each layer may contain
multiple predicted node prototypes.  The representation is deliberately
identity-free; a predicted artifact must not carry target-template node IDs or
successor edges.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


LAYER_H5_SCHEMA_VERSION = "scheduling-future-h5-layer-v1"


def validate_layer_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    horizon: int = 5,
) -> None:
    """Validate probabilities, contiguous layer offsets and safe identities."""

    if not isinstance(scenarios, (list, tuple)):
        raise ValueError("future_h5_layers must be a list of scenarios")
    if horizon < 0:
        raise ValueError("layer horizon must be non-negative")
    if not scenarios:
        return

    probabilities: list[float] = []
    scenario_ids: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            raise ValueError("future_h5_layers scenario must be an object")
        scenario_id = str(scenario.get("scenario_id") or "")
        if not scenario_id or scenario_id in scenario_ids:
            raise ValueError("future_h5_layers scenarios require unique scenario_id")
        scenario_ids.add(scenario_id)
        probability = float(scenario.get("scenario_probability", 0.0))
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError(f"invalid future layer scenario probability: {probability!r}")
        probabilities.append(probability)
        layers = scenario.get("layers")
        if not isinstance(layers, (list, tuple)):
            raise ValueError("future_h5_layers scenario requires a layers list")
        if len(layers) > horizon:
            raise ValueError("future_h5_layers contains more layers than requested horizon")
        offsets: list[int] = []
        for layer in layers:
            if not isinstance(layer, Mapping):
                raise ValueError("future H5 layer must be an object")
            offset = int(layer.get("layer_offset", 0))
            offsets.append(offset)
            nodes = layer.get("nodes")
            if not isinstance(nodes, (list, tuple)) or not nodes:
                raise ValueError("future H5 layers must contain at least one node prototype")
            for node in nodes:
                if not isinstance(node, Mapping):
                    raise ValueError("future H5 layer node must be an object")
                forbidden = {"node_id", "successors", "predecessors", "successor_node_ids", "predecessor_node_ids"}
                leaked = sorted(forbidden.intersection(node))
                if leaked:
                    raise ValueError(f"future H5 prediction leaks DAG identity fields: {leaked}")
        expected_offsets = list(range(1, len(layers) + 1))
        if offsets != expected_offsets:
            raise ValueError(
                f"future H5 layer offsets must be contiguous from 1: {offsets!r}"
            )

    total = sum(probabilities)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
        raise ValueError(f"future H5 scenario probabilities must sum to 1, got {total!r}")


def project_event_scenarios_to_unary_layers(
    scenarios: Sequence[Mapping[str, Any]],
    horizon: int = 5,
) -> list[dict[str, Any]]:
    """Project legacy event paths to layers for contract-only regression.

    This is not a topology predictor: each layer intentionally contains one
    node prototype.  The source is recorded so the resulting artifact cannot
    be mistaken for a learned multi-node topology forecast.
    """

    if horizon < 0:
        raise ValueError("layer horizon must be non-negative")
    result: list[dict[str, Any]] = []
    for scenario in scenarios:
        steps = scenario.get("steps") or []
        layers: list[dict[str, Any]] = []
        for offset, step in enumerate(steps[:horizon], 1):
            if not isinstance(step, Mapping):
                raise ValueError("legacy future scenario step must be an object")
            forbidden = {"node_id", "successors", "predecessors", "successor_node_ids", "predecessor_node_ids"}
            leaked = sorted(forbidden.intersection(step))
            if leaked:
                raise ValueError(f"legacy future step leaks DAG identity fields: {leaked}")
            node = dict(step)
            node["layer_offset"] = offset
            node["predicted_node_index"] = 0
            layers.append({"layer_offset": offset, "nodes": [node]})
        result.append(
            {
                "scenario_id": str(scenario.get("scenario_id") or ""),
                "scenario_probability": float(scenario.get("scenario_probability", 0.0)),
                "layers": layers,
                "synthetic_rollout": bool(scenario.get("synthetic_rollout", True)),
                "topology_source": "legacy_event_to_unary_layer_projection",
            }
        )
    validate_layer_scenarios(result, horizon)
    return result


def layer_node_count(scenarios: Sequence[Mapping[str, Any]], horizon: int = 5) -> list[int]:
    """Return per-scenario predicted node counts within the first H layers."""

    validate_layer_scenarios(scenarios, horizon)
    return [
        sum(len(layer.get("nodes") or []) for layer in scenario.get("layers") or [])
        for scenario in scenarios
    ]
