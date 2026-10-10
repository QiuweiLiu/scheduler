"""Pressure-grid transforms (exp C): arrival-scale alpha x pinned GPU capacity.

Frozen design: docs/research/2026-10-10_exp_c_pressure_grid.md

The arrival transform is the FROZEN stress-v2 builder's ``transform_episode``
(re-used by import, never re-implemented): every inter-arrival gap scales by the
same alpha, deadlines keep their absolute post-arrival budget, service classes
are set uniformly, episode ids / templates / membership are unchanged.  This
module adds the capacity pin (``gpu_topology_mb`` override, a disclosed
simulation counterfactual) and the per-cell aggregate integrity hash.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

_PREPROCESS = Path(__file__).resolve().parent / "preprocess"
if str(_PREPROCESS) not in sys.path:
    sys.path.insert(0, str(_PREPROCESS))

from build_stress_v2_episodes import transform_episode  # noqa: E402

PRESSURE_GRID_VERSION = "pressure-grid-v1"

ALPHAS: tuple[float, ...] = (1.00, 0.90, 0.80)
CAPACITY_LEVELS: tuple[str, ...] = ("cap2424", "cap3224", "cap3232")
CAPACITY_MB: Dict[str, tuple[float, float]] = {
    "cap2424": (24576.0, 24576.0),
    "cap3224": (32760.0, 24576.0),
    "cap3232": (32760.0, 32760.0),
}


def cell_id(alpha: float, capacity_key: str) -> str:
    return "a%03d_%s" % (round(float(alpha) * 100), capacity_key)


def cell_ids() -> List[str]:
    return [cell_id(alpha, capacity) for alpha in ALPHAS for capacity in CAPACITY_LEVELS]


def build_pressure_cell(
    episodes: Sequence[Dict[str, Any]],
    alpha: float,
    capacity_key: str,
    all_normal: bool = True,
) -> List[Dict[str, Any]]:
    """Deterministic in-memory cell transform for one (alpha, capacity) cell.

    Arrival transform via the frozen stress-v2 builder; capacity pin applied on
    top; the original ``episode_id`` is preserved (cross-cell pairing) while the
    provenance fields are recorded on every episode.
    """

    if capacity_key not in CAPACITY_MB:
        raise ValueError(f"unknown capacity key: {capacity_key!r}")
    capacities = CAPACITY_MB[capacity_key]
    transformed: List[Dict[str, Any]] = []
    for episode in episodes:
        base_id = str(episode.get("episode_id"))
        new = transform_episode(episode, float(alpha), all_normal)
        new["episode_id"] = base_id
        new["gpu_topology_mb"] = [float(value) for value in capacities]
        new["pressure_grid_version"] = PRESSURE_GRID_VERSION
        new["pressure_alpha"] = float(alpha)
        new["pressure_capacity_key"] = capacity_key
        new["pressure_capacity_mb"] = [float(value) for value in capacities]
        new["parent_episode_sha256"] = episode_sha256(episode)
        new["scenario_cell"] = "%s|%s" % (episode.get("scenario_cell"), PRESSURE_GRID_VERSION)
        transformed.append(new)
    return transformed


def episode_sha256(episode: Dict[str, Any]) -> str:
    blob = json.dumps(episode, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def cell_aggregate_sha256(episodes: Iterable[Dict[str, Any]]) -> str:
    """Order-stable aggregate over the transformed cell episodes."""

    digest = hashlib.sha256()
    for episode in sorted(episodes, key=lambda row: str(row.get("episode_id"))):
        digest.update(episode_sha256(episode).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
