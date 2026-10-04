"""Torpor-lifecycle-projection helpers (frozen 2026-10-04).

Adaptation of Torpor (USENIX ATC'25) to the single-GPU node-level substrate:
late binding, residency-aware model placement, interference-aware loading, and
swap-cost-aware eviction.  RRC tail-SLO queueing and NVLink GPU-to-GPU swapping
are unavailable here and are disclosed in the paper's table footnote.

Review constraints (see docs/research/2026-10-03_main_table_baselines_manifest.md):

* no weighted ``runtime + load + lambda * interference`` score -- Torpor has no
  such scheduler;
* job selection stays FCFS inside a service class (no runtime-SJF);
* placement is lexicographic: resident+available, then covered low-interference
  load, then covered higher-interference load, then the existing fail-closed
  uncovered path;
* eviction starts from the lowest measured ``swap_burden`` and never invents a
  penalty for uncovered interference cells.
"""

from __future__ import annotations

import statistics
from typing import Any, Mapping


def model_load_estimate_ms(
    train_stats: Mapping[str, Mapping[str, Any]],
    model_id: str,
) -> float:
    """Train-only load-time estimate for one model (median over its rows)."""

    values = []
    prefix = f"{model_id}|"
    for key, row in train_stats.items():
        if not str(key).startswith(prefix):
            continue
        value = row.get("load_p50_ms")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return statistics.median(values) if values else 0.0


def model_interference_cost_ms(
    interference_profile: Mapping[str, Any] | None,
    model_id: str,
) -> float:
    """Measured F4 interference cost when this model is the loaded side.

    Uses the maximum covered additive cell for this load model; uncovered
    combinations contribute nothing rather than an invented penalty.
    """

    if not interference_profile:
        return 0.0
    cells = interference_profile.get("cells") or []
    values = []
    for cell in cells:
        if cell.get("load") != model_id:
            continue
        value = cell.get("extra_ms")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0.0


def swap_burden_order(
    gpu: Any,
    train_stats: Mapping[str, Mapping[str, Any]],
    interference_profile: Mapping[str, Any] | None,
) -> list[str]:
    """Resident models ordered by ascending measured swap burden.

    ``swap_burden(m) = load_estimate(m) + max covered interference(m)``.  Ties
    break deterministically by model id: the engine does not track residency
    timestamps, so an LRU tie-break is not representable and is not faked.
    """

    residents = list(gpu.resident)
    return sorted(
        residents,
        key=lambda model_id: (
            model_load_estimate_ms(train_stats, model_id)
            + model_interference_cost_ms(interference_profile, model_id),
            str(model_id),
        ),
    )


def torpor_placement_rank(
    model_id: str,
    gpu: Any,
    load_cost_ms: float,
    interference_covered: bool,
    interference_extra_ms: float | None,
) -> tuple[int, float]:
    """Lexicographic placement rank for one (node, gpu) candidate.

    ``0`` resident+available, ``1`` covered load, ``2`` uncovered load.  The
    second element orders covered loads by measured cost only; uncovered loads
    never receive an invented penalty (their ordering falls back to the engine's
    fail-closed path).
    """

    if model_id in gpu.resident:
        return (0, 0.0)
    if interference_covered:
        return (1, float(load_cost_ms) + max(0.0, float(interference_extra_ms or 0.0)))
    return (2, float(load_cost_ms))
