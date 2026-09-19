#!/usr/bin/env python
"""Resource-v2: coherent discrete runtime distribution for the J-series predictor.

Library half of the Phase R experiment line (``EXP-20260919_j_series_resource_dist_v1``).
No training dependency.  Contains:

* train-only bin construction on ``log1p(ms)`` with a dedicated near-zero bin, the
  train p99.9 as the last finite edge, an overflow bin after it, and the coarse-band
  boundaries (1 s, 5 s) forced to be exact bin edges so no bin ever straddles a band;
* train-only **empirical mean** representatives per bin (empty bins fall back to the
  log midpoint, the overflow bin uses the train mean above the last finite edge), so
  the cheap cluster is representable and ``sum(p_k m_k)`` is a usable conditional mean;
* :class:`DiscreteRuntimeHead`: softmax -> CDF -> arbitrary quantiles, so quantile
  crossing is impossible by construction;
* the four loss terms (NLL, discrete ranked probability, raw-ms pseudo-Huber point,
  coarse-band CE) with the exact same bin-edge convention in training and evaluation;
* a four-layer evaluation report that keeps distribution calibration, distribution
  accuracy, node-level discrimination and scheduler consumption strictly apart.

Nothing here touches ``j_series_common`` or the frozen J3 checkpoint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

try:  # torch is optional so the bin/metric helpers stay importable without it
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch-less environments
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False

TAUS: Tuple[float, ...] = (0.50, 0.90, 0.95)
COARSE_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("ultra_cheap", 0.0, 1.0),
    ("light", 1.0, 1000.0),
    ("medium", 1000.0, 5000.0),
    ("expensive", 5000.0, math.inf),
)
BAND_NAMES = tuple(name for name, _, _ in COARSE_BANDS)
BAND_BOUNDARIES: Tuple[float, ...] = (1000.0, 5000.0)
SCHEMA_VERSION = "j-resource-dist-v2"
# the only acceptable overflow mass: the design puts it just above p99.9, so ties aside
# it should be ~0.1 %.  Anything above this is a construction failure, not a tail.
MAX_OVERFLOW_FRACTION = 0.005


# --------------------------------------------------------------------------- #
# bins
# --------------------------------------------------------------------------- #
def build_bin_spec(
    train_runtimes: Sequence[float],
    n_bins: int = 24,
    near_zero_edge_ms: float = 1.0,
    upper_quantile: float = 0.999,
) -> Dict[str, Any]:
    """Train-only bin edges, empirical representatives and the bin->band map.

    Layout (``n_bins`` bins, ``n_bins + 1`` edges)::

        [0, 1 ms)  then n_bins-2 log1p-spaced bins covering [1 ms, train p99.9]
        then the overflow bin [train p99.9, inf)

    The coarse-band boundaries (1 s, 5 s) are snapped onto the nearest interior
    edges so that no bin straddles a band boundary -- otherwise NLL (which only
    knows the bin) and the band CE would supervise the same sample differently.
    """

    values = np.asarray([float(v) for v in train_runtimes if float(v) > 0.0], dtype=np.float64)
    if values.size == 0:
        raise ValueError("build_bin_spec needs at least one positive runtime")
    if n_bins < 6:
        raise ValueError("n_bins must be >= 6 to hold the near-zero bin, the log grid, "
                         "the two snapped band edges and the overflow bin")
    if not 0.5 < upper_quantile <= 1.0:
        raise ValueError("upper_quantile must be in (0.5, 1.0]")

    log_lo = float(np.log1p(max(1.0, near_zero_edge_ms)))
    log_hi = float(np.quantile(np.log1p(values), upper_quantile))
    if log_hi <= log_lo:
        log_hi = log_lo + 1e-3
    inner = n_bins - 2  # log-spaced bins between 1 ms and p99.9
    interior = np.linspace(log_lo, log_hi, inner + 1)[1:]  # excludes 1 ms itself

    # snap the coarse-band boundaries onto the nearest interior edge
    snapped = []
    for boundary in BAND_BOUNDARIES:
        target = math.log1p(boundary)
        index = int(np.argmin(np.abs(interior - target)))
        interior[index] = target
        snapped.append(boundary)
    interior = np.unique(interior)  # collapsing is possible in principle; dedupe
    if len(interior) != inner:
        raise ValueError("band-boundary snapping collapsed log edges; reduce n_bins or bands")

    edges = np.concatenate(([0.0, float(near_zero_edge_ms)], np.expm1(interior), [math.inf]))
    # expm1(log1p(x)) is only exact to ~1e-12 relative, so pin the band boundaries
    # onto their exact values before any set-membership or straddle check.
    for boundary in BAND_BOUNDARIES:
        index = int(np.argmin(np.abs(edges[:-1] - boundary)))
        if abs(float(edges[index]) - boundary) > 1e-6 * boundary:
            raise ValueError("band boundary %s is not close to any bin edge" % boundary)
        edges[index] = boundary
    if len(edges) != n_bins + 1:
        raise ValueError("bin construction produced %d edges for n_bins=%d" % (len(edges), n_bins))
    if np.any(np.diff(edges[:-1]) <= 0):
        raise ValueError("bin edges must be strictly increasing before the overflow bin")
    for boundary in BAND_BOUNDARIES:
        if boundary not in set(edges.tolist()):
            raise ValueError("coarse-band boundary %s is not a bin edge" % boundary)

    reps = build_representatives(values, edges)
    band_of_bin = band_index(reps)
    counts = np.bincount(bin_index(values, edges), minlength=n_bins)
    overflow_fraction = float(counts[-1] / values.size)
    return {
        "schema_version": SCHEMA_VERSION,
        "n_bins": int(n_bins),
        "near_zero_edge_ms": float(near_zero_edge_ms),
        "upper_quantile": float(upper_quantile),
        "edges": [float(e) for e in edges],
        "representatives_ms": [float(r) for r in reps],
        "band_of_bin": [int(b) for b in band_of_bin],
        "band_names": list(BAND_NAMES),
        "band_boundaries_ms": list(BAND_BOUNDARIES),
        "fill_counts": counts.tolist(),
        "overflow_fraction": overflow_fraction,
        "train_positive_slots": int(values.size),
        "max_overflow_fraction": MAX_OVERFLOW_FRACTION,
    }


def edges_from_spec(spec: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(e) for e in spec["edges"]], dtype=np.float64)


def reps_from_spec(spec: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(r) for r in spec["representatives_ms"]], dtype=np.float64)


def band_of_bin_from_spec(spec: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([int(b) for b in spec["band_of_bin"]], dtype=np.int64)


def bin_index(values: Any, edges: np.ndarray) -> np.ndarray:
    """Bin index for each value; bins are left-closed ``[e_k, e_{k+1})``.

    ``side='right'`` makes a value exactly on an edge belong to the bin that starts
    there, so ``1.0`` lands in ``[1 ms, ...)`` and the near-zero bin is exactly
    ``[0, 1 ms)``.  The ranked-probability indicator is written in terms of the bin
    index, so training and evaluation cannot disagree on an edge value.
    """

    arr = np.asarray(values, dtype=np.float64)
    idx = np.searchsorted(edges, arr, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def build_representatives(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Train-only empirical mean of ``Y`` inside each bin (log midpoint if empty)."""

    midpoint = _log_midpoints(edges)
    idx = bin_index(values, edges)
    reps = midpoint.copy()
    for k in range(len(edges) - 1):
        sel = idx == k
        if sel.any():
            reps[k] = float(np.mean(values[sel]))
    # the overflow bin must stay above its lower edge and reflect the observed tail
    reps[-1] = max(reps[-1], math.nextafter(edges[-2], math.inf))
    return reps


def _log_midpoints(edges: np.ndarray) -> np.ndarray:
    reps = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if math.isinf(hi):
            reps.append(max(lo * 1.05, 1.0))
        elif lo <= 0.0:
            reps.append(max(hi / 2.0, 1e-3))
        else:
            reps.append(math.expm1((math.log1p(lo) + math.log1p(hi)) / 2.0))
    return np.asarray(reps, dtype=np.float64)


def band_index(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.zeros(arr.shape, dtype=np.int64)
    for i, (_, lo, hi) in enumerate(COARSE_BANDS):
        out[(arr >= lo) & (arr < hi)] = i
    return out


# --------------------------------------------------------------------------- #
# head + losses
# --------------------------------------------------------------------------- #
if _HAS_TORCH:

    class DiscreteRuntimeHead(nn.Module):
        def __init__(self, in_dim: int, n_bins: int, hidden: int | None = None):
            super().__init__()
            self.n_bins = int(n_bins)
            if hidden:
                self.net = nn.Sequential(
                    nn.Linear(int(in_dim), int(hidden)), nn.GELU(), nn.Linear(int(hidden), int(n_bins))
                )
            else:
                self.net = nn.Linear(int(in_dim), int(n_bins))

        def forward(self, features: Any) -> Any:
            return self.net(features)

    def _masked_mean(values: Any, mask: Any, eps: float = 1e-9) -> Any:
        m = mask.to(values.dtype)
        return (values * m).sum() / m.sum().clamp(min=eps)

    def nll_loss(logits: Any, target_idx: Any, mask: Any) -> Any:
        logp = F.log_softmax(logits, dim=-1)
        picked = logp.gather(-1, target_idx.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return _masked_mean(-picked, mask)

    def rps_loss(logits: Any, target_idx: Any, mask: Any) -> Any:
        """Discrete ranked probability score: mean_k (F_k - 1[bin(y) < k])^2.

        ``1[y <= e_k] == 1[bin(y) < k]`` under the same ``side='left'`` binning used
        by :func:`bin_index`, so training and evaluation agree on exact edges.
        """

        probs = F.softmax(logits, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)[..., :-1]
        k_idx = torch.arange(1, logits.shape[-1], device=logits.device).view(
            *([1] * (logits.dim() - 1)), -1
        )
        indicator = (target_idx.unsqueeze(-1) < k_idx).to(logits.dtype)
        return _masked_mean(((cdf - indicator) ** 2).mean(dim=-1), mask)

    def pseudo_huber_point(pred_ms: Any, target_ms: Any, scale: float, delta: float, mask: Any) -> Any:
        resid = (pred_ms - target_ms) / max(scale, 1e-6)
        return _masked_mean(delta * delta * (torch.sqrt(1.0 + (resid / delta) ** 2) - 1.0), mask)

    def coarse_band_ce(logits: Any, band_idx: Any, mask: Any, band_of_bin: np.ndarray) -> Any:
        """CE over the coarse bands, aggregated with the stored bin->band map."""

        probs = F.softmax(logits, dim=-1)
        groups = [
            torch.as_tensor(band_of_bin == b, device=probs.device) for b in range(len(BAND_NAMES))
        ]
        if any(not bool(g.any()) for g in groups):
            raise ValueError("a coarse band has no bin assigned; rebuild the bin spec")
        stacked = torch.stack([probs[..., g].sum(dim=-1) for g in groups], dim=-1).clamp(min=1e-9)
        picked = torch.log(stacked).gather(-1, band_idx.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return _masked_mean(-picked, mask)

    def derive_from_probs(probs: Any, reps: np.ndarray) -> Dict[str, Any]:
        """Quantiles + expectation from the categorical distribution (differentiable)."""

        rep_t = torch.as_tensor(np.asarray(reps, dtype=np.float64), dtype=probs.dtype, device=probs.device)
        cdf = torch.cumsum(probs, dim=-1)
        out: Dict[str, Any] = {"mean_ms": (probs * rep_t).sum(dim=-1)}
        for tau in TAUS:
            first = torch.argmax((cdf >= tau).to(probs.dtype), dim=-1)
            out["q%02d_ms" % round(tau * 100)] = rep_t[first]
        return out


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(np.asarray(x, dtype=np.float64)), _rank(np.asarray(y, dtype=np.float64))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx**2).sum()) * float((ry**2).sum()))
    return float((rx * ry).sum() / den) if den else float("nan")


def pinball(pred: np.ndarray, true: np.ndarray, tau: float) -> np.ndarray:
    diff = true - pred
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def tail_recall(pred: np.ndarray, true: np.ndarray, fraction: float = 0.10) -> float:
    n = len(true)
    k = max(1, int(round(n * fraction)))
    top_true = set(np.argsort(-true, kind="mergesort")[:k].tolist())
    top_pred = set(np.argsort(-pred, kind="mergesort")[:k].tolist())
    return len(top_true & top_pred) / k


BUCKET_EDGES: Tuple[float, ...] = (0.0, 0.5, 10.0, 1000.0, 2000.0, 5000.0, 8000.0, 12000.0, 30000.0)
EXPENSIVE_BUCKET: Tuple[float, float] = (8000.0, 12000.0)


def bucket_ratio(true: np.ndarray, pred: np.ndarray, lo: float, hi: float) -> Dict[str, Any]:
    """median(pred)/median(true) inside a true-runtime interval (numeric, no lookup)."""

    sel = (true >= lo) & (true < hi)
    if not sel.any():
        return {"n": 0, "pred_over_true": None}
    med_true = float(np.median(true[sel]))
    med_pred = float(np.median(pred[sel]))
    return {
        "n": int(sel.sum()),
        "median_true_ms": med_true,
        "median_pred_ms": med_pred,
        "pred_over_true": (med_pred / med_true) if med_true else None,
        "median_abs_log_error": float(np.median(np.abs(np.log1p(pred[sel]) - np.log1p(true[sel])))),
    }


def _bucket_report(true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    lo_edges = list(BUCKET_EDGES)
    hi_edges = lo_edges[1:] + [math.inf]
    out: Dict[str, Any] = {}
    for lo, hi in zip(lo_edges, hi_edges):
        block = bucket_ratio(true, pred, lo, hi)
        if block["n"]:
            out["%.1f-%.0f" % (lo, hi) if not math.isinf(hi) else ">30000"] = block
    return out


def _align_video_code(video_code: Any, n_slots: int) -> np.ndarray:
    vc = np.asarray(video_code).reshape(-1)
    if vc.size == n_slots:
        return vc
    if vc.size and n_slots % vc.size == 0:
        return np.repeat(vc, n_slots // vc.size)
    raise ValueError("video_code has %d entries but there are %d slot values" % (vc.size, n_slots))


def _layers(
    true_ms: np.ndarray,
    pred: Dict[float, np.ndarray],
    point: np.ndarray,
    mean_ms: np.ndarray | None,
    reps: np.ndarray,
    cdf: np.ndarray | None,
    edges: np.ndarray | None,
) -> Dict[str, Any]:
    layer_a: Dict[str, Any] = {}
    for tau in TAUS:
        cov = float(np.mean(true_ms <= pred[tau]))
        layer_a["q%02d" % round(tau * 100)] = {"coverage": cov, "calibration_error": cov - tau}
    layer_a["mean_abs_calibration_error"] = float(
        np.mean([abs(layer_a["q%02d" % round(t * 100)]["calibration_error"]) for t in TAUS])
    )
    crossing = int(np.sum((pred[0.50] > pred[0.90]) | (pred[0.90] > pred[0.95])))
    layer_a["quantile_crossing_rate"] = crossing / max(1, len(true_ms))

    y_mean = float(true_ms.mean())
    layer_b: Dict[str, Any] = {}
    for tau in TAUS:
        pb = float(np.mean(pinball(pred[tau], true_ms, tau)))
        layer_b["q%02d" % round(tau * 100)] = {
            "pinball_ms": pb,
            "normalized_pinball": pb / y_mean if y_mean else None,
        }
    layer_b["runtime_qscore_ms"] = float(np.mean([layer_b["q%02d" % round(t * 100)]["pinball_ms"] for t in TAUS]))
    if cdf is not None and edges is not None:
        # 1[y <= e_k] under side='left' binning == 1[bin(y) < k]
        indicator = (true_ms[:, None] < edges[1:][None, :]).astype(np.float64)
        layer_b["rps"] = float(np.mean(((cdf - indicator) ** 2)[:, :-1].mean(axis=-1)))
    else:
        layer_b["rps"] = None

    layer_c: Dict[str, Any] = {
        "spearman_point": spearman(point, true_ms),
        "tail_recall_top10_point": tail_recall(point, true_ms),
        "mae_log": float(np.mean(np.abs(np.log1p(point) - np.log1p(true_ms)))),
        "mae_raw_ms": float(np.mean(np.abs(point - true_ms))),
        "buckets": _bucket_report(true_ms, point),
    }
    lo, hi = EXPENSIVE_BUCKET
    layer_c["expensive_bucket_pred_over_true"] = bucket_ratio(true_ms, point, lo, hi)["pred_over_true"]

    layer_d = {
        "sum_q50_over_true": float(np.sum(pred[0.50]) / np.sum(true_ms)),
        "sum_q95_over_true": float(np.sum(pred[0.95]) / np.sum(true_ms)),
        "sum_mean_over_true": float(np.sum(mean_ms) / np.sum(true_ms)) if mean_ms is not None else None,
        "note": "only sum_mean_over_true is an additive-expectation diagnostic; "
                "the q50/q95 sums are consumption-scale quantities, not calibration",
    }
    return {
        "A_distribution_calibration": layer_a,
        "B_distribution_accuracy": layer_b,
        "C_node_discrimination": layer_c,
        "D_scheduler_consumption": layer_d,
    }


def evaluate_distribution(
    probs: np.ndarray,
    true_ms: np.ndarray,
    reps: np.ndarray,
    edges: np.ndarray,
    video_code: np.ndarray | None = None,
) -> Dict[str, Any]:
    probs = np.asarray(probs, dtype=np.float64)
    true_ms = np.asarray(true_ms, dtype=np.float64).reshape(-1)
    keep = true_ms > 0.0
    probs = probs.reshape(len(keep), -1)[keep]
    true_ms = true_ms[keep]
    video_code = _align_video_code(video_code, len(keep))[keep] if video_code is not None else None
    reps = np.asarray(reps, dtype=np.float64)
    cdf = np.cumsum(probs, axis=-1)
    mean_ms = probs @ reps
    pred = {tau: reps[np.argmax(cdf >= tau, axis=-1)] for tau in TAUS}
    out = _layers(true_ms, pred, mean_ms, mean_ms, reps, cdf, edges)
    out["video_clusters"] = int(len(set(video_code.tolist()))) if video_code is not None else None
    return {"schema_version": SCHEMA_VERSION, "n_slot_pairs": int(len(true_ms)), **out}


def evaluate_quantiles(
    q50: np.ndarray,
    q90: np.ndarray,
    q95: np.ndarray,
    true_ms: np.ndarray,
    slot_mask: np.ndarray | None = None,
    video_code: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Same four-layer report for a three-point quantile predictor (the J3 baseline)."""

    true_ms = np.asarray(true_ms, dtype=np.float64).reshape(-1)
    q50 = np.asarray(q50, dtype=np.float64).reshape(-1)
    q90 = np.asarray(q90, dtype=np.float64).reshape(-1)
    q95 = np.asarray(q95, dtype=np.float64).reshape(-1)
    keep = true_ms > 0.0
    if slot_mask is not None:
        keep &= np.asarray(slot_mask, dtype=bool).reshape(-1)
    true_ms, q50, q90, q95 = true_ms[keep], q50[keep], q90[keep], q95[keep]
    video_code = _align_video_code(video_code, len(keep))[keep] if video_code is not None else None
    pred = {0.50: q50, 0.90: q90, 0.95: q95}
    out = _layers(true_ms, pred, q50, None, np.asarray([1.0]), None, None)
    out["video_clusters"] = int(len(set(video_code.tolist()))) if video_code is not None else None
    return {"schema_version": SCHEMA_VERSION, "n_slot_pairs": int(len(true_ms)), **out}


def evaluate_gates(report: Mapping[str, Any], baseline: Mapping[str, Any], gates: Mapping[str, Any]) -> Dict[str, Any]:
    """Two-tier gates, all thresholds read from the config (no hard-coded numbers)."""

    a, c = report["A_distribution_calibration"], report["C_node_discrimination"]
    ba, bc = baseline["A_distribution_calibration"], baseline["C_node_discrimination"]

    def rel(name: str, default: float) -> float:
        return float(gates.get(name, default))

    integrity = {
        "calibration_not_worse": {
            "value": a["mean_abs_calibration_error"],
            "baseline": ba["mean_abs_calibration_error"],
            "limit": ba["mean_abs_calibration_error"] + rel("calibration_not_worse_by", 0.02),
            "pass": a["mean_abs_calibration_error"] <= ba["mean_abs_calibration_error"] + rel("calibration_not_worse_by", 0.02),
        },
        "no_quantile_crossing": {
            "value": a["quantile_crossing_rate"],
            "limit": rel("quantile_crossing_max", 0.0),
            "pass": a["quantile_crossing_rate"] <= rel("quantile_crossing_max", 0.0),
        },
    }
    viability = {
        "log_mae_min_relative": rel("viability_log_mae_max_relative", 0.95),
        "log_mae_relative": c["mae_log"] / bc["mae_log"] if bc["mae_log"] else None,
        "log_mae_pass": c["mae_log"] <= rel("viability_log_mae_max_relative", 0.95) * bc["mae_log"],
        "expensive_bucket_min": rel("viability_expensive_bucket_min", 0.25),
        "expensive_bucket_value": c["expensive_bucket_pred_over_true"],
        "expensive_bucket_pass": bool(
            c["expensive_bucket_pred_over_true"] is not None
            and c["expensive_bucket_pred_over_true"] >= rel("viability_expensive_bucket_min", 0.25)
        ),
        "spearman_delta": c["spearman_point"] - bc["spearman_point"],
        "spearman_pass": c["spearman_point"] >= bc["spearman_point"] + rel("viability_spearman_min_delta", 0.03),
        "tail_recall_delta": c["tail_recall_top10_point"] - bc["tail_recall_top10_point"],
        "tail_recall_pass": c["tail_recall_top10_point"] >= bc["tail_recall_top10_point"] + rel("viability_tail_recall_min_delta", 0.04),
    }
    viability["either_improved"] = bool(viability["spearman_pass"] or viability["tail_recall_pass"])
    viability["pass"] = bool(
        integrity["calibration_not_worse"]["pass"]
        and integrity["no_quantile_crossing"]["pass"]
        and viability["log_mae_pass"]
        and viability["expensive_bucket_pass"]
        and viability["either_improved"]
    )

    strong = {
        "spearman_pass": c["spearman_point"] >= bc["spearman_point"] + rel("spearman_min_delta", 0.08),
        "tail_recall_pass": c["tail_recall_top10_point"] >= bc["tail_recall_top10_point"] + rel("tail_recall_min_delta", 0.08),
        "log_mae_pass": c["mae_log"] <= rel("log_mae_max_relative", 0.90) * bc["mae_log"],
        "expensive_bucket_pass": bool(
            c["expensive_bucket_pred_over_true"] is not None
            and c["expensive_bucket_pred_over_true"] >= rel("expensive_bucket_min_pred_over_true", 0.50)
        ),
        "required": int(rel("improved_required_of_four", 3)),
    }
    strong["improved_count"] = int(sum(1 for key in ("spearman_pass", "tail_recall_pass", "log_mae_pass", "expensive_bucket_pass") if strong[key]))
    strong["pass"] = bool(
        strong["improved_count"] >= strong["required"]
        and integrity["calibration_not_worse"]["pass"]
        and integrity["no_quantile_crossing"]["pass"]
    )
    return {
        "integrity": integrity,
        "viability": viability,
        "strong": strong,
        "headline": {
            "integrity_pass": bool(integrity["calibration_not_worse"]["pass"] and integrity["no_quantile_crossing"]["pass"]),
            "viability_pass": viability["pass"],
            "strong_pass": strong["pass"],
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
