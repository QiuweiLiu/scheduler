#!/usr/bin/env python
"""Resource-v2: coherent discrete runtime distribution for the J-series predictor.

This module is the *library* half of the Phase R experiment line
(``EXP-20260919_j_series_resource_dist_v1``).  It contains, with no training
dependency:

* train-only bin construction on ``log1p(ms)`` with a dedicated near-zero bin and
  an overflow bin;
* :class:`DiscreteRuntimeHead`, a small head that predicts a categorical
  distribution over those bins (softmax -> CDF -> arbitrary quantiles, no
  quantile crossing possible);
* the four loss terms requested by the design review (NLL, ranked probability,
  raw-ms pseudo-Huber point term, coarse-band CE);
* a four-layer evaluation report that keeps *distribution calibration*,
  *distribution accuracy*, *node-level discrimination* and *scheduler
  consumption* strictly separate, plus the pre-registered failure-mode gates.

Nothing here touches ``j_series_common`` or the frozen J3 checkpoint: the head is
trained on features captured through a forward hook on the frozen resource head.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

try:  # torch is optional so bin/metric helpers stay importable without it
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only in torch-less environments
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
SCHEMA_VERSION = "j-resource-dist-v1"


# --------------------------------------------------------------------------- #
# bins
# --------------------------------------------------------------------------- #
def build_bin_edges(
    train_runtimes: Sequence[float],
    n_bins: int = 24,
    near_zero_edge_ms: float = 1.0,
    upper_quantile: float = 0.999,
) -> np.ndarray:
    """Train-only bin edges.

    Layout: ``[0, near_zero_edge_ms)`` then ``n_bins - 2`` log1p-spaced bins up to
    the train ``upper_quantile`` then one overflow bin closed at ``+inf``.
    Returns ``n_bins + 1`` edges so that ``searchsorted(edges, v, 'right') - 1``
    yields the bin index in ``[0, n_bins - 1]``.
    """

    values = np.asarray([float(v) for v in train_runtimes if float(v) > 0.0], dtype=np.float64)
    if values.size == 0:
        raise ValueError("build_bin_edges needs at least one positive runtime")
    if n_bins < 3:
        raise ValueError("n_bins must be >= 3 (near-zero bin + log bins + overflow)")
    if not 0.5 < upper_quantile <= 1.0:
        raise ValueError("upper_quantile must be in (0.5, 1.0]")
    upper = float(np.quantile(np.log1p(values), upper_quantile))
    lower = float(np.log1p(max(1.0, near_zero_edge_ms)))
    if upper <= lower:
        upper = lower + 1e-3
    # n_bins - 2 inner log edges + 1 upper edge before the overflow bin
    log_edges = np.linspace(lower, upper, n_bins)
    edges = np.concatenate(([0.0, near_zero_edge_ms], np.expm1(log_edges[1:])))
    edges[-1] = math.inf
    if len(edges) != n_bins + 1:
        raise ValueError("bin construction produced %d edges for n_bins=%d" % (len(edges), n_bins))
    if np.any(np.diff(edges[: len(edges) - 1]) <= 0):
        raise ValueError("bin edges must be strictly increasing before the overflow bin")
    return edges


def bin_index(values: Any, edges: np.ndarray) -> np.ndarray:
    """Bin index for each value; values are clamped into the overflow bin."""

    arr = np.asarray(values, dtype=np.float64)
    idx = np.searchsorted(edges, arr, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def bin_representatives(edges: np.ndarray) -> np.ndarray:
    """Representative runtime (ms) per bin: geometric mean inside, upper edge last."""

    reps = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if math.isinf(hi):
            reps.append(max(lo, 1.0))
        elif lo <= 0.0:
            reps.append(max(hi / 2.0, 1e-3))
        else:
            reps.append(math.expm1((math.log1p(lo) + math.log1p(hi)) / 2.0))
    return np.asarray(reps, dtype=np.float64)


def band_index(values: Any) -> np.ndarray:
    """Coarse band index for each value (0..3)."""

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
        """Predicts ``n_bins`` logits per slot from the frozen resource feature."""

        def __init__(self, in_dim: int, n_bins: int, hidden: int | None = None):
            super().__init__()
            self.n_bins = int(n_bins)
            if hidden:
                self.net = nn.Sequential(
                    nn.Linear(int(in_dim), int(hidden)),
                    nn.GELU(),
                    nn.Linear(int(hidden), int(n_bins)),
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

    def rps_loss(logits: Any, target_idx: Any, mask: Any, edges: np.ndarray) -> Any:
        """Discrete ranked probability score: sum_k (F_k - 1[y <= e_k])^2 / K.

        ``1[y <= e_k] == 1[bin(y) < k]`` because the bin index is the count of
        edges the value exceeds, so the indicator is built from the bin index
        directly and no edge values are needed here.
        """

        probs = F.softmax(logits, dim=-1)
        cdf = torch.cumsum(probs, dim=-1)[..., :-1]  # k = 1..K-1
        k_idx = torch.arange(1, logits.shape[-1], device=logits.device).view(
            *([1] * (logits.dim() - 1)), -1
        )
        indicator = (target_idx.unsqueeze(-1) < k_idx).to(logits.dtype)
        return _masked_mean(((cdf - indicator) ** 2).mean(dim=-1), mask)

    def pseudo_huber_point(pred_ms: Any, target_ms: Any, scale: float, delta: float, mask: Any) -> Any:
        resid = (pred_ms - target_ms) / max(scale, 1e-6)
        return _masked_mean(delta * delta * (torch.sqrt(1.0 + (resid / delta) ** 2) - 1.0), mask)

    def coarse_band_ce(logits: Any, band_idx: Any, mask: Any, edges: np.ndarray) -> Any:
        """CE on the 4 coarse bands obtained by aggregating the same distribution."""

        probs = F.softmax(logits, dim=-1)
        reps = torch.as_tensor(bin_representatives(edges), dtype=probs.dtype, device=probs.device)
        band_probs = []
        members = [torch.as_tensor(band_index(reps.detach().cpu().numpy()) == b, device=probs.device) for b in range(len(BAND_NAMES))]
        for member in members:
            band_probs.append(probs[..., member].sum(dim=-1))
        stacked = torch.stack(band_probs, dim=-1).clamp(min=1e-9)
        logp = torch.log(stacked)
        picked = logp.gather(-1, band_idx.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return _masked_mean(-picked, mask)

    def derive_from_probs(probs: Any, edges: np.ndarray) -> Dict[str, Any]:
        """Quantiles + expectation from a categorical distribution (differentiable)."""

        reps = torch.as_tensor(bin_representatives(edges), dtype=probs.dtype, device=probs.device)
        cdf = torch.cumsum(probs, dim=-1)
        out: Dict[str, Any] = {"mean_ms": (probs * reps).sum(dim=-1)}
        for tau in TAUS:
            hit = (cdf >= tau).to(probs.dtype)
            first = torch.argmax(hit, dim=-1)  # first bin whose cdf >= tau
            out["q%02d_ms" % round(tau * 100)] = reps[first]
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
    rx = rx - rx.mean()
    ry = ry - ry.mean()
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


def _bucket_report(true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    lo_edges = [0.0, 0.5, 10.0, 1000.0, 2000.0, 5000.0, 8000.0, 12000.0, 30000.0]
    hi_edges = lo_edges[1:] + [math.inf]
    out: Dict[str, Any] = {}
    for lo, hi in zip(lo_edges, hi_edges):
        sel = (true >= lo) & (true < hi)
        if not sel.any():
            continue
        label = "%.1f-%.0f" % (lo, hi) if not math.isinf(hi) else ">30000"
        out[label] = {
            "n": int(sel.sum()),
            "median_true_ms": float(np.median(true[sel])),
            "median_pred_ms": float(np.median(pred[sel])),
            "pred_over_true": float(np.median(pred[sel]) / np.median(true[sel])) if np.median(true[sel]) else None,
            "median_abs_log_error": float(np.median(np.abs(np.log1p(pred[sel]) - np.log1p(true[sel])))),
        }
    return out


def _align_video_code(video_code: Any, n_slots: int) -> np.ndarray:
    """Accept per-row or per-slot video codes and return a per-slot array."""

    vc = np.asarray(video_code).reshape(-1)
    if vc.size == n_slots:
        return vc
    if vc.size and n_slots % vc.size == 0:
        return np.repeat(vc, n_slots // vc.size)
    raise ValueError("video_code has %d entries but there are %d slot values" % (vc.size, n_slots))


def evaluate_distribution(
    probs: np.ndarray,
    true_ms: np.ndarray,
    edges: np.ndarray,
    video_code: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Four-layer report; see the module docstring for the layer semantics."""

    probs = np.asarray(probs, dtype=np.float64)
    true_ms = np.asarray(true_ms, dtype=np.float64)
    keep = true_ms > 0.0
    probs = probs[keep]
    true_ms = true_ms[keep]
    video_code = _align_video_code(video_code, len(keep))[keep] if video_code is not None else None
    reps = bin_representatives(edges)
    cdf = np.cumsum(probs, axis=-1)
    mean_ms = probs @ reps
    pred = {tau: reps[np.argmax(cdf >= tau, axis=-1)] for tau in TAUS}

    layer_a: Dict[str, Any] = {}
    for tau in TAUS:
        cov = float(np.mean(true_ms <= pred[tau]))
        layer_a["q%02d" % round(tau * 100)] = {"coverage": cov, "calibration_error": cov - tau}
    layer_a["mean_abs_calibration_error"] = float(
        np.mean([abs(layer_a["q%02d" % round(t * 100)]["calibration_error"]) for t in TAUS])
    )
    crossing = int(np.sum((pred[0.50] > pred[0.90]) | (pred[0.90] > pred[0.95])))
    layer_a["quantile_crossing_rate"] = crossing / max(1, len(true_ms))

    layer_b: Dict[str, Any] = {}
    y_mean = float(true_ms.mean())
    for tau in TAUS:
        pb = float(np.mean(pinball(pred[tau], true_ms, tau)))
        layer_b["q%02d" % round(tau * 100)] = {"pinball_ms": pb, "normalized_pinball": pb / y_mean if y_mean else None}
    layer_b["runtime_qscore_ms"] = float(
        np.mean([layer_b["q%02d" % round(t * 100)]["pinball_ms"] for t in TAUS])
    )
    layer_b["rps"] = float(np.mean(((cdf - (true_ms[:, None] <= edges[1:][None, :]).astype(float)) ** 2)[:, :-1].mean(axis=-1)))

    point = mean_ms
    layer_c: Dict[str, Any] = {
        "spearman_point": spearman(point, true_ms),
        "tail_recall_top10_point": tail_recall(point, true_ms),
        "mae_log": float(np.mean(np.abs(np.log1p(point) - np.log1p(true_ms)))),
        "mae_raw_ms": float(np.mean(np.abs(point - true_ms))),
        "buckets": _bucket_report(true_ms, point),
    }
    # the pre-registered failure-mode gate looks at the 5-12s region specifically
    exp_band = layer_c["buckets"].get("8000-12000")
    layer_c["expensive_bucket_pred_over_true"] = exp_band["pred_over_true"] if exp_band else None

    layer_d = {
        "sum_q50_over_true": float(np.sum(pred[0.50]) / np.sum(true_ms)),
        "sum_q95_over_true": float(np.sum(pred[0.95]) / np.sum(true_ms)),
        "sum_mean_over_true": float(np.sum(mean_ms) / np.sum(true_ms)),
        "note": "only sum_mean_over_true is an additive-expectation diagnostic; the q50/q95 ratios are consumption-scale only",
    }
    if video_code is not None and len(set(video_code.tolist())) > 1:
        layer_d["video_clusters"] = int(len(set(video_code.tolist())))

    return {
        "schema_version": SCHEMA_VERSION,
        "n_slot_pairs": int(len(true_ms)),
        "A_distribution_calibration": layer_a,
        "B_distribution_accuracy": layer_b,
        "C_node_discrimination": layer_c,
        "D_scheduler_consumption": layer_d,
    }


def evaluate_quantiles(
    q50: np.ndarray,
    q90: np.ndarray,
    q95: np.ndarray,
    true_ms: np.ndarray,
    slot_mask: np.ndarray | None = None,
    video_code: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Same four-layer report for a three-point quantile predictor (the J3 baseline).

    Identical metric code to :func:`evaluate_distribution` so the R1-vs-J3 gates
    compare like with like.  The point estimate used in layer C/D is ``q50``,
    because the old head has no conditional-mean output; layer D therefore does
    NOT report an additive-expectation diagnostic for this baseline.
    """

    true_ms = np.asarray(true_ms, dtype=np.float64).reshape(-1)
    q50 = np.asarray(q50, dtype=np.float64).reshape(-1)
    q90 = np.asarray(q90, dtype=np.float64).reshape(-1)
    q95 = np.asarray(q95, dtype=np.float64).reshape(-1)
    keep = true_ms > 0.0
    if slot_mask is not None:
        keep &= np.asarray(slot_mask, dtype=bool).reshape(-1)
    true_ms, q50, q90, q95 = true_ms[keep], q50[keep], q90[keep], q95[keep]
    video_code = _align_video_code(video_code, len(keep))[keep] if video_code is not None else None
    preds = {0.50: q50, 0.90: q90, 0.95: q95}

    layer_a: Dict[str, Any] = {}
    for tau, values in preds.items():
        cov = float(np.mean(true_ms <= values))
        layer_a["q%02d" % round(tau * 100)] = {"coverage": cov, "calibration_error": cov - tau}
    layer_a["mean_abs_calibration_error"] = float(
        np.mean([abs(layer_a["q%02d" % round(t * 100)]["calibration_error"]) for t in TAUS])
    )
    crossing = int(np.sum((q50 > q90) | (q90 > q95)))
    layer_a["quantile_crossing_rate"] = crossing / max(1, len(true_ms))

    y_mean = float(true_ms.mean())
    layer_b: Dict[str, Any] = {}
    for tau, values in preds.items():
        pb = float(np.mean(pinball(values, true_ms, tau)))
        layer_b["q%02d" % round(tau * 100)] = {"pinball_ms": pb, "normalized_pinball": pb / y_mean if y_mean else None}
    layer_b["runtime_qscore_ms"] = float(np.mean([layer_b["q%02d" % round(t * 100)]["pinball_ms"] for t in TAUS]))
    layer_b["rps"] = None  # not defined for a three-point predictor

    layer_c: Dict[str, Any] = {
        "spearman_point": spearman(q50, true_ms),
        "tail_recall_top10_point": tail_recall(q50, true_ms),
        "mae_log": float(np.mean(np.abs(np.log1p(q50) - np.log1p(true_ms)))),
        "mae_raw_ms": float(np.mean(np.abs(q50 - true_ms))),
        "buckets": _bucket_report(true_ms, q50),
    }
    exp_band = layer_c["buckets"].get("8000-12000")
    layer_c["expensive_bucket_pred_over_true"] = exp_band["pred_over_true"] if exp_band else None

    layer_d = {
        "sum_q50_over_true": float(np.sum(q50) / np.sum(true_ms)),
        "sum_q95_over_true": float(np.sum(q95) / np.sum(true_ms)),
        "sum_mean_over_true": None,
        "note": "three-point predictor: no conditional mean, so no additive diagnostic is available",
    }
    if video_code is not None:
        layer_d["video_clusters"] = int(len(set(video_code.tolist())))

    return {
        "schema_version": SCHEMA_VERSION,
        "n_slot_pairs": int(len(true_ms)),
        "A_distribution_calibration": layer_a,
        "B_distribution_accuracy": layer_b,
        "C_node_discrimination": layer_c,
        "D_scheduler_consumption": layer_d,
    }


def evaluate_gates(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    """Pre-registered R1 gates against a recorded J3 baseline report."""

    a, b, c = report["A_distribution_calibration"], report["B_distribution_accuracy"], report["C_node_discrimination"]
    ba, bc = baseline["A_distribution_calibration"], baseline["C_node_discrimination"]
    checks: Dict[str, Any] = {}
    checks["calibration_not_worse"] = {
        "value": a["mean_abs_calibration_error"],
        "baseline": ba["mean_abs_calibration_error"],
        "pass": a["mean_abs_calibration_error"] <= ba["mean_abs_calibration_error"] + 0.02,
    }
    checks["spearman_improved"] = {
        "value": c["spearman_point"],
        "baseline": bc["spearman_point"],
        "delta": c["spearman_point"] - bc["spearman_point"],
        "pass": c["spearman_point"] >= bc["spearman_point"] + 0.08,
    }
    checks["tail_recall_improved"] = {
        "value": c["tail_recall_top10_point"],
        "baseline": bc["tail_recall_top10_point"],
        "delta": c["tail_recall_top10_point"] - bc["tail_recall_top10_point"],
        "pass": c["tail_recall_top10_point"] >= bc["tail_recall_top10_point"] + 0.08,
    }
    checks["log_mae_improved"] = {
        "value": c["mae_log"],
        "baseline": bc["mae_log"],
        "relative": (c["mae_log"] / bc["mae_log"] - 1.0) if bc["mae_log"] else None,
        "pass": c["mae_log"] <= 0.90 * bc["mae_log"],
    }
    exp_ok = c["expensive_bucket_pred_over_true"] is not None and c["expensive_bucket_pred_over_true"] >= 0.50
    checks["expensive_bucket"] = {"value": c["expensive_bucket_pred_over_true"], "pass": bool(exp_ok)}
    checks["no_quantile_crossing"] = {"value": a["quantile_crossing_rate"], "pass": a["quantile_crossing_rate"] == 0.0}
    improving = [checks["spearman_improved"]["pass"], checks["tail_recall_improved"]["pass"],
                 checks["log_mae_improved"]["pass"], checks["expensive_bucket"]["pass"]]
    checks["headline"] = {
        "improved_count": int(sum(1 for v in improving if v)),
        "required": 3,
        "calibration_gate": checks["calibration_not_worse"]["pass"],
        "crossing_gate": checks["no_quantile_crossing"]["pass"],
        "pass": (sum(1 for v in improving if v) >= 3
                 and checks["calibration_not_worse"]["pass"]
                 and checks["no_quantile_crossing"]["pass"]),
    }
    return checks


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
