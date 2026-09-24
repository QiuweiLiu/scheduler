"""TIE-adapted (ICML 2026) with its OWN front end.

The review found a P0: the arm generated runtime_mean_ms / runtime_cvar90_ms in
train_resource_stats, but the generic estimate() path only carries
runtime_p50_ms / runtime_p90_ms / load_p50_ms / memory_p95_mb, so the policy silently
fell back to ``p50 + beta * p90``.  The 10/10 fidelity suite was a false positive: it
tested the bank and the pure formula but never that the real consumer used them.

Everything TIE needs now lives here, and nothing routes through the shared estimate():

  * the bank is keyed by ``(model_id, lane)`` ONLY.  resource_keys() would add
    sequence_index, which encodes workflow position and would leak progress into a
    current-node-only baseline.
  * a missing mean or CVaR raises.  It never falls back to a percentile.
  * the support question is settled by naming: this is an EMPIRICAL distribution over
    the train split with no post-hoc truncation, so the arm is called
    Empirical-TIE-adapted and the deviation is written down rather than papered over
    with a max(train) censoring point the paper does not have.
  * the waiting-time decay is its own pure function; it is not the aging arm.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Mapping, Sequence, Tuple

TIE_SCHEMA = "empirical-tie-current-node-bank-v1"
TIE_DEVIATION = (
    "the original request-specific max-token censoring has no direct runtime analogue "
    "in this workload; we use a train-only empirical current-node runtime distribution "
    "without post-hoc truncation"
)


def tie_key(node: Any) -> Tuple[str, str]:
    """The ONLY features TIE may condition on: the current node's visible identity.

    Deliberately not ``resource_keys()``, which also carries sequence_index and would
    give a current-node-only baseline extra knowledge of workflow position.
    """

    return (str(getattr(node, "model_id", "")), str(getattr(node, "lane", "")))


def upper_cvar(samples: Sequence[float], alpha: float) -> float:
    """Mean of the worst (1-alpha) tail, with a frozen small-sample rule.

    ``k = ceil((1 - alpha) * n)`` so the tail is never empty; for a singleton the CVaR
    is that sample, which the caller reports rather than hides.
    """

    if not samples:
        raise ValueError("CVaR needs at least one sample")
    ordered = sorted(float(s) for s in samples)
    k = max(1, int(math.ceil((1.0 - alpha) * len(ordered))))
    tail = ordered[-k:]
    return sum(tail) / len(tail)


def build_tie_bank(templates: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the TIE bank from TRAIN-ONLY templates, keyed by (model_id, lane)."""

    groups: Dict[Tuple[str, str], list] = defaultdict(list)
    for tpl in templates.values():
        if getattr(tpl, "split", None) != "train":
            continue
        for n in tpl.nodes:
            groups[tie_key(n)].append(float(n.runtime_ms))

    bank: Dict[str, Any] = {"schema": TIE_SCHEMA, "deviation": TIE_DEVIATION, "groups": {}}
    for key, samples in groups.items():
        bank["groups"]["%s|%s" % key] = {
            "model_id": key[0],
            "lane": key[1],
            "sample_count": len(samples),
            "runtime_mean_ms": sum(samples) / len(samples),
            "runtime_cvar90_ms": upper_cvar(samples, 0.90),
            "runtime_min_ms": min(samples),
            "runtime_max_ms": max(samples),
        }
    return bank


def tie_current_distribution(bank: Mapping[str, Any], node: Any) -> Dict[str, Any]:
    """Look up the current-node distribution, or FAIL CLOSED.

    A missing group, a missing mean or a missing CVaR is an error.  Falling back to a
    percentile is exactly the bug the review found, so it is not permitted here.
    """

    model_id, lane = tie_key(node)
    group = (bank.get("groups") or {}).get("%s|%s" % (model_id, lane))
    if group is None:
        raise KeyError("TIE bank has no group for (%r, %r)" % (model_id, lane))
    for field in ("runtime_mean_ms", "runtime_cvar90_ms"):
        value = group.get(field)
        if not isinstance(value, (int, float)):
            raise ValueError("TIE group (%r, %r) is missing %s" % (model_id, lane, field))
    return group


def tie_beta(queue_length: float, batch_capacity: float) -> float:
    """The paper's adaptive tail weight: clip(0.1 * L_q / B, 0.1, 0.5)."""

    b = max(1e-9, float(batch_capacity))
    return min(0.5, max(0.1, 0.1 * float(queue_length) / b))


def tie_current_score(mean_ms: float, cvar90_ms: float, beta: float, load_ms: float = 0.0) -> float:
    """TIE(X) = E[X] + beta * CVaR_0.9[X], plus the model-load surcharge."""

    return float(mean_ms) + float(beta) * float(cvar90_ms) + float(load_ms)


def tie_wait_adjust(score: float, wait_ms: float, gamma: float = 0.9, tau_ms: float = 30000.0) -> float:
    """Starvation prevention: the paper multiplies the score by gamma^(t_w / tau).

    Kept as a pure function of its own so it cannot be confused with the aging arm,
    which subtracts a wait credit from a remaining-work estimate.  This MULTIPLIES a
    tail-aware score by a decaying factor, so a longer wait yields a smaller (better)
    key under a min-ranking.
    """

    if tau_ms <= 0:
        raise ValueError("tau_ms must be positive")
    return float(score) * (float(gamma) ** (max(0.0, float(wait_ms)) / float(tau_ms)))


def bank_sample_report(bank: Mapping[str, Any]) -> Dict[str, Any]:
    """Report how thin the groups are; a singleton group makes CVaR degenerate."""

    groups = list((bank.get("groups") or {}).values())
    counts = [int(g.get("sample_count") or 0) for g in groups]
    return {
        "groups": len(groups),
        "sample_count_histogram": dict(sorted(Counter(counts).items())),
        "fraction_singleton": (sum(1 for c in counts if c == 1) / len(counts)) if counts else None,
        "fraction_lt_10": (sum(1 for c in counts if c < 10) / len(counts)) if counts else None,
        "total_samples": sum(counts),
    }
