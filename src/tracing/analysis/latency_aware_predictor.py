"""Latency-Aware-adapted: a train-only, REQUEST-CONDITIONED duration/memory predictor.

Paper: Latency-Aware-Orchestration (arXiv:2609.03335), three components
Predictor / Constructor / Scheduler.

The review's finding on the first version: the scheduler read ``node.compute_ms``, which
is ``max(0.1, runtime_ms - load_ms)`` -- the node's ACTUAL intrinsic compute time.  That is
the simulator's truth, so the "predictor" was reading the answer it was supposed to
predict, and every latency figure it produced was circular.

This module replaces it with a predictor trained on TRAIN-ONLY templates.  It is
conditioned on the request, not merely on the model: the same model serving a different
action can cost very differently, and a predictor keyed only by (model, lane) -- which is
what the TIE bank legitimately is -- would return one unchanged distribution for every
request.  ``non_degeneracy_report`` exists to make that distinction checkable rather than
asserted.

Tiers, finest first, each with a minimum support:
    T1 (model, lane, action_family, raw_action, batch_size)
    T2 (model, lane, action_family, batch_size)
    T3 (model, lane, action_family)
    T4 (model, lane)
The finest tier with enough support is used, so an unseen request degrades gracefully
instead of failing or, worse, silently borrowing another request's number.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PREDICTOR_SCHEMA = "latency-aware-request-predictor-v1"

# Minimum train samples before a tier is trusted.  Below it the next coarser tier is used.
DEFAULT_MIN_SUPPORT = 3


def _features(node: Any) -> Dict[str, Any]:
    """The request features the predictor conditions on.

    All of them are ontology fields the scheduler may legally see before dispatch:
    nothing here is an executed duration.
    """

    return {
        "model_id": str(getattr(node, "model_id", "") or ""),
        "lane": str(getattr(node, "lane", "") or ""),
        "action_family": str(getattr(node, "action_family", "") or "other"),
        "raw_action": str(getattr(node, "raw_action", "") or "other"),
        "batch_size": int(getattr(node, "batch_size", 1) or 1),
    }


def _key(tier: int, f: Mapping[str, Any]) -> Tuple[Any, ...]:
    if tier == 1:
        return (f["model_id"], f["lane"], f["action_family"], f["raw_action"], f["batch_size"])
    if tier == 2:
        return (f["model_id"], f["lane"], f["action_family"], f["batch_size"])
    if tier == 3:
        return (f["model_id"], f["lane"], f["action_family"])
    if tier == 4:
        return (f["model_id"], f["lane"])
    raise ValueError("unknown tier %r" % (tier,))


TIERS = (1, 2, 3, 4)


def build_latency_predictor(templates: Mapping[str, Any], *,
                            min_support: int = DEFAULT_MIN_SUPPORT) -> Dict[str, Any]:
    """Fit the predictor on TRAIN ONLY.

    Records, per tier and key, the mean run time, the mean peak memory and the mean load
    time of the nodes that match.  Validation and test templates are skipped and counted,
    so the train-only claim is checkable.
    """

    sums: Dict[int, Dict[Tuple[Any, ...], Dict[str, float]]] = {t: {} for t in TIERS}
    counts: Dict[int, Dict[Tuple[Any, ...], int]] = {t: {} for t in TIERS}
    skipped = 0
    n_nodes = 0

    for tpl in templates.values():
        if getattr(tpl, "split", None) != "train":
            skipped += 1
            continue
        for node in tpl.nodes:
            n_nodes += 1
            f = _features(node)
            run = float(getattr(node, "runtime_ms", 0.0) or 0.0)
            mem = float(getattr(node, "workspace_peak_mb", 0.0) or 0.0)
            load = float(getattr(node, "load_ms", 0.0) or 0.0)
            for tier in TIERS:
                key = _key(tier, f)
                slot = sums[tier].setdefault(key, {"run": 0.0, "mem": 0.0, "load": 0.0})
                slot["run"] += run
                slot["mem"] += mem
                slot["load"] += load
                counts[tier][key] = counts[tier].get(key, 0) + 1

    if n_nodes == 0:
        raise ValueError("no train nodes among the %d templates supplied; refusing to "
                         "build a latency predictor" % len(templates))

    tables: Dict[int, Dict[Tuple[Any, ...], Dict[str, float]]] = {}
    for tier in TIERS:
        table: Dict[Tuple[Any, ...], Dict[str, float]] = {}
        for key, slot in sums[tier].items():
            n = counts[tier][key]
            table[key] = {
                "n": n,
                "run_ms": slot["run"] / n,
                "peak_mem_mb": slot["mem"] / n,
                "load_ms": slot["load"] / n,
            }
        tables[tier] = table

    return {
        "schema": PREDICTOR_SCHEMA,
        "tables": tables,
        "min_support": int(min_support),
        "n_train_nodes": n_nodes,
        "skipped_non_train": skipped,
        "tier_keys": {
            1: ["model_id", "lane", "action_family", "raw_action", "batch_size"],
            2: ["model_id", "lane", "action_family", "batch_size"],
            3: ["model_id", "lane", "action_family"],
            4: ["model_id", "lane"],
        },
        "conditions_on": "request features only; no executed duration",
    }


def predict(predictor: Mapping[str, Any], node: Any) -> Dict[str, Any]:
    """Predict ``(run_ms, peak_mem_mb, load_ms)`` for a request.

    Uses the finest tier with enough support.  Returns the tier actually used so a
    fallback is visible rather than silent.
    """

    if predictor.get("schema") != PREDICTOR_SCHEMA:
        raise ValueError("latency predictor schema mismatch: %r" % (predictor.get("schema"),))
    f = _features(node)
    min_support = int(predictor.get("min_support", DEFAULT_MIN_SUPPORT))
    for tier in TIERS:
        row = predictor["tables"][tier].get(_key(tier, f))
        if row is not None and int(row["n"]) >= min_support:
            return {
                "run_ms": float(row["run_ms"]),
                "peak_mem_mb": float(row["peak_mem_mb"]),
                "load_ms": float(row["load_ms"]),
                "tier": tier,
                "n": int(row["n"]),
            }
    # nothing at any tier: (model, lane) itself is unknown.  Fail closed rather than
    # invent a number.
    raise KeyError(
        "latency predictor has no entry at any tier for model=%r lane=%r"
        % (f["model_id"], f["lane"])
    )


def non_degeneracy_report(predictor: Mapping[str, Any], *,
                          templates: Mapping[str, Any] | None = None
                          ) -> Dict[str, Any]:
    """Does the predictor depend on the REQUEST, measured on the PRODUCTION path?

    The first version compared ``tables[1]`` directly, which bypasses ``min_support`` and
    the production backoff entirely.  Tier-1 raw means could therefore look non-degenerate
    while the real ``predict()`` fell through to Tier 4 -- one answer per (model, lane) --
    for every request, which is exactly the vacuity this gate exists to catch.

    So the spread is now measured on EFFECTIVE predictions obtained through ``predict()``,
    and the tier actually used is reported per request.  ``templates`` supplies the
    requests to probe; when it is omitted the Tier-1 keys are used as the request set.
    """

    requests: List[Any] = []
    if templates is not None:
        for template in templates.values():
            requests.extend(list(template.nodes))
    if not requests:
        for key in predictor["tables"][1]:
            requests.append(_SyntheticRequest(*key))

    groups: Dict[Tuple[str, str], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    tier_usage: Dict[str, int] = defaultdict(int)
    for request in requests:
        f = _features(request)
        try:
            prediction = predict(predictor, request)
        except KeyError:
            continue
        tier_usage["T%d" % prediction["tier"]] += 1
        groups[(f["model_id"], f["lane"])].append((f["action_family"], prediction))

    per_group: Dict[str, Any] = {}
    non_degenerate = 0
    considered = 0
    for (model_id, lane), rows in groups.items():
        families = {fam for fam, _ in rows}
        if len(families) < 2:
            continue
        considered += 1
        runs = [r["run_ms"] for _, r in rows]
        mems = [r["peak_mem_mb"] for _, r in rows]
        loads = [r["load_ms"] for _, r in rows]
        spread = {
            "run_ms": max(runs) - min(runs),
            "peak_mem_mb": max(mems) - min(mems),
            "load_ms": max(loads) - min(loads),
        }
        degenerate = all(abs(v) <= 1e-12 for v in spread.values())
        if not degenerate:
            non_degenerate += 1
        per_group["%s|%s" % (model_id, lane)] = {
            "n_action_families": len(families),
            "spread": spread,
            "degenerate": degenerate,
        }

    total_requests = sum(tier_usage.values())
    return {
        "measured_on": "production predict(), not the raw Tier-1 table",
        "groups_with_multiple_actions": considered,
        "non_degenerate_groups": non_degenerate,
        "fraction_non_degenerate": (non_degenerate / considered) if considered else None,
        "tier_usage": dict(sorted(tier_usage.items())),
        "tier_usage_fraction": (
            {k: v / total_requests for k, v in sorted(tier_usage.items())}
            if total_requests else {}
        ),
        "per_group": per_group,
        "interpretation": (
            "A predictor keyed only by (model, lane) would report fraction 0.0 and use T4 "
            "for every request, because its answer cannot change with the request. This "
            "baseline must not report that."
        ),
    }


class _SyntheticRequest:
    """A minimal request carrying just the ontology fields the predictor keys on."""

    def __init__(self, model_id, lane, action_family, raw_action="", batch_size=1):
        self.model_id = model_id
        self.lane = lane
        self.action_family = action_family
        self.raw_action = raw_action or action_family
        self.batch_size = batch_size
        self.runtime_ms = 0.0
        self.workspace_peak_mb = 0.0
        self.load_ms = 0.0


def binds_to_request(predictor: Mapping[str, Any], template: Any,
                     action_family_a: str, action_family_b: str) -> Dict[str, Any]:
    """Compare predictions for two different actions on the SAME model and lane.

    This is the concrete form of the non-degeneracy gate: if changing the action does
    not move run time, memory or load, the predictor is not request-conditioned.
    """

    from dataclasses import replace

    nodes = list(template.nodes)
    if not nodes:
        raise ValueError("need a template with at least one node")
    base = nodes[0]
    # The gate must compare two requests under the SAME real (model, lane), which is what
    # ``replace`` below guarantees; relying on nodes[0] and nodes[1] of an arbitrary
    # template does not, because those usually differ in model too.
    # Hold model and lane fixed and change ONLY the request.  Using nodes[0] and nodes[1]
    # of an arbitrary template does not do that: those nodes usually differ in model too,
    # so the comparison would fall through to a coarse tier and report "no movement" for a
    # reason that has nothing to do with request conditioning.
    a = replace(base, action_family=action_family_a, raw_action=action_family_a)
    b = replace(base, action_family=action_family_b, raw_action=action_family_b)
    pa, pb = predict(predictor, a), predict(predictor, b)
    moved = {
        "run_ms": abs(pa["run_ms"] - pb["run_ms"]),
        "peak_mem_mb": abs(pa["peak_mem_mb"] - pb["peak_mem_mb"]),
        "load_ms": abs(pa["load_ms"] - pb["load_ms"]),
    }
    return {
        "same_model_and_lane": True,
        "action_a": action_family_a,
        "action_b": action_family_b,
        "prediction_a": pa,
        "prediction_b": pb,
        "moved": moved,
        "is_request_conditioned": any(v > 1e-12 for v in moved.values()),
    }
