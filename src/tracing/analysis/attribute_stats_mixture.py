"""Attribute-distribution x train-statistics mixture for future runtime cost.

Rationale (from the planning review)
------------------------------------
The legacy consumer ``predicted_future_cost`` reads the *argmax* predicted
attributes and looks up a single train-split statistic.  The artifact, however,
carries a full ``model_probabilities`` distribution per future step (the packer
emits every category with probability >= 0.01).  This module turns that
distribution into a runtime distribution.

Scope discipline (deliberate, from the review)
----------------------------------------------
* Only ``model_id`` is mixed, because only ``(model, lane, sequence_index)`` enters
  the statistics lookup key.  ``role`` and ``action_family`` are top-1-only in the
  artifact and do **not** enter the key, so they are recorded for diagnostics but
  never used to build the mixture.  No ``role x family x model`` Cartesian product:
  there is no evidence the attributes are conditionally independent.
* Two variants, kept separate so their effects can be attributed:
  - ``atom``  - attribute uncertainty only: each lookup group contributes its
                point statistic (``runtime_p50_ms``), weighted by attribute
                probability;
  - ``hist16`` - attribute uncertainty *plus* within-group empirical variability:
                each group contributes its training histogram over the shared
                16-bin schema.
* The ``>= 0.01`` truncation in the packer drops a little mass (observed sums
  0.977-1.000).  Both the raw and the renormalised distribution are reported; the
  renormalisation flag is written into every record so the choice is auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

# attribute categories the packer emits below this probability are absent from the
# artifact, so the mixture must be renormalised before it is used
PACKER_PROBABILITY_FLOOR = 0.01


@dataclass(frozen=True)
class GroupRuntime:
    """One lookup group's runtime evidence, from the training split only."""

    lookup_key: str
    lookup_level: str
    count: int
    runtime_p50_ms: float
    load_p50_ms: float
    # optional: normalised histogram over the shared bin schema (hist16 variant)
    runtime_bin_probs: Tuple[float, ...] | None = None


@dataclass
class AttributeLookupMixture:
    """Attribute probabilities paired with the lookup group each one selects."""

    raw_probability_sum: float
    renormalised: bool
    entries: List[Tuple[str, float, GroupRuntime]] = field(default_factory=list)
    # diagnostics only - never used to build the mixture
    attribute_entropy: Dict[str, float] = field(default_factory=dict)
    dropped_categories: int = 0

    @property
    def weight_sum(self) -> float:
        return sum(weight for _name, weight, _group in self.entries)


def _entropy(probabilities: Sequence[float]) -> float:
    total = 0.0
    for p in probabilities:
        if p > 0.0:
            total -= p * math.log(p)
    return total


def build_lookup_mixture(
    step: Mapping[str, Any],
    *,
    lookup_group: Any,
    renormalise: bool = True,
) -> AttributeLookupMixture:
    """Pair each predicted ``model_id`` candidate with its train-statistics group.

    ``lookup_group`` is a callable ``(model_id, lane) -> GroupRuntime | None`` so
    this module stays independent of the simulator's hierarchy (the review was
    explicit: do not invent a second lookup hierarchy).
    """

    probabilities = step.get("model_probabilities")
    lane = str(step.get("execution_lane") or "unknown")
    mixture = AttributeLookupMixture(raw_probability_sum=0.0, renormalised=False)

    if isinstance(probabilities, Mapping) and probabilities:
        items = [(str(k), float(v)) for k, v in probabilities.items()]
    else:
        # no distribution in the artifact: fall back to the argmax label as a
        # degenerate one-hot, and say so rather than inventing spread
        model_id = step.get("model_id")
        if model_id is None:
            raise ValueError("step carries neither model_probabilities nor model_id")
        items = [(str(model_id), 1.0)]

    raw_sum = sum(weight for _name, weight in items)
    mixture.raw_probability_sum = raw_sum
    if raw_sum <= 0.0:
        raise ValueError("model_probabilities sum to %.6f; refusing to normalise" % raw_sum)

    scale = (1.0 / raw_sum) if (renormalise and abs(raw_sum - 1.0) > 1e-9) else 1.0
    mixture.renormalised = scale != 1.0

    for name, weight in items:
        group = lookup_group(name, lane)
        if group is None:
            mixture.dropped_categories += 1
            continue
        mixture.entries.append((name, weight * scale, group))

    if not mixture.entries:
        raise ValueError("attribute mixture produced no usable lookup group")

    mixture.attribute_entropy = {
        "model_id": _entropy([weight for _name, weight in items]),
    }
    return mixture


def mix_atomic_runtime_distribution(
    mixture: AttributeLookupMixture,
) -> List[Tuple[float, float]]:
    """``atom`` variant: (runtime_ms, weight) atoms from each group's p50."""

    atoms: Dict[float, float] = {}
    for _name, weight, group in mixture.entries:
        atoms[float(group.runtime_p50_ms)] = atoms.get(float(group.runtime_p50_ms), 0.0) + weight
    return sorted(atoms.items())


def mix_empirical_runtime_distribution(
    mixture: AttributeLookupMixture,
    *,
    n_bins: int,
) -> Tuple[List[float], float]:
    """``hist16`` variant: per-bin probabilities after mixing group histograms.

    Returns ``(probs, mass_without_histogram)``.  Groups that carry no histogram
    (because the fallback level has no training samples) contribute their weight to
    the second return value instead of silently landing in some bin.
    """

    probs = [0.0] * n_bins
    missing_mass = 0.0
    for _name, weight, group in mixture.entries:
        histogram = group.runtime_bin_probs
        if histogram is None or len(histogram) != n_bins:
            missing_mass += weight
            continue
        for index, value in enumerate(histogram):
            probs[index] += weight * float(value)
    return probs, missing_mass


def canonical_views_from_atoms(atoms: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """mean / p50 / p90 / p95 / CVaR95 of a weighted atom list."""

    if not atoms:
        raise ValueError("no atoms to summarise")
    total_weight = sum(weight for _value, weight in atoms)
    if total_weight <= 0.0:
        raise ValueError("atom weights sum to %.6f" % total_weight)
    ordered = sorted(atoms)
    mean = sum(value * weight for value, weight in ordered) / total_weight

    def quantile(tau: float) -> float:
        running = 0.0
        for value, weight in ordered:
            running += weight / total_weight
            if running >= tau - 1e-12:
                return float(value)
        return float(ordered[-1][0])

    alpha = 0.95
    cvar = 0.0
    running = 0.0
    for value, weight in ordered:
        previous = running
        running += weight / total_weight
        if running > alpha:
            cvar += (running - max(previous, alpha)) * float(value)
    cvar /= max(1e-9, 1.0 - alpha)

    return {
        "runtime_mean_ms": mean,
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "cvar95_ms": cvar,
    }


def canonical_views_from_probs(
    probs: Sequence[float], representatives: Sequence[float], alpha: float = 0.95
) -> Dict[str, float]:
    """Same contract as the resource-v2 canonical views, for a bin histogram."""

    if len(probs) != len(representatives):
        raise ValueError("probs and representatives differ in length")
    cdf = 0.0
    mean = 0.0
    tail = 0.0
    previous = 0.0
    views: Dict[str, float] = {}
    for prob, rep in zip(probs, representatives):
        mean += float(prob) * float(rep)
        cdf += float(prob)
        if cdf > alpha:
            tail += (cdf - max(previous, alpha)) * float(rep)
        previous = cdf
    views["runtime_mean_ms"] = mean
    views["cvar95_ms"] = tail / max(1e-9, 1.0 - alpha)
    for tau, key in ((0.50, "p50"), (0.90, "p90"), (0.95, "p95")):
        running = 0.0
        for prob, rep in zip(probs, representatives):
            running += float(prob)
            if running >= tau - 1e-12:
                views[key] = float(rep)
                break
        else:
            views[key] = float(representatives[-1])
    return views
