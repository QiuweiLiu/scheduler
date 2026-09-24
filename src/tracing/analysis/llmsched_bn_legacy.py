"""LEGACY LLMSched front end -- kept ONLY as a regression oracle.

Superseded by `llmsched_bn.py` v2 (LLM-1).  The formal `policy="llmsched"` path
must not import anything from this module.  It is retained so that the old behaviour
can still be reconstructed and asserted against, which is what makes the retirement
verifiable rather than merely asserted.

Everything below is byte-equivalent to the retired implementation.

ORIGINAL DOCSTRING: LLMSched-adapted: a Bayesian workflow model + uncertainty-reduction scheduling.

Paper: LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM
Applications (IEEE ICDCS 2025).  Its front end is a DAG + Bayesian network built
from historical compound-workflow traces that models future STRUCTURE uncertainty
and DURATION uncertainty, with the posterior updated as stages complete.  Its back
end measures, with entropy, how much executing a ready stage reduces future
uncertainty, and mixes that exploration objective with a JCT/SRTF exploitation
objective through an epsilon-greedy choice.

Migrated faithfully:
    train-only traces -> BN/CPDs -> posterior evidence -> entropy-based
    uncertainty reduction -> epsilon-greedy EXPLORE/EXPLOIT

NOT migrated, stated explicitly:
  * ``sample_tasks(r)``: the paper runs a *fraction r* of a stage first to gather
    information.  Our execution unit is already an indivisible graph node, so
    intra-stage partial execution is omitted rather than faked with r=1.
  * the paper's own LLM/regular executor batching environment.

Relation to the project's "no artificial joint distribution" rule: this module
learns its CPDs directly from train-only execution traces, which is the baseline's
native model.  It does NOT assemble a joint out of F0's marginal 16-bin heads.

The epsilon-greedy contract is one coin per DECISION, not per candidate: a single
draw selects the mode, and the chosen mode then ranks the candidate pool.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Sequence

BN_SCHEMA = "llmsched-bayesian-workflow-model-v1-legacy"


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)


def build_bn_profiler(templates: Mapping[str, Any]) -> Dict[str, Any]:
    """Learn the Bayesian workflow model from TRAIN-ONLY templates.

    Structure: the workflow family fixes the stage alphabet; the CPDs are
    ``P(length = m | family)`` (structure uncertainty) and the per-position runtime
    statistics (duration uncertainty).
    """

    lengths: Dict[str, List[int]] = defaultdict(list)
    runtime_by_pos: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for tpl in templates.values():
        if getattr(tpl, "split", None) != "train":
            continue
        family = str(getattr(tpl, "baseline", "") or "unknown")
        order = sorted(tpl.nodes, key=lambda n: n.sequence_index)
        lengths[family].append(len(order))
        for i, n in enumerate(order):
            runtime_by_pos[family][i].append(float(n.runtime_ms))

    families: Dict[str, Any] = {}
    for family, ls in lengths.items():
        total = len(ls)
        length_counts = Counter(ls)
        families[family] = {
            "n_runs": total,
            "length_counts": {int(k): int(v) for k, v in sorted(length_counts.items())},
            "length_prob": {int(k): v / total for k, v in sorted(length_counts.items())},
            "pos_runtime_mean": {
                int(k): (sum(v) / len(v)) for k, v in sorted(runtime_by_pos[family].items())
            },
            "pos_runtime_samples": {
                int(k): sorted(v) for k, v in sorted(runtime_by_pos[family].items())
            },
        }
    return {"schema": BN_SCHEMA, "families": families}


def posterior_length_probs(profiler: Mapping[str, Any], family: str, consumed: int) -> Dict[int, float]:
    """``P(n = m | family, n >= consumed)`` -- the posterior over the total length."""

    fam = (profiler.get("families") or {}).get(str(family))
    if fam is None:
        raise KeyError("BN profiler has no family %r" % family)
    raw = {int(k): float(v) for k, v in fam["length_prob"].items()}
    kept = {m: p for m, p in raw.items() if m >= int(consumed)}
    if not kept:
        return {}
    z = sum(kept.values())
    return {m: p / z for m, p in kept.items()}


def posterior_entropy(profiler: Mapping[str, Any], family: str, consumed: int) -> float:
    """H(n | family, n >= consumed): uncertainty about how much work remains."""

    return _entropy(list(posterior_length_probs(profiler, family, consumed).values()))


def info_gain(profiler: Mapping[str, Any], family: str, consumed: int) -> float:
    """Expected entropy reduction about the total length from advancing one stage.

    Executing the node at position ``consumed`` reveals whether the workflow ends
    there, which moves the posterior from ``n >= consumed`` to ``n >= consumed + 1``.
    """

    h_now = posterior_entropy(profiler, family, consumed)
    h_next = posterior_entropy(profiler, family, consumed + 1)
    return max(0.0, h_now - h_next)


def expected_remaining_ms(profiler: Mapping[str, Any], family: str, consumed: int) -> float:
    """E[remaining runtime after the current node | posterior] (duration uncertainty)."""

    fam = (profiler.get("families") or {}).get(str(family))
    if fam is None:
        raise KeyError("BN profiler has no family %r" % family)
    post = posterior_length_probs(profiler, family, consumed)
    mean_by_pos = {int(k): float(v) for k, v in fam["pos_runtime_mean"].items()}
    if not post:
        return 0.0
    # for each surviving length m, sum the mean runtimes of positions consumed+1..m-1
    total = 0.0
    for m, p in post.items():
        total += p * sum(mean_by_pos.get(i, 0.0) for i in range(int(consumed) + 1, m))
    return total


def draw_mode(rng: Any, epsilon: float) -> str:
    """ONE coin per decision: returns 'EXPLORE' with probability epsilon."""

    return "EXPLORE" if float(rng.random()) < float(epsilon) else "EXPLOIT"


def duration_entropy(profiler: Mapping[str, Any], family: str, consumed: int, *, n_bins: int = 16) -> float:
    """Entropy of the runtime distribution of the stage at ``consumed`` (bits).

    This is the duration-uncertainty half of the paper's front end.  The empirical
    samples for that position are histogrammed over their own range, because the
    frozen runtime bin edges are not part of this baseline's own model.
    """

    fam = (profiler.get("families") or {}).get(str(family))
    if fam is None:
        raise KeyError("BN profiler has no family %r" % family)
    samples = list((fam.get("pos_runtime_samples") or {}).get(int(consumed), []))
    if len(samples) < 2:
        return 0.0
    lo, hi = min(samples), max(samples)
    if hi <= lo:
        return 0.0
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in samples:
        idx = min(n_bins - 1, int((v - lo) / width))
        counts[idx] += 1
    total = float(len(samples))
    probs = [c / total for c in counts if c > 0]
    return _entropy(probs)


def duration_range(profiler: Mapping[str, Any], family: str, consumed: int) -> float:
    """The paper's ``Range(Y_m)`` factor for the stage at ``consumed``."""

    fam = (profiler.get("families") or {}).get(str(family))
    if fam is None:
        raise KeyError("BN profiler has no family %r" % family)
    samples = list((fam.get("pos_runtime_samples") or {}).get(int(consumed), []))
    if not samples:
        return 0.0
    return max(samples) - min(samples)


def uncertainty_reduction(profiler: Mapping[str, Any], family: str, consumed: int) -> float:
    """R(X) in the paper's spirit: information x range.

    Structure term (mutual information about the total length) is included and is
    empirically ~0 on this corpus; the duration term carries the signal.
    """

    structural = info_gain(profiler, family, consumed)
    duration = duration_entropy(profiler, family, consumed)
    spread = duration_range(profiler, family, consumed)
    return (structural + duration) * max(1.0, spread)
