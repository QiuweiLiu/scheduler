"""Phase 21 — per-step calibration audit for a frozen prediction pack.

Used for the masked/fixed task-context pilot: both packs share anchors, batch,
checkpoint and truth, so any difference is attributable to the three restored
context fields.

For every anchor and future slot h in 1..H the script pairs the predicted
runtime quantiles with the measured runtime of the h-th following non-container
event (whole-run containers excluded, as established in Phase 17/18) and reports:

    R_tau       = sum(q_tau) / sum(y)          ratio of sums (1.0 = calibrated)
    C_tau       = P(y <= q_tau)                empirical coverage
    pinball_tau = quantile (pinball) loss in ms
    per slot h, per baseline, per model stack

Read-only; the only write is the JSON summary next to --out.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TAUS = (("p50", 0.50), ("p90", 0.90), ("p95", 0.95))
DEFAULT_SEED = 20260917
CONTAINER_EVENT_TYPE = "run"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def resolve_artifacts(path: Path, horizon: int) -> Path:
    """Locate the horizon artifact across packer and scheduler layouts.

    Accepts the packer's J-named files, the scheduler's b05-named copy, either
    layout directly or inside a ``prediction_artifacts`` subdirectory, and falls
    back to the parent directory when the caller passes a not-yet-existing
    subdirectory.
    """

    if path.is_file():
        return path
    names = (f"b05_future_h{int(horizon)}.jsonl.gz", f"j_future_h{int(horizon)}.jsonl.gz")
    for root in (path, path.parent):
        for candidate in [root / name for name in names] + [
            root / "prediction_artifacts" / name for name in names
        ]:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"no future_h{horizon} artifact under {path} or its parent")


def load_slots(path: Path, horizon: int) -> dict[str, list[dict[str, Any]]]:
    slots: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(path):
        node_id = str(row.get("node_id") or "")
        if not node_id:
            continue
        scenarios = row.get(f"future_h{int(horizon)}") or []
        slots[node_id] = (scenarios[0].get("steps") or []) if scenarios else []
    return slots


def truth_sequence(template: Mapping[str, Any], anchor: str, horizon: int) -> list[float]:
    """Measured runtimes of the next non-container events after the anchor."""

    nodes = {str(node["node_id"]): node for node in template.get("nodes") or []}
    if anchor not in nodes:
        return []
    order = {node_id: int(node.get("sequence_index") or 0) for node_id, node in nodes.items()}
    step_of = {}
    for node_id, node in nodes.items():
        steps = node.get("source_step_ids") or []
        step_of[node_id] = int(steps[0]) if steps and str(steps[0]).isdigit() else order[node_id]
    considered = [
        node_id
        for node_id, node in nodes.items()
        if str(node.get("event_type") or "") != CONTAINER_EVENT_TYPE
    ]
    considered.sort(key=lambda node_id: (step_of[node_id], order[node_id]))
    anchor_index = considered.index(anchor)
    tail = considered[anchor_index + 1 : anchor_index + 1 + int(horizon)]
    return [float(nodes[node_id].get("runtime_ms") or 0.0) for node_id in tail]


def pinball(prediction: float, truth: float, tau: float) -> float:
    delta = truth - prediction
    return max(tau * delta, (tau - 1.0) * delta)


def collect(
    slots: Mapping[str, Sequence[Mapping[str, Any]]],
    templates_by_node: Mapping[str, Mapping[str, Any]],
    horizon: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for anchor, steps in slots.items():
        template = templates_by_node.get(anchor)
        if template is None or not steps:
            continue
        truths = truth_sequence(template, anchor, horizon)
        if not truths:
            continue
        for index, step in enumerate(steps[: len(truths)]):
            resource = (step.get("resource") or {}).get("runtime_ms_quantiles") or {}
            record: dict[str, Any] = {
                "anchor": anchor,
                "slot": index + 1,
                "truth": float(truths[index]),
                "video_id": str(template.get("video_id") or ""),
                "baseline": str(template.get("baseline") or ""),
                "model_stack_id": str(template.get("model_stack_id") or ""),
            }
            for tau_name, _tau in TAUS:
                value = resource.get(tau_name)
                record[tau_name] = float(value) if isinstance(value, (int, float)) else None
            records.append(record)
    return records


def summarise(records: Sequence[Mapping[str, Any]], strata: str | None = None) -> dict[str, Any]:
    def block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"pairs": len(rows)}
        for tau_name, tau in TAUS:
            pairs = [(row[tau_name], row["truth"]) for row in rows if row[tau_name] is not None]
            if not pairs:
                continue
            sum_q = sum(q for q, _ in pairs)
            sum_y = sum(y for _, y in pairs)
            out[tau_name] = {
                "pairs": len(pairs),
                "ratio_of_sums": (sum_q / sum_y) if sum_y else None,
                "coverage": statistics.fmean([1.0 if y <= q else 0.0 for q, y in pairs]),
                "pinball_ms": statistics.fmean([pinball(q, y, tau) for q, y in pairs]),
            }
        return out

    summary: dict[str, Any] = {"overall": block(records)}
    for key in ("slot", "baseline", "model_stack_id", "video_id"):
        if strata and key != strata:
            continue
        buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in records:
            buckets[str(row[key])].append(row)
        summary[f"by_{key}"] = {name: block(rows) for name, rows in sorted(buckets.items())}
    return summary


def paired_delta(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    n_boot: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Per-pair pinball difference ((fixed) - (masked)), clustered by video."""

    indexed = {
        (row["anchor"], row["slot"]): row for row in right
    }
    out: dict[str, Any] = {}
    for tau_name, tau in TAUS:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in left:
            other = indexed.get((row["anchor"], row["slot"]))
            if other is None or row[tau_name] is None or other[tau_name] is None:
                continue
            delta = pinball(other[tau_name], other["truth"], tau) - pinball(row[tau_name], row["truth"], tau)
            grouped[str(row["video_id"])].append(delta)
        keys = sorted(grouped)
        if len(keys) < 2:
            out[tau_name] = {"mean": None, "ci": None, "pairs": 0}
            continue
        values = [value for key in keys for value in grouped[key]]
        rng = random.Random(seed)
        means = []
        for _ in range(int(n_boot)):
            draw = [value for _ in range(len(keys)) for value in grouped[keys[rng.randrange(len(keys))]]]
            if draw:
                means.append(statistics.fmean(draw))
        means.sort()
        out[tau_name] = {
            "mean": statistics.fmean(values),
            "ci": [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]],
            "pairs": len(values),
        }
    return out


def load_template_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for template in iter_jsonl(path):
        for node in template.get("nodes") or []:
            index[str(node["node_id"])] = template
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixed", type=Path, required=True, help="repaired prediction pack (dir or jsonl.gz)")
    parser.add_argument("--masked", type=Path, default=None, help="control pack; when given, a paired delta is reported")
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--video-allowlist", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    templates = load_template_index(args.templates)
    allowlist = None
    if args.video_allowlist:
        allowlist = {line.strip() for line in args.video_allowlist.read_text(encoding="utf-8").splitlines() if line.strip()}

    def records_for(path: Path) -> list[dict[str, Any]]:
        slots = load_slots(resolve_artifacts(path, args.horizon), args.horizon)
        if allowlist is not None:
            slots = {
                node: steps
                for node, steps in slots.items()
                if str((templates.get(node) or {}).get("video_id") or "") in allowlist
            }
        return collect(slots, templates, args.horizon)

    fixed_records = records_for(args.fixed)
    report: dict[str, Any] = {
        "fixed": summarise(fixed_records, strata=None),
        "fixed_video": summarise(fixed_records, strata="video_id"),
        "pairs": len(fixed_records),
    }
    if args.masked:
        masked_records = records_for(args.masked)
        report["masked"] = summarise(masked_records, strata=None)
        report["paired_delta_pinball_fixed_minus_masked"] = paired_delta(
            masked_records, fixed_records, args.bootstrap, args.seed
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    head = report.get("fixed", {}).get("overall", {})
    print(
        json.dumps(
            {
                "pairs": report["pairs"],
                "fixed": {name: head.get(name) for name, _ in TAUS},
                "delta": report.get("paired_delta_pinball_fixed_minus_masked"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
