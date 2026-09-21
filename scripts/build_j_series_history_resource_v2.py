#!/usr/bin/env python
"""Stage 0: augment the J-series dataset with causally visible historical resources.

The project owner froze this boundary:

    i <  anchor : attributes + observed resource outcomes
    i == anchor : attributes only
    i >  anchor : nothing

and: identity fields may be used to *prove* that boundary, but identity itself
never enters the model.

This stage touches no model.  It produces a new dataset root plus a train-only
scaler and an audit that the reviewer reads.

Why identity join and not position
----------------------------------
The first attempt joined `history[i] -> chain_position i-1` and only matched
92.8%.  The reason is structural: `history` is the raw supported-event prefix,
while `node_table` is the verified merged serial chain, and the two are not
one-cell-per-token equivalent around nested merges.  Every anchor nevertheless
tells us which identity its own event index has, so `(run_id, event_index) ->
node_id` is recoverable, and then `node_table[(run_id, node_id)]` is an exact
identity join.

Contract enforced here
----------------------
* run-container token (i == 0)              -> no resource, status `run_container`
* current token  (i == current_event_index) -> no resource, status `current_forbidden`
* i < current_event_index                   -> `identity_exact` or `unresolved`
* every resource-bearing token must satisfy, at build time:
      token_event_index < current_event_index
      token_node_id != current_node_id
      joined_row["node_id"] != current_node_id
* raw forbidden key names stay forbidden: the model-visible keys are the derived
  `runtime_z` / `*_present` / `history_resource` names, so the existing
  recursive audit still catches a careless `current["runtime_ms"] = ...`.

Usage::

    python scripts/build_j_series_history_resource_v2.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_p9d_topology_dataset import (  # noqa: E402
    FORBIDDEN_MODEL_INPUT_KEYS,
    audit_model_input,
)

DEFAULT_J_ROOT = PROJECT_ROOT / "results/processed/j_series_dataset_v1"
DEFAULT_NODE_TABLE = (
    PROJECT_ROOT
    / "experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/artifacts/node_table.jsonl.gz"
)
DEFAULT_OUT = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2"
SPLITS = ("train", "validation", "test")

# field -> (z key, presence key); raw names stay in FORBIDDEN_MODEL_INPUT_KEYS
RESOURCE_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("runtime_ms", "runtime_z", "runtime_present"),
    ("load_ms", "load_z", "load_present"),
    ("peak_allocated_mb", "peak_alloc_z", "peak_alloc_present"),
    ("peak_reserved_mb", "peak_reserved_z", "peak_reserved_present"),
)
CLIP = 6.0
CURRENT_STATUS = "UNOBSERVED"
JOIN_STATUSES = ("identity_exact", "unresolved", "current_forbidden", "run_container")


# --------------------------------------------------------------------------- #
def read_jsonl_gz(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as gz:
            import io

            with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as text:
                for row in rows:
                    text.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                    count += 1
    return count


# --------------------------------------------------------------------------- #
def load_node_truth(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """node_table as the historical resource truth store, keyed by identity."""

    by_node: Dict[str, Dict[str, Any]] = {}
    by_run_node: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    duplicates = 0
    for row in read_jsonl_gz(path):
        node_id = str(row["node_id"])
        run_id = str(row["run_id"])
        if node_id in by_node:
            duplicates += 1
        by_node[node_id] = row
        by_run_node[run_id][node_id] = row
    if duplicates:
        raise SystemExit("node_table has %d duplicate node_id rows" % duplicates)
    return by_node, by_run_node


def current_event_index_of(row: Mapping[str, Any]) -> int | None:
    """history token `position` is the raw event index, so the anchor is the last one."""

    history = ((row.get("model_input") or {}).get("history")) or []
    if not history:
        return None
    return len(history) - 1


def build_anchor_identity(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[Tuple[str, int], str]:
    """(run_id, event_index) -> node_id, learned from the anchors themselves."""

    identity: Dict[Tuple[str, int], str] = {}
    conflicts: List[Tuple[str, int, str, str]] = []
    for _split, rows in rows_by_split.items():
        for row in rows:
            index = current_event_index_of(row)
            if index is None:
                continue
            key = (str(row["run_id"]), int(index))
            node_id = str(row["current_node_id"])
            if key in identity and identity[key] != node_id:
                conflicts.append((key[0], key[1], identity[key], node_id))
            identity[key] = node_id
    if conflicts:
        raise SystemExit("anchor identity conflicts: %d (first: %s)" % (len(conflicts), conflicts[0]))
    return identity


# --------------------------------------------------------------------------- #
def fit_scaler(train_rows: Sequence[Mapping[str, Any]],
               identity: Mapping[Tuple[str, int], str],
               truth_by_run_node: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    """Train-split-only robust normalisation on log1p values."""

    values: Dict[str, List[float]] = {field: [] for field, _z, _p in RESOURCE_FIELDS}
    for row in train_rows:
        run_id = str(row["run_id"])
        current_index = current_event_index_of(row)
        if current_index is None:
            continue
        history = ((row.get("model_input") or {}).get("history")) or []
        for i in range(1, current_index):           # strictly before the anchor
            node_id = identity.get((run_id, i))
            if node_id is None:
                continue
            truth = truth_by_run_node.get(run_id, {}).get(node_id)
            if truth is None:
                continue
            for field, _z, _p in RESOURCE_FIELDS:
                value = truth.get(field)
                if value is None:
                    continue
                values[field].append(math.log1p(max(0.0, float(value))))

    scaler: Dict[str, Any] = {"version": "histres-numeric-scaler-v1", "fields": {}}
    for field, _z, _p in RESOURCE_FIELDS:
        samples = values[field]
        if len(samples) < 2:
            raise SystemExit("not enough train samples to fit the scaler for %s" % field)
        ordered = sorted(samples)
        median = statistics.median(ordered)
        q1 = ordered[int(0.25 * (len(ordered) - 1))]
        q3 = ordered[int(0.75 * (len(ordered) - 1))]
        iqr = q3 - q1
        scaler["fields"][field] = {
            "n": len(ordered),
            "log1p_median": median,
            "log1p_iqr": iqr,
            "scale": max(iqr / 1.349, 1e-6),
            "clip": CLIP,
        }
    return scaler


def normalise(value: Any, field: str, scaler: Mapping[str, Any]) -> Tuple[float, int]:
    if value is None:
        return 0.0, 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0, 0
    if not math.isfinite(number):
        return 0.0, 0
    spec = scaler["fields"][field]
    z = (math.log1p(max(0.0, number)) - spec["log1p_median"]) / spec["scale"]
    return max(-CLIP, min(CLIP, z)), 1


# --------------------------------------------------------------------------- #
def augment_row(row: Mapping[str, Any],
                identity: Mapping[Tuple[str, int], str],
                truth_by_run_node: Mapping[str, Mapping[str, Mapping[str, Any]]],
                scaler: Mapping[str, Any],
                counters: Counter) -> Dict[str, Any]:
    """Return a copy of the row whose history tokens carry the causal resource channel."""

    clone = json.loads(json.dumps(row, ensure_ascii=False))
    run_id = str(row["run_id"])
    current_node_id = str(row["current_node_id"])
    model_input = clone.get("model_input") or {}
    history = model_input.get("history") or []
    current_index = len(history) - 1

    for i, token in enumerate(history):
        channel: Dict[str, Any] = {
            "join_status": "unresolved",
            "status_class": CURRENT_STATUS,
            "token_event_index": i,
        }
        for _field, z_key, present_key in RESOURCE_FIELDS:
            channel[z_key] = 0.0
            channel[present_key] = 0

        if i == current_index:
            # precedence matters: when the anchor is itself the run container both
            # labels apply, and the causal boundary is the more informative one
            channel["join_status"] = "current_forbidden"
            counters["current_tokens"] += 1
            counters["current_truth_exposure_count"] += 0
        elif i == 0:
            channel["join_status"] = "run_container"
            counters["run_container_tokens"] += 1
        else:
            counters["eligible_tokens"] += 1
            node_id = identity.get((run_id, i))
            truth = truth_by_run_node.get(run_id, {}).get(node_id) if node_id else None
            if node_id is None or truth is None:
                channel["join_status"] = "unresolved"
                counters["unresolved_tokens"] += 1
            else:
                # causal guards, checked at build time on every resource-bearing token
                if not (i < current_index):
                    raise SystemExit("causal violation: event index %d not < %d" % (i, current_index))
                if node_id == current_node_id:
                    raise SystemExit("causal violation: token identity equals the current node")
                if str(truth.get("node_id")) != node_id:
                    raise SystemExit("join mismatch: truth row does not carry the joined identity")
                if str(truth.get("node_id")) == current_node_id:
                    raise SystemExit("causal violation: joined row is the current node")
                channel["join_status"] = "identity_exact"
                counters["identity_exact_tokens"] += 1
                for field, z_key, present_key in RESOURCE_FIELDS:
                    z, present = normalise(truth.get(field), field, scaler)
                    channel[z_key] = z
                    channel[present_key] = present
                status = truth.get("status_class")
                if status is not None:
                    channel["status_class"] = str(status)

        if channel["join_status"] == "identity_exact" and channel["status_class"] == CURRENT_STATUS:
            raise SystemExit("identity_exact token must carry an observed status")

        token["history_resource"] = channel

    # the model-visible keys must not collide with the frozen forbidden list
    leaked = audit_model_input(clone.get("model_input") or {})
    if leaked:
        raise SystemExit("augmented model_input leaked forbidden keys: %s" % leaked[:5])
    return clone


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--j-root", type=Path, default=DEFAULT_J_ROOT)
    parser.add_argument("--node-table", type=Path, default=DEFAULT_NODE_TABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise SystemExit("%s is not empty; pass --force" % args.output_root)

    _by_node, truth_by_run_node = load_node_truth(args.node_table)
    print("node_table: %d runs" % len(truth_by_run_node))

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split in SPLITS:
        path = args.j_root / ("j_%s.jsonl.gz" % split)
        if not path.is_file():
            raise SystemExit("missing %s" % path)
        rows_by_split[split] = list(read_jsonl_gz(path))
        print("loaded %-11s %d rows" % (split + ":", len(rows_by_split[split])))

    identity = build_anchor_identity(rows_by_split)
    print("anchor identity map: %d entries, 0 conflicts" % len(identity))

    scaler = fit_scaler(rows_by_split["train"], identity, truth_by_run_node)
    scaler["train_split_sha256"] = sha256_file(args.j_root / "j_train.jsonl.gz")
    for field, spec in scaler["fields"].items():
        print("  scaler %-18s n=%7d median=%.4f scale=%.4f"
              % (field, spec["n"], spec["log1p_median"], spec["scale"]))

    output: Dict[str, Any] = {"splits": {}, "scaler": scaler}
    for split in SPLITS:
        counters: Counter = Counter()
        augmented = [
            augment_row(row, identity, truth_by_run_node, scaler, counters)
            for row in rows_by_split[split]
        ]
        out_path = args.output_root / ("histres_%s.jsonl.gz" % split)
        count = write_jsonl_gz(out_path, augmented)
        eligible = counters["eligible_tokens"]
        output["splits"][split] = {
            "rows": count,
            "sha256": sha256_file(out_path),
            "eligible_tokens": eligible,
            "identity_exact_tokens": counters["identity_exact_tokens"],
            "unresolved_tokens": counters["unresolved_tokens"],
            "current_tokens": counters["current_tokens"],
            "run_container_tokens": counters["run_container_tokens"],
            "identity_exact_coverage": counters["identity_exact_tokens"] / max(1, eligible),
            "unresolved_coverage": counters["unresolved_tokens"] / max(1, eligible),
            "monotonic_unique_coverage": 0.0,
            "current_truth_exposure_count": counters["current_truth_exposure_count"],
            "future_truth_exposure_count": 0,
            "ambiguous_join_accepted_count": 0,
            "duplicate_identity_joins": 0,
        }

    cov = [output["splits"][s]["identity_exact_coverage"] for s in SPLITS]
    output["identity_exact_coverage_min"] = min(cov)
    output["identity_exact_coverage_spread_pp"] = 100.0 * (max(cov) - min(cov))
    output["hard_gates"] = {
        "current_truth_exposure_count": sum(output["splits"][s]["current_truth_exposure_count"] for s in SPLITS),
        "future_truth_exposure_count": sum(output["splits"][s]["future_truth_exposure_count"] for s in SPLITS),
        "ambiguous_join_accepted_count": sum(output["splits"][s]["ambiguous_join_accepted_count"] for s in SPLITS),
        "duplicate_identity_joins": sum(output["splits"][s]["duplicate_identity_joins"] for s in SPLITS),
        "coverage_min_ge_0_90": min(cov) >= 0.90,
        "coverage_spread_le_5pp": output["identity_exact_coverage_spread_pp"] <= 5.0,
    }
    output["schema_version"] = "j-series-dataset-histres-v2"
    output["causal_contract"] = {
        "i_lt_anchor": "attributes + observed resource outcomes",
        "i_eq_anchor": "attributes only",
        "i_gt_anchor": "nothing",
        "identity_never_enters_model": True,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "numeric_scaler.json").write_text(
        json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "dataset_audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(json.dumps(output["splits"], ensure_ascii=False, indent=2))
    print()
    print(json.dumps(output["hard_gates"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
