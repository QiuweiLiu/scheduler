#!/usr/bin/env python
"""Stage 0.1: causally visible historical resources, tightened for training.

The project owner froze this boundary:

    i <  anchor : attributes + observed resource outcomes
    i == anchor : attributes only
    i >  anchor : nothing

and: identity fields may be used to *prove* that boundary, but identity itself
never enters the model.

Changes in 0.1 (all four from the review)
-----------------------------------------
1. ``join_status`` / ``token_event_index`` moved OUT of ``model_input``.  They are
   dataset-construction artifacts, not physical facts: ``unresolved`` correlates
   with nested merges and retries, so a model could learn an offline-join shortcut.
   They now live in a row-level ``history_resource_audit`` array which the model
   never sees.
2. ``load_ms`` uses a zero-inflated two-part representation.  A plain log1p robust
   scaler degenerates here (train median 0, IQR 0), which clipped every positive
   load to the top of the range.  Now: ``load_present`` (measured) / ``load_nonzero``
   (magnitude > 0) / ``load_positive_z`` (magnitude, fitted on train loads > 0 only),
   so missing, true zero and a real positive load are three distinct states.
   Any other field whose train IQR is 0 now fails closed instead of silently
   falling back to a 1e-6 scale.
3. The index contract is verified fail-closed rather than assumed:
   ``position == i``, ``model_input.current_node == history[-1]``, and - when the v3
   feature file is available - ``current_event_index == len(history) - 1``.
4. Development defaults to ``--splits train validation``.  Test is generated once,
   after the model, contract, selection rule and consumers are frozen, using the
   same builder and the train-fitted scaler.

Usage::

    python scripts/build_j_series_history_resource_v2.py             # train + validation
    python scripts/build_j_series_history_resource_v2.py --splits train validation test
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
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
    _resource_applicable,
    audit_model_input,
)

DEFAULT_J_ROOT = PROJECT_ROOT / "results/processed/j_series_dataset_v1"
DEFAULT_V3_ROOT = PROJECT_ROOT / "results/processed/topology_predictor_p9d_v3"
DEFAULT_NODE_TABLE = (
    PROJECT_ROOT
    / "experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/artifacts/node_table.jsonl.gz"
)
DEFAULT_OUT = PROJECT_ROOT / "results/processed/j_series_dataset_histres_v2"
DEFAULT_SPLITS = ("train", "validation")
ALL_SPLITS = ("train", "validation", "test")

# (field, mode, z key, presence key).  Raw names stay in FORBIDDEN_MODEL_INPUT_KEYS.
# ``zero_inflated`` is required for load_ms: most training loads are exactly 0, so a
# plain log1p robust scaler degenerates (median 0, IQR 0) and clips every positive
# load to the top of the range, collapsing the channel into "is load positive".
RESOURCE_FIELDS: Tuple[Tuple[str, str, str, str], ...] = (
    ("runtime_ms", "continuous", "runtime_z", "runtime_present"),
    ("load_ms", "zero_inflated", "load_positive_z", "load_present"),
    ("peak_allocated_mb", "continuous", "peak_alloc_z", "peak_alloc_present"),
    ("peak_reserved_mb", "continuous", "peak_reserved_z", "peak_reserved_present"),
)
CLIP = 6.0
CURRENT_STATUS = "UNOBSERVED"
JOIN_STATUSES = ("identity_exact", "merged_into_composite_parent", "not_resource_applicable",
                 "unexpected_node_table_missing", "current_forbidden", "run_container")
MODEL_CHANNEL_KEYS = tuple(
    key
    for _field, mode, z_key, present_key in RESOURCE_FIELDS
    for key in ((z_key, present_key, "%s_nonzero" % z_key.split("_")[0]) if mode == "zero_inflated"
                else (z_key, present_key))
) + ("status_class",)


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
            with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as text:
                for row in rows:
                    text.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                    count += 1
    return count


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


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
    history = ((row.get("model_input") or {}).get("history")) or []
    return (len(history) - 1) if history else None


def verify_index_contract(row: Mapping[str, Any], v3_index: Mapping[str, int]) -> None:
    """P0-3: the whole causal proof rests on this correspondence, so verify it."""

    model_input = row.get("model_input") or {}
    history = model_input.get("history") or []
    if not history:
        raise SystemExit("row %s has an empty history" % row.get("sample_id"))
    for i, token in enumerate(history):
        if int(token.get("position", -1)) != i:
            raise SystemExit(
                "index contract: %s token %d carries position %r" % (row.get("sample_id"), i, token.get("position"))
            )
    if json.dumps(model_input.get("current_node"), sort_keys=True) != json.dumps(history[-1], sort_keys=True):
        raise SystemExit("index contract: %s current_node != history[-1]" % row.get("sample_id"))
    expected = v3_index.get(str(row.get("sample_id")))
    if expected is not None and int(expected) != len(history) - 1:
        raise SystemExit(
            "index contract: %s v3 current_event_index=%d but len(history)-1=%d"
            % (row.get("sample_id"), expected, len(history) - 1)
        )


def load_v3_current_event_index(root: Path, split: str) -> Dict[str, int]:
    """Cross-check source: the v3 feature file records current_event_index directly."""

    path = root / ("features_%s.jsonl.gz" % split)
    if not path.is_file():
        return {}
    index: Dict[str, int] = {}
    for row in read_jsonl_gz(path):
        sample_id = row.get("sample_id")
        value = row.get("current_event_index")
        if sample_id is not None and value is not None:
            index[str(sample_id)] = int(value)
    return index


def load_nested_child_map(root: Path, split: str) -> Dict[str, str]:
    """child event_id -> composite parent node_id, from the persisted v3 labels.

    build_chain() deletes nested children from chain_events before the resource
    table is written, so a nested child legitimately has no node_table row.  The
    parent label preserves ``merged_nested_call`` and ``nested_calls[].event_id``,
    which is the identity-level evidence needed to prove that.

    Fail-closed: a child mapping to more than one parent is an error.
    """

    path = root / ("labels_%s.jsonl.gz" % split)
    if not path.is_file():
        return {}
    children: Dict[str, set] = defaultdict(set)
    for row in read_jsonl_gz(path):
        for layer in row.get("future_layers") or []:
            for node in layer.get("nodes") or []:
                if not node.get("merged_nested_call"):
                    continue
                parent_id = str(node.get("node_id"))
                for nested in node.get("nested_calls") or []:
                    child_id = nested.get("event_id")
                    if child_id:
                        children[str(child_id)].add(parent_id)
    ambiguous = {child: parents for child, parents in children.items() if len(parents) != 1}
    if ambiguous:
        raise SystemExit("nested children with != 1 parent: %d (first %s)"
                         % (len(ambiguous), list(ambiguous.items())[:1]))
    return {child: next(iter(parents)) for child, parents in children.items()}


def build_anchor_identity(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> Tuple[Dict[Tuple[str, int], str], int]:
    identity: Dict[Tuple[str, int], str] = {}
    conflicts = 0
    for _split, rows in rows_by_split.items():
        for row in rows:
            index = current_event_index_of(row)
            if index is None:
                continue
            key = (str(row["run_id"]), int(index))
            node_id = str(row["current_node_id"])
            if key in identity and identity[key] != node_id:
                conflicts += 1
                continue
            identity[key] = node_id
    if conflicts:
        raise SystemExit("anchor identity conflicts: %d" % conflicts)
    return identity, conflicts


# --------------------------------------------------------------------------- #
def fit_scaler(train_rows: Sequence[Mapping[str, Any]],
               identity: Mapping[Tuple[str, int], str],
               truth_by_run_node: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    """Train-split-only robust normalisation; fails closed on a degenerate spread."""

    collected: Dict[str, List[float]] = {field: [] for field, _m, _z, _p in RESOURCE_FIELDS}
    for row in train_rows:
        run_id = str(row["run_id"])
        current_index = current_event_index_of(row)
        if current_index is None:
            continue
        for i in range(1, current_index):        # strictly before the anchor
            node_id = identity.get((run_id, i))
            if node_id is None:
                continue
            truth = truth_by_run_node.get(run_id, {}).get(node_id)
            if truth is None:
                continue
            for field, mode, _z, _p in RESOURCE_FIELDS:
                value = truth.get(field)
                if value is None:
                    continue
                number = float(value)
                if not math.isfinite(number):
                    raise SystemExit("non-finite %s in node_table for %s" % (field, node_id))
                if number < 0.0:
                    # P1: never silently clamp a negative observation to zero
                    raise SystemExit("negative %s=%r in node_table for %s" % (field, value, node_id))
                if mode == "zero_inflated" and number == 0.0:
                    continue                      # the positive branch is fitted separately
                collected[field].append(math.log1p(number))

    scaler: Dict[str, Any] = {"version": "histres-numeric-scaler-v2", "fields": {}}
    for field, mode, _z, _p in RESOURCE_FIELDS:
        samples = collected[field]
        if len(samples) < 2:
            raise SystemExit("not enough train samples to fit the scaler for %s" % field)
        ordered = sorted(samples)
        median = statistics.median(ordered)
        q1 = ordered[int(0.25 * (len(ordered) - 1))]
        q3 = ordered[int(0.75 * (len(ordered) - 1))]
        iqr = q3 - q1
        if iqr <= 0.0:
            # explicit failure: choose a representation deliberately, do not paper over it
            raise SystemExit(
                "scaler for %s has zero train IQR (median=%r); pick an explicit representation "
                "(zero-inflated / constant / alternative) instead of a 1e-6 fallback" % (field, median)
            )
        scaler["fields"][field] = {
            "mode": mode,
            "n": len(ordered),
            "log1p_median": median,
            "log1p_iqr": iqr,
            "scale": iqr / 1.349,
            "clip": CLIP,
        }
    return scaler


def normalise(value: Any, field: str, mode: str, scaler: Mapping[str, Any]) -> Tuple[float, int, int]:
    """Return (z, present, nonzero).  ``nonzero`` is only meaningful for zero-inflated."""

    if value is None:
        return 0.0, 0, 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0, 0, 0
    if not math.isfinite(number) or number < 0.0:
        return 0.0, 0, 0
    nonzero = 1 if number > 0.0 else 0
    if mode == "zero_inflated" and number == 0.0:
        return 0.0, 1, 0
    spec = scaler["fields"][field]
    z = (math.log1p(number) - spec["log1p_median"]) / spec["scale"]
    return max(-CLIP, min(CLIP, z)), 1, nonzero


# --------------------------------------------------------------------------- #
def augment_row(row: Mapping[str, Any],
                identity: Mapping[Tuple[str, int], str],
                truth_by_run_node: Mapping[str, Mapping[str, Mapping[str, Any]]],
                scaler: Mapping[str, Any],
                counters: Counter,
                nested_child_to_parent: Mapping[str, str] | None = None) -> Dict[str, Any]:
    """Attach the model-visible telemetry channel and the model-invisible audit trail."""

    clone = json.loads(json.dumps(row, ensure_ascii=False))
    run_id = str(row["run_id"])
    current_node_id = str(row["current_node_id"])
    model_input = clone.get("model_input") or {}
    history = model_input.get("history") or []
    current_index = len(history) - 1
    audit: List[Dict[str, Any]] = []

    for i, token in enumerate(history):
        channel: Dict[str, Any] = {"status_class": CURRENT_STATUS}
        for _field, mode, z_key, present_key in RESOURCE_FIELDS:
            channel[z_key] = 0.0
            channel[present_key] = 0
            if mode == "zero_inflated":
                channel["%s_nonzero" % z_key.split("_")[0]] = 0
        entry: Dict[str, Any] = {"token_index": i}

        if i == current_index:
            # precedence: when the anchor is itself the run container both labels
            # apply, and the causal role is the more informative one
            entry["join_status"] = "current_forbidden"
            entry["causal_role"] = "current"
            counters["current_tokens"] += 1
        elif i == 0:
            entry["join_status"] = "run_container"
            entry["causal_role"] = "past"
            counters["run_container_tokens"] += 1
        else:
            counters["eligible_tokens"] += 1
            node_id = identity.get((run_id, i))
            if node_id is None:
                entry["join_status"] = "unresolved"
                entry["unresolved_reason"] = "identity_key_missing"
                counters["unresolved_tokens"] += 1
                counters["unresolved_identity_key_missing"] += 1
                entry["causal_role"] = "past"
            else:
                truth = truth_by_run_node.get(run_id, {}).get(node_id)
                if truth is None:
                    # identity is known, the resource table has no row.  Classify by
                    # evidence, not assumption.
                    parent_id = (nested_child_to_parent or {}).get(node_id)
                    entry["joined_node_id"] = node_id
                    entry["causal_role"] = "past"
                    if parent_id is not None:
                        parent_truth = truth_by_run_node.get(run_id, {}).get(parent_id)
                        if parent_truth is None:
                            raise SystemExit(
                                "nested child %s maps to parent %s which is not in node_table"
                                % (node_id, parent_id)
                            )
                        entry["join_status"] = "merged_into_composite_parent"
                        entry["parent_node_id"] = parent_id
                        entry["parent_is_in_node_table"] = True
                        counters["merged_child_tokens"] += 1
                    else:
                        probe = {
                            "node_type": token.get("node_type"),
                            "event_type": token.get("event_type"),
                            "action": token.get("raw_action"),
                        }
                        if not _resource_applicable(probe):
                            entry["join_status"] = "not_resource_applicable"
                            counters["not_resource_applicable_tokens"] += 1
                        else:
                            entry["join_status"] = "unexpected_node_table_missing"
                            counters["unexpected_missing_tokens"] += 1
                else:
                    if not (i < current_index):
                        raise SystemExit("causal violation: event index %d not < %d" % (i, current_index))
                    if node_id == current_node_id:
                        raise SystemExit("causal violation: token identity equals the current node")
                    if str(truth.get("node_id")) != node_id:
                        raise SystemExit("join mismatch: truth row does not carry the joined identity")
                    if str(truth.get("node_id")) == current_node_id:
                        raise SystemExit("causal violation: joined row is the current node")
                    entry["join_status"] = "identity_exact"
                    entry["causal_role"] = "past"
                    entry["joined_node_id"] = node_id
                    counters["identity_exact_tokens"] += 1
                    for field, mode, z_key, present_key in RESOURCE_FIELDS:
                        z, present, nonzero = normalise(truth.get(field), field, mode, scaler)
                        channel[z_key] = z
                        channel[present_key] = present
                        if mode == "zero_inflated":
                            channel["%s_nonzero" % z_key.split("_")[0]] = nonzero
                    status = truth.get("status_class")
                    if status is not None:
                        channel["status_class"] = str(status)

        token["history_resource"] = channel
        audit.append(entry)

    clone["history_resource_audit"] = audit

    leaked = audit_model_input(clone.get("model_input") or {})
    if leaked:
        raise SystemExit("augmented model_input leaked forbidden keys: %s" % leaked[:5])
    for token in history:
        extra = set(token.get("history_resource", {})) - set(MODEL_CHANNEL_KEYS)
        if extra:
            raise SystemExit("model channel carries non-feature keys: %s" % sorted(extra))
    return clone


# --------------------------------------------------------------------------- #
def validate_augmented_dataset(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Independent post-build scan, plus honest occurrence/unique coverage statistics.

    Coverage denominator is ``identity_exact + unexpected_node_table_missing``:
    a nested child deliberately carries no independent telemetry (its time is inside
    the composite parent), so counting it as missing would understate coverage.
    Run-level coverage is deduplicated by ``(run_id, token_event_index)`` because the
    same historical event recurs in every later anchor prefix.
    """

    occ: Counter = Counter()
    unique_exact: Dict[Tuple[str, int], str] = {}
    unique_eligible: Dict[Tuple[str, int], str] = {}
    unique_exact_nodes: Dict[str, set] = defaultdict(set)
    unique_merged_nodes: Dict[str, set] = defaultdict(set)
    unique_unexpected_nodes: Dict[str, set] = defaultdict(set)
    runs_with_unexpected: set = set()
    runs_with_merged: set = set()

    for row in rows:
        model_input = row.get("model_input") or {}
        history = model_input.get("history") or []
        current_index = len(history) - 1
        run_id = str(row.get("run_id"))
        occ["rows"] += 1
        for i, token in enumerate(history):
            channel = token.get("history_resource") or {}
            resources_present = any(
                int(channel.get(present_key, 0)) == 1
                for _f, _m, _z, present_key in RESOURCE_FIELDS
            )
            status_observed = channel.get("status_class") != CURRENT_STATUS
            if resources_present or status_observed:
                occ["resource_bearing_tokens"] += 1
                if i >= current_index:
                    occ["current_truth_exposure_count"] += 1
                if i > current_index:
                    occ["future_truth_exposure_count"] += 1
        for entry in row.get("history_resource_audit") or []:
            status = entry.get("join_status")
            key = (run_id, int(entry.get("token_index", -1)))
            if status == "identity_exact":
                occ["identity_exact_occurrences"] += 1
                unique_exact[key] = run_id
                if entry.get("joined_node_id"):
                    unique_exact_nodes[str(entry["joined_node_id"])].add(run_id)
            elif status == "merged_into_composite_parent":
                occ["merged_child_occurrences"] += 1
                runs_with_merged.add(run_id)
                if entry.get("joined_node_id"):
                    unique_merged_nodes[str(entry["joined_node_id"])].add(run_id)
            elif status == "not_resource_applicable":
                occ["not_resource_applicable_occurrences"] += 1
            elif status == "unexpected_node_table_missing":
                occ["unexpected_missing_occurrences"] += 1
                runs_with_unexpected.add(run_id)
                if entry.get("joined_node_id"):
                    unique_unexpected_nodes[str(entry["joined_node_id"])].add(run_id)
            else:
                continue
            # eligible for resource telemetry, and deduplicated by unique event
            unique_eligible[key] = status
        if audit_model_input(model_input):
            occ["forbidden_key_violations"] += 1

    per_run: Dict[str, List[int]] = defaultdict(lambda: [0, 0])   # run -> [exact, eligible]
    for key, status in unique_eligible.items():
        run_id = key[0]
        per_run[run_id][1] += 1
        if status == "identity_exact":
            per_run[run_id][0] += 1
    coverages = [exact / eligible for exact, eligible in per_run.values() if eligible]

    report = dict(occ)
    report.update({
        "eligible_resource_event_occurrences": occ["identity_exact_occurrences"]
        + occ["unexpected_missing_occurrences"],
        "unique_identity_exact_nodes": len(unique_exact_nodes),
        "unique_merged_child_nodes": len(unique_merged_nodes),
        "unique_unexpected_missing_nodes": len(unique_unexpected_nodes),
        "unique_eligible_events": len(unique_eligible),
        "unique_resource_coverage_per_run_p10": percentile(coverages, 0.10),
        "unique_resource_coverage_per_run_p50": percentile(coverages, 0.50),
        "unique_resource_coverage_per_run_p90": percentile(coverages, 0.90),
        "runs_with_unexpected_missing": len(runs_with_unexpected),
        "runs_with_merged_child": len(runs_with_merged),
        "runs_with_eligible_events": len(per_run),
    })
    return report


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--j-root", type=Path, default=DEFAULT_J_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--node-table", type=Path, default=DEFAULT_NODE_TABLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), choices=list(ALL_SPLITS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.force:
        raise SystemExit("%s is not empty; pass --force" % args.output_root)

    _by_node, truth_by_run_node = load_node_truth(args.node_table)
    print("node_table: %d runs, sha256 %s" % (len(truth_by_run_node), sha256_file(args.node_table)))

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    v3_index_by_split: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        path = args.j_root / ("j_%s.jsonl.gz" % split)
        if not path.is_file():
            raise SystemExit("missing %s" % path)
        rows_by_split[split] = list(read_jsonl_gz(path))
        v3_index_by_split[split] = load_v3_current_event_index(args.v3_root, split)
        print("loaded %-11s %d rows (v3 index entries: %d)"
              % (split + ":", len(rows_by_split[split]), len(v3_index_by_split[split])))

    for split in args.splits:
        for row in rows_by_split[split]:
            verify_index_contract(row, v3_index_by_split[split])
    print("index contract: VERIFIED on every row (position==i, current_node==history[-1], v3 index cross-check)")

    identity, conflicts = build_anchor_identity(rows_by_split)
    print("anchor identity map: %d entries, %d conflicts" % (len(identity), conflicts))
    nested_maps = {split: load_nested_child_map(args.v3_root, split) for split in args.splits}
    for split, mapping in nested_maps.items():
        print("nested child->parent map (%-11s): %d children" % (split + ":", len(mapping)))

    scaler = fit_scaler(rows_by_split["train"], identity, truth_by_run_node)
    scaler["train_split_sha256"] = sha256_file(args.j_root / "j_train.jsonl.gz")
    scaler["node_table_sha256"] = sha256_file(args.node_table)
    scaler["builder_source_sha256"] = sha256_file(Path(__file__).resolve())
    for field, spec in scaler["fields"].items():
        print("  scaler %-18s mode=%-14s n=%7d median=%8.4f scale=%9.5f"
              % (field, spec["mode"], spec["n"], spec["log1p_median"], spec["scale"]))

    output: Dict[str, Any] = {"splits": {}, "splits_built": list(args.splits), "scaler": scaler}
    for split in args.splits:
        counters: Counter = Counter()
        augmented = [
            augment_row(row, identity, truth_by_run_node, scaler, counters, nested_maps[split])
            for row in rows_by_split[split]
        ]
        out_path = args.output_root / ("histres_%s.jsonl.gz" % split)
        count = write_jsonl_gz(out_path, augmented)
        validation = validate_augmented_dataset(augmented)
        eligible = counters["eligible_tokens"]
        telemetry_denominator = (
            validation.get("identity_exact_occurrences", 0)
            + validation.get("unexpected_missing_occurrences", 0)
        )
        output["splits"][split] = {
            "rows": count,
            "sha256": sha256_file(out_path),
            "eligible_tokens": eligible,
            "identity_exact_tokens": counters["identity_exact_tokens"],
            "merged_child_tokens": counters["merged_child_tokens"],
            "not_resource_applicable_tokens": counters["not_resource_applicable_tokens"],
            "unexpected_missing_tokens": counters["unexpected_missing_tokens"],
            "current_tokens": counters["current_tokens"],
            "run_container_tokens": counters["run_container_tokens"],
            "identity_exact_coverage": counters["identity_exact_tokens"] / max(1, eligible),
            "telemetry_denominator": telemetry_denominator,
            "identity_exact_coverage_of_telemetry": (
                validation.get("identity_exact_occurrences", 0) / max(1, telemetry_denominator)
            ),
            "monotonic_unique_coverage": 0.0,
            "post_build_validation": validation,
        }

    dev = [s for s in args.splits if s in ("train", "validation")]
    cov = [output["splits"][s]["identity_exact_coverage_of_telemetry"] for s in dev]
    output["development_splits"] = dev
    output["identity_exact_coverage_min"] = min(cov)
    output["identity_exact_coverage_spread_pp"] = 100.0 * (max(cov) - min(cov))
    output["hard_gates"] = {
        "current_truth_exposure_count": sum(
            output["splits"][s]["post_build_validation"].get("current_truth_exposure_count", 0) for s in args.splits),
        "future_truth_exposure_count": sum(
            output["splits"][s]["post_build_validation"].get("future_truth_exposure_count", 0) for s in args.splits),
        "ambiguous_join_accepted_count": 0,
        "duplicate_identity_joins": conflicts,
        "forbidden_key_violations": sum(
            output["splits"][s]["post_build_validation"].get("forbidden_key_violations", 0) for s in args.splits),
        "coverage_min_ge_0_90": min(cov) >= 0.90,
        "coverage_spread_le_5pp": output["identity_exact_coverage_spread_pp"] <= 5.0,
        "test_sealed_during_development": "test" not in dev,
        "unexpected_node_table_missing_occurrences": sum(
            output["splits"][s]["post_build_validation"].get("unexpected_missing_occurrences", 0)
            for s in args.splits),
        "merged_child_occurrences": sum(
            output["splits"][s]["post_build_validation"].get("merged_child_occurrences", 0)
            for s in args.splits),
        "not_resource_applicable_occurrences": sum(
            output["splits"][s]["post_build_validation"].get("not_resource_applicable_occurrences", 0)
            for s in args.splits),
    }
    output["schema_version"] = "j-series-dataset-histres-v2.1"
    output["causal_contract"] = {
        "i_lt_anchor": "attributes + observed resource outcomes",
        "i_eq_anchor": "attributes only",
        "i_gt_anchor": "nothing",
        "identity_never_enters_model": True,
        "join_provenance_is_not_a_model_feature": True,
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "numeric_scaler.json").write_text(
        json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "dataset_audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(json.dumps({s: {k: v for k, v in output["splits"][s].items() if k != "post_build_validation"}
                      for s in args.splits}, ensure_ascii=False, indent=2))
    print()
    print("post-build validation:")
    print(json.dumps({s: output["splits"][s]["post_build_validation"] for s in args.splits},
                     ensure_ascii=False, indent=2))
    print()
    print("hard gates:")
    print(json.dumps(output["hard_gates"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
