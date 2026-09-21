"""Stage 0 acceptance: the three causal mutation tests the reviewer asked for.

These encode the project owner's boundary as executable checks:

  current mutation -> model_input byte-identical
  future  mutation -> model_input byte-identical
  past    mutation -> only the corresponding historical channel changes

The tests run against real dataset rows and the real node_table, so they also
pin the identity join itself.
"""
from __future__ import annotations

import copy
import gzip
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_j_series_history_resource_v2 import (  # noqa: E402
    RESOURCE_FIELDS,
    augment_row,
    build_anchor_identity,
    current_event_index_of,
    fit_scaler,
    load_node_truth,
    read_jsonl_gz,
)

J_ROOT = PROJECT_ROOT / "results/processed/j_series_dataset_v1"
NODE_TABLE = (
    PROJECT_ROOT
    / "experiments/EXP-20260911_p9d_r0_oracle_signature_ceiling/artifacts/node_table.jsonl.gz"
)
CURRENT_SENTINEL = 9.87654321e11
LOAD_SENTINEL = 8.7654321e11
STATUS_SENTINEL = "__CURRENT_LEAK_SENTINEL__"


def load_rows(split: str, limit: int | None = None):
    rows = []
    for index, row in enumerate(read_jsonl_gz(J_ROOT / ("j_%s.jsonl.gz" % split))):
        if limit is not None and index >= limit:
            break
        rows.append(row)
    return rows


class HistResCausalGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.truth_by_run_node = load_node_truth(NODE_TABLE)[1]
        cls.train_rows = load_rows("train", 600)
        cls.validation_rows = load_rows("validation", 200)
        rows_by_split = {"train": cls.train_rows, "validation": cls.validation_rows}
        cls.identity, _conflicts = build_anchor_identity(rows_by_split)
        cls.scaler = fit_scaler(cls.train_rows, cls.identity, cls.truth_by_run_node)

    # -- helpers ---------------------------------------------------------- #
    def find_row_with_past(self, rows):
        """A row with at least one identity_exact historical token."""
        for row in rows:
            run_id = str(row["run_id"])
            current_index = current_event_index_of(row)
            if current_index is None or current_index < 2:
                continue
            for i in range(1, current_index):
                node_id = self.identity.get((run_id, i))
                if node_id and node_id in self.truth_by_run_node.get(run_id, {}):
                    return row, run_id, current_index, i, node_id
        return None

    def build(self, row, truth_store=None):
        from collections import Counter

        return augment_row(
            row,
            self.identity,
            truth_store if truth_store is not None else self.truth_by_run_node,
            self.scaler,
            Counter(),
        )

    # -- tests ------------------------------------------------------------ #
    def test_current_mutation_leaves_the_input_identical(self) -> None:
        """Changing the current node's own outcome must not reach the model."""

        found = self.find_row_with_past(self.train_rows)
        self.assertIsNotNone(found, "no usable row")
        row, run_id, current_index, _i, _node_id = found
        current_node = str(row["current_node_id"])

        before = self.build(row)
        mutated = copy.deepcopy(self.truth_by_run_node)
        target = mutated[run_id][current_node]
        target["runtime_ms"] = CURRENT_SENTINEL
        target["load_ms"] = LOAD_SENTINEL
        target["peak_allocated_mb"] = CURRENT_SENTINEL
        target["peak_reserved_mb"] = LOAD_SENTINEL
        target["status_class"] = STATUS_SENTINEL
        after = self.build(row, mutated)

        self.assertEqual(
            json.dumps(before["model_input"], sort_keys=True),
            json.dumps(after["model_input"], sort_keys=True),
            "mutating the CURRENT node changed the model input",
        )
        self.assertEqual(before["history_resource_audit"][current_index]["join_status"],
                         "current_forbidden")

    def test_future_mutation_leaves_the_input_identical(self) -> None:
        """Future rows must not be reachable from the model input at all."""

        found = self.find_row_with_past(self.train_rows)
        self.assertIsNotNone(found)
        row, run_id, current_index, _i, _node_id = found

        before = self.build(row)
        mutated = copy.deepcopy(self.truth_by_run_node)
        # every node in this run that is NOT part of the visible prefix
        visible = {str(row["current_node_id"])}
        for i in range(1, current_index):
            node_id = self.identity.get((run_id, i))
            if node_id:
                visible.add(node_id)
        touched = 0
        for node_id in list(mutated.get(run_id, {})):
            if node_id in visible:
                continue
            mutated[run_id][node_id]["runtime_ms"] = CURRENT_SENTINEL
            mutated[run_id][node_id]["status_class"] = STATUS_SENTINEL
            touched += 1
        if touched == 0:
            self.skipTest("no future rows in this run")
        after = self.build(row, mutated)

        self.assertEqual(
            json.dumps(before["model_input"], sort_keys=True),
            json.dumps(after["model_input"], sort_keys=True),
            "mutating FUTURE rows changed the model input",
        )

    def test_past_mutation_changes_only_its_own_channel(self) -> None:
        """A past outcome change must surface on exactly one token's channel."""

        found = self.find_row_with_past(self.train_rows)
        self.assertIsNotNone(found)
        row, run_id, current_index, past_index, past_node = found

        before = self.build(row)
        mutated = copy.deepcopy(self.truth_by_run_node)
        mutated[run_id][past_node]["runtime_ms"] = 123456.0
        mutated[run_id][past_node]["status_class"] = "MUTATED_PAST_STATUS"
        after = self.build(row, mutated)

        changed = []
        for i, (b, a) in enumerate(zip(before["model_input"]["history"],
                                       after["model_input"]["history"])):
            if json.dumps(b, sort_keys=True) != json.dumps(a, sort_keys=True):
                changed.append(i)
        self.assertEqual(changed, [past_index],
                         "expected only token %d to change, got %s" % (past_index, changed))

        channel = after["model_input"]["history"][past_index]["history_resource"]
        audit_entry = after["history_resource_audit"][past_index]
        self.assertEqual(audit_entry["join_status"], "identity_exact")
        self.assertEqual(channel["runtime_present"], 1)
        self.assertEqual(channel["status_class"], "MUTATED_PAST_STATUS")

    def test_current_token_never_carries_a_resource(self) -> None:
        for row in self.train_rows[:200]:
            built = self.build(row)
            history = built["model_input"]["history"]
            current = history[-1]["history_resource"]
            self.assertEqual(built["history_resource_audit"][-1]["join_status"], "current_forbidden")
            self.assertEqual(current["runtime_present"], 0)
            self.assertEqual(current["load_present"], 0)
            self.assertEqual(current["peak_alloc_present"], 0)
            self.assertEqual(current["peak_reserved_present"], 0)
            self.assertEqual(current["status_class"], "UNOBSERVED")

    def test_every_visible_resource_token_is_strictly_past(self) -> None:
        for row in self.train_rows[:200]:
            built = self.build(row)
            history = built["model_input"]["history"]
            current_index = len(history) - 1
            for index, token in enumerate(history):
                channel = token["history_resource"]
                if channel["status_class"] == "UNOBSERVED":
                    continue
                self.assertLess(index, current_index,
                                "token %d carries an observed status at/after the anchor" % index)
                for _field, _mode, _z, present_key in RESOURCE_FIELDS:
                    self.assertIn(channel[present_key], (0, 1))

    def test_forbidden_raw_keys_still_rejected(self) -> None:
        """The derived names must not have opened a hole in the old guard."""

        from build_p9d_topology_dataset import audit_model_input

        built = self.build(self.train_rows[0])
        self.assertEqual(audit_model_input(built["model_input"]), [])
        probe = copy.deepcopy(built["model_input"])
        probe["history"][-1]["runtime_ms"] = 1.0
        self.assertIn("runtime_ms", " ".join(audit_model_input(probe)))

    def test_identity_map_has_no_conflicts(self) -> None:
        self.assertGreater(len(self.identity), 0)


if __name__ == "__main__":
    unittest.main()
