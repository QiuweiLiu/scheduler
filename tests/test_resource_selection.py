from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tracing.analysis.select_resource_completion import select


class ResourceSelectionTests(unittest.TestCase):
    def test_selection_records_oversample_and_hits_exact_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.jsonl"
            completion = root / "completion.jsonl"
            output = root / "selected.jsonl"
            rejected = root / "rejected.jsonl"
            existing.write_text("\n".join(json.dumps({"run_id": f"e{i}"}) for i in range(2)) + "\n", encoding="utf-8")
            completion.write_text(
                "\n".join(json.dumps({"run_id": f"c{i}", "status": "success", "validator": "valid", "baseline": "star"}) for i in range(3)) + "\n",
                encoding="utf-8",
            )
            summary = select(existing, completion, output, rejected, expected=4)
            self.assertTrue(summary["formal_gate"])
            self.assertEqual(summary["selected_total"], 4)
            self.assertEqual(summary["rejected_completion"], 1)

    def test_selection_round_robins_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.jsonl"
            completion = root / "completion.jsonl"
            output = root / "selected.jsonl"
            rejected = root / "rejected.jsonl"
            existing.write_text("{}\n", encoding="utf-8")
            rows = [
                {"run_id": f"star{i}", "status": "success", "validator": "valid", "baseline": "star", "yolo_batch": 16}
                for i in range(3)
            ] + [
                {"run_id": f"react{i}", "status": "success", "validator": "valid", "baseline": "langgraph_react", "yolo_batch": 16}
                for i in range(3)
            ]
            completion.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            summary = select(existing, completion, output, rejected, expected=5)
            selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(summary["formal_gate"])
            self.assertEqual({row["baseline"] for row in selected[1:]}, {"star", "langgraph_react"})


if __name__ == "__main__":
    unittest.main()
