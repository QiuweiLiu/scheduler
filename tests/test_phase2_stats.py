from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tracing.analysis.phase2_trace_stats import summarize


def _event(run_id: str, event_id: str, action: str, runtime_ms: float) -> dict:
    return {
        "event_id": f"{run_id}:{event_id}",
        "event_type": "action",
        "action": action,
        "status": "success",
        "retry_of": None,
        "resource": {
            "runtime_ms": runtime_ms,
            "local_runtime_ms": runtime_ms,
            "api_wait_ms": 0.0,
            "load_ms": runtime_ms / 2,
            "peak_allocated_mb": 100.0 + runtime_ms,
        },
    }


class Phase2StatsTests(unittest.TestCase):
    def test_reports_paths_and_resource_variation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, actions in enumerate(
                (("frame-selector", "image-qa"), ("frame-selector", "image-grid-qa")),
                start=1,
            ):
                run_dir = root / f"run_{index}"
                run_dir.mkdir()
                (run_dir / "run_manifest.json").write_text(
                    json.dumps({"run_id": f"run_{index}", "status": "success", "baseline": "star"}),
                    encoding="utf-8",
                )
                events = [_event(f"run_{index}", str(pos), action, float(index * pos)) for pos, action in enumerate(actions, 1)]
                (run_dir / "trace.jsonl").write_text(
                    "".join(json.dumps(event) + "\n" for event in events),
                    encoding="utf-8",
                )
            result = summarize(root)
            self.assertEqual(result["runs"], 2)
            self.assertEqual(result["unique_paths"], 2)
            self.assertEqual(result["gates"]["H1_path_dynamicity_observation"], "pass")
            self.assertGreater(result["next_action_entropy_bits"], 0.0)
            self.assertIsNotNone(result["resource"]["local_runtime_ms_cv"])


if __name__ == "__main__":
    unittest.main()
