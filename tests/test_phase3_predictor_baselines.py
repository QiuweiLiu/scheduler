from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tracing.analysis.phase3_predictor_baselines import (
    PrefixTaskNB,
    build_examples,
    evaluate,
    load_runs,
)


def _write_run(root: Path, name: str, video: str, baseline: str, actions: tuple[str, ...]) -> None:
    run_dir = root / name
    run_dir.mkdir()
    manifest = {
        "run_id": name,
        "status": "success",
        "baseline": baseline,
        "model_name": "qwen3-vl-flash",
        "task_id": f"task_{video}_{name}",
        "video_path": f"/data/{video}.mp4",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    events = []
    for index, action in enumerate(actions):
        events.append(
            {
                "event_type": "action",
                "action": action,
                "resource": {"runtime_ms": float(index + 1)},
            }
        )
    (run_dir / "trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


class Phase3PredictorTests(unittest.TestCase):
    def test_loader_builds_terminal_prefix_examples_and_grouped_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "traces"
            root.mkdir()
            _write_run(root, "run_1", "v1", "star", ("a", "b"))
            _write_run(root, "run_2", "v1", "star", ("a", "c"))
            _write_run(root, "run_3", "v2", "star", ("a", "b"))
            manifest = Path(directory) / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"task_id": "task_v1_run_1", "video_path": "/data/v1.mp4", "question": "Which object?"})
                + "\n",
                encoding="utf-8",
            )
            runs = load_runs(root, manifest)
            examples = build_examples(runs)
            self.assertEqual(len(runs), 3)
            self.assertEqual(len(examples), 9)
            self.assertEqual(sum(example.target == "__END__" for example in examples), 3)
            report = evaluate(runs, examples)
            self.assertEqual(report["split"]["strategy"], "leave_one_video_out")
            self.assertEqual(report["data"]["videos"], 2)
            self.assertIn("markov2", report["classification"])
            self.assertIn("prefix_only_nb", report["classification"])
            self.assertIn("prefix_task_nb", report["classification"])
            self.assertIn(report["h2_observation"], {"preliminary_supported", "insufficient_evidence"})

    def test_prefix_task_model_returns_a_probability_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_run(root, "run_1", "v1", "star", ("a", "b"))
            _write_run(root, "run_2", "v2", "star", ("a", "c"))
            examples = build_examples(load_runs(root))
            model = PrefixTaskNB(sorted({example.target for example in examples}))
            model.fit(examples)
            probabilities = model.predict_proba(examples[0])
            self.assertAlmostEqual(sum(probabilities.values()), 1.0)
            self.assertEqual(set(probabilities), {"a", "b", "c", "__END__"})


if __name__ == "__main__":
    unittest.main()
