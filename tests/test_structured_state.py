from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tracing.collectors.structured_state import (
    EvidenceAccumulator,
    build_state_snapshot,
    canonical_action,
    convert_manifest_records,
    derive_task_structure,
    validate_state_snapshot,
    write_state_sidecar,
)
from tracing.collectors.videoseek_wrapper import TraceRecorder
from tracing.collectors.videotool_phase1 import ToolContext


class _VisibleFrames:
    video_info = {"duration": 100.0, "fps": 10.0}
    frames = [{"index": 0, "time_seconds": 0.0}, {"index": 50, "time_seconds": 5.0}]
    last_decode_ms = 0.0
    last_load_ms = 0.0

    def get_frame_count(self) -> int:
        return len(self.frames)


class StructuredStateTests(unittest.TestCase):
    def test_legacy_video_tools_have_scheduler_families(self) -> None:
        # These names occurred in the frozen traces.  They are frame selection
        # and spatial zoom tools, not an unknown ``other`` activity.
        self.assertEqual(canonical_action("image-grid-selector"), "sample_seek")
        self.assertEqual(canonical_action("ImageGridSelect"), "sample_seek")
        self.assertEqual(canonical_action("patch-zoomer"), "spatial_qa")
        self.assertEqual(canonical_action("PatchZoomer"), "spatial_qa")

    def test_task_structure_is_numeric_categorical_and_drops_answer(self) -> None:
        record = {
            "task_id": "task_1",
            "dataset": "videomme",
            "video_path": "/data/v1.mp4",
            "question": "Which item has the largest number after the event?",
            "options": "A. Apples.\nB. Candles.\nC. Berries.\nD. Same.",
            "answer": "C",
            "domain": "Knowledge",
            "sub_category": "Humanity & History",
            "baselines": ["star"],
        }
        structure = derive_task_structure(record)
        self.assertEqual(structure["question_type"], "comparison")
        self.assertEqual(structure["answer_type"], "multiple_choice")
        self.assertEqual(structure["option_count"], 4)
        converted = convert_manifest_records([record])[0]
        self.assertNotIn("answer", converted)
        self.assertNotIn("answer", converted["agent_input"])
        self.assertFalse(converted["task_structure"]["derivation"]["answer_label_used"])

    def test_evidence_accumulator_tracks_only_observed_intervals(self) -> None:
        accumulator = EvidenceAccumulator(duration_s=100.0, bin_count=10)
        accumulator.observe(
            intervals=[(10.0, 30.0)],
            frames_seen=8,
            modality="object",
            object_counts={"person": 2},
            confidence=0.8,
            source_event_id="event-1",
        )
        snapshot = accumulator.snapshot()
        self.assertEqual(snapshot["frames_seen"], 8)
        self.assertEqual(snapshot["object_counts"]["person"], 2)
        self.assertGreater(snapshot["coverage_ratio"], 0.0)
        self.assertFalse(snapshot["future_events_included"])
        self.assertFalse(snapshot["ground_truth_included"])

    def test_state_snapshot_has_no_length_cap_and_has_leakage_guards(self) -> None:
        accumulator = EvidenceAccumulator(duration_s=120.0)
        accumulator.observe(intervals=[(0.0, 15.0)], frames_seen=4, modality="scene")
        state = build_state_snapshot(
            run_id="run-1",
            task_id="task-1",
            video_id="video-1",
            baseline="star",
            step_id=8,
            task_structure={
                "question_type": "temporal",
                "answer_type": "multiple_choice",
                "temporal_scope": "ordered_events",
                "required_modalities": ["temporal"],
                "option_count": 4,
                "domain": "Knowledge",
                "sub_category": "Science",
            },
            video_metadata={"duration_s": 120.0, "fps": 25.0},
            prefix_actions=["overview", "frame-selector", "image-qa", "temporal-qa", "summarization-tool", "answer", "retry"],
            max_steps=12,
            evidence=accumulator.snapshot(),
        )
        self.assertEqual(state["prefix"]["observed_step_count"], 7)
        self.assertEqual(state["prefix"]["canonical_actions"][-1], "retry")
        self.assertLess(state["prefix"]["progress_ratio"], 1.0)
        self.assertEqual(validate_state_snapshot(state), [])
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "states" / "state_008.json"
            write_state_sidecar(sidecar, state)
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8"))["state_id"], "run-1:state:8")

    def test_state_validation_rejects_future_evidence(self) -> None:
        state = build_state_snapshot(
            run_id="run-1",
            task_id="task-1",
            video_id="video-1",
            baseline="star",
            step_id=0,
            task_structure={
                "question_type": "unknown",
                "answer_type": "multiple_choice",
                "temporal_scope": "unspecified",
                "required_modalities": ["unknown"],
                "option_count": 4,
                "domain": "unknown",
                "sub_category": "unknown",
            },
            video_metadata=None,
            prefix_actions=[],
        )
        state["evidence"]["future_events_included"] = True
        self.assertTrue(validate_state_snapshot(state))

    def test_manifest_output_is_jsonl_ready(self) -> None:
        record = {
            "task_id": "task_1",
            "dataset": "videomme",
            "video_path": "/data/v1.mp4",
            "question": "What is the content of the video?",
            "options": ["A. One", "B. Two"],
        }
        converted = convert_manifest_records([record])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in converted) + "\n", encoding="utf-8")
            loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(loaded[0]["manifest_schema"], "phase3-task-0.2")
            self.assertEqual(loaded[0]["task_structure"]["question_type"], "summary")

    def test_videotool_context_writes_initial_and_post_action_states(self) -> None:
        task_structure = {
            "question_type": "temporal",
            "answer_type": "multiple_choice",
            "temporal_scope": "ordered_events",
            "required_modalities": ["temporal"],
            "option_count": 4,
            "domain": "Knowledge",
            "sub_category": "Science",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = TraceRecorder(
                root,
                run_id="run_state",
                framework="videotool",
                dataset="videomme",
                task_id="task_state",
                model_id="local",
            )
            try:
                context = ToolContext(
                    _VisibleFrames(),
                    Path("/data/v1.mp4"),
                    root,
                    recorder,
                    None,
                    None,
                    baseline="star",
                    task_structure=task_structure,
                    max_steps=6,
                )
                context.record_observed_action(
                    action="frame-selector",
                    status="success",
                    runtime_ms=10.0,
                    api_wait_ms=0.0,
                    tool_metrics=None,
                    source_event_id="run_state:action:1",
                )
            finally:
                recorder.close()
            states = sorted((root / "states").glob("state_*.json"))
            self.assertEqual(len(states), 2)
            self.assertEqual(json.loads(states[0].read_text(encoding="utf-8"))["step_id"], 0)
            final = json.loads(states[-1].read_text(encoding="utf-8"))
            self.assertEqual(final["baseline"], "star")
            self.assertEqual(final["prefix"]["canonical_actions"], ["sample_seek"])


if __name__ == "__main__":
    unittest.main()
