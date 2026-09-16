from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from tracing.collectors.videotool_phase2_batch import _args_for_record


class Phase2BatchManifestTests(unittest.TestCase):
    def test_manifest_row_can_switch_model_stack(self) -> None:
        args = Namespace(
            output_root=Path("/tmp/out"),
            videotool_root=Path("/tmp/videotool"),
            model_name="default",
            planner_mode="local_qwen",
            max_iterations=6,
            yolo_model=None,
            yolo_python=None,
            yolo_batch=1,
            yolo_preobserve=False,
            qwen_model=None,
            qwen_python=None,
            qwen_max_new_tokens=96,
            planner_model=None,
            planner_python=None,
            planner_max_new_tokens=96,
            planner_constrained_json=False,
            answer_model=None,
            answer_python=None,
            answer_max_new_tokens=96,
            model_stack_id="default-stack",
            planner_model_id=None,
            visual_model_id=None,
            answer_model_id=None,
            detector_model_id=None,
        )
        record = {
            "task_id": "task",
            "video_path": "/video.mp4",
            "dataset": "videomme",
            "question": "question",
            "planner_mode": "local_split",
            "model_name": "Qwen2.5-VL-3B-Instruct",
            "qwen_model": "/qwen3b",
            "yolo_model": "/yolo.pt",
            "yolo_python": "/python-yolo",
            "qwen_max_new_tokens": 64,
            "model_stack_id": "stack-b",
        }
        row = _args_for_record(args, record, "task_r01")
        self.assertEqual(row.planner_mode, "local_split")
        self.assertEqual(row.model_name, "Qwen2.5-VL-3B-Instruct")
        self.assertEqual(row.qwen_model, Path("/qwen3b"))
        self.assertEqual(row.yolo_model, Path("/yolo.pt"))
        self.assertEqual(row.yolo_python, "/python-yolo")
        self.assertEqual(row.qwen_max_new_tokens, 64)
        self.assertEqual(row.model_stack_id, "stack-b")


if __name__ == "__main__":
    unittest.main()
