import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tracing.collectors.videoseek_wrapper import TraceRecorder
from tracing.collectors.videotool_phase1 import _build_parser, _worker_resource_metrics


class ModelStackContractTests(unittest.TestCase):
    def test_trace_event_can_override_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = TraceRecorder(
                root,
                run_id="model_stack",
                framework="test",
                dataset="test",
                task_id="task",
                model_id="stack-default",
            )
            recorder.record_event(
                event_type="action",
                step_id=1,
                action="image-qa",
                node_type="videotool_spatial",
                parent_step_ids=[],
                input_data={},
                start_time="2026-01-01T00:00:00Z",
                end_time="2026-01-01T00:00:00Z",
                runtime_ms=1.0,
                api_wait_ms=0.0,
                status="success",
                output_summary_ref=None,
                model_id="qwen2.5-vl-3b",
            )
            recorder.close()
            event = json.loads((root / "trace.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["model_id"], "qwen2.5-vl-3b")

    def test_phase1_parser_exposes_split_stack(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--video-path",
                "/tmp/video.mp4",
                "--output-root",
                "/tmp/out",
                "--videotool-root",
                "/tmp/videotool",
                "--question",
                "Q",
                "--planner-mode",
                "local_split",
                "--model-stack-id",
                "stack_b",
            ]
        )
        self.assertEqual(args.planner_mode, "local_split")
        self.assertEqual(args.model_stack_id, "stack_b")

    def test_worker_metrics_preserve_first_load_only(self) -> None:
        first = _worker_resource_metrics(
            {
                "request_index": 1,
                "model_load_ms": 100,
                "worker_startup_ms": 200,
                "inference_ms": 20,
            }
        )
        warm = _worker_resource_metrics(
            {
                "request_index": 2,
                "model_load_ms": 100,
                "worker_startup_ms": 200,
                "inference_ms": 20,
            }
        )
        self.assertEqual(first["load_ms"], 100)
        self.assertEqual(warm["load_ms"], 0.0)

    def test_gate_parser_rejects_prose_wrapped_json(self) -> None:
        path = Path(__file__).parents[1] / "scripts/run_qwen_planner_gate.py"
        spec = importlib.util.spec_from_file_location("planner_gate", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _, category = module._parse_object(
            'prose {"reasoning":"r","tool_name":"image-qa","tool_input":"x","info_sufficient":false}'
        )
        self.assertEqual(category, "non_json")


if __name__ == "__main__":
    unittest.main()
