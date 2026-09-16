from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tracing.collectors.videoseek_wrapper import (
    TraceRecorder,
    _git_provenance,
    _install_action_hook,
    _video_metadata,
)
from tracing.validators.validate_trace import validate_trace


def _event() -> dict:
    return {
        "schema_version": "0.1",
        "event_id": "run_example:action:1",
        "event_type": "action",
        "run_id": "run_example",
        "framework": "videoseek",
        "dataset": "lvbench",
        "task_id": "example",
        "step_id": 1,
        "parent_step_ids": [],
        "action": "overview",
        "node_type": "video_observation",
        "model_id": "openai/qwen3-vl-plus",
        "model_resident_before": None,
        "input": {"parameters": {}},
        "resource": {
            "gpu_id": None,
            "gpu_model": None,
            "queue_ms": None,
            "decode_ms": None,
            "load_ms": None,
            "runtime_ms": 12.5,
            "api_wait_ms": 0.0,
            "local_runtime_ms": 12.5,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        },
        "status": "success",
        "retry_of": None,
        "output_summary_ref": "trace.jsonl:action-1",
        "timestamp_start": "2026-07-31T00:00:00Z",
        "timestamp_end": "2026-07-31T00:00:01Z",
        "error": None,
    }


class TraceValidatorTests(unittest.TestCase):
    def _write(self, *events: object) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        trace_path = Path(directory.name) / "trace.jsonl"
        trace_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
        return trace_path

    def test_accepts_a_valid_cpu_only_event(self) -> None:
        self.assertEqual(validate_trace(self._write(_event())), [])

    def test_accepts_a_gpu_tool_event(self) -> None:
        payload = _event()
        payload["event_id"] = "run_example:action:2"
        payload["step_id"] = 2
        payload["parent_step_ids"] = [1]
        payload["action"] = "focus"
        payload["resource"].update(
            {
                "gpu_id": 0,
                "gpu_model": "RTX 4080 SUPER",
                "gpu_driver_version": "595.58.03",
                "system_gpu_index": 0,
                "system_gpu_memory_used_mib": 1024.0,
                "system_gpu_memory_total_mib": 32760.0,
                "system_gpu_utilization_pct": 53.0,
            }
        )
        self.assertEqual(validate_trace(self._write(payload)), [])

    def test_accepts_failure_and_retry_events(self) -> None:
        failed = _event()
        failed.update(
            {
                "event_id": "run_example:action:1",
                "status": "error",
                "error": "RuntimeError: transient tool failure",
                "output_summary_ref": None,
            }
        )
        retry = _event()
        retry.update(
            {
                "event_id": "run_example:action:2",
                "step_id": 2,
                "parent_step_ids": [1],
                "retry_of": "run_example:action:1",
            }
        )
        self.assertEqual(validate_trace(self._write(failed, retry)), [])

    def test_rejects_missing_action(self) -> None:
        payload = _event()
        payload.pop("action")
        errors = validate_trace(self._write(payload))
        self.assertTrue(any("missing fields: action" in error for error in errors))

    def test_rejects_negative_api_wait(self) -> None:
        payload = _event()
        payload["resource"]["api_wait_ms"] = -1
        errors = validate_trace(self._write(payload))
        self.assertTrue(any("resource.api_wait_ms" in error for error in errors))

    def test_rejects_retry_without_a_prior_event(self) -> None:
        payload = _event()
        payload["retry_of"] = "missing:event"
        errors = validate_trace(self._write(payload))
        self.assertTrue(any("retry_of must reference an earlier event_id" in error for error in errors))


class _Action:
    function_name = "overview"
    parameters = {"frames": 8}


class _Agent:
    def __init__(self, llm_call) -> None:
        self.trajectory_steps: list[object] = []
        self._llm_call = llm_call
        setattr(self, "_VideoSeekAgent__exec_action", self._execute)

    def _execute(self, action: _Action) -> object:
        return self._llm_call()


class TraceRecorderTests(unittest.TestCase):
    def _recorder(self) -> tuple[tempfile.TemporaryDirectory, TraceRecorder]:
        directory = tempfile.TemporaryDirectory()
        recorder = TraceRecorder(
            Path(directory.name),
            run_id="run_test",
            framework="videoseek",
            dataset="lvbench",
            task_id="example",
            model_id="openai/qwen3-vl-plus",
        )
        self.addCleanup(recorder.close)
        self.addCleanup(directory.cleanup)
        return directory, recorder

    def _events(self, recorder: TraceRecorder) -> list[dict]:
        recorder.close()
        return [
            json.loads(line)
            for line in recorder.trace_path.read_text(encoding="utf-8").splitlines()
        ]

    def test_recorder_emits_cpu_only_resource_fields(self) -> None:
        _, recorder = self._recorder()
        with patch("tracing.collectors.videoseek_wrapper._gpu_snapshot", return_value={}):
            recorder.record_event(
                event_type="action",
                step_id=1,
                action="overview",
                node_type="video_observation",
                parent_step_ids=[],
                input_data={"parameters": {}},
                start_time="2026-07-31T00:00:00Z",
                end_time="2026-07-31T00:00:01Z",
                runtime_ms=10.0,
                api_wait_ms=2.0,
                status="success",
                output_summary_ref="trace.jsonl:action-1",
            )
        event = self._events(recorder)[0]
        self.assertIsNone(event["resource"]["gpu_id"])
        self.assertIsNone(event["resource"]["gpu_model"])
        self.assertEqual(event["resource"]["local_runtime_ms"], 8.0)
        self.assertEqual(validate_trace(recorder.trace_path), [])

    def test_recorder_maps_a_sampled_gpu_to_gpu_id(self) -> None:
        _, recorder = self._recorder()
        sampled_gpu = {
            "system_gpu_index": 0,
            "gpu_model": "RTX 4080 SUPER",
            "gpu_driver_version": "595.58.03",
            "system_gpu_memory_used_mib": 128.0,
            "system_gpu_memory_total_mib": 32760.0,
            "system_gpu_utilization_pct": 7.0,
        }
        with patch(
            "tracing.collectors.videoseek_wrapper._gpu_snapshot",
            return_value=sampled_gpu,
        ):
            recorder.record_event(
                event_type="action",
                step_id=1,
                action="focus",
                node_type="video_observation",
                parent_step_ids=[],
                input_data={"parameters": {"frames": 8}},
                start_time="2026-07-31T00:00:00Z",
                end_time="2026-07-31T00:00:01Z",
                runtime_ms=12.0,
                api_wait_ms=0.0,
                status="success",
                output_summary_ref="trace.jsonl:action-1",
            )
        event = self._events(recorder)[0]
        self.assertEqual(event["resource"]["gpu_id"], 0)
        self.assertEqual(event["resource"]["gpu_driver_version"], "595.58.03")
        self.assertEqual(validate_trace(recorder.trace_path), [])

    def test_action_error_keeps_nested_api_call_accounting(self) -> None:
        _, recorder = self._recorder()

        def fail_llm_call() -> None:
            raise RuntimeError("transient API failure")

        agent = _Agent(recorder.wrap_llm_call(fail_llm_call, "fake.module"))
        restore = _install_action_hook(agent, recorder, "/tmp/video.mp4")
        try:
            with patch("tracing.collectors.videoseek_wrapper._gpu_snapshot", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "transient API failure"):
                    getattr(agent, "_VideoSeekAgent__exec_action")(_Action())
        finally:
            restore()

        events = self._events(recorder)
        self.assertEqual([event["event_type"] for event in events], ["api_call", "action"])
        action_event = events[-1]
        self.assertEqual(action_event["status"], "error")
        self.assertEqual(action_event["input"]["nested_api_call_count"], 1)
        self.assertGreaterEqual(action_event["resource"]["api_wait_ms"], 0.0)
        self.assertEqual(validate_trace(recorder.trace_path), [])

    def test_llm_none_result_is_an_error_event(self) -> None:
        _, recorder = self._recorder()
        wrapped = recorder.wrap_llm_call(lambda **_: None, "fake.module")
        with patch("tracing.collectors.videoseek_wrapper._gpu_snapshot", return_value={}):
            self.assertIsNone(wrapped(messages=[]))

        event = self._events(recorder)[0]
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error"], "VideoSeek call_llm_api returned None")
        self.assertEqual(validate_trace(recorder.trace_path), [])

    def test_action_failure_sentinel_is_an_error_event(self) -> None:
        _, recorder = self._recorder()
        agent = _Agent(lambda: "Tool execution failed.")
        restore = _install_action_hook(agent, recorder, "/tmp/video.mp4")
        try:
            with patch("tracing.collectors.videoseek_wrapper._gpu_snapshot", return_value={}):
                result = getattr(agent, "_VideoSeekAgent__exec_action")(_Action())
        finally:
            restore()

        self.assertEqual(result, "Tool execution failed.")
        event = self._events(recorder)[0]
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["error"], "VideoSeek tool returned its failure sentinel")
        self.assertEqual(validate_trace(recorder.trace_path), [])


class TraceMetadataTests(unittest.TestCase):
    def test_git_provenance_marks_empty_status_as_clean(self) -> None:
        with patch(
            "tracing.collectors.videoseek_wrapper._command_output",
            side_effect=["deadbeef", ""],
        ):
            provenance = _git_provenance(Path("/tmp/fake-repository"))
        self.assertEqual(provenance["commit"], "deadbeef")
        self.assertTrue(provenance["working_tree_clean"])

    def test_video_metadata_tolerates_non_object_ffprobe_output(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        video_path = Path(directory.name) / "video.mp4"
        video_path.write_bytes(b"not-a-real-video")
        with patch(
            "tracing.collectors.videoseek_wrapper._command_output",
            return_value="[]",
        ):
            metadata = _video_metadata(video_path)
        self.assertEqual(metadata["size_bytes"], len(b"not-a-real-video"))
        self.assertIsNone(metadata["duration_seconds"])
        self.assertIsNone(metadata["resolution"])


if __name__ == "__main__":
    unittest.main()
