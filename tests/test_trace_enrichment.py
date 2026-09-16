from __future__ import annotations

import json
from pathlib import Path

from tracing.analysis.build_trace_enrichment import build


def _resource(runtime_ms: float = 1.0) -> dict[str, object]:
    return {
        "gpu_id": 0,
        "gpu_model": "test-gpu",
        "queue_ms": None,
        "decode_ms": 0.0,
        "load_ms": 0.0,
        "runtime_ms": runtime_ms,
        "api_wait_ms": 0.0,
        "local_runtime_ms": runtime_ms,
        "peak_allocated_mb": 10.0,
        "peak_reserved_mb": 12.0,
    }


def _event(
    *,
    run_id: str,
    event_id: str,
    event_type: str,
    action: str,
    node_type: str,
    step_id: int,
    status: str = "success",
    parent_step_ids: list[int] | None = None,
    runtime_ms: float = 1.0,
) -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "event_id": event_id,
        "event_type": event_type,
        "run_id": run_id,
        "framework": "test",
        "dataset": "testset",
        "task_id": "task_video_r01",
        "step_id": step_id,
        "parent_step_ids": parent_step_ids or [],
        "action": action,
        "node_type": node_type,
        "model_id": "test-model",
        "model_resident_before": None,
        "input": {"standard_tool_call": {"name": action, "args": {}}},
        "resource": _resource(runtime_ms),
        "status": status,
        "retry_of": None,
        "output_summary_ref": f"trace.jsonl:{event_id}",
        "timestamp_start": f"2026-08-03T00:00:0{step_id}Z",
        "timestamp_end": f"2026-08-03T00:00:0{step_id + 1}Z",
        "error": None,
    }


def _write_run(root: Path, suffix: str) -> Path:
    run_id = f"run_{suffix}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "0.1",
        "run_id": run_id,
        "framework": "test",
        "dataset": "testset",
        "task_id": "task_video_r01",
        "video_path": "/tmp/video.mp4",
        "baseline": "star",
        "model_stack_id": "stack_test",
        "model_name": "test-model",
        "seed": 1,
        "status": "success",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    events = [
        _event(
            run_id=run_id,
            event_id=f"{run_id}:run:1",
            event_type="run",
            action="baseline_start",
            node_type="run_control",
            step_id=0,
            runtime_ms=0.0,
        ),
        _event(
            run_id=run_id,
            event_id=f"{run_id}:action:2",
            event_type="action",
            action="frame-selector",
            node_type="video_observation",
            step_id=1,
            parent_step_ids=[],
        ),
        _event(
            run_id=run_id,
            event_id=f"{run_id}:run:3",
            event_type="run",
            action="answer",
            node_type="answer_generation",
            step_id=2,
            parent_step_ids=[1],
        ),
    ]
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    return run_dir


def test_build_creates_two_views_and_prefixes_without_future_input(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_run(root, "a")
    _write_run(root, "b")
    output = tmp_path / "derived"

    summary = build([root], output)

    assert summary["selected_snapshot_runs"] == 1
    assert summary["duplicate_candidate_runs"] == 1
    semantic = [json.loads(line) for line in (output / "semantic_events_v0_1.jsonl").read_text().splitlines()]
    prefixes = [json.loads(line) for line in (output / "prefix_samples_v0_1.jsonl").read_text().splitlines()]
    assert [row["raw_action"] for row in semantic] == ["frame-selector", "answer"]
    assert prefixes[0]["target_next_raw_action"] == "frame-selector"
    assert prefixes[0]["prefix_raw_actions"] == []
    assert all(row["future_events_included_in_input"] is False for row in prefixes)
    assert all(row["ground_truth_included_in_input"] is False for row in prefixes)


def test_include_errors_keeps_failure_for_compute_audit(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    run_dir = _write_run(root, "a")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "error"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    output = tmp_path / "derived"

    summary = build([root], output, include_errors=True)

    assert summary["selected_snapshot_runs"] == 1
    row = json.loads((output / "candidate_snapshot_v0_1.jsonl").read_text().splitlines()[0])
    assert row["status"] == "error"
