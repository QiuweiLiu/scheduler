from __future__ import annotations

import json
from pathlib import Path

from tracing.analysis.reproduce_prediction_baselines import (
    PathLengthModel,
    PlannerTransitionModel,
    Prefix,
    StructuredBackoffModel,
    TransitionModel,
    build_report,
)


def _row(run_id: str, video_id: str, split: str, position: int, prefix: list[str], target: str) -> dict[str, object]:
    return {
        "prefix_id": f"{run_id}:semantic:{position}",
        "run_id": run_id,
        "video_id": video_id,
        "split": split,
        "baseline": "star",
        "prefix_activities": prefix,
        "prefix_raw_actions": prefix,
        "target_next_activity": target,
        "target_next_raw_action": target,
        "remaining_steps": 2 - position if target != "__END__" else 0,
        "remaining_runtime_ms": 1.0,
        "future_events_included_in_input": False,
        "ground_truth_included_in_input": False,
        "source_trace_sha256": "trace-hash",
        "task_structure": {"question_type": "temporal", "domain": "Knowledge", "option_count": 4},
    }


def test_report_rejects_missing_formal_split(tmp_path: Path) -> None:
    prefix = tmp_path / "prefix.jsonl"
    prefix.write_text(json.dumps(_row("r1", "v1", "unassigned", 0, [], "sample_seek")) + "\n", encoding="utf-8")
    try:
        build_report(prefix)
    except ValueError as exc:
        assert "split-manifest" in str(exc)
    else:
        raise AssertionError("missing formal split was accepted")


def test_report_has_fixed_split_and_no_future_leakage(tmp_path: Path) -> None:
    prefix = tmp_path / "prefix.jsonl"
    rows = []
    split_rows = []
    for index in range(64):
        video_id = f"v{index:02d}"
        split = "train" if index < 48 else "validation" if index < 56 else "test"
        split_rows.append({"video_id": video_id, "split": split})
        run_id = f"run_{video_id}"
        rows.extend([
            _row(run_id, video_id, "unassigned", 0, [], "sample_seek"),
            _row(run_id, video_id, "unassigned", 1, ["sample_seek"], "__END__"),
        ])
    prefix.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    split = tmp_path / "split.jsonl"
    split.write_text("\n".join(json.dumps(row) for row in split_rows) + "\n", encoding="utf-8")
    report = build_report(prefix, split)
    assert report["fixed_split"]["fixed_split_gate"] is True
    assert report["leakage_checks"]["future_events_included_in_input"] is False
    assert report["fixed_split"]["metrics"]["test"]["classification"]["dyorc_markov2"]["n"] == 16


def _prefix(run_id: str, position: int, prefix: list[str], target: str) -> Prefix:
    return Prefix(
        run_id=run_id,
        video_id=run_id,
        split="train",
        baseline="star",
        prefix=tuple(prefix),
        raw_prefix=tuple(prefix),
        target=target,
        raw_target=target,
        remaining_steps=3 - position,
        remaining_runtime_ms=1.0,
        task_structure={},
        position=position,
    )


def test_empty_prefix_uses_position_conditioned_prior() -> None:
    rows = [
        _prefix("a", 0, [], "spatial_qa"),
        _prefix("b", 0, [], "spatial_qa"),
        _prefix("c", 0, [], "temporal_qa"),
        _prefix("a", 1, ["spatial_qa"], "answer"),
        _prefix("b", 1, ["spatial_qa"], "answer"),
    ]
    model = TransitionModel(["answer", "spatial_qa", "temporal_qa"], order=1)
    model.fit(rows)
    probabilities = model.predict(_prefix("test", 0, [], "spatial_qa"))
    assert probabilities["spatial_qa"] > probabilities["temporal_qa"]


def test_planner_markov_keeps_planner_priors_separate() -> None:
    rows = [
        Prefix("a", "v1", "train", "star", (), (), "sample_seek", "sample_seek", 1, 1.0, {}, "unknown", "planner_a", 0),
        Prefix("b", "v2", "train", "star", (), (), "sample_seek", "sample_seek", 1, 1.0, {}, "unknown", "planner_a", 0),
        Prefix("c", "v3", "train", "star", (), (), "temporal_qa", "temporal_qa", 1, 1.0, {}, "unknown", "planner_b", 0),
        Prefix("d", "v4", "train", "star", (), (), "temporal_qa", "temporal_qa", 1, 1.0, {}, "unknown", "planner_b", 0),
    ]
    model = PlannerTransitionModel(["sample_seek", "temporal_qa"], order=1).fit(rows)
    probabilities = model.predict(rows[-1])
    assert probabilities["temporal_qa"] > probabilities["sample_seek"]


def test_structured_backoff_uses_coarse_task_context() -> None:
    rows = [
        Prefix(
            "a", "v1", "train", "star", (), (), "temporal_qa", "temporal_qa", 0, 1.0,
            {"question_type": "temporal", "official_task_type": "Temporal Perception", "answer_type": "multiple_choice", "temporal_scope": "global"},
            position=0,
        ),
        Prefix(
            "b", "v2", "train", "star", (), (), "temporal_qa", "temporal_qa", 0, 1.0,
            {"question_type": "temporal", "official_task_type": "Temporal Perception", "answer_type": "multiple_choice", "temporal_scope": "global"},
            position=0,
        ),
        Prefix(
            "c", "v3", "train", "star", (), (), "spatial_qa", "spatial_qa", 0, 1.0,
            {"question_type": "spatial", "official_task_type": "Spatial Perception", "answer_type": "multiple_choice", "temporal_scope": "global"},
            position=0,
        ),
    ]
    model = StructuredBackoffModel(["spatial_qa", "temporal_qa"]).fit(rows)
    query = rows[0]
    probabilities = model.predict(query)
    assert probabilities["temporal_qa"] > probabilities["spatial_qa"]


def test_path_model_returns_future_suffix_not_prefix_plus_target() -> None:
    rows = [
        _prefix("train", 0, [], "spatial_qa"),
        _prefix("train", 1, ["spatial_qa"], "summarize"),
        _prefix("train", 2, ["spatial_qa", "summarize"], "answer"),
        _prefix("train", 3, ["spatial_qa", "summarize", "answer"], "__END__"),
    ]
    model = PathLengthModel(["__END__", "answer", "spatial_qa", "summarize"])
    model.fit(rows)
    _, candidates, _, _ = model.predict(_prefix("test", 1, ["spatial_qa"], "summarize"))
    assert candidates[0] == ["summarize", "answer", "__END__"]
