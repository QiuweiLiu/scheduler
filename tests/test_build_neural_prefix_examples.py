from __future__ import annotations

from tracing.analysis.build_neural_prefix_examples import build_examples, build_report
from tracing.analysis.reproduce_prediction_baselines import Prefix


def _row(run_id: str, position: int, prefix: tuple[str, ...], target: str) -> Prefix:
    return Prefix(
        run_id=run_id,
        video_id="video-1",
        split="train",
        baseline="star",
        prefix=prefix,
        raw_prefix=prefix,
        target=target,
        raw_target=target,
        remaining_steps=3 - position,
        remaining_runtime_ms=1.0,
        task_structure={"question_type": "temporal", "answer_text": "do not expose"},
        state_features={"prefix": {"last_actions": ["sample_seek"]}},
        position=position,
    )


def test_builder_separates_prefix_from_future_labels() -> None:
    examples = build_examples(
        [
            _row("run-1", 0, (), "sample_seek"),
            _row("run-1", 1, ("sample_seek",), "answer"),
            _row("run-1", 2, ("sample_seek", "answer"), "__END__"),
        ]
    )
    assert examples[0]["input"]["prefix_nodes"] == []
    assert examples[0]["target"]["suffix"] == ["sample_seek", "answer", "__END__"]
    assert "answer_text" not in examples[0]["input"]["task_context"]
    assert examples[0]["leakage_checks"]["future_events_in_input"] is False


def test_report_counts_rows_and_groups() -> None:
    examples = build_examples([_row("run-1", 0, (), "sample_seek")])
    report = build_report(examples)
    assert report["rows"] == 1
    assert report["runs"] == 1
    assert report["groups"] == 1
