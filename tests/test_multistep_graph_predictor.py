from __future__ import annotations

from tracing.analysis.multistep_graph_predictor import (
    GraphBeamPredictor,
    SuffixIndexPredictor,
    TransitionBeamPredictor,
    future_suffixes,
)
from tracing.analysis.reproduce_prediction_baselines import Prefix


def _row(run_id: str, position: int, prefix: tuple[str, ...], target: str, baseline: str = "star") -> Prefix:
    return Prefix(
        run_id=run_id,
        video_id=run_id,
        split="train",
        baseline=baseline,
        prefix=prefix,
        raw_prefix=prefix,
        target=target,
        raw_target=target,
        remaining_steps=3 - position,
        remaining_runtime_ms=1.0,
        task_structure={},
        position=position,
    )


def _rows() -> list[Prefix]:
    return [
        _row("r1", 0, (), "a"),
        _row("r1", 1, ("a",), "b"),
        _row("r1", 2, ("a", "b"), "__END__"),
        _row("r2", 0, (), "a"),
        _row("r2", 1, ("a",), "c"),
        _row("r2", 2, ("a", "c"), "__END__"),
    ]


def test_future_suffixes_are_scoring_only_and_ordered() -> None:
    suffixes = future_suffixes(_rows())
    assert suffixes[("r1", 0)] == ("a", "b", "__END__")
    assert suffixes[("r2", 1)] == ("c", "__END__")


def test_graph_beam_emits_only_train_observed_edges() -> None:
    model = GraphBeamPredictor().fit(_rows())
    assert model.context_model is not None
    paths = model.predict(_rows()[0], horizon=2, beam_size=5)
    assert {path.path for path in paths} == {("a", "b"), ("a", "c")}
    assert all(model.graph.is_legal("star", (), path.path) for path in paths)
    assert not model.graph.is_legal("star", (), ("a", "missing"))


def test_markov_and_suffix_models_return_short_paths() -> None:
    rows = _rows()
    markov = TransitionBeamPredictor(["a", "b", "c", "__END__"]).fit(rows)
    paths = markov.predict(rows[0], horizon=3, beam_size=2)
    assert paths and len(paths[0].path) <= 3
    suffix = SuffixIndexPredictor(GraphBeamPredictor().fit(rows)).fit(rows)
    indexed, fallback = suffix.predict(rows[0], horizon=3, beam_size=2)
    assert indexed[0].path == ("a", "b", "__END__")
    assert fallback is False


def test_suffix_model_marks_unseen_prefix_as_fallback() -> None:
    rows = _rows()
    model = SuffixIndexPredictor(GraphBeamPredictor().fit(rows)).fit(rows)
    query = _row("unseen", 1, ("unknown",), "a")
    _, fallback = model.predict(query, horizon=2, beam_size=2)
    assert fallback is True
