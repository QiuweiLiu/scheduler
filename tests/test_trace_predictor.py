from __future__ import annotations

import unittest

from tracing.analysis.reproduce_prediction_baselines import Prefix
from tracing.analysis.trace_predictor import (
    PlannerAwareTracePredictor,
    SchedulerClassPredictor,
    scheduler_class,
)


def _prefix(position: int, target: str, *, planner: str = "unknown", last_actions: list[str] | None = None) -> Prefix:
    return Prefix(
        run_id=f"r{position}",
        video_id="v",
        split="train",
        baseline="star",
        prefix=tuple(["spatial_qa"] * position),
        raw_prefix=tuple(["image-qa"] * position),
        target=target,
        raw_target=target,
        remaining_steps=1,
        remaining_runtime_ms=1.0,
        task_structure={},
        planner_model_id=planner,
        position=position,
        state_features={"prefix": {"last_actions": list(last_actions or [])}},
    )


class TracePredictorTests(unittest.TestCase):
    def test_scheduler_class_keeps_fine_tools_in_observe_family(self) -> None:
        self.assertEqual(scheduler_class("sample_seek"), "observe")
        self.assertEqual(scheduler_class("summarize"), "summarize")
        self.assertEqual(scheduler_class("__END__"), "end")

    def test_scheduler_model_uses_position_conditioned_routing(self) -> None:
        model = SchedulerClassPredictor(["answer", "end", "observe", "summarize"])
        model.fit([_prefix(0, "spatial_qa"), _prefix(0, "temporal_qa"), _prefix(1, "answer")])
        probabilities = model.predict(_prefix(1, "answer"))
        self.assertGreater(probabilities["answer"], probabilities["observe"])

    def test_planner_aware_model_does_not_mix_planner_priors(self) -> None:
        rows = [
            _prefix(0, "sample_seek", planner="planner_a"),
            _prefix(0, "sample_seek", planner="planner_a"),
            _prefix(0, "temporal_qa", planner="planner_b"),
            _prefix(0, "temporal_qa", planner="planner_b"),
        ]
        model = PlannerAwareTracePredictor(["sample_seek", "temporal_qa"]).fit(rows)
        probabilities = model.predict(_prefix(0, "temporal_qa", planner="planner_b"))
        self.assertGreater(probabilities["temporal_qa"], probabilities["sample_seek"])

    def test_v04_omits_mixed_broad_position_backoff(self) -> None:
        levels = {name for name, _, _ in PlannerAwareTracePredictor.LEVEL_WEIGHTS}
        self.assertIn("planner_last_actions", levels)
        self.assertIn("baseline_last", levels)
        self.assertNotIn("baseline_position", levels)
        self.assertNotIn("position", levels)


if __name__ == "__main__":
    unittest.main()
