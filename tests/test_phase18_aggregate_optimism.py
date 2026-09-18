from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

import analysis_phase18_aggregate_optimism as phase18  # noqa: E402


def event(node_id: str, sequence_index: int, step: str, runtime_ms: float, event_type: str = "api_call",
          lane: str = "gpu", predecessors: list[str] | None = None) -> dict:
    return {
        "node_id": node_id,
        "sequence_index": sequence_index,
        "source_step_ids": [step],
        "predecessor_node_ids": predecessors or [],
        "event_type": event_type,
        "execution_lane": lane,
        "runtime_ms": runtime_ms,
        "load_ms": None,
    }


def template_nodes() -> list[dict]:
    """3 steps + one whole-run container whose runtime equals the workload total."""

    return [
        event("a1", 0, "1", 100.0),
        event("a2", 1, "1", 50.0, event_type="action"),
        event("b1", 2, "2", 200.0),
        event("c1", 3, "3", 650.0, event_type="action"),
        event("r1", 4, "4", 1000.0, event_type="run"),
    ]


def make_template() -> phase18.TemplateSteps:
    return phase18.TemplateSteps({"nodes": template_nodes()})


def make_step(p50: float, p90: float, p95: float, occurrence: float = 0.0, load_p95: float = 0.0) -> dict:
    return {
        "execution_lane": "gpu",
        "resource": {
            "runtime_ms_quantiles": {"p50": p50, "p90": p90, "p95": p95},
            "load_occurrence_probability": occurrence,
            "load_duration_ms_quantiles": {"p50": load_p95, "p90": load_p95, "p95": load_p95},
        },
    }


def make_artifact(node_id: str, steps: list[dict], length: int | None = None) -> dict:
    return {
        "row": {"node_id": node_id, "run_id": "run-1", "baseline": "star", "predicted_future_length": length or len(steps)},
        "steps": steps,
    }


INFO = {"template_id": "t1", "video_id": "video-a", "baseline": "star", "split": "validation"}


class PredictedSumsTests(unittest.TestCase):
    def test_sums_gated_load_and_violations(self) -> None:
        steps = [
            make_step(100.0, 200.0, 300.0, 0.6, 50.0),
            make_step(100.0, 200.0, 300.0, 0.4, 50.0),
            make_step(100.0, 200.0, 300.0, 0.9, 50.0),
        ]
        result = phase18.predicted_sums(steps, "a1")
        self.assertEqual(result["sums"], {"p50": 300.0, "p90": 600.0, "p95": 900.0})
        self.assertEqual(result["gated_steps"], 2)
        self.assertEqual(result["extra_load_p95"], 100.0)

    def test_non_monotone_quantiles_are_counted_not_repaired(self) -> None:
        result = phase18.predicted_sums([make_step(100.0, 50.0, 300.0)], "a1")
        self.assertEqual(result["quantile_violations"], 1)
        self.assertEqual(result["sums"]["p90"], 50.0)

    def test_missing_quantile_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            phase18.predicted_sums([{"resource": {"runtime_ms_quantiles": {"p50": 10.0}}}], "a1")
        with self.assertRaises(ValueError):
            phase18.predicted_sums([{"resource": {}}], "a1")
        with self.assertRaises(ValueError):
            phase18.predicted_sums([make_step(10.0, 20.0, float("nan"))], "a1")

    def test_gated_step_without_load_duration_fails_closed(self) -> None:
        step = make_step(10.0, 20.0, 30.0, occurrence=0.9)
        step["resource"].pop("load_duration_ms_quantiles")
        with self.assertRaises(ValueError):
            phase18.predicted_sums([step], "a1")

    def test_empty_chain_sums_to_zero(self) -> None:
        result = phase18.predicted_sums([], "a1")
        self.assertEqual(result["n_steps"], 0)
        self.assertEqual(result["sums"], {"p50": 0.0, "p90": 0.0, "p95": 0.0})


class TemplateStepsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.template = make_template()

    def test_containers_are_excluded_from_steps(self) -> None:
        self.assertEqual(self.template.containers, ["r1"])
        self.assertFalse(self.template.has("r1"))
        self.assertEqual(sorted(self.template.step_numbers()), [1, 2, 3])

    def test_next_k_events_are_events_not_steps(self) -> None:
        # one slot is one event: a1 -> [a2] (same step), then b1, then c1
        self.assertEqual(self.template.next_k_events("a1", 1), ["a2"])
        self.assertEqual(self.template.next_k_events("a1", 2), ["a2", "b1"])
        self.assertEqual(self.template.next_k_events("a1", 4), ["a2", "b1", "c1"])
        self.assertEqual(self.template.next_k_events("c1", 1), [])

    def test_remaining_after_includes_the_rest_of_the_own_step(self) -> None:
        self.assertEqual(self.template.remaining_after("a1"), ["a2", "b1", "c1"])
        self.assertEqual(self.template.remaining_after("a2"), ["b1", "c1"])
        self.assertEqual(self.template.remaining_after("c1"), [])

    def test_remaining_step_count_and_branching(self) -> None:
        self.assertEqual(self.template.remaining_step_count("a1"), 2)
        self.assertEqual(self.template.remaining_step_count("c1"), 0)
        self.assertEqual(self.template.events_per_remaining_step("a1"), [1, 1])

    def test_workload_total_matches_the_container_wall_clock(self) -> None:
        self.assertEqual(self.template.workload_total_ms(), 1000.0)
        self.assertEqual(self.template.wall_clock_ms, 1000.0)

    def test_missing_runtime_fails_closed(self) -> None:
        nodes = template_nodes()
        nodes[0]["runtime_ms"] = None
        template = phase18.TemplateSteps({"nodes": nodes})
        with self.assertRaises(ValueError):
            template.runtime_ms("a1")


class TruthDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.template = make_template()

    def test_next_k_and_remaining_all(self) -> None:
        truth = phase18.truth_definitions(self.template, "a1", n_pred_steps=1)
        self.assertEqual(truth["s_true_next_k"], 50.0)  # a2, the next event
        self.assertEqual(truth["s_true_remaining_all"], 900.0)  # a2 + b1 + c1
        self.assertEqual(truth["n_remaining_steps"], 2)
        self.assertFalse(truth["branching"])

    def test_container_runtime_is_reported_separately(self) -> None:
        truth = phase18.truth_definitions(self.template, "a1", n_pred_steps=2)
        self.assertEqual(truth["s_true_next_k"], 250.0)  # a2 + b1
        self.assertEqual(truth["s_container_total"], 1000.0)

    def test_last_step_anchor_has_no_remaining_work(self) -> None:
        truth = phase18.truth_definitions(self.template, "c1", n_pred_steps=1)
        self.assertEqual(truth["n_remaining_steps"], 0)
        self.assertEqual(truth["s_true_next_k"], 0.0)
        self.assertEqual(truth["s_true_remaining_all"], 0.0)
        self.assertEqual(truth["s_container_total"], 1000.0)


class WallClockCheckTests(unittest.TestCase):
    def test_ratio_is_one_for_a_consistent_template(self) -> None:
        check = phase18.wall_clock_check({"t1": make_template()})
        self.assertEqual(check["n"], 1)
        self.assertAlmostEqual(check["median"], 1.0)


class AnchorMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.template = make_template()

    def test_ratios_use_event_level_next_k_truth(self) -> None:
        artifact = make_artifact("a1", [make_step(50.0, 80.0, 100.0)] * 1, length=1)
        record = phase18.anchor_metrics(artifact, self.template, "a1", INFO)

        self.assertEqual(record["n_pred_steps"], 1)
        self.assertEqual(record["s_true_next_k"], 50.0)  # the next event, a2
        self.assertAlmostEqual(record["r_next_k_p50"], 50.0 / 50.0)
        self.assertAlmostEqual(record["r_remaining_all_p95"], 100.0 / 900.0)
        self.assertEqual(record["b_next_k_p50"], 0.0)  # 50 predicted vs 50 true (the next event)

    def test_last_step_anchor_ratio_is_none_but_bias_is_defined(self) -> None:
        record = phase18.anchor_metrics(make_artifact("c1", [make_step(10.0, 20.0, 30.0)]), self.template, "c1", INFO)

        self.assertEqual(record["n_remaining_steps"], 0)
        self.assertIsNone(record["r_next_k_p95"])
        self.assertEqual(record["b_next_k_p95"], -30.0)

    def test_zero_pred_steps_are_kept_and_show_omission(self) -> None:
        record = phase18.anchor_metrics(make_artifact("a1", []), self.template, "a1", INFO)

        self.assertEqual(record["n_pred_steps"], 0)
        self.assertEqual(record["shat_next_k_p50"], 0.0)
        # "the next 0 steps" has no comparable truth, so that ratio stays undefined ...
        self.assertEqual(record["s_true_next_k"], 0.0)
        self.assertIsNone(record["r_next_k_p50"])
        # ... while the omission is fully visible against the remaining workload
        self.assertEqual(record["r_remaining_all_p50"], 0.0)
        self.assertEqual(record["b_remaining_all_p50"], record["s_true_remaining_all"])

    def test_extra_load_enters_only_the_plus_load_ratio(self) -> None:
        artifact = make_artifact("a1", [make_step(50.0, 80.0, 100.0, 0.9, 25.0)], length=1)
        record = phase18.anchor_metrics(artifact, self.template, "a1", INFO)

        self.assertEqual(record["extra_load_p95"], 25.0)
        self.assertEqual(record["shat_next_k_p95"], 100.0)
        self.assertAlmostEqual(record["r_next_k_p95_plus_load"], 125.0 / 50.0)

    def test_video_id_comes_from_template_info(self) -> None:
        record = phase18.anchor_metrics(make_artifact("a1", [make_step(1.0, 2.0, 3.0)]), self.template, "a1", INFO)
        self.assertEqual(record["video_id"], "video-a")
        self.assertEqual(record["anchor_event_type"], "api_call")


class BootstrapTests(unittest.TestCase):
    def test_paired_filtering_keeps_values_and_clusters_aligned(self) -> None:
        with_none = [(None, "v1"), (2.0, "v1"), (3.0, "v2"), (4.0, "v2")]
        self.assertEqual(
            phase18.paired_bootstrap_ci(with_none, n_boot=200, seed=7),
            phase18.bootstrap_ci([2.0, 3.0, 4.0], ["v1", "v2", "v2"], n_boot=200, seed=7),
        )
        self.assertIsNone(phase18.paired_bootstrap_ci([(None, "v1")], n_boot=10, seed=7))

    def test_bootstrap_rejects_unaligned_inputs(self) -> None:
        with self.assertRaises(ValueError):
            phase18.bootstrap_ci([1.0, 2.0], ["v1"], n_boot=10, seed=7)

    def test_single_cluster_returns_none(self) -> None:
        self.assertIsNone(phase18.bootstrap_ci([1.0, 2.0], ["v1", "v1"], n_boot=50, seed=7))


class SummarizeTests(unittest.TestCase):
    """Integration: real anchor_metrics() records feed summarize() directly."""

    def _records(self) -> list[dict]:
        template = make_template()
        two = [make_step(50.0, 80.0, 100.0)] * 2
        return [
            phase18.anchor_metrics(make_artifact("a1", two, length=2), template, "a1", INFO),
            phase18.anchor_metrics(make_artifact("b1", two, length=2), template, "b1", INFO),
            phase18.anchor_metrics(make_artifact("c1", two, length=2), template, "c1", dict(INFO, video_id="video-b")),
            phase18.anchor_metrics(make_artifact("a2", []), template, "a2", dict(INFO, video_id="video-c")),
        ]

    def test_real_records_summarize_without_key_errors(self) -> None:
        summary = phase18.summarize(self._records(), bootstrap=200, seed=7, wall_clock_ratio={"n": 1, "median": 1.0})

        primary = summary["populations"]["primary_gpu_with_remaining"]
        self.assertEqual(primary["anchors"], 3)  # a1, b1, a2 have remaining steps; c1 does not
        self.assertEqual(primary["no_remaining_step_anchors"], 0)
        self.assertEqual(summary["populations"]["last_step_gpu"]["anchors"], 1)
        self.assertIsNotNone(primary["ratio_of_sums_next_k_p50"])

    def test_ratio_of_sums_is_a_ratio_of_sums_not_a_mean_of_ratios(self) -> None:
        summary = phase18.summarize(self._records(), bootstrap=200, seed=7, wall_clock_ratio={"n": 1, "median": 1.0})
        primary = summary["populations"]["primary_gpu_with_remaining"]

        # event-level next_k truth: a1 -> a2+b1 = 250; b1 -> c1 = 650; a2 excluded (0 predicted)
        expected = (100.0 + 100.0) / (250.0 + 650.0)
        mean_of_ratios = (100.0 / 250.0 + 100.0 / 650.0) / 2.0
        self.assertAlmostEqual(primary["ratio_of_sums_next_k_p50"], expected)
        self.assertNotAlmostEqual(primary["ratio_of_sums_next_k_p50"], mean_of_ratios)

    def test_counts_and_populations(self) -> None:
        summary = phase18.summarize(self._records(), bootstrap=100, seed=7, wall_clock_ratio={"n": 1, "median": 1.0})
        self.assertEqual(summary["counts"]["joined_anchors"], 4)
        self.assertEqual(summary["counts"]["distinct_videos"], 3)
        self.assertEqual(summary["counts"]["distinct_templates"], 1)
        self.assertEqual(summary["counts"]["zero_pred_step_anchors"], 1)
        # this fixture has one event per remaining step, so no branching population exists
        self.assertNotIn("branching", summary["populations"])
        self.assertEqual(summary["populations"]["primary_gpu_with_remaining"]["branching_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
