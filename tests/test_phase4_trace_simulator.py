import unittest

from tracing.analysis.phase4_trace_simulator import Action, JobTemplate, run_experiment


class Phase4SimulatorTests(unittest.TestCase):
    def test_replay_reports_all_policies_and_no_missing_jobs(self):
        actions = (
            Action("frame-selector", "sample_seek", "cpu", 1.0, 0.0, 1.0),
            Action("image-qa", "spatial_qa", "qwen", 10.0, 5.0, 100.0),
            Action("image-grid-qa", "spatial_qa", "qwen", 8.0, 0.0, 100.0),
        )
        jobs = [JobTemplate(f"job-{index}", f"video-{index}", "star", actions) for index in range(4)]
        report = run_experiment(jobs, [200.0, 200.0], [0, 1], 0.0)
        self.assertEqual(report["data"]["runs"], 4)
        self.assertEqual({row["policy"] for row in report["aggregate"]}, {
            "round_robin", "least_loaded", "myopic", "static_template", "predictive", "oracle",
        })
        self.assertTrue(all(row["completed_jobs"] == 4 for row in report["aggregate"]))


if __name__ == "__main__":
    unittest.main()
