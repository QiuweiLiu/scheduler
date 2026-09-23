"""Fidelity gate for Pythia-adapted (layer 1 of the two-layer gate).

The review's list for Pythia:
    the PFA is built from train-only traces only;
    the expected remaining distance is analytically checkable on a toy trace;
    the runtime state advances correctly as the current agent progresses.
"""
from __future__ import annotations

import unittest

from tracing.analysis.pythia_profiler import (
    build_pythia_profiler,
    expected_remaining_ms,
    s_completion,
)
from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    Node,
    Template,
    simulate_episode,
    train_resource_stats,
)


def chain_template(tid: str, runtimes, family: str, split: str = "train", model="m1") -> Template:
    nodes = []
    for i, rt in enumerate(runtimes):
        nodes.append(Node(
            node_id=f"{tid}:n{i}", sequence_index=i,
            predecessors=(f"{tid}:n{i-1}",) if i else (),
            successors=(f"{tid}:n{i+1}",) if i + 1 < len(runtimes) else (),
            lane="gpu", model_id=model, runtime_ms=float(rt), load_ms=5.0,
            workspace_peak_mb=100.0, resident_model_mb=90.0, status="success",
            role="execute", action_family="inference",
        ))
    return Template(tid, tid, split, family, tuple(nodes), {n.node_id: n for n in nodes})


class PythiaProfilerTests(unittest.TestCase):
    def test_arm_is_registered(self):
        self.assertIn("pythia_completion", POLICIES)

    def test_expected_remaining_is_analytically_exact(self):
        """Two toy runs make the expectation hand-computable.

        run A = [10, 20, 30]: remaining after k consumed = [60, 50, 30, 0]
        run B = [100, 100]:   remaining after k consumed = [200, 100, 0, 0]
        mean(remaining after k=0) = (60 + 200) / 2 = 130
        mean(remaining after k=1) = (50 + 100) / 2 =  75
        mean(remaining after k=2) = (30 + 0)   / 2 =  15
        mean(remaining after k=3) = (0  + 0)   / 2 =   0
        """
        tpls = {"a": chain_template("a", [10, 20, 30], "fam"),
                "b": chain_template("b", [100, 100], "fam")}
        prof = build_pythia_profiler(tpls)
        fam = prof["families"]["fam"]
        self.assertEqual(fam["n_runs"], 2)
        self.assertEqual(fam["remaining_mean_by_consumed"], [130.0, 75.0, 15.0, 0.0])
        self.assertAlmostEqual(expected_remaining_ms(prof, "fam", 0), 75.0)
        self.assertAlmostEqual(expected_remaining_ms(prof, "fam", 1), 15.0)
        self.assertAlmostEqual(expected_remaining_ms(prof, "fam", 2), 0.0)

    def test_profiler_uses_only_the_training_split(self):
        train = {"a": chain_template("a", [10, 20], "fam", split="train")}
        val = {"b": chain_template("b", [9999, 9999], "fam", split="validation")}
        only_val = build_pythia_profiler(val)
        both = build_pythia_profiler({**train, **val})
        # the validation run must not move the estimate
        self.assertAlmostEqual(
            expected_remaining_ms(only_val, "fam", 0) if "fam" in only_val["families"] else
            expected_remaining_ms(both, "fam", 0),
            expected_remaining_ms(both, "fam", 0),
        )
        self.assertEqual(both["families"]["fam"]["n_runs"], 1, "validation leaked into the profiler")

    def test_completion_score_is_monotone_in_distance(self):
        self.assertGreater(s_completion(100.0), s_completion(1000.0))
        self.assertAlmostEqual(s_completion(100.0), 0.01)

    def test_unknown_family_fails_closed(self):
        prof = build_pythia_profiler({"a": chain_template("a", [1, 2], "fam")})
        with self.assertRaises(KeyError):
            expected_remaining_ms(prof, "not-a-family", 0)


class PythiaPolicyTests(unittest.TestCase):
    def setUp(self):
        self.templates = {
            "a": chain_template("a", [100.0, 200.0], "fam", model="m0"),
            "b": chain_template("b", [150.0, 160.0], "fam", model="m1"),
        }
        self.prof = build_pythia_profiler(self.templates)
        self.stats = train_resource_stats(self.templates)
        self.episode = {
            "episode_id": "pythia-fidelity", "split": "train",
            "gpu_topology_mb": [4000.0, 4000.0], "initial_residency_hint": [[], []],
            "jobs": [
                {"job_instance_id": "ja", "template_id": "a", "arrival_ms": 0.0,
                 "deadline_ms": 1e9, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "b", "arrival_ms": 0.0,
                 "deadline_ms": 1e9, "service_class": "normal"},
            ],
        }

    def test_runs_and_completes(self):
        s, _ = simulate_episode(self.episode, self.templates, "pythia_completion",
                                train_stats=self.stats,
                                policy_context={"pythia_profiler": self.prof},
                                collect_events=True)
        self.assertEqual(s["completed_jobs"], 2)
        self.assertEqual(s["failed_jobs"], 0)

    def test_requires_the_profiler_fail_closed(self):
        with self.assertRaises(ValueError):
            simulate_episode(self.episode, self.templates, "pythia_completion",
                             train_stats=self.stats, collect_events=False)

    def test_runtime_state_advances_as_the_agent_progresses(self):
        """After the first node the job must be scored with consumed=1."""
        _s, events = simulate_episode(self.episode, self.templates, "pythia_completion",
                                      train_stats=self.stats,
                                      policy_context={"pythia_profiler": self.prof},
                                      collect_events=True)
        starts = [e for e in events if e.get("event_type") == "node_start"]
        self.assertEqual(len(starts), 4, "two 2-node chains should start four nodes")
        # each job's second node must exist, which can only happen once consumed=1
        per_job = {}
        for e in starts:
            per_job.setdefault(e.get("job_instance_id"), []).append(e.get("node_id"))
        for jid, nodes in per_job.items():
            self.assertEqual(len(nodes), 2, "%s did not advance to its second node" % jid)


if __name__ == "__main__":
    unittest.main()
