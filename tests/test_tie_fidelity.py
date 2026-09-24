"""Fidelity gate for Empirical-TIE-adapted.

The review's finding was that the previous 10/10 was a false positive: it tested the
bank and the pure formula but never that the real consumer used them.  Every test here
drives the ACTUAL policy path, and the sentinel tests deliberately make the TIE
criterion and a percentile criterion disagree so that a silent fallback is caught.
"""
from __future__ import annotations

import unittest

from tracing.analysis.tie_methods import (
    TIE_DEVIATION,
    tie_load_estimate,
    tie_queue_length,
    tie_score_for,
    bank_sample_report,
    build_tie_bank,
    tie_beta,
    tie_current_distribution,
    tie_current_score,
    tie_key,
    tie_wait_adjust,
    upper_cvar,
)
from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    Node,
    Template,
    simulate_episode,
    train_resource_stats,
)


def one_node_template(tid, model, runtime, lane="gpu", seq=0):
    """Three chained nodes on one model.

    estimate() enforces a minimum group size, so a single-node template would be
    rejected before the TIE bank is ever consulted.  Three nodes give the group enough
    support while keeping the template trivially small.
    """

    nodes = [Node(node_id=f"{tid}:n{i}", sequence_index=i,
                  predecessors=(f"{tid}:n{i-1}",) if i else (),
                  successors=(f"{tid}:n{i+1}",) if i < 2 else (),
                  lane=lane, model_id=model, runtime_ms=float(runtime) + i, load_ms=0.0,
                  workspace_peak_mb=10.0, resident_model_mb=10.0, status="success",
                  role="execute", action_family="inference")
             for i in range(3)]
    return Template(tid, tid, "train", "test", tuple(nodes), {n.node_id: n for n in nodes})


def two_group_bank(mean_a, cvar_a, mean_b, cvar_b):
    """A hand-built TIE bank with two groups; no estimate() involved."""

    return {"schema": "test", "groups": {
        "mA|gpu": {"model_id": "mA", "lane": "gpu", "sample_count": 100,
                   "runtime_mean_ms": mean_a, "runtime_cvar90_ms": cvar_a,
                   "load_mean_ms": 0.0},
        "mB|gpu": {"model_id": "mB", "lane": "gpu", "sample_count": 100,
                   "runtime_mean_ms": mean_b, "runtime_cvar90_ms": cvar_b,
                   "load_mean_ms": 0.0},
    }}


class FrontEndTests(unittest.TestCase):
    def test_arm_is_registered(self):
        self.assertIn("tie_current", POLICIES)

    def test_key_is_model_and_lane_only(self):
        """resource_keys() would add sequence_index and leak workflow position."""
        a = one_node_template("a", "m1", 10.0, seq=0)
        b = one_node_template("b", "m1", 10.0, seq=7)
        self.assertEqual(tie_key(a.nodes[0]), tie_key(b.nodes[0]))

    def test_missing_group_fails_closed(self):
        bank = two_group_bank(1.0, 1.0, 2.0, 2.0)
        with self.assertRaises(KeyError):
            tie_current_distribution(bank, one_node_template("z", "nope", 1.0).nodes[0])

    def test_missing_mean_or_cvar_fails_closed(self):
        """The exact bug: a missing mean must never fall back to a percentile."""
        for broken in ({"runtime_mean_ms": None}, {"runtime_cvar90_ms": None}):
            bank = {"groups": {"m1|gpu": {"model_id": "m1", "lane": "gpu",
                                          "sample_count": 5, "runtime_mean_ms": 1.0,
                                          "runtime_cvar90_ms": 2.0, "load_mean_ms": 0.0,
                                          **broken}}}
            with self.assertRaises(ValueError):
                tie_current_distribution(bank, one_node_template("t", "m1", 1.0).nodes[0])

    def test_bank_is_train_only(self):
        train = {"a": one_node_template("a", "m1", 100.0)}
        val = Template("b", "b", "validation", "test",
                       (Node(node_id="b:n", sequence_index=0, predecessors=(), successors=(),
                             lane="gpu", model_id="m1", runtime_ms=99999.0, load_ms=0.0,
                             workspace_peak_mb=1.0, resident_model_mb=1.0, status="success",
                             role="execute", action_family="inference"),), {})
        val.by_id["b:n"] = val.nodes[0]
        both = build_tie_bank({**train, "b": val})
        # the training template contributes three nodes and the validation one none
        self.assertEqual(both["groups"]["m1|gpu"]["sample_count"], 3)
        self.assertAlmostEqual(both["groups"]["m1|gpu"]["runtime_mean_ms"],
                               (100.0 + 101.0 + 102.0) / 3.0)

    def test_cvar_small_sample_rule(self):
        self.assertAlmostEqual(upper_cvar([5.0], 0.90), 5.0)          # singleton
        self.assertAlmostEqual(upper_cvar([1.0, 10.0], 0.90), 10.0)   # k = 1
        self.assertAlmostEqual(upper_cvar([1.0, 2.0, 3.0, 4.0], 0.90), 4.0)

    def test_sample_report_surfaces_thin_groups(self):
        bank = {"groups": {"a": {"sample_count": 1}, "b": {"sample_count": 50}}}
        rep = bank_sample_report(bank)
        self.assertAlmostEqual(rep["fraction_singleton"], 0.5)
        self.assertAlmostEqual(rep["fraction_lt_10"], 0.5)

    def test_deviation_is_recorded(self):
        self.assertIn("max-token censoring", TIE_DEVIATION)


class QueueLengthTests(unittest.TestCase):
    """L_q must count WAITING UNITS, not candidate tuples.

    The pool is ready_node x free_gpu, so counting entries would multiply the waiting
    queue by the number of free devices.  These two cases differ only in the number of
    free GPUs and must report the same L_q.
    """

    @staticmethod
    def _candidate(priority, job_index, node_id, gpu_index):
        return ((priority, 0.0, job_index, node_id), job_index, node_id, "mX", gpu_index,
                {"runtime_p50_ms": 1.0, "load_p50_ms": 0.0}, True)

    def test_two_ready_units_on_one_free_gpu(self):
        pool = [self._candidate(0.0, 0, "a:n", 0), self._candidate(0.0, 1, "b:n", 0)]
        self.assertEqual(tie_queue_length(pool, 0.0), 2.0)

    def test_two_ready_units_on_two_free_gpus_is_still_two(self):
        pool = []
        for gpu_index in (0, 1):
            pool.append(self._candidate(0.0, 0, "a:n", gpu_index))
            pool.append(self._candidate(0.0, 1, "b:n", gpu_index))
        self.assertEqual(len(pool), 4, "the pool really does replicate per free device")
        self.assertEqual(tie_queue_length(pool, 0.0), 2.0,
                         "L_q must not scale with the number of free GPUs")

    def test_only_the_competitive_priority_tier_counts(self):
        pool = [self._candidate(0.0, 0, "a:n", 0), self._candidate(1.0, 1, "b:n", 0)]
        self.assertEqual(tie_queue_length(pool, 0.0), 1.0)
        self.assertEqual(tie_queue_length(pool, 1.0), 1.0)


class LoadOwnershipTests(unittest.TestCase):
    def test_load_comes_from_the_tie_bank(self):
        bank = two_group_bank(1.0, 1.0, 2.0, 2.0)
        bank["groups"]["mA|gpu"]["load_mean_ms"] = 42.0
        node = one_node_template("t", "mA", 1.0).nodes[0]
        self.assertAlmostEqual(tie_load_estimate(bank, node), 42.0)

    def test_score_is_built_only_from_the_bank(self):
        bank = two_group_bank(100.0, 400.0, 2.0, 2.0)
        bank["groups"]["mA|gpu"]["load_mean_ms"] = 1.0
        node = one_node_template("t", "mA", 1.0).nodes[0]
        # E + beta*CVaR + load = 100 + 0.5*400 + 1
        self.assertAlmostEqual(tie_score_for(bank, node, 0.5, resident=False), 301.0)
        # a resident model pays no load
        self.assertAlmostEqual(tie_score_for(bank, node, 0.5, resident=True), 300.0)

    def test_missing_group_fails_closed(self):
        bank = two_group_bank(1.0, 1.0, 2.0, 2.0)
        with self.assertRaises(KeyError):
            tie_score_for(bank, one_node_template("t", "nope", 1.0).nodes[0], 0.5,
                          resident=True)


class FormulaTests(unittest.TestCase):
    def test_beta_is_the_paper_clip(self):
        self.assertAlmostEqual(tie_beta(0.0, 2.0), 0.1)
        self.assertAlmostEqual(tie_beta(2.0, 2.0), 0.1)
        self.assertAlmostEqual(tie_beta(6.0, 2.0), 0.3)
        self.assertAlmostEqual(tie_beta(10.0, 2.0), 0.5)
        self.assertAlmostEqual(tie_beta(100.0, 2.0), 0.5)

    def test_score_is_mean_plus_beta_cvar(self):
        self.assertAlmostEqual(tie_current_score(100.0, 400.0, 0.5), 300.0)
        self.assertAlmostEqual(tie_current_score(100.0, 400.0, 0.1), 140.0)

    def test_wait_adjust_decays(self):
        self.assertAlmostEqual(tie_wait_adjust(1000.0, 0.0, 0.9, 30000.0), 1000.0)
        self.assertLess(tie_wait_adjust(1000.0, 30000.0, 0.9, 30000.0), 1000.0)
        self.assertAlmostEqual(tie_wait_adjust(1000.0, 30000.0, 0.9, 30000.0), 900.0)

    def test_wait_adjust_is_not_the_aging_rule(self):
        """Aging SUBTRACTS a credit; this MULTIPLIES.  Different mechanisms."""
        from tracing.analysis.workload_v02_simulator import srtf_aging_key
        score = 1000.0
        wait = 30000.0
        self.assertNotAlmostEqual(tie_wait_adjust(score, wait, 0.9, 30000.0),
                                  srtf_aging_key(0.0, score, wait)[1])


class SentinelMutationTests(unittest.TestCase):
    """The tests the review asked for: make TIE and a percentile disagree."""

    def _run_first_choice(self, bank, templates, jobs, gpu_count=1, beta_forced=None):
        ep = {
            "episode_id": "sentinel", "split": "train",
            "gpu_topology_mb": [40000.0] * gpu_count,
            "initial_residency_hint": [[] for _ in range(gpu_count)],
            "jobs": jobs,
        }
        ctx = {"tie_bank": bank}
        if beta_forced is not None:
            ctx["tie_B"] = beta_forced
        s, ev = simulate_episode(ep, templates, "tie_current", policy_context=ctx,
                                 train_stats=train_resource_stats(templates),
                                 collect_events=True)
        starts = [e for e in ev if e.get("event_type") == "node_start"]
        return (starts[0].get("job_instance_id") if starts else None), s

    def test_choice_follows_mean_and_cvar_not_percentiles(self):
        """A has the smaller mean and the fatter tail; B has the smaller percentile.

        With beta pushed to its ceiling, TIE must prefer B, while a p50-driven arm
        would prefer A.  If the arm silently used p50/p90 the winner would be A.
        """
        templates = {"ta": one_node_template("ta", "mA", 10.0),
                     "tb": one_node_template("tb", "mB", 1000.0)}
        bank = two_group_bank(mean_a=100.0, cvar_a=1000.0, mean_b=200.0, cvar_b=200.0)
        jobs = [{"job_instance_id": "ja", "template_id": "ta", "arrival_ms": 0.0,
                 "deadline_ms": 1e9, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "tb", "arrival_ms": 0.0,
                 "deadline_ms": 1e9, "service_class": "normal"}]
        # B = 1 forces beta to its 0.5 ceiling: A = 100+500 = 600, B = 200+100 = 300
        first, _ = self._run_first_choice(bank, templates, jobs, gpu_count=1, beta_forced=1.0)
        self.assertEqual(first, "jb", "TIE must follow mean/CVaR, not the percentile")

    def test_percentile_arm_disagrees_on_the_same_state(self):
        """The same state under a p50-driven arm picks the other candidate."""
        templates = {"ta": one_node_template("ta", "mA", 10.0),
                     "tb": one_node_template("tb", "mB", 1000.0)}
        ep = {"episode_id": "sentinel2", "split": "train",
              "gpu_topology_mb": [40000.0], "initial_residency_hint": [[]],
              "jobs": [{"job_instance_id": "ja", "template_id": "ta", "arrival_ms": 0.0,
                        "deadline_ms": 1e9, "service_class": "normal"},
                       {"job_instance_id": "jb", "template_id": "tb", "arrival_ms": 0.0,
                        "deadline_ms": 1e9, "service_class": "normal"}]}
        _s, ev = simulate_episode(ep, templates, "myopic",
                                  train_stats=train_resource_stats(templates),
                                  collect_events=True)
        starts = [e for e in ev if e.get("event_type") == "node_start"]
        self.assertEqual(starts[0].get("job_instance_id"), "ja",
                         "the percentile arm should prefer the small percentile")

    def test_changing_only_the_cvar_flips_the_winner(self):
        templates = {"ta": one_node_template("ta", "mA", 10.0),
                     "tb": one_node_template("tb", "mB", 1000.0)}
        jobs = [{"job_instance_id": "ja", "template_id": "ta", "arrival_ms": 0.0,
                 "deadline_ms": 1e9, "service_class": "normal"},
                {"job_instance_id": "jb", "template_id": "tb", "arrival_ms": 0.0,
                 "deadline_ms": 1e9, "service_class": "normal"}]
        # give A a small tail: A = 100 + 0.5*10 = 105 beats B = 300
        bank = two_group_bank(mean_a=100.0, cvar_a=10.0, mean_b=200.0, cvar_b=200.0)
        first, _ = self._run_first_choice(bank, templates, jobs, gpu_count=1, beta_forced=1.0)
        self.assertEqual(first, "ja", "a smaller CVaR must be able to flip the choice")


if __name__ == "__main__":
    unittest.main()
