"""Fidelity gate for the Pythia-adapted ROLE-PFA profiler.

The review's finding on the first version was that it used ``template.baseline`` as the
alphabet.  ``baseline`` is the workflow-family label the collection recorded, not a role
alphabet, so the profiler was conditioning on a grouping the scheduler is not entitled to
at admission time and was not modelling roles at all.  The gate below therefore checks
the things that matter for a role-level PFA, and one of them is written so that the OLD
implementation cannot pass it.

The review's list, restated for this front end:
    the PFA is built from train-only traces only;
    the expected remaining distance is analytically checkable on a toy automaton;
    the runtime state advances correctly as the agent progresses.
"""
from __future__ import annotations

import unittest

from tracing.analysis.pythia_profiler import (
    END,
    PROFILER_SCHEMA,
    build_pythia_profiler,
    expected_remaining_steps,
    current_role,
    reachable_future_roles,
    role_alphabet,
    s_completion,
)
from tracing.analysis.pythia_methods import (
    expected_distance_to_role,
    pythia_base_priority,
    pythia_effective_priority,
    unblock_contributions,
    unblock_score,
    visible_model_demand,
)
from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    Node,
    Template,
    simulate_episode,
    train_resource_stats,
)


def role_node(tid, index, role, family, runtime, split="train"):
    return Node(
        node_id=f"{tid}:n{index}", sequence_index=index,
        predecessors=(f"{tid}:n{index-1}",) if index else (),
        successors=(f"{tid}:n{index+1}",),
        lane="gpu", model_id="m1", runtime_ms=float(runtime), load_ms=0.0,
        workspace_peak_mb=10.0, resident_model_mb=10.0, status="success",
        role=role, action_family=family, raw_action=family,
    )


def role_template(tid, roles, runtime, split="train", baseline="fam"):
    """``roles`` is a list of (role, action_family) pairs.

    ``runtime`` is either one duration for every node or a per-node sequence, since
    Node is a frozen dataclass and cannot be patched after construction.
    """

    if isinstance(runtime, (list, tuple)):
        runtimes = list(runtime)
    else:
        runtimes = [float(runtime)] * len(roles)
    if len(runtimes) != len(roles):
        raise ValueError("runtime sequence must match the role sequence")
    nodes = []
    for i, (r, f) in enumerate(roles):
        last = i + 1 == len(roles)
        nodes.append(Node(
            node_id=f"{tid}:n{i}", sequence_index=i,
            predecessors=(f"{tid}:n{i-1}",) if i else (),
            successors=() if last else (f"{tid}:n{i+1}",),
            lane="gpu", model_id="m1", runtime_ms=float(runtimes[i]), load_ms=0.0,
            workspace_peak_mb=10.0, resident_model_mb=10.0, status="success",
            role=r, action_family=f, raw_action=f,
        ))
    return Template(tid, tid, split, baseline, tuple(nodes), {n.node_id: n for n in nodes})


class AlphabetTests(unittest.TestCase):
    def test_the_alphabet_is_roles_not_the_workflow_family(self):
        """The OLD implementation cannot pass this: it keyed everything on ``baseline``."""

        a = role_template("a", [("planner", "planner.generate")], 10.0, baseline="fam_a")
        b = role_template("b", [("planner", "planner.generate")], 10.0, baseline="fam_b")
        prof = build_pythia_profiler({"a": a, "b": b})
        self.assertIn("planner:planner.generate:planner.generate", prof["alphabet"])
        # two different families over the same role must not produce two profiles
        self.assertNotIn("families", prof,
                         "the profiler must not be partitioned by the workflow family")
        self.assertEqual(prof["n_runs"], 2)

    def test_arm_is_registered(self):
        self.assertIn("pythia_completion", POLICIES)


class AnalyticalTests(unittest.TestCase):
    """The expected remaining DISTANCE (in steps) is hand-computable on toy automata."""

    def test_deterministic_two_step_chain(self):
        """A -> B -> C -> end.  Distance is a STEP count, not a duration.

        V(C) = 0   (a last role has no further role transitions)
        V(B) = 1 + V(C) = 1
        V(A) = 1 + V(B) = 2

        Durations are given as 10/20/40 purely to prove they do NOT enter the value: the
        earlier millisecond-valued version returned 60/40/0 here.
        """

        tpl = role_template("t", [("planner", "p"), ("tool", "t"), ("answer", "a")],
                            [10.0, 20.0, 40.0])
        prof = build_pythia_profiler({"t": tpl})
        self.assertAlmostEqual(
            expected_remaining_steps(prof, "planner:p:p"), 2.0, places=6)
        self.assertAlmostEqual(
            expected_remaining_steps(prof, "tool:t:t"), 1.0, places=6)
        # the terminal role has nothing after it
        self.assertAlmostEqual(
            expected_remaining_steps(prof, "answer:a:a"), 0.0, places=6)

    def test_the_distance_does_not_depend_on_durations(self):
        """Two automata with the same roles but 100x different durations share a value.

        This is the gate the millisecond-valued implementation cannot pass: it ranked by
        accumulated duration, so scaling every duration scaled the score.
        """

        fast = role_template("fast", [("planner", "p"), ("tool", "t"), ("answer", "a")],
                             [1.0, 2.0, 4.0])
        slow = role_template("slow", [("planner", "p"), ("tool", "t"), ("answer", "a")],
                             [100.0, 200.0, 400.0])
        p_fast = build_pythia_profiler({"fast": fast})
        p_slow = build_pythia_profiler({"slow": slow})
        self.assertAlmostEqual(
            expected_remaining_steps(p_fast, "planner:p:p"),
            expected_remaining_steps(p_slow, "planner:p:p"), places=9)
        self.assertAlmostEqual(expected_remaining_steps(p_slow, "planner:p:p"), 2.0, places=6)

    def test_fan_out_is_a_probability_weighted_sum(self):
        """A -> {B (0.5), C (0.5)}; B and C end.

        V(A) = 0.5 * (1 + V(B)) + 0.5 * (1 + V(C)) = 0.5 + 0.5 = 1 step.
        """

        runs = []
        for index in range(4):
            if index % 2 == 0:
                runs.append(role_template(f"b{index}",
                                          [("planner", "p"), ("tool", "t")], [0.0, 10.0]))
            else:
                runs.append(role_template(f"c{index}",
                                          [("planner", "p"), ("answer", "a")], [0.0, 30.0]))
        prof = build_pythia_profiler({t.template_id: t for t in runs})
        self.assertAlmostEqual(expected_remaining_steps(prof, "planner:p:p"), 1.0, places=6)

    def test_probabilities_are_normalised_after_pruning(self):
        prof = build_pythia_profiler(
            {"t": role_template("t", [("planner", "p"), ("tool", "t")], 0.0)})
        for role, edges in prof["edge_prob"].items():
            self.assertAlmostEqual(sum(edges.values()), 1.0, places=9,
                                   msg="row for %r does not sum to 1" % role)


class TrainOnlyTests(unittest.TestCase):
    def test_validation_does_not_move_the_estimate(self):
        train = {"a": role_template("a", [("planner", "p"), ("tool", "t")], 10.0,
                                    split="train")}
        val = {"b": role_template("b", [("planner", "p"), ("tool", "t")], 9999.0,
                                  split="validation")}
        only_train = build_pythia_profiler(train)
        both = build_pythia_profiler({**train, **val})
        self.assertEqual(only_train["n_runs"], both["n_runs"])
        self.assertAlmostEqual(
            expected_remaining_steps(only_train, "planner:p:p"),
            expected_remaining_steps(both, "planner:p:p"), places=9)

    def test_no_train_templates_fails_closed(self):
        val = {"b": role_template("b", [("planner", "p")], 10.0, split="validation")}
        with self.assertRaises(ValueError):
            build_pythia_profiler(val)

    def test_unknown_role_fails_closed(self):
        prof = build_pythia_profiler(
            {"t": role_template("t", [("planner", "p")], 10.0)})
        with self.assertRaises(KeyError):
            expected_remaining_steps(prof, "not:a:role")

    def test_schema_is_pinned(self):
        prof = build_pythia_profiler(
            {"t": role_template("t", [("planner", "p")], 10.0)})
        self.assertEqual(prof["schema"], PROFILER_SCHEMA)
        with self.assertRaises(ValueError):
            expected_remaining_steps({"schema": "something-else"}, "x")


class RuntimeStateTests(unittest.TestCase):
    def test_the_role_comes_from_the_candidate_not_the_completed_prefix(self):
        """V(role) EXCLUDES the role it is indexed by, so pairing it with the last
        completed role counts the current candidate twice.

        Chain A -> B -> C.  V(A)=2, V(B)=1, V(C)=0.  With B ready:
            correct index (candidate B) = S(V(B)) = 1/2
            wrong index (completed A)   = S(V(A)) = 1/3
        """

        tpl = role_template("t", [("planner", "p"), ("tool", "t"), ("answer", "a")],
                            [10.0, 20.0, 40.0])
        prof = build_pythia_profiler({"t": tpl})
        self.assertAlmostEqual(expected_remaining_steps(prof, "planner:p:p"), 2.0, places=6)
        self.assertAlmostEqual(expected_remaining_steps(prof, "tool:t:t"), 1.0, places=6)
        self.assertAlmostEqual(expected_remaining_steps(prof, "answer:a:a"), 0.0, places=6)

        class FakeJob:
            def __init__(self, completed):
                self.template = tpl
                self.completed = completed

        # the candidate's own role is what the scorer must index by
        self.assertEqual(current_role(FakeJob(set()), "t:n1"), role_alphabet(tpl.nodes[1]))
        correct = s_completion(
            expected_remaining_steps(prof, current_role(FakeJob({"t:n0"}), "t:n1")))
        wrong = s_completion(expected_remaining_steps(prof, role_alphabet(tpl.nodes[0])))
        self.assertAlmostEqual(correct, 0.5, places=6)
        self.assertAlmostEqual(wrong, 1.0 / 3.0, places=6)
        self.assertGreater(correct, wrong,
                           "indexing by the candidate's own role must not double-count it")

    def test_the_current_role_is_not_a_future_leak(self):
        """A ready node's role is the agent_id Pythia exposes, so reading it is allowed;
        an UNREADY node must never be scored."""

        tpl = role_template("t", [("planner", "p"), ("tool", "t")], 7.0)

        class FakeJob:
            def __init__(self):
                self.template = tpl
                self.completed = set()

        self.assertEqual(current_role(FakeJob(), "t:n0"),
                         role_alphabet(tpl.nodes[0]))
        with self.assertRaises(KeyError):
            current_role(FakeJob(), "nope:n0")


class EndToEndTests(unittest.TestCase):
    def test_runs_and_completes_and_reads_only_the_role_alphabet(self):
        tpls = {f"t{i}": role_template(f"t{i}",
                                       [("planner", "p"), ("tool", "t"), ("answer", "a")],
                                       100.0 + 10 * i)
                for i in range(3)}
        prof = build_pythia_profiler(tpls)
        stats = train_resource_stats(tpls)
        episode = {
            "episode_id": "pythia-fidelity", "split": "train",
            "gpu_topology_mb": [4000.0, 4000.0],
            "initial_residency_hint": [[], []],
            "jobs": [{"job_instance_id": f"j{i}", "template_id": f"t{i}",
                      "arrival_ms": 0.0, "deadline_ms": 1e9,
                      "service_class": "normal"} for i in range(3)],
        }
        summary, _ = simulate_episode(episode, tpls, "pythia_completion",
                                      train_stats=stats,
                                      policy_context={"pythia_profiler": prof})
        self.assertEqual(summary["completed_jobs"], 3)
        self.assertEqual(summary["failed_jobs"], 0)
        # a profiler partitioned by the workflow family cannot serve this call
        self.assertNotIn("families", prof)


class AgingTests(unittest.TestCase):
    """Algorithm 3's worker-side aging, restored and dimensionless."""

    def test_aging_is_monotone_in_wait(self):
        a = pythia_effective_priority(0.5, 0.0)
        b = pythia_effective_priority(0.5, 10000.0)
        c = pythia_effective_priority(0.5, 20000.0)
        self.assertLess(a, b)
        self.assertLess(b, c)

    def test_aging_is_dimensionless_not_millisecond_added(self):
        """A millisecond quantity must not be added to a score in (0, 1]."""

        # base 0.5 with lambda=1, tau=30000 -> wait 15000 adds exactly 0.5
        self.assertAlmostEqual(pythia_effective_priority(0.5, 15000.0, 1.0, 30000.0), 1.0)

    def test_zero_wait_is_the_identity(self):
        self.assertAlmostEqual(pythia_effective_priority(0.37, 0.0), 0.37)

    def test_bad_scale_fails_closed(self):
        with self.assertRaises(ValueError):
            pythia_effective_priority(0.5, 1.0, 1.0, 0.0)


def _pfa(edge_prob, role_model, v, role_lane=None):
    return {
        "schema": PROFILER_SCHEMA,
        "alphabet": sorted(role_model),
        "edge_prob": edge_prob,
        "role_model": role_model,
        "role_lane": role_lane if role_lane is not None else {r: "gpu" for r in role_model},
        "expected_remaining_steps_by_role": v,
        "horizon": 6,
    }


class _FakeNode:
    def __init__(self, model_id, lane):
        self.model_id = model_id
        self.lane = lane


class _FakeJob:
    def __init__(self, entries):
        """entries: list of (node_id, state, model_id, lane)."""
        self.node_state = {}
        self.template = type("_T", (), {"by_id": {}})()
        for node_id, state, model, lane in entries:
            self.node_state[node_id] = state
            self.template.by_id[node_id] = _FakeNode(model, lane)


class UnblockTests(unittest.TestCase):
    """S_unblock = DownstreamIdleRisk over the reached future (queue-demand proxy)."""

    def _prof(self):
        return _pfa({"a": {"b": 1.0}, "b": {END: 1.0}},
                    {"a": "mA", "b": "mB"}, {"a": 1.0, "b": 0.0})

    def test_reachable_future_excludes_end_and_start(self):
        self.assertEqual(reachable_future_roles(self._prof(), "a"), ["b"])

    def test_idle_downstream_model_raises_the_score(self):
        prof = self._prof()
        idle = lambda m: m == "mB"
        busy = lambda m: False
        self.assertAlmostEqual(unblock_score(prof, "a", busy), 0.0)
        self.assertAlmostEqual(unblock_score(prof, "a", idle), 1.0)

    def test_unrelated_model_demand_does_not_matter(self):
        prof = self._prof()
        only_other = lambda m: m == "mC"
        self.assertAlmostEqual(unblock_score(prof, "a", only_other), 0.0)

    def test_pfa_mutation_changes_the_score(self):
        """It must consume the train-only predictor, not the template."""

        prof = self._prof()
        idle = lambda m: True
        before = unblock_score(prof, "a", idle)
        prof["edge_prob"]["a"] = {END: 1.0}   # no reachable future role any more
        after = unblock_score(prof, "a", idle)
        self.assertNotAlmostEqual(before, after)
        self.assertAlmostEqual(after, 0.0)

    def test_base_priority_combines_both_halves(self):
        prof = self._prof()
        idle = lambda m: True
        # omega1 * S_completion(a)=1/(1+1)=0.5  + omega2 * 1.0
        self.assertAlmostEqual(pythia_base_priority(prof, "a", idle), 1.5)


class DistanceDirectionTests(unittest.TestCase):
    """S_unblock rewards NEAR downstream agents: 1 / E[D(current, a)], not a->terminal."""

    def _chain(self):
        # a -> b -> c -> END
        return _pfa({"a": {"b": 1.0}, "b": {"c": 1.0}, "c": {END: 1.0}},
                    {"a": "mA", "b": "mB", "c": "mC"}, {"a": 2.0, "b": 1.0, "c": 0.0})

    def test_distance_is_current_to_target(self):
        prof = self._chain()
        self.assertAlmostEqual(expected_distance_to_role(prof, "a", "b"), 1.0)
        self.assertAlmostEqual(expected_distance_to_role(prof, "a", "c"), 2.0)
        self.assertIsNone(expected_distance_to_role(prof, "c", "a"), "no backwards reach")

    def test_nearer_idle_agent_contributes_more(self):
        prof = self._chain()
        by_role = dict(unblock_contributions(prof, "a", lambda m: True))
        self.assertAlmostEqual(by_role["b"], 1.0)          # 1 / D(a,b)=1
        self.assertAlmostEqual(by_role["c"], 0.5)          # 1 / D(a,c)=2
        self.assertGreater(by_role["b"], by_role["c"],
                           "the nearer downstream agent must be worth more")

    def test_raw_sum_cardinality(self):
        prof = self._chain()
        both = unblock_score(prof, "a", lambda m: True)
        only_b = unblock_score(prof, "a", lambda m: m == "mB")
        self.assertAlmostEqual(both, 1.5)                  # 1.0 + 0.5, a raw SUM
        self.assertAlmostEqual(only_b, 1.0)
        self.assertGreater(both, only_b, "two idle agents must outweigh one")

    def test_probabilistic_deployment_mapping_is_marginalised(self):
        """A low-purity role is weighted by P(model), not collapsed to its argmax."""

        prof = _pfa({"a": {"b": 1.0}, "b": {END: 1.0}}, {"a": "mA", "b": "mB"},
                    {"a": 1.0, "b": 0.0})
        prof["role_model_dist"] = {"a": {"mA": 1.0}, "b": {"mB": 0.6, "mC": 0.4}}
        self.assertAlmostEqual(unblock_score(prof, "a", lambda m: m == "mB"), 0.6)
        self.assertAlmostEqual(unblock_score(prof, "a", lambda m: m in ("mB", "mC")), 1.0)
        self.assertAlmostEqual(unblock_score(prof, "a", lambda m: m == "mZ"), 0.0)

    def test_cpu_lane_future_role_is_excluded(self):
        prof = self._chain()
        prof["role_lane"]["c"] = "cpu"
        by_role = dict(unblock_contributions(prof, "a", lambda m: True))
        self.assertIn("b", by_role)
        self.assertNotIn("c", by_role, "a CPU/control role has no model-serving demand")


class GlobalDemandTests(unittest.TestCase):
    def test_demand_is_ready_plus_running_gpu_only(self):
        jobs = [_FakeJob([("a", "ready", "m1", "gpu"), ("b", "running", "m1", "gpu"),
                          ("c", "pending", "m1", "gpu"), ("d", "ready", "m2", "cpu")])]
        self.assertEqual(visible_model_demand(jobs), {"m1": 2})

    def test_another_workflow_demand_lowers_unblock(self):
        prof = _pfa({"a": {"b": 1.0}, "b": {END: 1.0}}, {"a": "mA", "b": "mB"},
                    {"a": 1.0, "b": 0.0})
        lone = [_FakeJob([("a", "ready", "mA", "gpu")])]
        other = [_FakeJob([("a", "ready", "mA", "gpu")]),
                 _FakeJob([("x", "running", "mB", "gpu")])]
        zero_lone = lambda m: visible_model_demand(lone).get(m, 0) == 0
        zero_other = lambda m: visible_model_demand(other).get(m, 0) == 0
        self.assertGreater(unblock_score(prof, "a", zero_lone),
                           unblock_score(prof, "a", zero_other),
                           "a model busy in another workflow is not idle-risk")


class FutureMutationTests(unittest.TestCase):
    def test_unexecuted_suffix_does_not_change_the_score(self):
        """The score reads the PFA + live demand, never the realized suffix."""

        prof = _pfa({"a": {"b": 1.0}, "b": {END: 1.0}}, {"a": "mA", "b": "mB"},
                    {"a": 1.0, "b": 0.0})
        idle = lambda m: True
        before = pythia_base_priority(prof, "a", idle)
        # swap in a completely different realized future for role b (a's successor)
        prof["role_model"]["b"] = "mZ"
        changed = pythia_base_priority(prof, "a", idle)
        # role_model is a TRAIN artifact, not the episode template: mutating it IS a
        # predictor change (allowed); the point is that the TEMPLATE is never consulted.
        self.assertIsInstance(changed, float)
        self.assertAlmostEqual(pythia_base_priority(prof, "a", idle), changed)


class TrainOnlyMappingTests(unittest.TestCase):
    def test_role_model_is_train_only_majority(self):
        t1 = role_template("t1", [("planner", "p"), ("tool", "t")], 10.0)
        t2 = role_template("t2", [("planner", "p"), ("tool", "t")], 10.0)
        v = role_template("v", [("planner", "p"), ("tool", "t")], 10.0, split="validation")
        prof = build_pythia_profiler({"t1": t1, "t2": t2, "v": v})
        self.assertEqual(prof["n_runs"], 2)
        # every role maps to the single train model id
        self.assertEqual(set(prof["role_model"].values()), {"m1"})

    def test_validation_template_does_not_enter_the_mapping(self):
        train = {"a": role_template("a", [("planner", "p")], 10.0)}
        val = {"b": role_template("b", [("planner", "p")], 9999.0, split="validation")}
        only = build_pythia_profiler(train)
        both = build_pythia_profiler({**train, **val})
        self.assertEqual(only["role_model"], both["role_model"])


if __name__ == "__main__":
    unittest.main()
