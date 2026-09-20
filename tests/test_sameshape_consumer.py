from __future__ import annotations

import unittest

from tracing.analysis.workload_v02_simulator import (
    POLICIES,
    SAMESHAPE_POLICIES,
    Node,
    Template,
    sameshape_future_cost,
    simulate_episode,
    train_resource_stats,
)


def make_template(template_id: str, runtime_ms: float, load_ms: float) -> Template:
    node = Node(
        node_id=f"{template_id}:gpu",
        sequence_index=0,
        predecessors=(),
        successors=(),
        lane="gpu",
        model_id="model-a",
        runtime_ms=runtime_ms,
        load_ms=load_ms,
        workspace_peak_mb=2100.0,
        resident_model_mb=2000.0,
        status="success",
        role="execute",
        action_family="inference",
    )
    return Template(template_id, template_id, "train", "test", (node,), {node.node_id: node})


def make_episode() -> dict[str, object]:
    return {
        "episode_id": "sameshape-consumer-smoke",
        "split": "train",
        "gpu_topology_mb": [4000.0, 4000.0],
        "initial_residency_hint": [[], []],
        "jobs": [
            {
                "job_instance_id": "priority",
                "template_id": "t0",
                "arrival_ms": 0.0,
                "deadline_ms": 100000.0,
                "service_class": "priority",
            },
            {
                "job_instance_id": "normal-a",
                "template_id": "t1",
                "arrival_ms": 0.0,
                "deadline_ms": 100000.0,
                "service_class": "normal",
            },
            {
                "job_instance_id": "normal-b",
                "template_id": "t2",
                "arrival_ms": 0.0,
                "deadline_ms": 100000.0,
                "service_class": "normal",
            },
        ],
    }


def make_artifacts(node_ids: tuple[str, ...], p50: float, p90: float, p95: float, horizon: int = 5) -> dict:
    step = {
        "model_id": "model-a",
        "execution_lane": "gpu",
        "resource": {
            "runtime_ms_quantiles": {"p50": p50, "p90": p90, "p95": p95},
            "load_occurrence_probability": 0.0,
        },
    }
    return {
        node_id: {
            f"future_h{horizon}": [
                {"scenario_probability": 1.0, "steps": [dict(step) for _ in range(horizon)]}
            ]
        }
        for node_id in node_ids
    }


class SameShapeConsumerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.templates = {
            f"t{index}": make_template(f"t{index}", 10.0 + 5.0 * index, 5.0) for index in range(3)
        }
        self.episode = make_episode()
        self.train_stats = train_resource_stats(self.templates)
        self.artifacts = make_artifacts(("t0:gpu", "t1:gpu", "t2:gpu"), 100.0, 300.0, 400.0)

    def test_arms_are_registered_policies(self) -> None:
        for policy in SAMESHAPE_POLICIES:
            self.assertIn(policy, POLICIES)

    def test_p95_arm_is_exactly_the_q95_champion_consumer(self) -> None:
        """sameshape_h5_p95 differs from predopt_h5_q95 in name only."""

        champion, _champion_events = simulate_episode(
            self.episode,
            self.templates,
            "predopt_h5_q95",
            future_artifacts=self.artifacts,
            train_stats=self.train_stats,
            collect_events=False,
        )
        candidate, _candidate_events = simulate_episode(
            self.episode,
            self.templates,
            "sameshape_h5_p95",
            future_artifacts=self.artifacts,
            train_stats=self.train_stats,
            collect_events=False,
        )
        self.assertEqual(
            {key: value for key, value in champion.items() if key != "policy"},
            {key: value for key, value in candidate.items() if key != "policy"},
        )

    def test_future_statistic_is_monotone_and_truth_arm_runs(self) -> None:
        node_id = "t0:gpu"
        p50_cost = sameshape_future_cost(node_id, self.artifacts, self.train_stats, 5, "p50")
        p95_cost = sameshape_future_cost(node_id, self.artifacts, self.train_stats, 5, "p95")
        self.assertAlmostEqual(p50_cost, 500.0, places=6)
        self.assertAlmostEqual(p95_cost, 2000.0, places=6)
        self.assertLess(p50_cost, p95_cost)

        truth_summary, _events = simulate_episode(
            self.episode,
            self.templates,
            "sameshape_h5_truth",
            future_artifacts=self.artifacts,
            train_stats=self.train_stats,
            collect_events=False,
        )
        self.assertEqual(truth_summary["completed_jobs"], 3)
        self.assertEqual(truth_summary["failed_jobs"], 0)

    def test_arms_require_future_artifacts(self) -> None:
        with self.assertRaises(ValueError):
            simulate_episode(
                self.episode,
                self.templates,
                "sameshape_h5_p50",
                train_stats=self.train_stats,
                collect_events=False,
            )


# --------------------------------------------------------------------------- #
# resource-v2 consumption functionals
RESOURCE_V2_ARTIFACT_NODE = "t0:gpu"
TEST_EDGES = (0.0, 1000.0, 5000.0)
TEST_REPS = (1.0, 3000.0, 25000.0)


def _sha256_file(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def views_for(probs):
    from tracing.analysis.workload_v02_simulator import _resource_v2_views

    return _resource_v2_views(probs, TEST_REPS, 0.95)


def make_v2_artifacts(
    node_ids=("t0:gpu", "t1:gpu", "t2:gpu"),
    probs=(0.80, 0.15, 0.05),
    horizon=5,
    occurrence=0.0,
    load_p95=0.0,
    drop=(),
):
    views = views_for(probs)
    resource = {
        "runtime_probs": list(probs),
        "runtime_ms_quantiles": {"p50": views["p50"], "p90": views["p90"], "p95": views["p95"]},
        "runtime_mean_ms": views["runtime_mean_ms"],
        "cvar95_ms": views["cvar95_ms"],
        "bin_schema_id": "test_3",
        "load_occurrence_probability": occurrence,
    }
    if load_p95:
        resource["load_duration_ms_quantiles"] = {"p95": load_p95}
    for key in drop:
        resource.pop(key, None)
    step = {"model_id": "model-a", "execution_lane": "gpu", "resource": dict(resource)}
    return {
        node_id: {
            f"future_h{horizon}": [
                {"scenario_probability": 1.0, "steps": [dict(step) for _ in range(horizon)]}
            ]
        }
        for node_id in node_ids
    }


class ResourceV2ConsumerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.templates = {
            f"t{index}": make_template(f"t{index}", 10.0 + 5.0 * index, 5.0) for index in range(3)
        }
        self.episode = make_episode()
        self.train_stats = train_resource_stats(self.templates)

    def test_new_arms_are_registered(self) -> None:
        for policy in ("sameshape_h5_condmean", "sameshape_h5_stepcvar95"):
            self.assertIn(policy, POLICIES)
            self.assertIn(policy, SAMESHAPE_POLICIES)

    def test_condmean_and_stepcvar_match_their_fields(self) -> None:
        """The new arms read the canonical view, not the p95."""

        artifacts = make_v2_artifacts()
        views = views_for((0.80, 0.15, 0.05))
        for stat, key in (("condmean", "runtime_mean_ms"), ("stepcvar95", "cvar95_ms")):
            got = sameshape_future_cost(RESOURCE_V2_ARTIFACT_NODE, artifacts, self.train_stats, 5, stat)
            self.assertAlmostEqual(got, views[key] * 5, places=6, msg=stat)
        # and they differ from the legacy p95 consumption
        p95 = sameshape_future_cost(RESOURCE_V2_ARTIFACT_NODE, artifacts, self.train_stats, 5, "p95")
        self.assertNotAlmostEqual(p95, views["runtime_mean_ms"] * 5, places=6)

    def test_new_arms_are_fail_closed(self) -> None:
        """A missing / NaN / negative field must raise, never silently fall back."""

        for key in ("runtime_mean_ms", "cvar95_ms"):
            stat = "condmean" if key == "runtime_mean_ms" else "stepcvar95"
            with self.assertRaises(ValueError, msg=f"{key} removed"):
                sameshape_future_cost(
                    RESOURCE_V2_ARTIFACT_NODE,
                    make_v2_artifacts(drop=(key,)),
                    self.train_stats, 5, stat,
                )
            for bad in (float("nan"), float("inf"), -1.0):
                artifacts = make_v2_artifacts()
                artifacts[RESOURCE_V2_ARTIFACT_NODE]["future_h5"][0]["steps"][0]["resource"][key] = bad
                with self.assertRaises(ValueError, msg=f"{key}={bad!r}"):
                    sameshape_future_cost(
                        RESOURCE_V2_ARTIFACT_NODE, artifacts, self.train_stats, 5, stat
                    )

    def test_legacy_p95_still_falls_back_but_the_new_arms_do_not(self) -> None:
        """The frozen champion keeps its documented fallback; the arms may not."""

        stripped = make_v2_artifacts(drop=("runtime_ms_quantiles", "runtime_mean_ms", "cvar95_ms"))
        # legacy path falls back to the statistics estimate instead of raising
        sameshape_future_cost(RESOURCE_V2_ARTIFACT_NODE, stripped, self.train_stats, 5, "p95")
        for stat in ("condmean", "stepcvar95"):
            with self.assertRaises(ValueError, msg=stat):
                sameshape_future_cost(RESOURCE_V2_ARTIFACT_NODE, stripped, self.train_stats, 5, stat)

    def test_load_rule_is_shared_by_every_functional(self) -> None:
        """Only the runtime functional may change; the load surcharge is common."""

        plain = make_v2_artifacts()
        loaded = make_v2_artifacts(occurrence=0.9, load_p95=777.0)
        for stat in ("p95", "condmean", "stepcvar95"):
            base = sameshape_future_cost(RESOURCE_V2_ARTIFACT_NODE, plain, self.train_stats, 5, stat)
            with_load = sameshape_future_cost(RESOURCE_V2_ARTIFACT_NODE, loaded, self.train_stats, 5, stat)
            self.assertAlmostEqual(with_load - base, 777.0 * 5, places=6, msg=stat)

    def test_new_arms_run_end_to_end(self) -> None:
        artifacts = make_v2_artifacts()
        for policy in ("sameshape_h5_p95", "sameshape_h5_condmean", "sameshape_h5_stepcvar95"):
            summary, _events = simulate_episode(
                self.episode, self.templates, policy,
                future_artifacts=artifacts, train_stats=self.train_stats, collect_events=False,
            )
            self.assertEqual(summary["completed_jobs"], 3)
            self.assertEqual(summary["failed_jobs"], 0)
            self.assertEqual(summary["policy"], policy)


class ResourceV2OverlayTests(unittest.TestCase):
    """The overlay loader must refuse to return a partially repaired artifact."""

    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base_root = self.root / "base"
        self.arm_root = self.root / "arm"
        self.base_root.mkdir()
        self.arm_root.mkdir()
        self.write_packs()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # -- helpers ---------------------------------------------------------- #
    def base_step(self, node_id, index, runtime_p95):
        return {
            "step_index": index,
            "model_id": "model-a",
            "execution_lane": "gpu",
            "role": "execute",
            "resource": {
                "runtime_ms_quantiles": {"p50": runtime_p95 * 0.2, "p90": runtime_p95 * 0.8, "p95": runtime_p95},
                "load_occurrence_probability": 0.0,
            },
        }

    def v2_step(self, index, probs):
        views = views_for(probs)
        return {
            "step_index": index,
            "model_id": "model-a",
            "execution_lane": "gpu",
            "role": "execute",
            "resource": {
                "runtime_probs": list(probs),
                "runtime_ms_quantiles": {"p50": views["p50"], "p90": views["p90"], "p95": views["p95"]},
                "runtime_mean_ms": views["runtime_mean_ms"],
                "cvar95_ms": views["cvar95_ms"],
                "bin_schema_id": "test_3",
                "load_occurrence_probability": 0.0,
            },
        }

    def write_gz(self, path, rows):
        import gzip
        import json

        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def write_packs(self, node_ids=("t0:gpu",), probs=(0.80, 0.15, 0.05), base_only=()):
        base_rows = [
            {"node_id": nid, "future_h5": [{"scenario_probability": 1.0, "steps": [self.base_step(nid, i, 400.0) for i in range(3)]}]}
            for nid in node_ids
        ]
        self.write_gz(self.base_root / "b05_node_h1.jsonl.gz", [{"node_id": nid, "future_h1": [{"scenario_probability": 1.0, "steps": [self.base_step(nid, 0, 400.0)]}]} for nid in node_ids])
        self.write_gz(self.base_root / "b05_future_h3.jsonl.gz", [{"node_id": nid, "future_h3": [{"scenario_probability": 1.0, "steps": [self.base_step(nid, i, 400.0) for i in range(3)]}]} for nid in node_ids])
        self.write_gz(self.base_root / "b05_future_h5.jsonl.gz", base_rows)
        arm_ids = [nid for nid in node_ids if nid not in base_only]
        self.write_gz(self.arm_root / "b05_future_h5.jsonl.gz", [
            {"node_id": nid, "future_h5": [{"scenario_probability": 1.0, "steps": [self.v2_step(i, probs) for i in range(3)]}]}
            for nid in arm_ids
        ])
        manifest = {
            "schema_version": "resource-v2-artifact-manifest-v2",
            "artifact_id": "resource_v2_test",
            # real digests: the loader recomputes both and refuses on mismatch (P0-3)
            "artifact_sha256": _sha256_file(self.arm_root / "b05_future_h5.jsonl.gz"),
            "base_pack_sha256": _sha256_file(self.base_root / "b05_future_h5.jsonl.gz"),
            "producer_checkpoint_sha256": "0" * 64,
            "resource_head_id": "test_seed11",
            "bin_schema_id": "test_3",
            "bin_edges_ms": list(TEST_EDGES),
            "bin_representatives_ms": list(TEST_REPS),
            "cvar_alpha": 0.95,
            "horizon": 5,
        }
        (self.arm_root / "resource_v2_manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")

    def load(self):
        from tracing.analysis.workload_v02_simulator import load_resource_v2_overlay

        return load_resource_v2_overlay(self.base_root, self.arm_root)

    # -- tests ------------------------------------------------------------ #
    def test_happy_path_splices_only_the_h5_block(self) -> None:
        artifacts, preflight = self.load()
        self.assertEqual(preflight["node_count"], 1)
        self.assertEqual(preflight["step_count"], 3)
        self.assertEqual(preflight["unupgraded_steps"], 0)
        self.assertEqual(preflight["nonresource_mismatch_count"], 0)
        self.assertLessEqual(preflight["prob_sum_max_abs_error"], 1e-9)
        self.assertLessEqual(preflight["canonical_view_max_abs_error"], 1e-6)
        # the base H1/H3 files are preserved and the H5 is replaced
        self.assertIn("future_h1", artifacts["t0:gpu"])
        self.assertIn("runtime_probs", artifacts["t0:gpu"]["future_h5"][0]["steps"][0]["resource"])

    def test_node_set_mismatch_raises(self) -> None:
        self.write_packs(node_ids=("t0:gpu", "t1:gpu"), base_only=("t1:gpu",))
        with self.assertRaises(ValueError):
            self.load()

    def test_view_tampering_raises(self) -> None:
        import gzip
        import json

        path = self.arm_root / "b05_future_h5.jsonl.gz"
        rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
        rows[0]["future_h5"][0]["steps"][0]["resource"]["runtime_ms_quantiles"]["p95"] = 12345.0
        self.write_gz(path, rows)
        with self.assertRaises(ValueError):
            self.load()

    def test_unupgraded_step_raises(self) -> None:
        import gzip
        import json

        path = self.arm_root / "b05_future_h5.jsonl.gz"
        rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
        del rows[0]["future_h5"][0]["steps"][1]["resource"]["runtime_probs"]
        self.write_gz(path, rows)
        with self.assertRaises(ValueError):
            self.load()

    def test_nonresource_drift_raises(self) -> None:
        import gzip
        import json

        path = self.arm_root / "b05_future_h5.jsonl.gz"
        rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
        rows[0]["future_h5"][0]["steps"][0]["role"] = "plan"
        self.write_gz(path, rows)
        with self.assertRaises(ValueError):
            self.load()

    def test_missing_load_field_raises(self) -> None:
        import gzip
        import json

        path = self.arm_root / "b05_future_h5.jsonl.gz"
        rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
        del rows[0]["future_h5"][0]["steps"][0]["resource"]["load_occurrence_probability"]
        self.write_gz(path, rows)
        with self.assertRaises(ValueError):
            self.load()

    def test_manifest_sha_mismatch_raises(self) -> None:
        """P0-3: the loader must recompute the digests, not trust the manifest."""

        import json

        path = self.arm_root / "resource_v2_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["artifact_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.load()

    def test_base_pack_sha_mismatch_raises(self) -> None:
        import json

        path = self.arm_root / "resource_v2_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["base_pack_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.load()

    def test_out_of_range_probability_raises(self) -> None:
        """P0-2: bool and out-of-range values must not slip through."""

        import gzip
        import json

        for bad in (1.5, -0.1, True):
            self.write_packs()
            path = self.arm_root / "b05_future_h5.jsonl.gz"
            rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
            rows[0]["future_h5"][0]["steps"][0]["resource"]["runtime_probs"][0] = bad
            self.write_gz(path, rows)
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.load()


if __name__ == "__main__":
    unittest.main()
