"""Tests for the train-set prior isolation re-run (exp F)."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_prior_lib import (  # noqa: E402
    build_prior_pack,
    compute_train_prior,
    prior_sha256,
)


def j_row(steps, length=None):
    future = steps if length is None else steps[:length]
    return {"future": [dict(step) for step in future]}


def step(model, runtime=100.0, load=0.0):
    return {"model_class": model, "targets": {"runtime_ms": runtime, "load_ms": load}}


def prior_fixture():
    rows = [
        j_row([step("Qwen3-4B", 100.0, 0.0), step("cpu-metadata-adapter-v1", 50.0, 0.0)]),
        j_row([step("Qwen3-4B", 300.0, 10.0), step("Qwen3-VL-8B-Instruct", 200.0, 0.0),
               step("Qwen3-4B", 400.0, 20.0)]),
        j_row([step("Qwen3-VL-8B-Instruct", 200.0, 0.0)]),
    ]
    return rows


def pack_fixture():
    step_template = {
        "model_probabilities": {"Qwen3-4B": 1.0},
        "model_probability": 1.0,
        "execution_lane": "gpu",
        "raw_action": "planner.generate",
        "role": "plan",
        "role_probability": 0.9,
        "action_family": "other",
        "family_probability": 0.9,
        "is_retry": False,
        "is_retry_probability": 0.1,
        "merged_nested_call": False,
        "merged_nested_call_probability": 0.1,
        "resource": {
            "bin_schema_id": "mass_balanced_16",
            "resource_head_id": "F0_seed11_f16",
            "cvar95_ms": 8998.0,
            "runtime_mean_ms": 1447.0,
            "runtime_ms_quantiles": {"p50": 307.0, "p90": 5589.0, "p95": 6460.0},
            "load_occurrence_probability": 0.02,
            "load_duration_ms_quantiles": {"p50": 949.0, "p90": 1859.0, "p95": 2956.0},
        },
        "step_offset": 1,
    }
    return {
        "n:1": {
            "length_probabilities": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "predicted_future_length": 1,
            "future_h5": [{"steps": [copy.deepcopy(step_template), copy.deepcopy(step_template)]}],
        }
    }


class TrainPriorStatsTests(unittest.TestCase):
    def test_prior_channels(self) -> None:
        prior = compute_train_prior(prior_fixture())
        # step-1 model distribution: 4B x2, 8B x1 -> 2/3, 1/3
        self.assertAlmostEqual(prior["model_dist"][0]["Qwen3-4B"], 2 / 3)
        self.assertAlmostEqual(prior["model_dist"][0]["Qwen3-VL-8B-Instruct"], 1 / 3)
        # step-2: cpu x1, 8B x1
        self.assertAlmostEqual(prior["model_dist"][1]["cpu-metadata-adapter-v1"], 0.5)
        # length: lengths [2,3,1] -> index 1: 1/3, index 2: 1/3, index 3: 1/3
        self.assertAlmostEqual(prior["length_dist"][1], 1 / 3)
        self.assertAlmostEqual(prior["length_dist"][3], 1 / 3)
        self.assertAlmostEqual(prior["length_dist"][5], 0.0)
        # runtime step 1: [100, 300, 200] -> mean 200
        self.assertAlmostEqual(prior["runtime"][0]["mean"], 200.0)
        self.assertAlmostEqual(prior["runtime"][0]["p50"], 200.0)
        # load step 1: one positive of 3 -> 1/3
        self.assertAlmostEqual(prior["load"][0]["occurrence"], 1 / 3)
        self.assertAlmostEqual(prior["load"][0]["p50"], 10.0)
        # lane step 1: gpu x2, cpu x1 -> gpu
        self.assertEqual(prior["lane_majority"][0], "gpu")
        # lane step 2: cpu x1, 8B x1 -> cpu count 1/2 -> cpu wins tie? gpu needs *2 >= n -> 1*2 >= 2 True -> gpu
        self.assertEqual(prior["lane_majority"][1], "gpu")

    def test_prior_deterministic(self) -> None:
        first = prior_sha256(compute_train_prior(prior_fixture()))
        second = prior_sha256(compute_train_prior(prior_fixture()))
        self.assertEqual(first, second)


class PriorPackTransformTests(unittest.TestCase):
    def test_partial_replaces_only_model_and_length(self) -> None:
        prior = compute_train_prior(prior_fixture())
        pack = pack_fixture()
        before = copy.deepcopy(pack)
        out = build_prior_pack(pack, prior, "partial")
        row = out["n:1"]
        self.assertEqual(pack, before, "input pack mutated")
        steps = row["future_h5"][0]["steps"]
        self.assertEqual(steps[0]["model_probabilities"], prior["model_dist"][0])
        self.assertEqual(steps[1]["model_probabilities"], prior["model_dist"][1])
        self.assertAlmostEqual(steps[0]["model_probability"], max(prior["model_dist"][0].values()))
        self.assertEqual(row["length_probabilities"], prior["length_dist"])
        # runtime / load / lane untouched in partial mode
        self.assertEqual(steps[0]["execution_lane"], "gpu")
        self.assertEqual(steps[0]["resource"], before["n:1"]["future_h5"][0]["steps"][0]["resource"])
        # untouched identity fields byte-preserved
        self.assertEqual(steps[0]["raw_action"], "planner.generate")
        self.assertEqual(steps[0]["is_retry"], False)

    def test_full_replaces_runtime_load_and_lane(self) -> None:
        prior = compute_train_prior(prior_fixture())
        out = build_prior_pack(pack_fixture(), prior, "full")
        steps = out["n:1"]["future_h5"][0]["steps"]
        resource = steps[0]["resource"]
        self.assertAlmostEqual(resource["runtime_mean_ms"], prior["runtime"][0]["mean"])
        self.assertAlmostEqual(resource["runtime_ms_quantiles"]["p95"], prior["runtime"][0]["p95"])
        self.assertAlmostEqual(resource["load_occurrence_probability"], prior["load"][0]["occurrence"])
        self.assertAlmostEqual(resource["load_duration_ms_quantiles"]["p50"], prior["load"][0]["p50"])
        # provenance labels stay
        self.assertEqual(resource["bin_schema_id"], "mass_balanced_16")
        # identity fields stay
        self.assertEqual(steps[1]["role"], "plan")
        self.assertEqual(steps[1]["action_family"], "other")

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_prior_pack(pack_fixture(), compute_train_prior(prior_fixture()), "bogus")

    def test_row_without_future_steps_passes_through(self) -> None:
        prior = compute_train_prior(prior_fixture())
        pack = {"tail": {"length_probabilities": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "future_h5": []}}
        out = build_prior_pack(pack, prior, "full")
        self.assertEqual(out["tail"], pack["tail"])


if __name__ == "__main__":
    unittest.main()
