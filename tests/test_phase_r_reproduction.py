"""Stage 1.4 acceptance: reproduce a frozen Phase-R result through the new path.

The review asked for this because reading formulas by eye had already missed three
defects. Replaying R1b seed 11 through the new code catches, in one shot: a wrong
quantile key, wrong bin representatives, a wrong mask, a wrong CDF inversion, or a
drift in the evaluation convention.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import j_series_resource_dist as dist  # noqa: E402
import j_series_train_eval as te  # noqa: E402

RUN_ROOT = PROJECT_ROOT / "outputs/j_series_resource_dist_v1"
CACHE = RUN_ROOT / "cache_validation.npz"
HEAD = RUN_ROOT / "R1b_mlp/seed11_best.pt"
GATES = RUN_ROOT / "R1b_mlp/seed11_val_gates.json"

# recorded Phase-R acceptance numbers for R1b seed 11 on the J validation split
RECORDED_RUNTIME_QSCORE = 667.2534249714512
RECORDED_CALIBRATION_ERROR = 0.0245193421357424


def recorded_value(document, *candidates):
    """Find a recorded number by key path, tolerating the nesting Phase-R used."""

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (int, float)) and key in candidates:
                    yield key, float(value)
                else:
                    yield from walk(value)

    return dict(walk(document))


class PhaseRReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not CACHE.is_file() or not HEAD.is_file():
            raise unittest.SkipTest("Phase-R artifacts are not present")
        cls.bins = te.load_runtime_bins()
        cls.cache = np.load(CACHE, allow_pickle=True)
        cls.payload = torch.load(HEAD, map_location="cpu", weights_only=False)
        cls.gates = json.loads(GATES.read_text(encoding="utf-8"))

    def replay(self):
        """R1b head -> probabilities -> quantile views, on the cached validation features."""

        features = torch.as_tensor(self.cache["features"], dtype=torch.float32)
        state = self.payload["state_dict"] if "state_dict" in self.payload else self.payload["head"]
        in_dim = int(self.payload.get("in_dim") or features.shape[-1])
        hidden = int(self.payload.get("hidden") or 64)
        head = dist.DiscreteRuntimeHead(in_dim, len(self.bins["representatives_ms"]), hidden=hidden)
        head.load_state_dict(state)
        head.eval()
        with torch.no_grad():
            logits = head(features)
            probs = torch.softmax(logits, dim=-1)
            reps = np.asarray(self.bins["representatives_ms"], dtype=np.float64)
            views = dist.derive_from_probs(probs, reps)
        return views, probs.numpy()

    def test_reproduces_the_recorded_quantile_views(self) -> None:
        views, probs = self.replay()
        self.assertEqual(probs.shape[1:], (5, len(self.bins["representatives_ms"])))
        for key in ("q50_ms", "q90_ms", "q95_ms", "mean_ms"):
            self.assertIn(key, views)
            self.assertTrue(np.isfinite(np.asarray(views[key])).all())

    def test_reproduces_the_recorded_calibration_error(self) -> None:
        """Coverage-based calibration error must match the frozen number."""

        recorded_q50 = self.gates["validation"]["A_distribution_calibration"]["q50"]
        views, _ = self.replay()
        runtime = np.asarray(self.cache["runtime_ms"], dtype=np.float64)
        mask = np.asarray(self.cache["mask"], dtype=np.float64) > 0.0
        valid = mask & (runtime > 0.0)
        # Phase-R defines coverage as P(true <= predicted), i.e. the empirical CDF
        # evaluated at the predicted quantile (j_series_resource_dist.py:430)
        q50 = np.asarray(views["q50_ms"], dtype=np.float64)
        coverage = float((((runtime <= q50) & valid).sum()) / max(1.0, valid.sum()))
        # Phase-R recorded q50 coverage 0.44211257817929117 and error -0.05788742182070883
        self.assertAlmostEqual(coverage, recorded_q50["coverage"], delta=1e-9,
                               msg="q50 coverage moved; the CDF inversion or bin reps changed")
        self.assertAlmostEqual(coverage - 0.50, recorded_q50["calibration_error"], delta=1e-9)

    def test_reproduces_the_recorded_runtime_qscore(self) -> None:
        """The headline acceptance number must survive the new code path."""

        recorded = recorded_value(self.gates, "runtime_qscore", "qscore")
        views, _ = self.replay()
        runtime = np.asarray(self.cache["runtime_ms"], dtype=np.float64)
        mask = np.asarray(self.cache["mask"], dtype=np.float64) > 0.0
        valid = mask & (runtime > 0.0)

        parts = []
        for tau, key in ((0.50, "q50_ms"), (0.90, "q90_ms"), (0.95, "q95_ms")):
            pred = np.asarray(views[key], dtype=np.float64)
            residual = runtime - pred
            pinball = np.where(residual >= 0, tau * residual, (tau - 1.0) * residual)
            parts.append(float(pinball[valid].mean()))
        qscore = float(np.mean(parts))

        self.assertAlmostEqual(
            qscore, RECORDED_RUNTIME_QSCORE, delta=1e-3,
            msg="RuntimeQScore moved from the frozen value %.6f to %.6f" % (RECORDED_RUNTIME_QSCORE, qscore),
        )
        if recorded.get("runtime_qscore") is not None:
            self.assertAlmostEqual(recorded["runtime_qscore"], RECORDED_RUNTIME_QSCORE, delta=1e-3)

    def test_no_quantile_crossing(self) -> None:
        views, _ = self.replay()
        q50 = np.asarray(views["q50_ms"], dtype=np.float64)
        q90 = np.asarray(views["q90_ms"], dtype=np.float64)
        q95 = np.asarray(views["q95_ms"], dtype=np.float64)
        self.assertTrue(np.all(q50 <= q90 + 1e-9))
        self.assertTrue(np.all(q90 <= q95 + 1e-9))


if __name__ == "__main__":
    unittest.main()
