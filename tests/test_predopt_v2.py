import unittest

from tracing.analysis import workload_v02_simulator as simulator
from tracing.analysis.workload_v02_simulator import POLICIES


class PredOptV2PolicyContractTests(unittest.TestCase):
    def test_v2_horizons_are_registered(self):
        self.assertIn("predopt_v2_h1", POLICIES)
        self.assertIn("predopt_v2_h3", POLICIES)
        self.assertIn("predopt_v2_h5", POLICIES)

    def test_legacy_predopt_policies_remain_registered(self):
        self.assertIn("predopt_h1", POLICIES)
        self.assertIn("predopt_h3", POLICIES)
        self.assertIn("predopt_h5", POLICIES)

    def test_v2_weights_are_explicit_and_positive(self):
        self.assertGreater(simulator.PREDOPT_V2_PRIORITY_WEIGHT, 0.0)
        self.assertGreater(simulator.PREDOPT_V2_FUTURE_WEIGHT, 0.0)
        self.assertGreaterEqual(simulator.PREDOPT_V2_MEMORY_WEIGHT, 0.0)


if __name__ == "__main__":
    unittest.main()
