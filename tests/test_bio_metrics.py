from __future__ import annotations

import unittest

import numpy as np

from snn_bench.eval.metrics import bio_plausibility_metrics


class BioPlausibilityMetricsTest(unittest.TestCase):
    def test_bio_plausibility_metrics_shapes(self) -> None:
        probs = np.array([0.1, 0.7, 0.85, 0.2, 0.9, 0.4, 0.8, 0.1], dtype=np.float32)
        metrics = bio_plausibility_metrics(probs, dt_ms=1.0, threshold=0.5, stability_window=4)

        self.assertIn("spike_sparsity", metrics)
        self.assertIn("firing_rate_distribution", metrics)
        self.assertIn("temporal_precision", metrics)
        self.assertIn("stability", metrics)
        self.assertTrue(0.0 <= metrics["spike_sparsity"] <= 1.0)
        self.assertTrue(metrics["firing_rate_distribution"]["mean_hz"] >= 0.0)
        self.assertTrue(0.0 <= metrics["temporal_precision"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
