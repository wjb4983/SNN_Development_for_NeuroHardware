from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from snn_bench.models.zoo import save_prediction_artifacts


class PredictionArtifactMetadataTest(unittest.TestCase):
    def test_prediction_artifact_includes_target_summary(self) -> None:
        y_true = np.array([0, 1, 1], dtype=np.int64)
        y_prob = np.array([0.2, 0.7, 0.8], dtype=np.float32)
        target_summary = {
            "task_name": "direction_5m_distribution",
            "horizon": "5m",
            "label_type": "binary",
            "classes": ["down_or_flat", "up"],
            "label_semantics": "1 if 5-minute forward return > 0, else 0",
        }

        with tempfile.TemporaryDirectory() as td:
            artifact_path = save_prediction_artifacts(Path(td), "mlp", y_true, y_prob, target_summary=target_summary)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertIn("target_summary", payload)
        self.assertEqual(payload["target_summary"]["horizon"], "5m")
        self.assertEqual(payload["target_summary"]["label_semantics"], target_summary["label_semantics"])


if __name__ == "__main__":
    unittest.main()
