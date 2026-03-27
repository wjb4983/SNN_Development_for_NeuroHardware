from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from snn_bench.models.zoo import ModelSpec, ModelZoo, save_prediction_artifacts, set_global_seed


class ModelZooInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        set_global_seed(7)
        self.x = np.random.randn(120, 6).astype(np.float32)
        self.y = (self.x[:, 0] + self.x[:, 1] > 0).astype(np.int64)

    def _assert_model_roundtrip(self, model_name: str) -> None:
        model = ModelZoo.create(ModelSpec(name=model_name, family="zoo", params={"epochs": 1, "batch_size": 16}), input_dim=self.x.shape[1])
        train_info = model.fit(self.x, self.y, epochs=1)
        proba = model.predict_proba(self.x)
        metrics = model.evaluate(self.x, self.y)

        self.assertIn("accuracy", metrics)
        self.assertEqual(len(proba), len(self.x))
        self.assertTrue(0.0 <= float(np.min(proba)) <= 1.0)
        self.assertTrue(0.0 <= float(np.max(proba)) <= 1.0)

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / f"{model_name}.bin"
            model.save_checkpoint(ckpt)
            self.assertTrue(ckpt.exists())
            pred = save_prediction_artifacts(Path(td), model_name, self.y, proba)
            self.assertTrue(pred.exists())

        self.assertIsInstance(train_info, dict)

    def test_baseline_models(self):
        for model_name in ["logreg", "gbm", "mlp"]:
            with self.subTest(model=model_name):
                self._assert_model_roundtrip(model_name)

    def test_snn_models(self):
        for model_name in ["snntorch_lif", "norse_lsnn", "spikingjelly_lif"]:
            with self.subTest(model=model_name):
                self._assert_model_roundtrip(model_name)


if __name__ == "__main__":
    unittest.main()
