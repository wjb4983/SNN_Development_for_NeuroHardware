from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from snn_bench.models.zoo import ModelSpec, ModelZoo, TorchSNNAdapter, save_prediction_artifacts, set_global_seed


class ModelZooInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        set_global_seed(7)
        self.x = np.random.randn(120, 6).astype(np.float32)
        self.x_seq = np.random.randn(120, 8, 6).astype(np.float32)
        self.y = (self.x[:, 0] + self.x[:, 1] > 0).astype(np.int64)

    def _assert_model_roundtrip(self, model_name: str, params: dict[str, object] | None = None, seq: bool = False) -> None:
        model_params = {"epochs": 1, "batch_size": 16}
        if params:
            model_params.update(params)

        x_input = self.x_seq if seq else self.x
        model = ModelZoo.create(ModelSpec(name=model_name, family="zoo", params=model_params), input_dim=self.x.shape[1])
        train_info = model.fit(x_input, self.y, epochs=1)
        proba = model.predict_proba(x_input)
        metrics = model.evaluate(x_input, self.y)

        self.assertIn("accuracy", metrics)
        self.assertEqual(proba.shape, (len(self.x),))
        self.assertTrue(0.0 <= float(np.min(proba)) <= 1.0)
        self.assertTrue(0.0 <= float(np.max(proba)) <= 1.0)

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / f"{model_name}.bin"
            model.save_checkpoint(ckpt)
            self.assertTrue(ckpt.exists())

            reloaded = ModelZoo.create(ModelSpec(name=model_name, family="zoo", params=model_params), input_dim=self.x.shape[1])
            reloaded.load_checkpoint(ckpt)
            reloaded_proba = reloaded.predict_proba(x_input)
            self.assertEqual(reloaded_proba.shape, proba.shape)
            if isinstance(model, TorchSNNAdapter):
                np.testing.assert_allclose(proba, reloaded_proba, atol=1e-5, rtol=1e-4)

            pred = save_prediction_artifacts(Path(td), model_name, self.y, proba)
            self.assertTrue(pred.exists())

        self.assertIsInstance(train_info, dict)

    def test_baseline_models(self):
        for model_name in ["naive_persistence", "logreg", "gbm", "mlp"]:
            with self.subTest(model=model_name):
                self._assert_model_roundtrip(model_name)

    def test_snn_models_tabular(self):
        sweep = [
            ("snntorch_lif", {"hidden_sizes": [48, 24], "depth": 2, "dropout": 0.1, "surrogate_type": "tanh", "reset_mode": "zero"}),
            ("snntorch_alif", {"hidden_sizes": [32], "depth": 1, "dropout": 0.0, "surrogate_type": "sigmoid", "reset_mode": "subtract"}),
            ("norse_lsnn", {"hidden_sizes": [40], "depth": 1, "dropout": 0.1, "surrogate_type": "fast_sigmoid", "reset_mode": "zero"}),
            ("spikingjelly_lif", {"hidden_sizes": [32, 16], "depth": 2, "dropout": 0.0, "surrogate_type": "tanh", "reset_mode": "subtract"}),
        ]
        for model_name, params in sweep:
            with self.subTest(model=model_name):
                self._assert_model_roundtrip(model_name, params=params, seq=False)

    def test_temporal_models_sequence_inputs(self):
        sweep = [
            ("spikingjelly_temporal_conv", {"hidden_sizes": [24], "depth": 1, "dropout": 0.1, "surrogate_type": "tanh", "reset_mode": "zero"}),
            ("tcn_spike", {"hidden_sizes": [20], "depth": 1, "dropout": 0.05, "surrogate_type": "sigmoid", "reset_mode": "subtract"}),
            ("norse_recurrent_lsnn", {"hidden_sizes": [28], "depth": 1, "dropout": 0.0, "surrogate_type": "fast_sigmoid", "reset_mode": "zero"}),
        ]
        for model_name, params in sweep:
            with self.subTest(model=model_name):
                self._assert_model_roundtrip(model_name, params=params, seq=True)


if __name__ == "__main__":
    unittest.main()
