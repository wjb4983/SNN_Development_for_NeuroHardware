from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import f1_score

from snn_bench.models.zoo import ModelSpec, ModelZoo, set_global_seed


class RegimeSwitchingModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seed = 17
        cls.n_samples = 96
        cls.n_features = 6
        cls.seq_len = 10
        cls.n_classes = 4

    def setUp(self) -> None:
        set_global_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        t = np.linspace(0.0, 6.0 * np.pi, self.n_samples, dtype=np.float32)
        base = np.stack(
            [
                np.sin(t),
                np.cos(t),
                np.sin(0.5 * t),
                np.cos(0.75 * t),
                np.sin(1.25 * t),
                np.cos(1.5 * t),
            ],
            axis=1,
        ).astype(np.float32)
        noise = rng.normal(0.0, 0.05, size=base.shape).astype(np.float32)
        self.x_tabular = base + noise

        regime_scores = np.stack(
            [
                self.x_tabular[:, 0] - self.x_tabular[:, 1],
                self.x_tabular[:, 2] + 0.2 * self.x_tabular[:, 3],
                self.x_tabular[:, 4] - 0.1 * self.x_tabular[:, 5],
                -self.x_tabular[:, 0] - 0.2 * self.x_tabular[:, 2],
            ],
            axis=1,
        )
        self.y_regime = np.argmax(regime_scores, axis=1).astype(np.int64)

        self.x_sequence = np.stack(
            [
                np.roll(self.x_tabular, shift=s, axis=0)
                for s in range(self.seq_len)
            ],
            axis=1,
        ).astype(np.float32)

    def _fit_predict_assertions(self, model_name: str, x_input: np.ndarray, extra_params: dict[str, object] | None = None) -> None:
        params: dict[str, object] = {
            "seed": self.seed,
            "epochs": 1,
            "batch_size": 16,
            "lr": 1e-3,
            "output_dim": self.n_classes,
            "training_strategy": "multiclass",
            "hidden_sizes": [24],
            "depth": 1,
            "dropout": 0.0,
            "surrogate_type": "tanh",
            "reset_mode": "zero",
        }
        if extra_params:
            params.update(extra_params)

        model = ModelZoo.create(
            ModelSpec(name=model_name, family="regime_classification", params=params),
            input_dim=self.n_features,
        )
        model.fit(x_input, self.y_regime, epochs=1)
        proba = model.predict_proba(x_input)

        self.assertEqual(proba.shape, (self.n_samples, self.n_classes))
        self.assertFalse(np.isnan(proba).any())
        self.assertTrue(np.isfinite(proba).all())
        np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-5)

        y_pred = np.argmax(proba, axis=1)
        macro_f1 = f1_score(self.y_regime, y_pred, average="macro", zero_division=0)
        self.assertTrue(np.isfinite(macro_f1))

    def test_lightweight_regime_classification_models(self) -> None:
        model_cases = [
            ("snntorch_lif", self.x_tabular, {"hidden_sizes": [20]}),
            ("snntorch_alif", self.x_tabular, {"hidden_sizes": [20], "surrogate_type": "sigmoid"}),
            ("norse_recurrent_lsnn", self.x_sequence, {"hidden_sizes": [20], "surrogate_type": "fast_sigmoid"}),
            ("spikingjelly_temporal_conv", self.x_sequence, {"hidden_sizes": [20]}),
        ]
        for model_name, x_input, extra in model_cases:
            with self.subTest(model=model_name):
                self._fit_predict_assertions(model_name, x_input, extra_params=extra)

    def test_markov_and_snn_smoke_under_same_task_config(self) -> None:
        task_family = "regime_classification"
        markov = ModelZoo.create(
            ModelSpec(
                name="markov_discrete",
                family=task_family,
                params={
                    "n_states": self.n_classes,
                    "n_return_bins": 4,
                    "n_vol_bins": 3,
                    "smoothing": 1e-2,
                },
            ),
            input_dim=self.n_features,
        )
        snn = ModelZoo.create(
            ModelSpec(
                name="snntorch_lif",
                family=task_family,
                params={
                    "seed": self.seed,
                    "epochs": 1,
                    "batch_size": 16,
                    "output_dim": self.n_classes,
                    "training_strategy": "multiclass",
                    "hidden_sizes": [16],
                    "depth": 1,
                    "dropout": 0.0,
                },
            ),
            input_dim=self.n_features,
        )

        markov.fit(self.x_tabular, self.y_regime)
        snn.fit(self.x_tabular, self.y_regime, epochs=1)

        markov_proba = markov.predict_proba(self.x_tabular)
        snn_proba = snn.predict_proba(self.x_tabular)

        self.assertEqual(markov_proba.shape, (self.n_samples, self.n_classes))
        self.assertEqual(snn_proba.shape, (self.n_samples, self.n_classes))


if __name__ == "__main__":
    unittest.main()
