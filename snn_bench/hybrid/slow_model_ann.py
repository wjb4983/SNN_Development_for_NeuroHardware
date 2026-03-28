from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SlowModelOutput:
    score: np.ndarray
    confidence: np.ndarray


class SlowANNModel:
    """Minute/hour model over slower-moving multi-factor features."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        alpha: float = 1e-4,
        random_state: int = 7,
    ) -> None:
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation="relu",
                        alpha=alpha,
                        learning_rate_init=1e-3,
                        max_iter=350,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.residual_std = 1.0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "SlowANNModel":
        self.pipeline.fit(x.values, y.values)
        preds = self.pipeline.predict(x.values)
        resid = y.values - preds
        self.residual_std = float(np.std(resid) + 1e-8)
        return self

    def predict(self, x: pd.DataFrame) -> SlowModelOutput:
        score = self.pipeline.predict(x.values)
        confidence = np.exp(-np.abs(score) / (3.0 * self.residual_std + 1e-8))
        return SlowModelOutput(score=score.astype(np.float32), confidence=confidence.astype(np.float32))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"pipeline": self.pipeline, "residual_std": self.residual_std}, f)

    @classmethod
    def load(cls, path: Path) -> "SlowANNModel":
        with path.open("rb") as f:
            state = pickle.load(f)
        model = cls()
        model.pipeline = state["pipeline"]
        model.residual_std = float(state["residual_std"])
        return model
