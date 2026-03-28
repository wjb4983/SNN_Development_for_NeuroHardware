from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class QuantModel(Protocol):
    """Shared train/predict contract for all models in this template."""

    name: str

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


class ANNModel(QuantModel, Protocol):
    """ANN family marker interface."""


class SNNModel(QuantModel, Protocol):
    """SNN family marker interface."""


@dataclass(frozen=True)
class ModelSpec:
    family: str
    name: str
