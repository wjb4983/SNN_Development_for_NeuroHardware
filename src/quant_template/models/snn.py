from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier


class SNNProxyClassifier:
    """SNN interface placeholder using online surrogate training semantics."""

    name = "snn_proxy_sgd"

    def __init__(self, random_state: int = 42):
        self.model = SGDClassifier(loss="log_loss", random_state=random_state)
        self._classes: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._classes = np.unique(y_train)
        self.model.partial_fit(X_train, y_train, classes=self._classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob = self.model.predict_proba(X)
        if prob.ndim == 1:
            prob = np.column_stack([1.0 - prob, prob])
        return prob
