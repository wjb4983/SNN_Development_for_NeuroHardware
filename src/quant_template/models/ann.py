from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class ANNBaselineClassifier:
    """Simple ANN baseline interface backed by logistic regression."""

    name = "ann_logistic_baseline"

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(max_iter=300, random_state=random_state)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
