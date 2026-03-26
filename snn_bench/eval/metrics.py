from __future__ import annotations

import numpy as np


def binary_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)
    return float((preds == labels).mean())
