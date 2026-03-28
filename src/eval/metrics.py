from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15) -> float:
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)

    ece = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.any():
            ece += (m.mean()) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    try:
        out["auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        out["auc_ovr"] = float("nan")
    return out


def calibration_points(y_true: np.ndarray, y_prob_up: np.ndarray, bins: int = 15):
    return calibration_curve((y_true == 2).astype(int), y_prob_up, n_bins=bins)
