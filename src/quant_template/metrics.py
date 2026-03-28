from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def ml_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] > 1:
        try:
            out["auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            out["auc_ovr"] = float("nan")
    return out


def trading_kpis(returns: np.ndarray, positions: np.ndarray, cost_per_turnover_bps: float = 1.0) -> dict[str, float]:
    turns = np.abs(np.diff(positions, prepend=0.0))
    gross = positions * returns
    costs = turns * cost_per_turnover_bps * 1e-4
    net = gross - costs
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity / np.maximum(peak, 1e-12)) - 1.0

    sharpe = float(np.sqrt(252.0) * net.mean() / (net.std() + 1e-12))
    return {
        "gross_return": float(gross.sum()),
        "net_return": float(net.sum()),
        "turnover": float(turns.sum()),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "hit_rate": float((net > 0).mean()),
    }
