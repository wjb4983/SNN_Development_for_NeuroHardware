from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef


def directional_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
    }


def pnl_proxy(y_true: np.ndarray, y_prob: np.ndarray, transaction_cost_bps: float) -> dict[str, float]:
    signal = np.where(y_prob >= 0.5, 1.0, -1.0)
    realized = np.where(y_true > 0.5, 1.0, -1.0)
    gross = signal * realized
    turnover = np.abs(np.diff(signal, prepend=signal[0]))
    costs = turnover * (transaction_cost_bps / 1e4)
    net = gross - costs
    return {
        "gross_mean": float(np.mean(gross)),
        "net_mean": float(np.mean(net)),
        "net_sharpe_proxy": float(np.mean(net) / (np.std(net) + 1e-8)),
    }


def latency_adjusted_throughput(latency_ms: np.ndarray, target_budget_ms: float = 5.0) -> dict[str, float]:
    return {
        "latency_mean_ms": float(np.mean(latency_ms)),
        "latency_p95_ms": float(np.percentile(latency_ms, 95)),
        "budget_hit_rate": float(np.mean(latency_ms <= target_budget_ms)),
    }
