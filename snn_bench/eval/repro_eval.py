from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, roc_auc_score

EPS = 1e-12


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 1.0
    spread_bps: float = 1.0
    impact_coef: float = 0.2


def _one_hot(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y_true), n_classes), dtype=float)
    out[np.arange(len(y_true)), y_true] = 1.0
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if not np.any(mask):
            continue
        ece += np.abs(correct[mask].mean() - conf[mask].mean()) * (mask.sum() / len(y_true))
    return float(ece)


def compute_ml_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = y_prob.argmax(axis=1)
    n_classes = y_prob.shape[1]

    metrics: dict[str, float] = {
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "logloss": float(log_loss(y_true, y_prob, labels=np.arange(n_classes))),
        "brier": float(np.mean(np.sum((y_prob - _one_hot(y_true, n_classes)) ** 2, axis=1))),
        "calibration": expected_calibration_error(y_true, y_prob),
    }
    try:
        if n_classes == 2:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def positions_from_probabilities(
    y_prob: np.ndarray,
    class_map: tuple[int, int, int] = (-1, 0, 1),
    confidence_threshold: float = 0.55,
) -> np.ndarray:
    probs = np.asarray(y_prob, dtype=float)
    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    pos = np.array([class_map[idx] for idx in pred_idx], dtype=float)
    pos[conf < confidence_threshold] = 0.0
    return pos


def strategy_returns_with_costs(
    positions: np.ndarray,
    future_returns: np.ndarray,
    realized_vol: np.ndarray,
    cost_model: CostModel,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=float)
    ret = np.asarray(future_returns, dtype=float)
    vol = np.nan_to_num(np.asarray(realized_vol, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    turnover = np.abs(np.diff(np.insert(pos, 0, 0.0)))
    linear_cost = turnover * ((cost_model.fee_bps + cost_model.spread_bps) / 10_000.0)
    impact_cost = turnover * cost_model.impact_coef * vol
    total_cost = linear_cost + impact_cost

    gross = pos * ret
    net = gross - total_cost
    return net, turnover


def _max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max
    return float(drawdown.min())


def compute_trading_metrics(strategy_returns: np.ndarray, turnover: np.ndarray, periods_per_year: int) -> dict[str, float]:
    r = np.asarray(strategy_returns, dtype=float)
    eq = np.cumsum(r)
    downside = r[r < 0]

    sharpe = np.sqrt(periods_per_year) * r.mean() / (r.std() + EPS)
    sortino = np.sqrt(periods_per_year) * r.mean() / (downside.std() + EPS) if len(downside) else 0.0
    return {
        "net_pnl": float(r.sum()),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": _max_drawdown(eq),
        "turnover": float(np.mean(turnover)),
    }


def evaluate_direction_task(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    future_returns: np.ndarray,
    realized_vol: np.ndarray,
    confidence_threshold: float,
    periods_per_year: int,
    cost_model: CostModel,
) -> dict[str, dict[str, float]]:
    ml = compute_ml_metrics(y_true=y_true, y_prob=y_prob)
    pos = positions_from_probabilities(y_prob=y_prob, confidence_threshold=confidence_threshold)
    strat_returns, turnover = strategy_returns_with_costs(
        positions=pos,
        future_returns=future_returns,
        realized_vol=realized_vol,
        cost_model=cost_model,
    )
    trading = compute_trading_metrics(strat_returns, turnover=turnover, periods_per_year=periods_per_year)
    return {"ml": ml, "trading": trading}


def no_leakage_walkforward_check(train_end_idx: int, prediction_start_idx: int) -> None:
    if prediction_start_idx <= train_end_idx:
        raise ValueError("prediction_start_idx must be strictly greater than train_end_idx to avoid lookahead")
