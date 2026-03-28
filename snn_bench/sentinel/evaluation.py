from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_score, recall_score

from snn_bench.sentinel.gate import GateState


@dataclass(slots=True)
class EvaluationResult:
    precision: float
    recall: float
    avg_detection_delay: float
    drawdown_reduction: float


def _window_detection_delay(pred_pos: np.ndarray, stress: np.ndarray) -> float:
    starts = np.flatnonzero((stress == 1) & (np.roll(stress, 1) == 0))
    if stress.size > 0 and stress[0] == 1:
        starts = np.insert(starts, 0, 0)
    delays = []
    for s in starts:
        end = s
        while end < len(stress) and stress[end] == 1:
            end += 1
        hits = np.flatnonzero(pred_pos[s:end] == 1)
        if hits.size == 0:
            delays.append(float(end - s))
        else:
            delays.append(float(hits[0]))
    return float(np.mean(delays)) if delays else 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-6)
    return float(np.min(dd))


def evaluate_sentinel(stress_labels: np.ndarray, states: list[GateState], pnl: np.ndarray) -> EvaluationResult:
    pred = np.array([1 if s in {GateState.WARNING, GateState.BLOCK} else 0 for s in states], dtype=np.int64)
    precision = float(precision_score(stress_labels, pred, zero_division=0))
    recall = float(recall_score(stress_labels, pred, zero_division=0))
    delay = _window_detection_delay(pred, stress_labels)

    baseline_equity = np.cumsum(pnl)
    gated_pnl = pnl.copy()
    gated_pnl[np.array([s == GateState.BLOCK for s in states])] = 0.0
    gated_equity = np.cumsum(gated_pnl)
    base_dd = abs(_max_drawdown(baseline_equity))
    gated_dd = abs(_max_drawdown(gated_equity))
    reduction = float((base_dd - gated_dd) / max(base_dd, 1e-6))
    return EvaluationResult(precision=precision, recall=recall, avg_detection_delay=delay, drawdown_reduction=reduction)
