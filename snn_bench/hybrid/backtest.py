from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from snn_bench.hybrid.risk_gate import RiskState


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    pnl: pd.Series
    position: pd.Series
    turnover: pd.Series
    metrics: dict[str, float]
    attribution: dict[str, float]


def _max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = equity_curve / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd))


def _annualize_from_minute() -> float:
    return np.sqrt(252.0 * 390.0)


def run_backtest(
    market: pd.DataFrame,
    fused_score: np.ndarray,
    fused_conf: np.ndarray,
    risk_state: np.ndarray,
    leverage: np.ndarray,
    slow_component: np.ndarray,
    fast_component: np.ndarray,
    *,
    score_to_pos: float = 75.0,
    max_position: float = 1.0,
    max_turnover_step: float = 0.35,
    cooldown_steps: int = 5,
    tx_cost_bps: float = 1.5,
    slippage_bps: float = 0.75,
    latency_steps: int = 1,
) -> BacktestResult:
    n = len(market)
    target_raw = np.tanh(fused_score * score_to_pos) * max_position
    target_raw = target_raw * leverage

    position = np.zeros(n, dtype=np.float32)
    turnover = np.zeros(n, dtype=np.float32)
    cooldown = 0

    for t in range(1, n):
        prev = position[t - 1]
        target = float(target_raw[t])

        if risk_state[t] == RiskState.BLOCK.value:
            cooldown = cooldown_steps
            target = 0.0
        elif cooldown > 0:
            cooldown -= 1
            target = 0.0

        delta = np.clip(target - prev, -max_turnover_step, max_turnover_step)
        position[t] = prev + delta
        turnover[t] = abs(delta)

    exec_pos = np.roll(position, latency_steps)
    exec_pos[:latency_steps] = 0.0

    ret = market["ret_1"].values.astype(np.float32)
    gross = exec_pos * ret
    costs = turnover * ((tx_cost_bps + slippage_bps) * 1e-4)
    pnl = gross - costs
    equity = np.cumprod(1.0 + pnl)

    ann = _annualize_from_minute()
    pnl_std = float(np.std(pnl) + 1e-12)
    downside = pnl[pnl < 0]
    down_std = float(np.std(downside) + 1e-12)

    metrics = {
        "sharpe": float((np.mean(pnl) / pnl_std) * ann),
        "sortino": float((np.mean(pnl) / down_std) * ann),
        "max_drawdown": _max_drawdown(equity),
        "turnover": float(np.sum(turnover)),
        "hit_rate": float(np.mean(pnl > 0)),
    }

    attribution = {
        "slow_component_pnl": float(np.mean(slow_component * ret)),
        "fast_component_pnl": float(np.mean(fast_component * ret)),
        "risk_drag_pnl": float(np.mean((target_raw - np.tanh(fused_score * score_to_pos)) * ret)),
    }

    return BacktestResult(
        equity_curve=pd.Series(equity, index=market.index, name="equity"),
        pnl=pd.Series(pnl, index=market.index, name="pnl"),
        position=pd.Series(exec_pos, index=market.index, name="position"),
        turnover=pd.Series(turnover, index=market.index, name="turnover"),
        metrics=metrics,
        attribution=attribution,
    )
