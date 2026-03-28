from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestConfig:
    execution_lag: int = 1
    fee_bps: float = 0.5
    spread_bps: float = 0.5


class EventDrivenBacktester:
    """Minimal event-driven backtester with explicit transaction costs."""

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def run(self, prices: np.ndarray, signals: np.ndarray) -> dict[str, np.ndarray | float]:
        if len(prices) != len(signals):
            raise ValueError("prices and signals must have same length")

        positions = np.sign(signals).astype(float)
        executed = np.roll(positions, self.cfg.execution_lag)
        executed[: self.cfg.execution_lag] = 0.0

        returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices, 1e-12)
        gross = executed * returns

        turns = np.abs(np.diff(executed, prepend=0.0))
        total_cost_bps = self.cfg.fee_bps + self.cfg.spread_bps
        costs = turns * total_cost_bps * 1e-4
        net = gross - costs

        equity = np.cumprod(1.0 + net)
        return {
            "returns": net,
            "equity_curve": equity,
            "net_pnl": float(net.sum()),
            "gross_pnl": float(gross.sum()),
            "total_cost": float(costs.sum()),
            "turnover": float(turns.sum()),
        }
