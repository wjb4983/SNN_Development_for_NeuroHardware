from __future__ import annotations

import numpy as np


def pnl_simulation(
    mid_prices: np.ndarray,
    pred_cls: np.ndarray,
    latency_steps: int = 2,
    fee_bps: float = 0.2,
    spread_bps: float = 1.0,
) -> dict:
    """Simple cost-aware PnL simulation: class 2=long, 0=short, 1=flat."""
    pos = np.where(pred_cls == 2, 1.0, np.where(pred_cls == 0, -1.0, 0.0))
    pos_exec = np.roll(pos, latency_steps)
    pos_exec[:latency_steps] = 0.0

    ret = np.diff(np.log(mid_prices + 1e-9), prepend=np.log(mid_prices[0] + 1e-9))
    gross = pos_exec * ret
    turn = np.abs(np.diff(pos_exec, prepend=0.0))
    cost = (fee_bps + spread_bps) * 1e-4 * turn
    net = gross - cost

    sharpe = np.sqrt(252 * 6.5 * 3600) * net.mean() / (net.std() + 1e-9)
    return {
        "net_pnl": float(net.sum()),
        "gross_pnl": float(gross.sum()),
        "cost": float(cost.sum()),
        "turnover": float(turn.sum()),
        "sharpe_like": float(sharpe),
    }
