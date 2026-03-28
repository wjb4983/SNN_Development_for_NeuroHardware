from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .schema import EventType


def _safe_div(numer: float, denom: float) -> float:
    if abs(denom) < 1e-9:
        return 0.0
    return float(numer / denom)


def build_feature_frame(events: pd.DataFrame, top_k: int = 5, lookback_events: int = 50) -> pd.DataFrame:
    """Build causal per-event features used by policy models."""
    rows: list[dict[str, float]] = []
    mid_hist: deque[float] = deque(maxlen=max(lookback_events, 2))
    trade_ts_hist: deque[int] = deque(maxlen=lookback_events)
    last_fill_ts: int | None = None

    for row in events.itertuples(index=False):
        bid_prices = np.asarray(row.bid_prices, dtype=np.float32)
        ask_prices = np.asarray(row.ask_prices, dtype=np.float32)
        bid_sizes = np.asarray(row.bid_sizes, dtype=np.float32)
        ask_sizes = np.asarray(row.ask_sizes, dtype=np.float32)

        best_bid = float(bid_prices[0]) if bid_prices.size > 0 else 0.0
        best_ask = float(ask_prices[0]) if ask_prices.size > 0 else 0.0
        spread = max(best_ask - best_bid, 0.0)
        mid = 0.5 * (best_ask + best_bid) if best_ask > 0 and best_bid > 0 else 0.0
        mid_hist.append(mid)

        if row.event_type == EventType.MARKET_TRADE:
            trade_ts_hist.append(int(row.ts_ns))
        if row.event_type == EventType.OWN_FILL:
            last_fill_ts = int(row.ts_ns)

        vol = 0.0
        if len(mid_hist) > 2:
            arr = np.asarray(mid_hist, dtype=np.float32)
            rets = np.diff(arr) / np.clip(arr[:-1], 1e-6, None)
            vol = float(np.std(rets))

        horizon_ns = 1_000_000_000
        recent_trades = sum(1 for ts in trade_ts_hist if (row.ts_ns - ts) <= horizon_ns)
        trade_intensity = float(recent_trades) / (horizon_ns / 1e9)

        level_feats: dict[str, float] = {}
        depth_bid = float(np.sum(bid_sizes))
        depth_ask = float(np.sum(ask_sizes))
        imbalance = _safe_div(depth_bid - depth_ask, depth_bid + depth_ask)
        for i in range(top_k):
            level_feats[f"bid_px_{i+1}"] = float(bid_prices[i]) if i < len(bid_prices) else 0.0
            level_feats[f"ask_px_{i+1}"] = float(ask_prices[i]) if i < len(ask_prices) else 0.0
            level_feats[f"bid_sz_{i+1}"] = float(bid_sizes[i]) if i < len(bid_sizes) else 0.0
            level_feats[f"ask_sz_{i+1}"] = float(ask_sizes[i]) if i < len(ask_sizes) else 0.0

        time_since_fill_s = 0.0 if last_fill_ts is None else max((row.ts_ns - last_fill_ts) / 1e9, 0.0)
        queue_position = float(row.queue_position) if row.queue_position is not None else 0.0

        ev_onehot = {f"ev_{et.value}": float(row.event_type == et) for et in EventType}

        rows.append(
            {
                "ts_ns": int(row.ts_ns),
                "spread": float(spread),
                "mid": float(mid),
                "imbalance": float(imbalance),
                "short_vol": float(vol),
                "trade_intensity": float(trade_intensity),
                "queue_position": queue_position,
                "time_since_fill_s": float(time_since_fill_s),
                **level_feats,
                **ev_onehot,
            }
        )

    return pd.DataFrame(rows)
