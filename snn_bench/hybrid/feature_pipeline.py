from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HybridDataset:
    market: pd.DataFrame
    slow_features: pd.DataFrame
    fast_features: pd.DataFrame
    target: pd.Series


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_synthetic_hybrid_data(n_steps: int = 2000, seed: int = 7) -> HybridDataset:
    """Create a reproducible synthetic market dataset with both slow and fast signals."""
    rng = _rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_steps, freq="min")

    macro = np.cumsum(rng.normal(0.0, 0.02, n_steps))
    sector = np.cumsum(rng.normal(0.0, 0.015, n_steps))
    size = np.cumsum(rng.normal(0.0, 0.01, n_steps))

    lob_imb = rng.normal(0.0, 1.0, n_steps)
    trade_flow = rng.normal(0.0, 1.0, n_steps)
    cancel_rate = np.abs(rng.normal(0.6, 0.2, n_steps))

    base_noise = rng.normal(0.0, 0.0008, n_steps)
    latent = 0.0005 * np.tanh(macro) + 0.00035 * lob_imb + 0.0002 * trade_flow
    returns = latent + base_noise
    price = 100 * np.exp(np.cumsum(returns))

    market = pd.DataFrame(
        {
            "price": price,
            "ret_1": returns,
            "macro_proxy": macro,
            "factor_sector": sector,
            "factor_size": size,
            "lob_imbalance": lob_imb,
            "trade_flow": trade_flow,
            "cancel_rate": cancel_rate,
        },
        index=idx,
    )

    slow = build_slow_features(market)
    fast = build_fast_features(market)
    target = market["ret_1"].shift(-1).fillna(0.0)
    return HybridDataset(market=market, slow_features=slow, fast_features=fast, target=target)


def build_slow_features(market: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=market.index)
    ret = market["ret_1"]
    df["ret_1"] = ret
    df["ret_5"] = ret.rolling(5, min_periods=1).sum()
    df["ret_30"] = ret.rolling(30, min_periods=1).sum()
    df["vol_15"] = ret.rolling(15, min_periods=2).std().fillna(0.0)
    df["vol_60"] = ret.rolling(60, min_periods=2).std().fillna(0.0)
    df["macro_proxy"] = market["macro_proxy"]
    df["factor_sector"] = market["factor_sector"]
    df["factor_size"] = market["factor_size"]
    return df.fillna(0.0)


def build_fast_features(market: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=market.index)
    df["lob_imbalance"] = market["lob_imbalance"]
    df["trade_flow"] = market["trade_flow"]
    df["cancel_rate"] = market["cancel_rate"]
    df["flow_ema_8"] = market["trade_flow"].ewm(span=8, adjust=False).mean()
    df["imbalance_diff"] = market["lob_imbalance"].diff().fillna(0.0)
    df["trade_sign"] = np.sign(market["trade_flow"]) 
    return df.fillna(0.0)
