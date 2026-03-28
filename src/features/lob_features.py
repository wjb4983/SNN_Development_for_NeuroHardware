from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


HORIZON_MAP_MS = {"100ms": 100, "500ms": 500, "1s": 1000, "5s": 5000}


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    return a / (b.abs() + eps)


def build_lob_features(df: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    data = df.copy()
    data["mid_price"] = (data["bid_price_1"] + data["ask_price_1"]) / 2.0
    data["spread"] = data["ask_price_1"] - data["bid_price_1"]

    feature_cols: list[str] = ["spread"]
    data["mid_delta"] = data["mid_price"].diff().fillna(0.0)
    feature_cols.append("mid_delta")

    bid_depth = np.zeros(len(data))
    ask_depth = np.zeros(len(data))
    for lvl in range(1, levels + 1):
        bid_depth += data.get(f"bid_size_{lvl}", 0.0)
        ask_depth += data.get(f"ask_size_{lvl}", 0.0)
        data[f"bid_delta_{lvl}"] = data.get(f"bid_size_{lvl}", 0.0).diff().fillna(0.0)
        data[f"ask_delta_{lvl}"] = data.get(f"ask_size_{lvl}", 0.0).diff().fillna(0.0)
        feature_cols.extend([f"bid_delta_{lvl}", f"ask_delta_{lvl}"])

    data["imbalance"] = _safe_div(pd.Series(bid_depth), pd.Series(bid_depth + ask_depth))
    feature_cols.append("imbalance")

    data["depth_slope"] = _safe_div(
        data.get("ask_price_5", data["ask_price_1"]) - data["ask_price_1"],
        data.get("ask_size_5", data["ask_size_1"]) + 1.0,
    ) - _safe_div(
        data["bid_price_1"] - data.get("bid_price_5", data["bid_price_1"]),
        data.get("bid_size_5", data["bid_size_1"]) + 1.0,
    )
    feature_cols.append("depth_slope")

    data["trade_sign"] = np.sign(data.get("trade_price", data["mid_price"]).diff().fillna(0.0))
    data["trade_intensity"] = data.get("trade_qty", 0.0).rolling(20, min_periods=1).mean()
    data["cancel_intensity"] = data.get("cancel_qty", 0.0).rolling(20, min_periods=1).mean()
    feature_cols.extend(["trade_sign", "trade_intensity", "cancel_intensity"])

    features = data[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    features = (features - features.mean()) / (features.std(ddof=0) + 1e-6)
    return features.astype(np.float32)


def make_horizon_labels(
    df: pd.DataFrame,
    horizons: Iterable[str] = ("100ms", "500ms", "1s", "5s"),
    move_threshold_bps: float = 0.5,
) -> dict[str, np.ndarray]:
    if "mid_price" not in df:
        mid = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
    else:
        mid = df["mid_price"]

    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.datetime64):
        ts_ms = ts.astype("int64") // 1_000_000
    else:
        ts_ms = pd.to_numeric(ts, errors="coerce").fillna(0).astype(np.int64)

    labels: dict[str, np.ndarray] = {}
    bps_denom = 1e-4
    for h in horizons:
        h_ms = HORIZON_MAP_MS[h]
        target_time = ts_ms + h_ms
        fut_idx = np.searchsorted(ts_ms.to_numpy(), target_time.to_numpy(), side="left")
        fut_idx = np.clip(fut_idx, 0, len(mid) - 1)
        future_mid = mid.to_numpy()[fut_idx]
        ret_bps = ((future_mid - mid.to_numpy()) / (mid.to_numpy() + 1e-9)) / bps_denom

        y = np.ones(len(ret_bps), dtype=np.int64)
        y[ret_bps > move_threshold_bps] = 2
        y[ret_bps < -move_threshold_bps] = 0
        labels[h] = y

    return labels
