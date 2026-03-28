from __future__ import annotations

import numpy as np
import pandas as pd

from snn_bench.multistream.schema import FeatureConfig


def _signed_side(s: pd.Series) -> np.ndarray:
    map_side = {"buy": 1.0, "sell": -1.0, "bid": 1.0, "ask": -1.0}
    return s.fillna("buy").astype(str).str.lower().map(map_side).fillna(0.0).to_numpy(dtype=np.float32)


def build_feature_matrix(frame: pd.DataFrame, cfg: FeatureConfig) -> tuple[np.ndarray, dict[str, int], list[str]]:
    work = frame.copy()
    assets = sorted({c[: -len("_price")] for c in work.columns if c.endswith("_price") and c != "target_price"})

    event_vocab = {evt: i for i, evt in enumerate(cfg.event_type_vocab)}
    feature_cols: list[np.ndarray] = []
    feature_names: list[str] = []

    target_price = work["target_price"].to_numpy(dtype=np.float32)
    target_size = work["target_size"].to_numpy(dtype=np.float32)
    target_side = _signed_side(work["target_side"])
    target_lr = np.r_[0.0, np.diff(np.log(np.clip(target_price, 1e-9, None)))]

    feature_cols += [target_lr, np.log1p(np.abs(target_size)) * target_side]
    feature_names += ["target_log_return", "target_signed_size"]

    target_evt_idx = work["target_event_type"].fillna("trade").astype(str).map(event_vocab).fillna(0).to_numpy(dtype=np.int64)
    target_evt_one_hot = np.eye(len(event_vocab), dtype=np.float32)[target_evt_idx]
    for i, evt in enumerate(cfg.event_type_vocab):
        feature_cols.append(target_evt_one_hot[:, i])
        feature_names.append(f"target_evt_{evt}")

    ts_ns = work["timestamp"].astype("int64").to_numpy(dtype=np.int64)
    for asset in assets:
        p = work[f"{asset}_price"].ffill().fillna(work["target_price"])
        s = work[f"{asset}_size"].fillna(0.0)
        side = _signed_side(work[f"{asset}_side"])
        evt = work[f"{asset}_event_type"].fillna("trade").astype(str)
        evt_idx = evt.map(event_vocab).fillna(0).to_numpy(dtype=np.int64)
        evt_one_hot = np.eye(len(event_vocab), dtype=np.float32)[evt_idx]

        p_np = p.to_numpy(dtype=np.float32)
        spread_like = (work["target_price"].to_numpy(dtype=np.float32) - p_np) / np.clip(target_price, 1e-9, None)
        signed_flow = np.log1p(np.abs(s.to_numpy(dtype=np.float32))) * side

        rel_ts = work[f"{asset}_timestamp"].astype("int64").to_numpy(dtype=np.float64)
        rel_ms = np.clip((ts_ns - rel_ts) / 1e6, 0.0, 1e9)
        rel_ms[np.isnan(rel_ms)] = 1e6

        feature_cols += [spread_like, signed_flow, rel_ms.astype(np.float32)]
        feature_names += [f"{asset}_basis", f"{asset}_signed_size", f"{asset}_lag_ms"]
        for i, evt_name in enumerate(cfg.event_type_vocab):
            feature_cols.append(evt_one_hot[:, i])
            feature_names.append(f"{asset}_evt_{evt_name}")

    x = np.column_stack(feature_cols).astype(np.float32)
    return x, event_vocab, feature_names


def drop_nan_targets(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y).all(axis=1)
    return x[mask], y[mask]
