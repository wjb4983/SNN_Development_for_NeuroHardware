from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from snn_bench.multistream.schema import DatasetConfig


REQUIRED_COLS = {"timestamp", "event_type", "price", "size", "side"}


def load_event_streams(cfg: DatasetConfig) -> dict[str, pd.DataFrame]:
    streams: dict[str, pd.DataFrame] = {}
    for stream_cfg in cfg.streams:
        frame = pd.read_csv(stream_cfg.path)
        missing = REQUIRED_COLS.difference(frame.columns)
        if missing:
            raise ValueError(f"stream {stream_cfg.asset} missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame["asset"] = stream_cfg.asset
        frame["max_staleness_ms"] = stream_cfg.max_staleness_ms
        streams[stream_cfg.asset] = frame
    if cfg.target_asset not in streams:
        raise ValueError(f"target asset {cfg.target_asset} not present in stream configs")
    return streams


def causal_synchronize(streams: dict[str, pd.DataFrame], target_asset: str) -> pd.DataFrame:
    target_stream = streams[target_asset].copy()
    if "asset" not in target_stream.columns:
        target_stream["asset"] = target_asset
    target = target_stream[["timestamp", "asset", "price", "event_type", "size", "side"]].copy()
    target = target.rename(
        columns={
            "price": "target_price",
            "event_type": "target_event_type",
            "size": "target_size",
            "side": "target_side",
        }
    )

    merged = target
    for asset, frame in streams.items():
        if asset == target_asset:
            continue
        staleness_ms = int(frame["max_staleness_ms"].iloc[0])
        source = frame[["timestamp", "price", "size", "side", "event_type"]].copy()
        source = source.rename(
            columns={
                "price": f"{asset}_price",
                "size": f"{asset}_size",
                "side": f"{asset}_side",
                "event_type": f"{asset}_event_type",
            }
        )
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            source.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(milliseconds=staleness_ms),
        )
        # last source timestamp used to produce relative lag feature.
        ts_source = frame[["timestamp"]].rename(columns={"timestamp": f"{asset}_timestamp"})
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            ts_source.sort_values(f"{asset}_timestamp"),
            left_on="timestamp",
            right_on=f"{asset}_timestamp",
            direction="backward",
            tolerance=pd.Timedelta(milliseconds=staleness_ms),
        )

    return merged.sort_values("timestamp").reset_index(drop=True)


def make_multi_horizon_targets(frame: pd.DataFrame, horizons_s: tuple[int, ...]) -> pd.DataFrame:
    out = frame.copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    t_ns = out["timestamp"].astype("int64").to_numpy()
    px = out["target_price"].to_numpy(dtype=np.float64)

    for hz in horizons_s:
        future_ts = t_ns + int(hz * 1_000_000_000)
        idx = np.searchsorted(t_ns, future_ts, side="left")
        valid = idx < len(out)
        fwd = np.full(len(out), np.nan, dtype=np.float64)
        fwd[valid] = px[idx[valid]]
        ret = (fwd - px) / np.clip(px, 1e-9, None)
        out[f"y_{hz}s_return"] = ret
        out[f"y_{hz}s_direction"] = np.where(np.isnan(ret), np.nan, (ret > 0.0).astype(float))

    return out


def config_to_dict(cfg: DatasetConfig) -> dict:
    return asdict(cfg)
