from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EPS = 1e-12


@dataclass(frozen=True)
class TaskConfig:
    task_name: str
    enabled: bool
    raw: dict


def load_task_configs(config_dir: str | Path) -> list[TaskConfig]:
    root = Path(config_dir)
    configs: list[TaskConfig] = []
    for path in sorted(root.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        configs.append(TaskConfig(task_name=str(payload["task_name"]), enabled=bool(payload.get("enabled", True)), raw=payload))
    return configs


def _future_log_return(log_price: pd.Series, horizon: int) -> pd.Series:
    return log_price.shift(-horizon) - log_price


def direction_with_neutral_band(future_log_return: pd.Series, neutral_band_bps: float) -> pd.Series:
    threshold = float(neutral_band_bps) / 10_000.0
    labels = np.where(future_log_return > threshold, 1, np.where(future_log_return < -threshold, -1, 0))
    return pd.Series(labels.astype(np.int8), index=future_log_return.index, name="direction_label")


def quantile_distribution_labels(
    future_log_return: pd.Series,
    bins: int = 5,
    fitted_edges: np.ndarray | None = None,
) -> tuple[pd.Series, np.ndarray]:
    valid = future_log_return.dropna().to_numpy(dtype=float)
    if fitted_edges is None:
        fitted_edges = np.quantile(valid, np.linspace(0.0, 1.0, bins + 1))

    labels = np.digitize(future_log_return.to_numpy(dtype=float), fitted_edges[1:-1], right=False)
    return pd.Series(labels.astype(np.int64), index=future_log_return.index, name="distribution_label"), fitted_edges


def build_direction_distribution_targets(
    close: pd.Series,
    horizon: int,
    neutral_band_bps: float,
    bins: int = 5,
    fitted_edges: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    log_close = np.log(close.astype(float).replace(0, np.nan))
    fwd = _future_log_return(log_close, horizon=horizon)
    direction = direction_with_neutral_band(fwd, neutral_band_bps=neutral_band_bps)
    distribution, edges = quantile_distribution_labels(fwd, bins=bins, fitted_edges=fitted_edges)
    out = pd.DataFrame({"future_log_return": fwd, "direction_label": direction, "distribution_label": distribution})
    return out, edges


def next_window_realized_vol(close: pd.Series, window: int) -> pd.Series:
    log_close = np.log(close.astype(float).replace(0, np.nan))
    r = log_close.diff()
    rv = r.shift(-window + 1).rolling(window).std() * np.sqrt(window)
    return rv.rename("realized_vol_target")


def iv_skew_movement_labels(skew: pd.Series, horizon: int, threshold: float) -> pd.Series:
    move = skew.shift(-horizon) - skew
    labels = np.where(move > threshold, 1, np.where(move < -threshold, -1, 0))
    return pd.Series(labels.astype(np.int8), index=skew.index, name="iv_skew_move_label")


def assert_no_lookahead(features: pd.DataFrame, labels: pd.DataFrame) -> None:
    if not features.index.equals(labels.index):
        raise ValueError("feature and label indexes must align exactly for walk-forward training")
    if not features.index.is_monotonic_increasing:
        raise ValueError("feature index must be monotonic increasing")
