from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-8


@dataclass
class WalkForwardWindow:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class WalkForwardSplitter:
    """Simple walk-forward splitter for time-indexed datasets."""

    def __init__(
        self,
        train_size: int,
        val_size: int,
        test_size: int,
        step_size: int | None = None,
    ) -> None:
        if min(train_size, val_size, test_size) <= 0:
            raise ValueError("train/val/test sizes must be positive")
        self.train_size = int(train_size)
        self.val_size = int(val_size)
        self.test_size = int(test_size)
        self.step_size = int(step_size or test_size)

    def split(self, n_samples: int | Iterable[object]) -> list[WalkForwardWindow]:
        if not isinstance(n_samples, int):
            n_samples = len(list(n_samples))
        n_samples = int(n_samples)

        windows: list[WalkForwardWindow] = []
        start = 0
        total = self.train_size + self.val_size + self.test_size

        while start + total <= n_samples:
            train_end = start + self.train_size
            val_end = train_end + self.val_size
            test_end = val_end + self.test_size
            windows.append(
                WalkForwardWindow(
                    train_idx=np.arange(start, train_end),
                    val_idx=np.arange(train_end, val_end),
                    test_idx=np.arange(val_end, test_end),
                )
            )
            start += self.step_size
        return windows


class ForecastFeaturePipeline:
    """Leakage-safe feature engineering for 5m/30m forecasting from bars + options snapshots."""

    def __init__(
        self,
        vol_windows: tuple[int, ...] = (5, 20, 60),
        vol_z_window: int = 60,
        horizon_map: dict[str, int] | None = None,
        n_return_bins: int = 3,
    ) -> None:
        self.vol_windows = vol_windows
        self.vol_z_window = int(vol_z_window)
        self.horizon_map = horizon_map or {"ret_5m": 5, "ret_30m": 30, "rv_30m": 30}
        self.n_return_bins = int(n_return_bins)

        self._fitted = False
        self.feature_columns_: list[str] = []
        self._means: pd.Series | None = None
        self._stds: pd.Series | None = None
        self._regime_edges: np.ndarray | None = None
        self._ret_edges: dict[str, np.ndarray] = {}

    @staticmethod
    def _to_dt_index(values: pd.Series) -> pd.DatetimeIndex:
        idx = pd.to_datetime(values, utc=True, errors="coerce")
        if idx.isna().any():
            raise ValueError("timestamp column contains unparsable values")
        return pd.DatetimeIndex(idx)

    def _build_price_volume_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        frame["timestamp"] = self._to_dt_index(frame["t"])
        frame = frame.sort_values("timestamp").set_index("timestamp")

        close = frame["c"].astype(float)
        high = frame["h"].astype(float)
        low = frame["l"].astype(float)
        open_ = frame["o"].astype(float)
        volume = frame["v"].astype(float)
        trades = frame["n"].astype(float)

        out = pd.DataFrame(index=frame.index)
        out["log_ret_1"] = np.log(close.replace(0, np.nan)).diff()

        for window in self.vol_windows:
            out[f"rv_{window}"] = out["log_ret_1"].rolling(window).std() * np.sqrt(window)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr_like_14"] = tr.rolling(14).mean()

        vol_mean = volume.rolling(self.vol_z_window).mean()
        vol_std = volume.rolling(self.vol_z_window).std()
        out["vol_z"] = (volume - vol_mean) / (vol_std + EPS)

        out["close_location"] = ((close - low) / (high - low + EPS)) - 0.5
        out["signed_vol"] = np.sign(close - open_) * volume
        out["trade_size"] = volume / (trades + 1.0)
        out["trade_imbalance_proxy"] = np.sign(close - open_) * trades
        return out

    @staticmethod
    def _normalize_option_columns(options: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            "call_put": "option_type",
            "type": "option_type",
            "cp": "option_type",
            "openInterest": "open_interest",
            "oi": "open_interest",
            "iv": "implied_volatility",
            "impliedVolatility": "implied_volatility",
            "expiry": "expiration",
            "exp": "expiration",
            "underlier_price": "underlying_price",
            "underlying": "underlying_price",
        }
        normalized = options.rename(columns={k: v for k, v in col_map.items() if k in options.columns}).copy()
        required = [
            "t",
            "option_type",
            "volume",
            "open_interest",
            "implied_volatility",
            "strike",
            "expiration",
        ]
        missing = [c for c in required if c not in normalized.columns]
        if missing:
            raise ValueError(f"options frame missing required columns: {missing}")

        normalized["timestamp"] = pd.to_datetime(normalized["t"], utc=True, errors="coerce")
        normalized["expiration"] = pd.to_datetime(normalized["expiration"], utc=True, errors="coerce")
        normalized = normalized.dropna(subset=["timestamp", "expiration"])
        normalized["option_type"] = normalized["option_type"].astype(str).str.lower().str[0]
        normalized["delta"] = pd.to_numeric(normalized.get("delta"), errors="coerce")
        if "underlying_price" not in normalized.columns:
            normalized["underlying_price"] = np.nan
        return normalized

    def _build_options_features(self, options: pd.DataFrame, bar_index: pd.DatetimeIndex, spot: pd.Series) -> pd.DataFrame:
        opts = self._normalize_option_columns(options)
        opts = opts.sort_values("timestamp")

        out_rows: list[dict[str, float | pd.Timestamp]] = []
        for ts, grp in opts.groupby("timestamp"):
            g = grp.copy()
            u = g["underlying_price"].astype(float)
            if u.notna().any():
                spot_px = float(u.dropna().iloc[-1])
            else:
                spot_px = float(spot.reindex([ts], method="nearest").iloc[0])

            call_mask = g["option_type"].eq("c")
            put_mask = g["option_type"].eq("p")

            call_vol = g.loc[call_mask, "volume"].astype(float).sum()
            put_vol = g.loc[put_mask, "volume"].astype(float).sum()
            cp_ratio = (call_vol + 1.0) / (put_vol + 1.0)

            mny = (g["strike"].astype(float) / (spot_px + EPS)) - 1.0
            oi = g["open_interest"].astype(float)
            near = oi[(mny.abs() <= 0.02)].sum()
            otm_call = oi[(mny > 0.02)].sum()
            otm_put = oi[(mny < -0.02)].sum()
            total_oi = oi.sum() + EPS
            oi_conc = max(near, otm_call, otm_put) / total_oi

            dte_days = (g["expiration"] - ts).dt.total_seconds() / (24 * 3600)
            dte_days = dte_days.clip(lower=0.25)
            near_w = np.exp(-dte_days / 7.0)
            signed_oi = np.where(call_mask, 1.0, -1.0) * oi.to_numpy()
            near_exp_pressure = float(np.sum(signed_oi * near_w.to_numpy()) / (np.sum(np.abs(oi.to_numpy())) + EPS))

            iv = g["implied_volatility"].astype(float)
            iv_level = float(iv.mean())

            if g["delta"].notna().any():
                put_iv25 = iv.iloc[(g["delta"] + 0.25).abs().argmin()]
                call_iv25 = iv.iloc[(g["delta"] - 0.25).abs().argmin()]
            else:
                put_candidates = iv[mny < 0]
                call_candidates = iv[mny > 0]
                put_iv25 = float(put_candidates.mean()) if len(put_candidates) else iv_level
                call_iv25 = float(call_candidates.mean()) if len(call_candidates) else iv_level
            skew_proxy = float(put_iv25 - call_iv25)

            iv_term = g.groupby(g["expiration"]) ["implied_volatility"].mean().sort_index()
            dte_term = (iv_term.index - ts).days.astype(float)
            valid = dte_term > 0
            if valid.sum() >= 2:
                dte_use = dte_term[valid][:3]
                iv_use = iv_term.values[valid][:3]
                term_slope = float((iv_use[1] - iv_use[0]) / (dte_use[1] - dte_use[0] + EPS))
            else:
                term_slope = 0.0

            out_rows.append(
                {
                    "timestamp": ts,
                    "cp_volume_ratio": cp_ratio,
                    "oi_concentration": float(oi_conc),
                    "near_exp_pressure": near_exp_pressure,
                    "iv_level": iv_level,
                    "skew_proxy": skew_proxy,
                    "term_slope": term_slope,
                }
            )

        out = pd.DataFrame(out_rows).set_index("timestamp").sort_index()
        out["iv_change"] = out["iv_level"].diff()
        return out.reindex(bar_index).ffill().fillna(0.0)

    @staticmethod
    def _time_context(index: pd.DatetimeIndex) -> pd.DataFrame:
        minute = index.hour * 60 + index.minute
        day_minutes = 24 * 60
        phase = 2 * np.pi * minute / day_minutes
        out = pd.DataFrame(index=index)
        out["tod_sin"] = np.sin(phase)
        out["tod_cos"] = np.cos(phase)
        return out

    def _compute_regime(self, rv: pd.Series) -> pd.Series:
        if self._regime_edges is None:
            raise RuntimeError("pipeline not fit")
        bins = np.digitize(rv.fillna(0.0).to_numpy(), self._regime_edges[1:-1], right=False)
        return pd.Series(bins.astype(float), index=rv.index, name="vol_regime")

    def _build_feature_frame(self, bars: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
        pv = self._build_price_volume_features(bars)
        opt = self._build_options_features(options, pv.index, spot=bars.set_index(pd.to_datetime(bars["t"], utc=True))["c"])
        ctx = self._time_context(pv.index)
        features = pd.concat([pv, opt, ctx], axis=1)
        if self._regime_edges is not None:
            features["vol_regime"] = self._compute_regime(features[f"rv_{self.vol_windows[-1]}"])
        return features

    def _build_targets(self, bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        frame["timestamp"] = self._to_dt_index(frame["t"])
        frame = frame.sort_values("timestamp").set_index("timestamp")
        log_close = np.log(frame["c"].astype(float).replace(0, np.nan))

        out = pd.DataFrame(index=frame.index)
        ret_5 = (log_close.shift(-self.horizon_map["ret_5m"]) - log_close)
        ret_30 = (log_close.shift(-self.horizon_map["ret_30m"]) - log_close)
        rv_h = (
            log_close.diff().shift(-self.horizon_map["rv_30m"] + 1).rolling(self.horizon_map["rv_30m"]).std()
            * np.sqrt(self.horizon_map["rv_30m"])
        )

        out["ret_5m"] = ret_5
        out["ret_30m"] = ret_30
        out["rv_30m"] = rv_h
        return out

    def fit(self, bars: pd.DataFrame, options: pd.DataFrame) -> "ForecastFeaturePipeline":
        features = self._build_feature_frame(bars, options)

        reg_src = features[f"rv_{self.vol_windows[-1]}"].fillna(0.0).to_numpy()
        self._regime_edges = np.quantile(reg_src, [0.0, 1 / 3, 2 / 3, 1.0])
        features["vol_regime"] = self._compute_regime(features[f"rv_{self.vol_windows[-1]}"])

        self.feature_columns_ = list(features.columns)
        self._means = features.mean(numeric_only=True)
        self._stds = features.std(numeric_only=True).replace(0, 1.0)

        targets = self._build_targets(bars)
        for key in ("ret_5m", "ret_30m"):
            valid = targets[key].dropna().to_numpy()
            edges = np.quantile(valid, np.linspace(0, 1, self.n_return_bins + 1))
            self._ret_edges[key] = edges

        self._fitted = True
        return self

    def transform(self, bars: pd.DataFrame, options: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self._fitted:
            raise RuntimeError("call fit before transform")
        features = self._build_feature_frame(bars, options)
        features = features.reindex(columns=self.feature_columns_)
        x = ((features - self._means) / (self._stds + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        targets_raw = self._build_targets(bars)
        y = pd.DataFrame(index=targets_raw.index)
        for key in ("ret_5m", "ret_30m"):
            bins = np.digitize(targets_raw[key].to_numpy(), self._ret_edges[key][1:-1], right=False)
            y[f"{key}_label"] = bins.astype(np.int64)
        y["rv_30m_target"] = targets_raw["rv_30m"].astype(float)

        valid = y["rv_30m_target"].notna()
        return x.loc[valid], y.loc[valid]

    def fit_transform(self, bars: pd.DataFrame, options: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.fit(bars, options)
        return self.transform(bars, options)
