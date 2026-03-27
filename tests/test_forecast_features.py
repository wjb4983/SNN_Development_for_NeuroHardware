from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from snn_bench.feature_pipelines.forecast_features import ForecastFeaturePipeline, WalkForwardSplitter


class ForecastFeaturePipelineTest(unittest.TestCase):
    @staticmethod
    def _make_bars(n: int = 240) -> pd.DataFrame:
        ts = pd.date_range("2025-01-02 14:30:00+00:00", periods=n, freq="min")
        base = 100 + np.cumsum(np.sin(np.arange(n) / 10.0) * 0.05 + 0.01)
        o = base + np.random.default_rng(1).normal(0, 0.02, n)
        c = base + np.random.default_rng(2).normal(0, 0.02, n)
        h = np.maximum(o, c) + 0.05
        l = np.minimum(o, c) - 0.05
        v = 1000 + np.arange(n) * 2
        ntr = 80 + (np.arange(n) % 10)
        return pd.DataFrame({"t": ts, "o": o, "h": h, "l": l, "c": c, "v": v, "n": ntr})

    @staticmethod
    def _make_options(bars: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        expiries = [
            pd.Timestamp("2025-01-10", tz="UTC"),
            pd.Timestamp("2025-01-17", tz="UTC"),
            pd.Timestamp("2025-02-21", tz="UTC"),
        ]
        deltas = [-0.4, -0.25, -0.1, 0.1, 0.25, 0.4]
        strikes_off = [-0.1, -0.03, -0.01, 0.01, 0.03, 0.1]
        for i, row in bars.iloc[::5].iterrows():
            spot = float(row["c"])
            ts = row["t"]
            for exp in expiries:
                for d, off in zip(deltas, strikes_off):
                    is_call = d > 0
                    rows.append(
                        {
                            "t": ts,
                            "option_type": "call" if is_call else "put",
                            "volume": 10 + (i % 7) + (3 if is_call else 0),
                            "open_interest": 100 + (i % 13) * 5,
                            "implied_volatility": 0.18 + abs(off) * 0.25 + (0.01 if exp == expiries[0] else 0),
                            "strike": spot * (1 + off),
                            "expiration": exp,
                            "delta": d,
                            "underlying_price": spot,
                        }
                    )
        return pd.DataFrame(rows)

    def test_feature_and_target_shapes(self):
        bars = self._make_bars()
        options = self._make_options(bars)
        pipe = ForecastFeaturePipeline()

        x, y = pipe.fit_transform(bars.iloc[:180], options[options["t"] < bars.iloc[180]["t"]])

        self.assertGreater(x.shape[1], 10)
        self.assertEqual(len(x), len(y))
        self.assertTrue({"ret_5m_label", "ret_30m_label", "rv_30m_target"}.issubset(y.columns))
        self.assertFalse(x.isna().any().any())

    def test_leakage_safe_scaling(self):
        bars = self._make_bars()
        options = self._make_options(bars)

        train_cut = 150
        b_train = bars.iloc[:train_cut]
        o_train = options[options["t"] < bars.iloc[train_cut]["t"]]

        pipe_a = ForecastFeaturePipeline().fit(b_train, o_train)
        x_a, _ = pipe_a.transform(bars, options)

        bars_mut = bars.copy()
        bars_mut.loc[bars_mut.index >= train_cut, "c"] *= 5.0
        pipe_b = ForecastFeaturePipeline().fit(b_train, o_train)
        x_b, _ = pipe_b.transform(bars_mut, options)

        pd.testing.assert_series_equal(x_a.iloc[:100, 0], x_b.iloc[:100, 0], check_names=False)


class WalkForwardSplitterTest(unittest.TestCase):
    def test_walk_forward_windows(self):
        splitter = WalkForwardSplitter(train_size=50, val_size=20, test_size=10, step_size=10)
        windows = splitter.split(120)

        self.assertEqual(len(windows), 5)
        self.assertEqual(windows[0].train_idx[0], 0)
        self.assertEqual(windows[0].val_idx[0], 50)
        self.assertEqual(windows[0].test_idx[0], 70)
        self.assertEqual(windows[-1].test_idx[-1], 119)


class SmokeIntegrationTest(unittest.TestCase):
    def test_end_to_end_smoke(self):
        bars = ForecastFeaturePipelineTest._make_bars(360)
        options = ForecastFeaturePipelineTest._make_options(bars)
        splitter = WalkForwardSplitter(train_size=180, val_size=60, test_size=60, step_size=60)
        windows = splitter.split(len(bars))
        self.assertGreaterEqual(len(windows), 2)

        first = windows[0]
        pipe = ForecastFeaturePipeline()
        pipe.fit(
            bars.iloc[first.train_idx],
            options[options["t"].isin(set(bars.iloc[first.train_idx]["t"]))],
        )

        x_val, y_val = pipe.transform(
            bars.iloc[np.concatenate([first.val_idx, first.test_idx])],
            options[options["t"].isin(set(bars.iloc[np.concatenate([first.val_idx, first.test_idx])]["t"]))],
        )

        self.assertGreater(len(x_val), 0)
        self.assertEqual(len(x_val), len(y_val))


if __name__ == "__main__":
    unittest.main()
