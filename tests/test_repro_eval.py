from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from snn_bench.eval.repro_eval import CostModel, no_leakage_walkforward_check, strategy_returns_with_costs
from snn_bench.tasks.performance_realism import build_direction_distribution_targets


class SlippageModelTest(unittest.TestCase):
    def test_higher_volatility_increases_impact_cost(self):
        positions = np.array([0, 1, 1, -1, -1, 0], dtype=float)
        future_returns = np.array([0.0, 0.002, -0.001, 0.003, -0.002, 0.0], dtype=float)
        low_vol = np.full_like(future_returns, 0.005)
        high_vol = np.full_like(future_returns, 0.050)

        model = CostModel(fee_bps=1.0, spread_bps=1.0, impact_coef=0.5)

        pnl_low, _ = strategy_returns_with_costs(positions, future_returns, low_vol, model)
        pnl_high, _ = strategy_returns_with_costs(positions, future_returns, high_vol, model)

        self.assertLess(pnl_high.sum(), pnl_low.sum())


class NoLeakageTest(unittest.TestCase):
    def test_walkforward_guard_raises_when_overlap(self):
        with self.assertRaises(ValueError):
            no_leakage_walkforward_check(train_end_idx=100, prediction_start_idx=100)

    def test_distribution_bins_reuse_train_edges(self):
        close_train = pd.Series(np.linspace(100, 110, 200))
        train_targets, edges = build_direction_distribution_targets(
            close_train,
            horizon=5,
            neutral_band_bps=2.0,
            bins=5,
        )

        close_test = close_train.copy()
        close_test.iloc[-30:] = close_test.iloc[-30:] * 10.0

        test_targets, _ = build_direction_distribution_targets(
            close_test,
            horizon=5,
            neutral_band_bps=2.0,
            bins=5,
            fitted_edges=edges,
        )

        np.testing.assert_array_equal(
            train_targets["distribution_label"].iloc[:120].to_numpy(),
            test_targets["distribution_label"].iloc[:120].to_numpy(),
        )


if __name__ == "__main__":
    unittest.main()
