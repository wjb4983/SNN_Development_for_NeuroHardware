from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from snn_bench.tasks.registry import TaskRegistry, assert_aligned_not_empty


class TaskRegistryTest(unittest.TestCase):
    @staticmethod
    def _make_bars(n: int = 240) -> pd.DataFrame:
        ts = pd.date_range("2025-01-02 14:30:00+00:00", periods=n, freq="min")
        base = 100 + np.cumsum(np.sin(np.arange(n) / 5.0) * 0.08 + 0.01)
        o = base + np.random.default_rng(1).normal(0, 0.02, n)
        c = base + np.random.default_rng(2).normal(0, 0.02, n)
        h = np.maximum(o, c) + 0.05
        l = np.minimum(o, c) - 0.05
        v = 1000 + np.arange(n) * 2
        ntr = 80 + (np.arange(n) % 10)
        return pd.DataFrame({"t": ts, "o": o, "h": h, "l": l, "c": c, "v": v, "n": ntr})

    def test_task_yamls_build_non_empty_aligned_datasets(self):
        bars = self._make_bars(360)
        registry = TaskRegistry("snn_bench/configs/tasks")

        task_names = [
            "direction_5m_distribution",
            "direction_30m_distribution",
            "realized_vol_30m",
            "options_iv_skew_movement",
        ]

        for task_name in task_names:
            with self.subTest(task_name=task_name):
                spec = registry.resolve(task_name=task_name)
                x, y = registry.build_dataset(bars, spec)
                assert_aligned_not_empty(x, y)
                self.assertGreater(len(x), 0)
                self.assertEqual(len(x), len(y))


if __name__ == "__main__":
    unittest.main()
