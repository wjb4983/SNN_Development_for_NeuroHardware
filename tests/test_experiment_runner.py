from __future__ import annotations

import unittest

from snn_bench.scripts.run_experiments import _build_run_config, _deep_merge


class ExperimentRunnerConfigTest(unittest.TestCase):
    def test_deep_merge_recursively_merges_nested_dicts(self):
        base = {"model": {"name": "mlp", "params": {"epochs": 1, "lr": 0.001}}, "seed": 7}
        override = {"model": {"params": {"epochs": 3}}}

        merged = _deep_merge(base, override)

        self.assertEqual(merged["model"]["name"], "mlp")
        self.assertEqual(merged["model"]["params"]["epochs"], 3)
        self.assertEqual(merged["model"]["params"]["lr"], 0.001)
        self.assertEqual(merged["seed"], 7)

    def test_build_run_config_promotes_name_to_run_name(self):
        defaults = {"ticker": "AAPL", "model": {"name": "mlp"}}
        run_item = {"name": "my_run", "model": {"name": "logreg"}}

        cfg = _build_run_config(defaults, run_item)

        self.assertEqual(cfg["run_name"], "my_run")
        self.assertEqual(cfg["ticker"], "AAPL")
        self.assertEqual(cfg["model"]["name"], "logreg")


if __name__ == "__main__":
    unittest.main()
