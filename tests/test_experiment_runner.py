from __future__ import annotations

import unittest

from snn_bench.scripts.run_experiments import (
    _build_leaderboard,
    _build_run_config,
    _deep_merge,
    _metric_direction,
    _task_primary_metrics,
)


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

    def test_primary_metric_resolution_prefers_task_evaluation(self):
        metrics = {
            "task": {"evaluation": {"primary_ml_metric": "f1", "primary_trading_metric": "sharpe"}},
            "eval": {"roc_auc": 0.7, "f1": 0.6},
        }
        ml, trading = _task_primary_metrics(metrics)
        self.assertEqual(ml, "f1")
        self.assertEqual(trading, "sharpe")


    def test_primary_metric_resolution_falls_back_to_f1_macro_for_multiclass_eval(self):
        metrics = {
            "task": {"evaluation": {}},
            "eval": {"accuracy": 0.72, "f1_macro": 0.64, "f1": 0.83},
        }
        ml, trading = _task_primary_metrics(metrics)
        self.assertEqual(ml, "f1_macro")
        self.assertIsNone(trading)

    def test_build_leaderboard_supports_f1_macro(self):
        completed = [
            {"run_id": "r1", "model": "snntorch_lif", "task": {"name": "regime_classification"}, "eval": {"f1_macro": 0.51}},
            {"run_id": "r2", "model": "hmm_gaussian", "task": {"name": "regime_classification"}, "eval": {"f1_macro": 0.66}},
            {"run_id": "r3", "model": "markov_chain", "task": {"name": "regime_classification"}, "eval": {"f1_macro": 0.49}},
        ]
        leaderboard = _build_leaderboard(completed, "eval.f1_macro", "ml", "desc")
        self.assertEqual([row["run_id"] for row in leaderboard], ["r2", "r1", "r3"])
        self.assertEqual([row["metric_key"] for row in leaderboard], ["eval.f1_macro", "eval.f1_macro", "eval.f1_macro"])

    def test_metric_direction(self):
        self.assertEqual(_metric_direction("roc_auc"), "desc")
        self.assertEqual(_metric_direction("rmse"), "asc")
        self.assertEqual(_metric_direction("max_drawdown"), "asc")

    def test_build_leaderboard_ranks_by_metric(self):
        completed = [
            {"run_id": "r1", "model": "mlp", "task": {"name": "direction_5m_distribution"}, "eval": {"roc_auc": 0.61}},
            {"run_id": "r2", "model": "logreg", "task": {"name": "direction_5m_distribution"}, "eval": {"roc_auc": 0.67}},
            {"run_id": "r3", "model": "gbm", "task": {"name": "direction_5m_distribution"}, "eval": {"roc_auc": 0.58}},
        ]
        leaderboard = _build_leaderboard(completed, "eval.roc_auc", "ml", "desc")
        self.assertEqual([row["run_id"] for row in leaderboard], ["r2", "r1", "r3"])
        self.assertEqual([row["rank"] for row in leaderboard], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
