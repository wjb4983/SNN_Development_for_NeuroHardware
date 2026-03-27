from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from snn_bench.eval.reporting import generate_run_report


class ReportingModuleTest(unittest.TestCase):
    def _write_run_files(self, run_dir: Path, include_trading: bool) -> None:
        pred_path = run_dir / "mlp_predictions.json"
        pred_payload = {
            "model": "mlp",
            "target_summary": {"task_name": "direction_5m_distribution"},
            "y_true": [0, 1, 0, 1, 1, 0, 1, 0],
            "y_prob": [0.1, 0.9, 0.2, 0.8, 0.7, 0.4, 0.95, 0.3],
            "reference_close": [100.0, 101.0, 100.5, 102.0, 101.8, 101.0, 103.0, 102.5],
            "reference_next_close": [101.0, 100.5, 102.0, 101.8, 101.0, 103.0, 102.5, 103.2],
        }
        pred_path.write_text(json.dumps(pred_payload, indent=2), encoding="utf-8")

        eval_cfg = {"trading_metrics": ["net_pnl"]} if include_trading else {"trading_metrics": []}
        metrics_payload = {
            "run_id": "sample_run",
            "eval": {"accuracy": 0.75, "roc_auc": 0.81},
            "predictions": str(pred_path),
            "task": {"evaluation": eval_cfg},
        }
        (run_dir / "train_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    def test_generate_run_report_creates_required_plots(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_1"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_files(run_dir, include_trading=False)

            report_path = generate_run_report(run_dir)

            self.assertTrue(report_path.exists())
            self.assertTrue((run_dir / "plots" / "confusion_matrix.png").exists())
            self.assertTrue((run_dir / "plots" / "roc_curve.png").exists())
            self.assertTrue((run_dir / "plots" / "pr_curve.png").exists())
            self.assertTrue((run_dir / "plots" / "calibration_plot.png").exists())
            self.assertTrue((run_dir / "plots" / "probability_histogram.png").exists())
            self.assertTrue((run_dir / "plots" / "next_bar_prediction_vs_outcome.png").exists())
            self.assertFalse((run_dir / "plots" / "equity_curve.png").exists())

            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("plots/confusion_matrix.png", report_text)
            self.assertIn("Directional hit rate", report_text)

    def test_generate_run_report_includes_equity_plot_when_trading_available(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run_2"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_files(run_dir, include_trading=True)

            report_path = generate_run_report(run_dir)

            self.assertTrue(report_path.exists())
            self.assertTrue((run_dir / "plots" / "equity_curve.png").exists())


if __name__ == "__main__":
    unittest.main()
