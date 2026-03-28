from __future__ import annotations

import json
from pathlib import Path

import yaml

from snn_bench.hybrid.cli import cmd_backtest_hybrid, cmd_train_fast, cmd_train_fusion, cmd_train_slow


def test_hybrid_training_and_backtest(tmp_path: Path) -> None:
    cfg = {
        "reproducibility": {"seed": 9},
        "data": {"n_steps": 600, "seed": 9},
        "artifacts": {
            "slow_model_path": str(tmp_path / "slow.pkl"),
            "fast_model_path": str(tmp_path / "fast.pkl"),
            "fusion_model_path": str(tmp_path / "fusion.pkl"),
            "backtest_report_path": str(tmp_path / "report.json"),
        },
        "fast_model": {"epochs": 3, "hidden_dim": 10},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cmd_train_slow(config_path)
    cmd_train_fast(config_path)
    cmd_train_fusion(config_path)
    cmd_backtest_hybrid(config_path)

    report_path = Path(cfg["artifacts"]["backtest_report_path"])
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert "attribution" in payload
    assert set(payload["risk_state_counts"].keys()) == {"NORMAL", "WARNING", "BLOCK"}
