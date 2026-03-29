from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from snn_bench.hybrid.cli import cmd_backtest_hybrid, cmd_train_fast, cmd_train_fusion, cmd_train_slow
from snn_bench.hybrid.fusion import RegimeAwareFusion


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


def test_regime_aware_fusion_deterministic_outputs() -> None:
    fusion = RegimeAwareFusion(
        slow_weight=0.6,
        fast_weight=0.4,
        regime_conf_sensitivity=1.0,
        low_confidence_threshold=0.3,
    )
    slow_score = np.array([0.2, 0.2], dtype=np.float32)
    slow_conf = np.array([0.8, 0.8], dtype=np.float32)
    fast_score = np.array([0.9, 0.9], dtype=np.float32)
    fast_conf = np.array([0.8, 0.8], dtype=np.float32)
    post = np.array([[0.99, 0.01], [0.5, 0.5]], dtype=np.float32)
    out = fusion.blend(slow_score, slow_conf, fast_score, fast_conf, regime_posteriors=post)
    assert out.score.shape == (2,)
    assert out.confidence.shape == (2,)
    assert out.snn_trust is not None
    assert out.regime_confidence is not None
    assert out.snn_trust[0] > out.snn_trust[1]
    assert out.fast_component[0] > out.fast_component[1]


def test_regime_aware_fusion_low_confidence_fallback() -> None:
    fusion = RegimeAwareFusion(
        slow_weight=0.5,
        fast_weight=0.5,
        regime_conf_sensitivity=1.0,
        low_confidence_threshold=0.95,
        low_confidence_snn_scale=0.3,
    )
    slow_score = np.array([0.4], dtype=np.float32)
    slow_conf = np.array([1.0], dtype=np.float32)
    fast_score = np.array([1.0], dtype=np.float32)
    fast_conf = np.array([1.0], dtype=np.float32)
    post = np.array([[0.5, 0.5]], dtype=np.float32)
    out = fusion.blend(slow_score, slow_conf, fast_score, fast_conf, regime_posteriors=post)
    assert out.snn_trust is not None
    assert out.snn_trust[0] <= 0.3
    assert out.score[0] < 0.7  # closer to slow component due to low trust in SNN


def test_hybrid_backtest_with_regime_fusion_markov_artifact(tmp_path: Path) -> None:
    n_steps = 120
    markov_posterior_path = tmp_path / "markov_post.npy"
    posterior = np.tile(np.array([[0.9, 0.1]], dtype=np.float32), (n_steps, 1))
    np.save(markov_posterior_path, posterior)

    cfg = {
        "reproducibility": {"seed": 12},
        "data": {"n_steps": n_steps, "seed": 12},
        "artifacts": {
            "slow_model_path": str(tmp_path / "slow.pkl"),
            "fast_model_path": str(tmp_path / "fast.pkl"),
            "fusion_model_path": str(tmp_path / "fusion.pkl"),
            "backtest_report_path": str(tmp_path / "report.json"),
        },
        "fast_model": {"epochs": 2, "hidden_dim": 8},
        "fusion": {"type": "regime_aware"},
        "regime_fusion": {"markov_posterior_path": str(markov_posterior_path)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cmd_train_slow(config_path)
    cmd_train_fast(config_path)
    cmd_train_fusion(config_path)
    cmd_backtest_hybrid(config_path)

    report_path = Path(cfg["artifacts"]["backtest_report_path"])
    assert report_path.exists()
