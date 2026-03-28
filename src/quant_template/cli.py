from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from .backtest import BacktestConfig
from .models import ANNBaselineClassifier, SNNProxyClassifier
from .pipeline import run_experiment
from .splits import PurgedEmbargoWalkForward


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    extends = cfg.pop("extends", None)
    if extends:
        base = _load_yaml(path.parent / extends)
        base.update(cfg)
        return base
    return cfg


def _build_model(model_cfg: dict):
    family = model_cfg.get("family", "ann")
    if family == "ann":
        return ANNBaselineClassifier()
    if family == "snn":
        return SNNProxyClassifier()
    raise ValueError(f"Unsupported model family: {family}")


def _make_demo_data(n_samples: int = 500, n_features: int = 8):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    raw = X[:, 0] * 0.7 - X[:, 1] * 0.2 + rng.normal(scale=0.8, size=n_samples)
    y = (raw > np.median(raw)).astype(int)
    prices = 100 + np.cumsum(rng.normal(loc=0.02, scale=0.5, size=n_samples))
    return X, y, prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quant template experiment")
    parser.add_argument("--config", default="configs/template/default.yaml")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    model = _build_model(cfg["model"])
    split = PurgedEmbargoWalkForward(**cfg["split"])
    bt_cfg = BacktestConfig(**cfg["backtest"])
    X, y, prices = _make_demo_data()

    artifacts = run_experiment(
        model=model,
        X=X,
        y=y,
        prices=prices,
        output_dir=cfg.get("output_dir", "artifacts/template"),
        seed=cfg.get("seed", 42),
        splitter=split,
        backtest_cfg=bt_cfg,
    )
    print(f"Run complete: {artifacts.run_id} -> {artifacts.run_dir}")
    print(artifacts.summary_metrics)


if __name__ == "__main__":
    main()
