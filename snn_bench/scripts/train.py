from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from snn_bench.configs.settings import BenchmarkConfig, ModelSelectionConfig, SmokeConfig
from snn_bench.data_connectors.backtest_store import BacktestBarStoreConnector
from snn_bench.data_connectors.snapshot_cache import SnapshotCacheConnector
from snn_bench.feature_pipelines.basic_features import BasicFeaturePipeline
from snn_bench.models import ModelSpec, ModelZoo, save_prediction_artifacts, set_global_seed
from snn_bench.utils.secrets import load_massive_api_key


def _available_years(index_blob: dict) -> list[int]:
    years = index_blob.get("years") or []
    if not years:
        raise ValueError("index.json does not contain years")
    return sorted(int(y) for y in years)


def _load_cfg_from_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _apply_smoke(cfg: BenchmarkConfig, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if not cfg.smoke.enabled:
        return x, y, cfg.epochs
    limit = min(len(x), int(cfg.smoke.sample_size))
    return x[:limit], y[:limit], int(cfg.smoke.epochs)


def run_training(cfg: BenchmarkConfig, out_dir: Path, max_years: int = 0) -> dict:
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    load_massive_api_key(cfg.massive_api_key_file)

    snapshot = SnapshotCacheConnector(cfg.data_paths.snapshot_dir, cfg.data_paths.external_snapshot_dir)
    bars = BacktestBarStoreConnector(cfg.data_paths.backtest_root)

    rows_snapshot = len(snapshot.load_frame(cfg.ticker))
    years = _available_years(bars.load_index(cfg.ticker, cfg.timeframe))
    selected_years = years if max_years <= 0 else years[:max_years]
    frames = [bars.load_year(cfg.ticker, cfg.timeframe, y) for y in selected_years]
    frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    x, y = BasicFeaturePipeline().transform(frame)
    x, y, epochs = _apply_smoke(cfg, x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y.astype(np.int64),
        test_size=0.2,
        random_state=cfg.seed,
        shuffle=True,
        stratify=y.astype(np.int64) if len(np.unique(y)) > 1 else None,
    )

    model_spec = ModelSpec(name=cfg.model.name, family="zoo", params={**cfg.model.params, "seed": cfg.seed})
    model = ModelZoo.create(model_spec, input_dim=x_train.shape[1])
    train_info = model.fit(x_train, y_train, epochs=epochs)

    y_prob = model.predict_proba(x_test)
    eval_metrics = model.evaluate(x_test, y_test)

    run_id = f"{cfg.run_name}_{cfg.model.name}_{cfg.ticker}_{cfg.timeframe}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = run_dir / f"{cfg.model.name}_checkpoint.bin"
    model.save_checkpoint(checkpoint)
    pred_artifact = save_prediction_artifacts(run_dir, cfg.model.name, y_test, y_prob)

    metrics = {
        "run_id": run_id,
        "ticker": cfg.ticker,
        "timeframe": cfg.timeframe,
        "years": selected_years,
        "rows_snapshot": rows_snapshot,
        "rows_train_total": int(len(x)),
        "rows_train": int(len(x_train)),
        "rows_eval": int(len(x_test)),
        "epochs": epochs,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "model": cfg.model.name,
        "model_params": cfg.model.params,
        "train_info": train_info,
        "eval": eval_metrics,
        "checkpoint": str(checkpoint),
        "predictions": str(pred_artifact),
    }
    (run_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model-zoo benchmark model")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--timeframe", default="1D")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--max-years", type=int, default=0, help="0=all available years; otherwise use earliest N years")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-sample-size", type=int, default=256)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_data: dict = {}
    if args.config:
        cfg_data = _load_cfg_from_yaml(args.config)

    cfg = BenchmarkConfig.model_validate(
        {
            "ticker": args.ticker,
            "timeframe": args.timeframe,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "run_name": args.run_name,
            "seed": args.seed,
            "model": ModelSelectionConfig(name=args.model, params={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}),
            "smoke": SmokeConfig(enabled=args.smoke, sample_size=args.smoke_sample_size, epochs=args.smoke_epochs),
            **cfg_data,
        }
    )

    metrics = run_training(cfg, Path(args.out_dir), max_years=args.max_years)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
