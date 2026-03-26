from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from snn_bench.configs.settings import BenchmarkConfig
from snn_bench.data_connectors.backtest_store import BacktestBarStoreConnector
from snn_bench.data_connectors.snapshot_cache import SnapshotCacheConnector
from snn_bench.eval.metrics import binary_accuracy
from snn_bench.feature_pipelines.basic_features import BasicFeaturePipeline
from snn_bench.models.dummy_snn import DummySNN
from snn_bench.tasks.binary_direction import BinaryDirectionDataset
from snn_bench.trainers.basic_trainer import BasicTrainer


def _safe_years(index_blob: dict) -> list[int]:
    years = index_blob.get("years") or []
    return sorted(int(y) for y in years)


def run_smoke(cfg: BenchmarkConfig) -> dict:
    snap = SnapshotCacheConnector(
        primary_dir=cfg.data_paths.snapshot_dir,
        fallback_dir=cfg.data_paths.external_snapshot_dir,
    )
    backtest = BacktestBarStoreConnector(cfg.data_paths.backtest_root)

    snapshot_rows = len(snap.load_frame(cfg.ticker))
    index_blob = backtest.load_index(cfg.ticker, cfg.timeframe)
    years = _safe_years(index_blob)
    if not years:
        raise ValueError("index.json does not contain years")

    bars = backtest.load_year(cfg.ticker, cfg.timeframe, years[0])
    x, y = BasicFeaturePipeline().transform(bars)

    model = DummySNN(input_dim=x.shape[1])
    trainer = BasicTrainer(model=model)
    loader = DataLoader(BinaryDirectionDataset(x, y), batch_size=cfg.batch_size, shuffle=False)

    avg_loss = trainer.train_epoch(loader)
    logits = trainer.predict_logits(x).detach().cpu().numpy()
    acc = binary_accuracy(logits=logits, labels=y)

    return {
        "ticker": cfg.ticker,
        "timeframe": cfg.timeframe,
        "rows_snapshot": snapshot_rows,
        "rows_bars": int(len(bars)),
        "loss": float(avg_loss),
        "accuracy": float(acc),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark smoke pipeline")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--timeframe", default="1D")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(ticker=args.ticker, timeframe=args.timeframe)
    metrics = run_smoke(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
