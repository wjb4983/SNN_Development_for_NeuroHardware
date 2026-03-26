from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from snn_bench.configs.settings import BenchmarkConfig
from snn_bench.data_connectors.backtest_store import BacktestBarStoreConnector
from snn_bench.data_connectors.snapshot_cache import SnapshotCacheConnector
from snn_bench.eval.metrics import binary_accuracy
from snn_bench.feature_pipelines.basic_features import BasicFeaturePipeline
from snn_bench.models.dummy_snn import DummySNN
from snn_bench.tasks.binary_direction import BinaryDirectionDataset
from snn_bench.trainers.basic_trainer import BasicTrainer
from snn_bench.utils.secrets import load_massive_api_key


def _first_year(index_blob: dict) -> int:
    years = index_blob.get("years") or []
    if not years:
        raise ValueError("index.json does not contain years")
    return int(sorted(years)[0])


def run_training(cfg: BenchmarkConfig, out_dir: Path) -> dict:
    load_massive_api_key(cfg.massive_api_key_file)
    snapshot = SnapshotCacheConnector(cfg.data_paths.snapshot_dir, cfg.data_paths.external_snapshot_dir)
    bars = BacktestBarStoreConnector(cfg.data_paths.backtest_root)

    rows_snapshot = len(snapshot.load_frame(cfg.ticker))
    year = _first_year(bars.load_index(cfg.ticker, cfg.timeframe))
    frame = bars.load_year(cfg.ticker, cfg.timeframe, year)
    x, y = BasicFeaturePipeline().transform(frame)

    dataset = BinaryDirectionDataset(x, y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = DummySNN(input_dim=x.shape[1])
    trainer = BasicTrainer(model=model, lr=cfg.lr)

    losses: list[float] = []
    for _ in range(cfg.epochs):
        losses.append(trainer.train_epoch(loader))

    logits = trainer.predict_logits(x).detach().cpu().numpy()
    acc = binary_accuracy(logits, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{cfg.ticker}_{cfg.timeframe}_dummy.pt"
    torch.save(model.state_dict(), checkpoint)

    metrics = {
        "ticker": cfg.ticker,
        "timeframe": cfg.timeframe,
        "year": year,
        "rows_snapshot": rows_snapshot,
        "rows_train": int(len(dataset)),
        "epochs": cfg.epochs,
        "loss_last": float(losses[-1]),
        "accuracy": float(acc),
        "checkpoint": str(checkpoint),
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dummy benchmark model")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--timeframe", default="1D")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(
        ticker=args.ticker,
        timeframe=args.timeframe,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    metrics = run_training(cfg, Path(args.out_dir))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
