from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from snn_bench.sentinel.calibration import save_threshold_config, tune_thresholds
from snn_bench.sentinel.data import FEATURE_COLUMNS, SentinelDataModule, load_stream_csv
from snn_bench.sentinel.model import SentinelConfig, StreamingSNNSentinel, infer_stream, sentinel_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train streaming SNN risk sentinel")
    p.add_argument("--input", type=Path, required=True, help="CSV with required stream features")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/sentinel"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--normalization-window", type=int, default=256)
    return p.parse_args()


def _plot_training(losses: list[float], path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Sentinel training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_anomaly(scores: np.ndarray, stress: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="anomaly_score")
    plt.fill_between(np.arange(len(stress)), 0, np.max(scores), where=stress > 0, alpha=0.2, label="stress")
    plt.legend(loc="upper right")
    plt.title("Anomaly score vs stress labels")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    frame = load_stream_csv(args.input, normalization_window=args.normalization_window)
    module = SentinelDataModule(
        frame=frame,
        feature_columns=FEATURE_COLUMNS,
        regime_column="regime_label",
        stress_column="stress_label",
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    train_loader, val_loader, arrays = module.build()

    model = StreamingSNNSentinel(SentinelConfig(input_dim=len(FEATURE_COLUMNS), hidden_dim=args.hidden_dim, regime_classes=3))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_losses: list[float] = []

    for _ in range(args.epochs):
        model.train()
        batch_losses = []
        for xb, regime, _ in train_loader:
            recon, logits, latent = model(xb)
            loss, _parts = sentinel_loss(xb, recon, logits, regime, latent)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        epoch_losses.append(float(np.mean(batch_losses) if batch_losses else 0.0))

    model_path = out_dir / "sentinel_checkpoint.pt"
    model.save_checkpoint(model_path)

    val_feats = arrays["features"][int(arrays["cut"][0]) :]
    val_stress = arrays["stress"][int(arrays["cut"][0]) :]
    infer = infer_stream(model, val_feats)
    cfg = tune_thresholds(infer["anomaly_score"], val_stress, target_fpr=args.target_fpr)
    thresh_path = out_dir / "thresholds.json"
    save_threshold_config(thresh_path, cfg)

    _plot_training(epoch_losses, plots_dir / "training_loss.png")
    _plot_anomaly(infer["anomaly_score"], val_stress, plots_dir / "anomaly_vs_stress.png")

    metrics = {
        "epochs": args.epochs,
        "train_loss_last": epoch_losses[-1] if epoch_losses else None,
        "target_fpr": args.target_fpr,
        "thresholds": cfg.__dict__,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = out_dir / "report.md"
    report.write_text(
        "\n".join(
            [
                "# Streaming SNN Risk Sentinel Report",
                "",
                "## Outputs",
                "- (a) anomaly score from reconstruction error",
                "- (b) regime class from spiking classifier head",
                "- (c) binary risk gate derived from calibrated thresholds",
                "",
                "## Training Summary",
                f"- epochs: {args.epochs}",
                f"- final_train_loss: {metrics['train_loss_last']}",
                f"- target_fpr: {args.target_fpr}",
                f"- threshold_config: `{thresh_path.name}`",
                "",
                "## Charts",
                "![training loss](plots/training_loss.png)",
                "![anomaly vs stress](plots/anomaly_vs_stress.png)",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"checkpoint": str(model_path), "thresholds": str(thresh_path), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
