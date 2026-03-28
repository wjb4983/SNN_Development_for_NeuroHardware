from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import LOBSequenceDataset, load_lob_dataframe
from src.eval.backtest import pnl_simulation
from src.eval.validation import PurgedWalkForwardSplit
from src.features import build_lob_features, make_horizon_labels, rate_code, ttfs_code
from src.models import ANNBaselineLSTM, LOBSNNModel
from src.train import TrainConfig, fit_model
from src.train.utils import save_json, set_seed


def _load_cfg(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(cfg: dict, in_channels: int):
    mcfg = cfg["model"]
    if mcfg["name"].lower() == "snn":
        return LOBSNNModel(in_channels=in_channels, conv_channels=mcfg["conv_channels"], hidden_size=mcfg["hidden_size"])
    return ANNBaselineLSTM(in_channels=in_channels, hidden_size=mcfg["hidden_size"])


def _prepare_data(cfg: dict):
    df = load_lob_dataframe(cfg["source"], cfg["data_path"])
    feats = build_lob_features(df, levels=cfg["features"]["levels"])
    labels_map = make_horizon_labels(df, horizons=[cfg["horizon"]])
    y = labels_map[cfg["horizon"]]

    enc = cfg["features"].get("encoding", "none")
    x = feats.to_numpy()
    if enc == "rate":
        x = rate_code(x, timesteps=cfg["features"].get("timesteps", 16)).mean(axis=0)
    elif enc == "ttfs":
        x = ttfs_code(x, timesteps=cfg["features"].get("timesteps", 16)).sum(axis=0)

    return df, x.astype(np.float32), y.astype(np.int64)


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    set_seed(cfg["seed"])
    df, x, y = _prepare_data(cfg)

    splitter = PurgedWalkForwardSplit(**cfg["eval"])
    train_idx, valid_idx = next(splitter.split(len(x)))
    train_ds = LOBSequenceDataset(x[train_idx], y[train_idx], window=cfg["window"], stride=cfg["stride"])
    valid_ds = LOBSequenceDataset(x[valid_idx], y[valid_idx], window=cfg["window"], stride=cfg["stride"])

    model = _build_model(cfg, in_channels=x.shape[1])
    tcfg = TrainConfig(seed=cfg["seed"], **cfg["train"])
    out_dir = Path(cfg["output_dir"]) / f"{cfg['model']['name']}_{cfg['horizon']}"

    result = fit_model(model, train_ds, valid_ds, out_dir, tcfg)
    save_json(out_dir / "train_summary.json", result)
    print(f"Training completed. Results in {out_dir}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    df, x, y = _prepare_data(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model = _build_model(cfg, in_channels=x.shape[1])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ds = LOBSequenceDataset(x, y, window=cfg["window"], stride=cfg["stride"])
    probs, labels = [], []
    for xb, yb in ds:
        with torch.no_grad():
            out = model(xb.unsqueeze(0))[:, -1, :]
            p = torch.softmax(out, dim=-1).squeeze(0).numpy()
        probs.append(p)
        labels.append(int(yb))

    probs = np.vstack(probs)
    pred = probs.argmax(axis=1)
    start = cfg["window"]
    mid = ((df["bid_price_1"] + df["ask_price_1"]) / 2.0).to_numpy()[start : start + len(pred)]
    bt = pnl_simulation(
        mid,
        pred,
        latency_steps=cfg["eval"]["latency_steps"],
        fee_bps=cfg["eval"]["fee_bps"],
        spread_bps=cfg["eval"]["spread_bps"],
    )

    out = {"pred_dist": pred.tolist(), "labels": labels, "backtest": bt}
    out_path = Path(cfg["output_dir"]) / "evaluation.json"
    save_json(out_path, out)
    print(f"Saved evaluation: {out_path}")


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    pred = np.asarray(payload["pred_dist"], dtype=np.int64)
    df = load_lob_dataframe(cfg["source"], cfg["data_path"])
    mid = ((df["bid_price_1"] + df["ask_price_1"]) / 2.0).to_numpy()[: len(pred)]

    bt = pnl_simulation(mid, pred, cfg["eval"]["latency_steps"], cfg["eval"]["fee_bps"], cfg["eval"]["spread_bps"])
    print(bt)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LOB SNN alpha CLI")
    sub = p.add_subparsers(required=True)

    t = sub.add_parser("train", help="train model")
    t.add_argument("--config", default="configs/lob_alpha.yaml")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate checkpoint")
    e.add_argument("--config", default="configs/lob_alpha.yaml")
    e.add_argument("--checkpoint", required=True)
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("backtest", help="backtest prediction artifact")
    b.add_argument("--config", default="configs/lob_alpha.yaml")
    b.add_argument("--predictions", required=True)
    b.set_defaults(func=cmd_backtest)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
