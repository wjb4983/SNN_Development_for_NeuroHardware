from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import classification_metrics
from src.train.utils import save_json, set_seed


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    clip_grad: float = 1.0
    patience: int = 5
    tbptt_steps: int = 32
    amp: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _tbptt_loss(logits: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module, steps: int) -> torch.Tensor:
    total = 0.0
    chunks = 0
    for t in range(0, logits.size(1), steps):
        total = total + loss_fn(logits[:, t : t + steps].reshape(-1, logits.size(-1)), y[:, t : t + steps].reshape(-1))
        chunks += 1
    return total / max(chunks, 1)


def fit_model(model: nn.Module, train_ds, valid_ds, out_dir: str | Path, cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    loss_fn = nn.CrossEntropyLoss()

    best = {"val_loss": float("inf"), "epoch": -1}
    wait = 0
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            yb_seq = yb.unsqueeze(1).repeat(1, xb.size(1))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.amp):
                logits = model(xb)
                loss = _tbptt_loss(logits, yb_seq, loss_fn, cfg.tbptt_steps)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(opt)
            scaler.update()
            train_losses.append(loss.item())

        model.eval()
        y_true, y_pred, y_prob = [], [], []
        val_losses = []
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                yb_seq = yb.unsqueeze(1).repeat(1, xb.size(1))
                logits = model(xb)
                val_losses.append(_tbptt_loss(logits, yb_seq, loss_fn, cfg.tbptt_steps).item())

                p = torch.softmax(logits[:, -1, :], dim=-1)
                y_true.append(yb.cpu().numpy())
                y_pred.append(p.argmax(-1).cpu().numpy())
                y_prob.append(p.cpu().numpy())

        y_true_a = np.concatenate(y_true)
        y_pred_a = np.concatenate(y_pred)
        y_prob_a = np.concatenate(y_prob)
        metrics = classification_metrics(y_true_a, y_pred_a, y_prob_a)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(np.mean(val_losses)),
            }
        )
        history.append(metrics)

        if metrics["val_loss"] < best["val_loss"]:
            best = metrics
            wait = 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "best": best}, out_dir / "checkpoint.pt")
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    save_json(out_dir / "results.json", {"best": best, "history": history, "config": asdict(cfg)})
    return {"best": best, "history": history}
