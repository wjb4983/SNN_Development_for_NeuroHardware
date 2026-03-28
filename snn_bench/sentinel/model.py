from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class SentinelConfig:
    input_dim: int
    hidden_dim: int = 48
    regime_classes: int = 3
    beta: float = 0.9


class StreamingSNNSentinel(nn.Module):
    def __init__(self, cfg: SentinelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.enc = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.mem_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.dec = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        self.classifier = nn.Linear(cfg.hidden_dim, cfg.regime_classes)
        self.beta = float(cfg.beta)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B,T,F]
        bsz, steps, _ = x.shape
        mem = torch.zeros((bsz, self.cfg.hidden_dim), device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(steps):
            cur = self.enc(x[:, t, :]) + self.mem_proj(mem)
            mem = self.beta * mem + (1.0 - self.beta) * cur
            spk = torch.sigmoid(mem)
            spikes.append(spk)
        z = torch.stack(spikes, dim=1)
        recon = self.dec(z)
        logits = self.classifier(z.mean(dim=1))
        return recon, logits, z

    @staticmethod
    def anomaly_score(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return ((x - recon) ** 2).mean(dim=(1, 2))

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.state_dict(), "config": self.cfg.__dict__}, path)

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str = "cpu") -> "StreamingSNNSentinel":
        payload = torch.load(path, map_location=map_location)
        cfg = SentinelConfig(**payload["config"])
        model = cls(cfg)
        model.load_state_dict(payload["state"])
        return model


def sentinel_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    logits: torch.Tensor,
    regime_target: torch.Tensor,
    latent: torch.Tensor,
    *,
    recon_weight: float = 1.0,
    cls_weight: float = 1.0,
    smooth_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    rec = torch.nn.functional.mse_loss(recon, x)
    cls = torch.nn.functional.cross_entropy(logits, regime_target)
    latent_delta = latent[:, 1:, :] - latent[:, :-1, :]
    smooth = torch.mean(latent_delta**2) if latent_delta.numel() else torch.tensor(0.0, device=x.device)
    total = recon_weight * rec + cls_weight * cls + smooth_weight * smooth
    return total, {
        "reconstruction_loss": float(rec.detach().cpu().item()),
        "classification_loss": float(cls.detach().cpu().item()),
        "smoothness_loss": float(smooth.detach().cpu().item()),
    }


def infer_stream(model: StreamingSNNSentinel, features: np.ndarray, batch_size: int = 256) -> dict[str, np.ndarray]:
    model.eval()
    x = torch.from_numpy(features.astype(np.float32))
    scores: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i : i + batch_size][:, None, :]
            recon, logits, _ = model(xb)
            anomaly = model.anomaly_score(xb, recon)
            reg_prob = torch.softmax(logits, dim=-1)
            scores.append(anomaly.cpu().numpy())
            probs.append(reg_prob.cpu().numpy())
    anomaly_score = np.concatenate(scores, axis=0)
    regime_prob = np.concatenate(probs, axis=0)
    regime_class = np.argmax(regime_prob, axis=1)
    return {
        "anomaly_score": anomaly_score,
        "regime_prob": regime_prob,
        "regime_class": regime_class,
    }
