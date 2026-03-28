from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class FastModelOutput:
    score: np.ndarray
    confidence: np.ndarray


class _SimpleLIFNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 24, beta: float = 0.85) -> None:
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem = torch.zeros((x.shape[0], self.fc_in.out_features), device=x.device)
        h = torch.tanh(self.fc_in(x))
        mem = self.beta * mem + h
        spikes = (mem > 0).float()
        score = self.fc_out(spikes).squeeze(-1)
        conf = torch.sigmoid(torch.abs(mem).mean(dim=1))
        return score, conf


class FastSNNModel:
    """Event-level model approximating microstructure response using spiking dynamics."""

    def __init__(self, input_dim: int, hidden_dim: int = 24, lr: float = 1e-3, epochs: int = 35, seed: int = 7) -> None:
        torch.manual_seed(seed)
        self.model = _SimpleLIFNet(in_dim=input_dim, hidden_dim=hidden_dim)
        self.lr = lr
        self.epochs = epochs
        self.feature_cols: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "FastSNNModel":
        self.feature_cols = list(x.columns)
        xt = torch.tensor(x.values, dtype=torch.float32)
        yt = torch.tensor(y.values, dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred, _ = self.model(xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()
        return self

    def predict(self, x: pd.DataFrame) -> FastModelOutput:
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(x[self.feature_cols].values, dtype=torch.float32)
            score, conf = self.model(xt)
        return FastModelOutput(score=score.cpu().numpy().astype(np.float32), confidence=conf.cpu().numpy().astype(np.float32))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.model.fc_in.in_features,
            "hidden_dim": self.model.fc_in.out_features,
            "lr": self.lr,
            "epochs": self.epochs,
            "feature_cols": self.feature_cols,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "FastSNNModel":
        with path.open("rb") as f:
            payload = pickle.load(f)
        obj = cls(
            input_dim=payload["input_dim"],
            hidden_dim=payload["hidden_dim"],
            lr=payload["lr"],
            epochs=payload["epochs"],
        )
        obj.model.load_state_dict(payload["state_dict"])
        obj.feature_cols = payload["feature_cols"]
        return obj
