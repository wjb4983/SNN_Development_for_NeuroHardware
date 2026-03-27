from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


class BasicTrainer:
    """Simple CPU trainer for smoke checks."""

    def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:
        self.model = model
        self.optim = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb_t = torch.as_tensor(xb, dtype=torch.float32)
            yb_t = torch.as_tensor(yb, dtype=torch.float32)
            logits = self.model(xb_t)
            loss = self.loss_fn(logits, yb_t)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(float(loss.item()))
        return sum(losses) / max(1, len(losses))

    @torch.no_grad()
    def predict_logits(self, xb):
        self.model.eval()
        xb_t = torch.as_tensor(xb, dtype=torch.float32)
        return self.model(xb_t)
