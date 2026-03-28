from __future__ import annotations

import torch
import torch.nn as nn


class ANNBaselineLSTM(nn.Module):
    """Fair ANN baseline using LSTM + linear head."""

    def __init__(self, in_channels: int, hidden_size: int = 128, classes: int = 3):
        super().__init__()
        self.net = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_size, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.net(x)
        return self.head(o)
