from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class LIFCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, decay: float = 0.9) -> None:
        super().__init__()
        self.ff = nn.Linear(input_dim, hidden_dim)
        self.decay = decay

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mem, spk = state
        mem = self.decay * mem + self.ff(x_t) - spk
        spk = torch.sigmoid(5.0 * (mem - 1.0))
        return spk, (mem, spk)


class PerAssetSpikingEncoder(nn.Module):
    def __init__(self, input_dim: int, encoder_dim: int, decay: float) -> None:
        super().__init__()
        self.cell = LIFCell(input_dim, encoder_dim, decay=decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        mem = torch.zeros(b, self.cell.ff.out_features, device=x.device)
        spk = torch.zeros_like(mem)
        out = []
        for i in range(t):
            s, (mem, spk) = self.cell(x[:, i, :], (mem, spk))
            out.append(s)
        return torch.stack(out, dim=1)


class SparseCoupling(nn.Module):
    def __init__(self, n_assets: int, top_k_edges: int) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.top_k_edges = max(1, min(top_k_edges, n_assets * n_assets))
        self.raw = nn.Parameter(torch.randn(n_assets, n_assets) / math.sqrt(max(1, n_assets)))

    def coupling_matrix(self) -> torch.Tensor:
        flat = self.raw.flatten()
        top_idx = torch.topk(torch.abs(flat), self.top_k_edges, sorted=False).indices
        mask = torch.zeros_like(flat)
        mask[top_idx] = 1.0
        sparse = (flat * mask).reshape(self.n_assets, self.n_assets)
        return torch.tanh(sparse)

    def forward(self, per_asset_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [B, T, A, D]
        c = self.coupling_matrix()
        mixed = torch.einsum("ij,btjd->btid", c, per_asset_repr)
        return mixed, c


class RecurrentSpikingFusion(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int, decay: float) -> None:
        super().__init__()
        self.cell = LIFCell(input_dim, fusion_dim, decay=decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        mem = torch.zeros(b, self.cell.ff.out_features, device=x.device)
        spk = torch.zeros_like(mem)
        states = []
        for i in range(t):
            s, (mem, spk) = self.cell(x[:, i, :], (mem, spk))
            states.append(s)
        return torch.stack(states, dim=1)


class MultiStreamSNN(nn.Module):
    def __init__(self, per_asset_dim: int, n_assets: int, encoder_dim: int, fusion_dim: int, decay: float, top_k_edges: int, n_horizons: int) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.encoder = PerAssetSpikingEncoder(per_asset_dim, encoder_dim, decay)
        self.coupling = SparseCoupling(n_assets=n_assets, top_k_edges=top_k_edges)
        self.fusion = RecurrentSpikingFusion(input_dim=n_assets * encoder_dim, fusion_dim=fusion_dim, decay=decay)
        self.head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, n_horizons))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, A, F]
        b, t, a, f = x.shape
        enc = []
        for i in range(a):
            enc.append(self.encoder(x[:, :, i, :]))
        per_asset = torch.stack(enc, dim=2)
        mixed, couplings = self.coupling(per_asset)
        fused = self.fusion(mixed.reshape(b, t, -1))
        logits = self.head(fused[:, -1, :])
        return logits, couplings, fused


class MultiStreamANNBaseline(nn.Module):
    def __init__(self, per_asset_dim: int, n_assets: int, hidden_dim: int, n_horizons: int, mode: str = "lstm") -> None:
        super().__init__()
        self.mode = mode
        in_dim = per_asset_dim * n_assets
        if mode == "tcn":
            self.backbone = nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
            )
        else:
            self.backbone = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, a, f = x.shape
        merged = x.reshape(b, t, a * f)
        if self.mode == "tcn":
            h = self.backbone(merged.transpose(1, 2)).transpose(1, 2)
            out = h[:, -1, :]
        else:
            h, _ = self.backbone(merged)
            out = h[:, -1, :]
        return self.head(out)


@dataclass
class FitArtifacts:
    train_loss: list[float]
    val_loss: list[float]
    class_weights: list[float]


def balanced_bce_loss(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(targets > 0.5, class_weights[1], class_weights[0])
    return (loss * weights).mean()


def class_weights_from_targets(y: np.ndarray) -> np.ndarray:
    flat = y.reshape(-1)
    p = float(np.mean(flat > 0.5))
    p = min(max(p, 1e-3), 1 - 1e-3)
    return np.asarray([1.0 / (1 - p), 1.0 / p], dtype=np.float32)
