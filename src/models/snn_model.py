from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn


class LOBSNNModel(nn.Module):
    """Conv1D + 2-layer recurrent ALIF/LIF style spiking network."""

    def __init__(self, in_channels: int, conv_channels: int = 64, hidden_size: int = 128, classes: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc_in = nn.Linear(conv_channels, hidden_size)
        self.spk1 = snn.RLeaky(beta=0.95, linear_features=hidden_size)
        self.spk2 = snn.RSynaptic(alpha=0.9, beta=0.95, linear_features=hidden_size)
        self.head = nn.Linear(hidden_size, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.conv(x).transpose(1, 2)  # [B, T, F]

        mem1 = self.spk1.init_rleaky()
        syn2, mem2 = self.spk2.init_rsynaptic()
        outs = []
        for t in range(x.size(1)):
            z = self.fc_in(x[:, t, :])
            s1, mem1 = self.spk1(z, mem1)
            s2, syn2, mem2 = self.spk2(s1, syn2, mem2)
            outs.append(self.head(s2))
        return torch.stack(outs, dim=1)
