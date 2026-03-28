from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn


class RecurrentSpikingPolicy(nn.Module):
    """Recurrent spiking policy for action + size bucket prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_actions: int = 6,
        num_sizes: int = 4,
        value_head: bool = True,
        beta: float = 0.95,
    ) -> None:
        super().__init__()
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.lif = snn.RLeaky(beta=beta, linear_features=hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.size_head = nn.Linear(hidden_dim, num_sizes)
        self.value_head = nn.Linear(hidden_dim, 1) if value_head else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [batch, seq, feat]."""
        batch, seq, _ = x.shape
        spk = torch.zeros((batch, self.policy_head.in_features), device=x.device)
        mem = torch.zeros((batch, self.policy_head.in_features), device=x.device)
        spk_out = []
        for t in range(seq):
            z = torch.relu(self.enc(x[:, t, :]))
            spk, mem = self.lif(z, spk, mem)
            spk_out.append(spk)
        h = torch.stack(spk_out, dim=1)
        tail = h[:, -1, :]
        out = {
            "action_logits": self.policy_head(tail),
            "size_logits": self.size_head(tail),
        }
        if self.value_head is not None:
            out["value"] = self.value_head(tail).squeeze(-1)
        return out


class ANNBaselinePolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_actions: int = 6, num_sizes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.size_head = nn.Linear(hidden_dim, num_sizes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = x[:, -1, :]
        h = self.net(flat)
        return {
            "action_logits": self.action_head(h),
            "size_logits": self.size_head(h),
        }
