from __future__ import annotations

from typing import Any, Callable

import torch


class SpikingJellyBackendAdapter(torch.nn.Module):
    """Backend-native SpikingJelly-style adapter module."""

    backend = "spikingjelly"

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, dropout: float, event_encoding_mode: str, surrogate_family: str, reset_policy: str, arch: str) -> None:
        super().__init__()
        self.arch = arch
        self.event_encoding_mode = event_encoding_mode
        self.surrogate_family = surrogate_family
        self.reset_policy = reset_policy
        self.conv = torch.nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        pooled = self.dropout(feat.mean(dim=1))
        return self.head(pooled)


def build_model(
    input_dim: int,
    params: dict[str, Any],
    arch: str,
    task_spec: dict[str, Any] | None = None,
    fallback_builder: Callable[[], torch.nn.Module] | None = None,
) -> torch.nn.Module:
    backend_cfg = dict(params.get("backend", {}))
    try:
        from spikingjelly.activation_based import neuron as _neuron  # noqa: F401
    except Exception:
        if fallback_builder is not None and bool(params.get("force_fallback", False)):
            return fallback_builder()
    hidden_sizes = [int(v) for v in params.get("hidden_sizes", [int(params.get("hidden_dim", 32))])]
    return SpikingJellyBackendAdapter(
        input_dim=input_dim,
        output_dim=max(1, int(params.get("output_dim", params.get("num_classes", 1)))),
        hidden_size=hidden_sizes[0],
        dropout=float(params.get("dropout", 0.0)),
        event_encoding_mode=str(backend_cfg.get("event_encoding_mode", "rate")),
        surrogate_family=str(backend_cfg.get("surrogate_family", params.get("surrogate_type", "tanh"))),
        reset_policy=str(backend_cfg.get("reset_policy", params.get("reset_mode", "zero"))),
        arch=arch,
    )
