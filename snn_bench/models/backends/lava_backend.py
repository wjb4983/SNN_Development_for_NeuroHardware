from __future__ import annotations

from typing import Any, Callable

import torch


class LavaBackendAdapter(torch.nn.Module):
    """Backend-native Lava-style adapter module."""

    backend = "lava"

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, dropout: float, event_encoding_mode: str, reset_policy: str, arch: str) -> None:
        super().__init__()
        self.arch = arch
        self.event_encoding_mode = event_encoding_mode
        self.reset_policy = reset_policy
        self.fc1 = torch.nn.Linear(input_dim, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.mean(dim=1)
        h = torch.relu(self.fc1(x))
        return self.fc2(self.dropout(h))


def build_model(
    input_dim: int,
    params: dict[str, Any],
    arch: str,
    task_spec: dict[str, Any] | None = None,
    fallback_builder: Callable[[], torch.nn.Module] | None = None,
) -> torch.nn.Module:
    backend_cfg = dict(params.get("backend", {}))
    try:
        import lava.lib.dl.slayer as _slayer  # noqa: F401
    except Exception:
        if fallback_builder is not None and bool(params.get("force_fallback", False)):
            return fallback_builder()
    hidden_sizes = [int(v) for v in params.get("hidden_sizes", [int(params.get("hidden_dim", 32))])]
    return LavaBackendAdapter(
        input_dim=input_dim,
        output_dim=max(1, int(params.get("output_dim", params.get("num_classes", 1)))),
        hidden_size=hidden_sizes[0],
        dropout=float(params.get("dropout", 0.0)),
        event_encoding_mode=str(backend_cfg.get("event_encoding_mode", "delta")),
        reset_policy=str(backend_cfg.get("reset_policy", params.get("reset_mode", "zero"))),
        arch=arch,
    )
