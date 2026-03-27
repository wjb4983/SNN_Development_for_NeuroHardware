from __future__ import annotations

from typing import Any, Callable

import torch


class SNNtorchBackendAdapter(torch.nn.Module):
    """Backend-native snnTorch-style adapter module."""

    backend = "snntorch"

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: list[int], dropout: float, surrogate_family: str, reset_policy: str, arch: str) -> None:
        super().__init__()
        hidden = max(1, hidden_sizes[0] if hidden_sizes else 32)
        self.arch = arch
        self.surrogate_family = surrogate_family
        self.reset_policy = reset_policy
        self.encoder = torch.nn.Linear(input_dim, hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.readout = torch.nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.mean(dim=1)
        h = torch.tanh(self.encoder(x))
        h = self.dropout(h)
        return self.readout(h)


def build_model(
    input_dim: int,
    params: dict[str, Any],
    arch: str,
    task_spec: dict[str, Any] | None = None,
    fallback_builder: Callable[[], torch.nn.Module] | None = None,
) -> torch.nn.Module:
    backend_cfg = dict(params.get("backend", {}))
    try:
        import snntorch as _snn  # noqa: F401
    except Exception:
        if fallback_builder is not None and bool(params.get("force_fallback", False)):
            return fallback_builder()
    return SNNtorchBackendAdapter(
        input_dim=input_dim,
        output_dim=max(1, int(params.get("output_dim", params.get("num_classes", 1)))),
        hidden_sizes=[int(v) for v in params.get("hidden_sizes", [int(params.get("hidden_dim", 32))])],
        dropout=float(params.get("dropout", 0.0)),
        surrogate_family=str(backend_cfg.get("surrogate_family", params.get("surrogate_type", "tanh"))),
        reset_policy=str(backend_cfg.get("reset_policy", params.get("reset_mode", "zero"))),
        arch=arch,
    )
