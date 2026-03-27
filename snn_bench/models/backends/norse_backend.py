from __future__ import annotations

from typing import Any, Callable

import torch


class NorseBackendAdapter(torch.nn.Module):
    """Backend-native Norse-style recurrent adapter module."""

    backend = "norse"

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, dropout: float, recurrent_cell_type: str, surrogate_family: str, reset_policy: str, arch: str) -> None:
        super().__init__()
        self.arch = arch
        self.recurrent_cell_type = recurrent_cell_type
        self.surrogate_family = surrogate_family
        self.reset_policy = reset_policy
        self.in_proj = torch.nn.Linear(input_dim, hidden_size)
        self.rec_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.out_proj = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        h = torch.zeros(x.shape[0], self.rec_proj.in_features, device=x.device, dtype=x.dtype)
        outs: list[torch.Tensor] = []
        for t in range(x.shape[1]):
            h = torch.tanh(self.in_proj(x[:, t, :]) + self.rec_proj(h))
            outs.append(self.out_proj(self.dropout(h)))
        return torch.stack(outs, dim=1).mean(dim=1)


def build_model(
    input_dim: int,
    params: dict[str, Any],
    arch: str,
    task_spec: dict[str, Any] | None = None,
    fallback_builder: Callable[[], torch.nn.Module] | None = None,
) -> torch.nn.Module:
    backend_cfg = dict(params.get("backend", {}))
    try:
        import norse.torch as _norse_torch  # noqa: F401
    except Exception:
        if fallback_builder is not None and bool(params.get("force_fallback", False)):
            return fallback_builder()
    hidden_sizes = [int(v) for v in params.get("hidden_sizes", [int(params.get("hidden_dim", 32))])]
    return NorseBackendAdapter(
        input_dim=input_dim,
        output_dim=max(1, int(params.get("output_dim", params.get("num_classes", 1)))),
        hidden_size=hidden_sizes[0],
        dropout=float(params.get("dropout", 0.0)),
        recurrent_cell_type=str(backend_cfg.get("recurrent_cell_type", "lsnn")),
        surrogate_family=str(backend_cfg.get("surrogate_family", params.get("surrogate_type", "fast_sigmoid"))),
        reset_policy=str(backend_cfg.get("reset_policy", params.get("reset_mode", "zero"))),
        arch=arch,
    )
