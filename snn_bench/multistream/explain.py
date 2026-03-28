from __future__ import annotations

import numpy as np
import torch


def coupling_edge_importance(coupling_matrix: torch.Tensor, asset_names: list[str]) -> list[dict[str, float | str]]:
    matrix = coupling_matrix.detach().cpu().numpy()
    edges: list[dict[str, float | str]] = []
    for i, src in enumerate(asset_names):
        for j, dst in enumerate(asset_names):
            edges.append({"source": src, "target": dst, "weight": float(matrix[i, j]), "importance": float(abs(matrix[i, j]))})
    edges.sort(key=lambda x: x["importance"], reverse=True)
    return edges


def temporal_attribution_snapshots(fused_states: torch.Tensor, logits: torch.Tensor, n_snapshots: int = 5) -> list[dict[str, float | int]]:
    # surrogate: gradient norm of recent fused states wrt max horizon logit.
    idx = torch.linspace(0, fused_states.shape[1] - 1, steps=min(n_snapshots, fused_states.shape[1])).long()
    rows = []
    target = logits[:, -1].mean()
    grads = torch.autograd.grad(target, fused_states, retain_graph=True, allow_unused=True)[0]
    if grads is None:
        return []
    g = grads.detach().abs().mean(dim=0).mean(dim=-1)
    for i in idx.tolist():
        rows.append({"timestep": int(i), "attribution": float(g[i].item())})
    return rows


def summarize_latency(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float32)
    return {
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }
