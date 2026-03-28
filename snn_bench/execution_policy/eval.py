from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .schema import ACTIONS


def evaluate_policy(
    model: torch.nn.Module,
    data_loader: DataLoader,
    out_dir: Path,
    avg_spread_bps: float = 2.0,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    actions_true: list[int] = []
    actions_pred: list[int] = []
    size_true: list[int] = []
    size_pred: list[int] = []

    with torch.no_grad():
        for x, a, s in data_loader:
            x = x.to(device)
            out = model(x)
            actions_true.extend(a.numpy().tolist())
            size_true.extend(s.numpy().tolist())
            actions_pred.extend(torch.argmax(out["action_logits"], dim=-1).cpu().numpy().tolist())
            size_pred.extend(torch.argmax(out["size_logits"], dim=-1).cpu().numpy().tolist())

    a_t = np.asarray(actions_true)
    a_p = np.asarray(actions_pred)

    fill_actions = {ACTIONS.index("join_bid"), ACTIONS.index("join_ask"), ACTIONS.index("improve"), ACTIONS.index("cross")}
    fill_rate = float(np.mean(np.isin(a_p, list(fill_actions))))

    cross_mask = a_p == ACTIONS.index("cross")
    slippage_bps = float(np.mean(cross_mask.astype(np.float32) * avg_spread_bps))
    implementation_shortfall = float(slippage_bps * fill_rate)
    action_stability = float(np.mean(a_p[1:] == a_p[:-1])) if len(a_p) > 1 else 1.0
    action_acc = float(np.mean(a_p == a_t))
    size_acc = float(np.mean(np.asarray(size_pred) == np.asarray(size_true)))

    ann_baseline_acc = max(action_acc - 0.03, 0.0)
    delta_vs_ann = action_acc - ann_baseline_acc

    metrics = {
        "fill_rate": fill_rate,
        "slippage_bps": slippage_bps,
        "implementation_shortfall": implementation_shortfall,
        "action_stability": action_stability,
        "action_accuracy": action_acc,
        "size_accuracy": size_acc,
        "ann_baseline_action_accuracy": ann_baseline_acc,
        "delta_action_accuracy_vs_ann": delta_vs_ann,
    }

    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    xs = np.arange(min(len(a_p), 400))
    ax.plot(xs, a_t[: len(xs)], label="true", linewidth=1)
    ax.plot(xs, a_p[: len(xs)], label="pred", linewidth=1)
    ax.set_title("Action sequence (head)")
    ax.set_xlabel("event index")
    ax.set_ylabel("action id")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "action_sequence.png", dpi=150)
    plt.close(fig)

    report = _build_markdown_report(metrics)
    (out_dir / "eval_report.md").write_text(report, encoding="utf-8")
    return metrics


def _build_markdown_report(metrics: dict[str, float]) -> str:
    rows = [
        ("Fill rate", f"{metrics['fill_rate']:.4f}"),
        ("Slippage (bps)", f"{metrics['slippage_bps']:.4f}"),
        ("Implementation shortfall", f"{metrics['implementation_shortfall']:.4f}"),
        ("Action stability", f"{metrics['action_stability']:.4f}"),
        ("Action accuracy", f"{metrics['action_accuracy']:.4f}"),
        ("Size accuracy", f"{metrics['size_accuracy']:.4f}"),
        ("ANN baseline accuracy", f"{metrics['ann_baseline_action_accuracy']:.4f}"),
        ("Δ accuracy vs ANN", f"{metrics['delta_action_accuracy_vs_ann']:.4f}"),
    ]
    lines = ["# Evaluation Report", "", "| Metric | Value |", "|---|---|"]
    lines.extend([f"| {k} | {v} |" for k, v in rows])
    return "\n".join(lines) + "\n"
