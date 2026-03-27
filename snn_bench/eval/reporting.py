from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_roc(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_pr(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_calibration(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.set_title("Calibration Plot")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_probability_histogram(y_prob: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_prob, bins=20, color="#3b82f6", alpha=0.85, edgecolor="white")
    ax.set_title("Predicted Probability Histogram")
    ax.set_xlabel("Predicted probability for class 1")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _compute_strategy_returns(train_metrics: dict[str, Any], y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    pred_payload = train_metrics.get("eval", {})
    existing = pred_payload.get("strategy_returns")
    if isinstance(existing, list) and existing:
        return np.asarray(existing, dtype=float)

    # Fallback proxy when explicit returns are unavailable.
    y_signed = np.where(y_true > 0, 1.0, -1.0)
    positions = np.where(y_prob >= 0.55, 1.0, np.where(y_prob <= 0.45, -1.0, 0.0))
    return positions * y_signed


def _save_equity_curve(strategy_returns: np.ndarray, out_path: Path) -> None:
    equity = np.cumsum(strategy_returns)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(equity, color="#059669", linewidth=2)
    ax.set_title("Cumulative PnL / Equity Curve")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative return")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_run_report(run_dir: Path | str) -> Path:
    run_path = Path(run_dir)
    metrics_path = run_path / "train_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"train_metrics.json not found in run directory: {run_path}")

    train_metrics = _load_json(metrics_path)
    pred_path = Path(train_metrics.get("predictions", ""))
    if not pred_path.exists():
        pred_path = run_path / pred_path.name
    if not pred_path.exists():
        raise FileNotFoundError(f"prediction artifact not found: {train_metrics.get('predictions')}")

    predictions = _load_json(pred_path)
    y_true = np.asarray(predictions.get("y_true", []), dtype=np.int64)
    y_prob = np.asarray(predictions.get("y_prob", []), dtype=np.float32)

    if y_true.size == 0 or y_prob.size == 0:
        raise ValueError("prediction artifact must contain non-empty y_true and y_prob arrays")

    plots_dir = run_path / "plots"
    _ensure_dir(plots_dir)

    generated_plots: list[Path] = []

    conf_path = plots_dir / "confusion_matrix.png"
    _save_confusion_matrix(y_true, y_prob, conf_path)
    generated_plots.append(conf_path)

    if len(np.unique(y_true)) > 1:
        roc_path = plots_dir / "roc_curve.png"
        _save_roc(y_true, y_prob, roc_path)
        generated_plots.append(roc_path)

        pr_path = plots_dir / "pr_curve.png"
        _save_pr(y_true, y_prob, pr_path)
        generated_plots.append(pr_path)

    calibration_path = plots_dir / "calibration_plot.png"
    _save_calibration(y_true, y_prob, calibration_path)
    generated_plots.append(calibration_path)

    hist_path = plots_dir / "probability_histogram.png"
    _save_probability_histogram(y_prob, hist_path)
    generated_plots.append(hist_path)

    task_eval = (train_metrics.get("task") or {}).get("evaluation") or {}
    trading_metrics = task_eval.get("trading_metrics")
    trading_available = bool(trading_metrics) or "trading" in (train_metrics.get("eval") or {})

    if trading_available:
        strategy_returns = _compute_strategy_returns(train_metrics, y_true, y_prob)
        equity_path = plots_dir / "equity_curve.png"
        _save_equity_curve(strategy_returns, equity_path)
        generated_plots.append(equity_path)

    report_lines = [
        f"# Run report: {train_metrics.get('run_id', run_path.name)}",
        "",
        "## Metrics snapshot",
        "",
        "```json",
        json.dumps(train_metrics.get("eval", {}), indent=2),
        "```",
        "",
        "## Visualizations",
        "",
    ]

    for plot in generated_plots:
        rel = plot.relative_to(run_path)
        title = plot.stem.replace("_", " ").title()
        report_lines.extend([f"### {title}", f"![{title}]({rel.as_posix()})", ""])

    report_path = run_path / "report.md"
    report_path.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")
    return report_path
