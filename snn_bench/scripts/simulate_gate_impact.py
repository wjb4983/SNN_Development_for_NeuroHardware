from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from snn_bench.sentinel.calibration import load_threshold_config
from snn_bench.sentinel.evaluation import evaluate_sentinel
from snn_bench.sentinel.gate import RiskGate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate risk gate impact from sentinel outputs")
    p.add_argument("--scores", type=Path, required=True, help="NPY anomaly scores")
    p.add_argument("--stress-labels", type=Path, required=True, help="NPY stress labels")
    p.add_argument("--pnl", type=Path, required=True, help="NPY per-step pnl")
    p.add_argument("--threshold-config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/sentinel_sim"))
    p.add_argument("--min-warning-steps", type=int, default=3)
    p.add_argument("--min-block-steps", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots = args.out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    scores = np.load(args.scores)
    stress = np.load(args.stress_labels).astype(np.int64)
    pnl = np.load(args.pnl).astype(np.float32)

    thresh = load_threshold_config(args.threshold_config)
    gate = RiskGate(thresh.to_gate_config(min_warning_steps=args.min_warning_steps, min_block_steps=args.min_block_steps))
    states = gate.run(scores)
    eval_res = evaluate_sentinel(stress, states, pnl)

    state_code = np.array([0 if s.value == "NORMAL" else (1 if s.value == "WARNING" else 2) for s in states], dtype=np.int64)
    baseline_equity = np.cumsum(pnl)
    gated = pnl.copy()
    gated[state_code == 2] = 0.0
    gated_equity = np.cumsum(gated)

    plt.figure(figsize=(10, 4))
    plt.plot(baseline_equity, label="baseline")
    plt.plot(gated_equity, label="gated")
    plt.title("Gate impact on equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "equity_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(scores, label="anomaly_score")
    plt.plot(state_code, label="gate_state_code", alpha=0.7)
    plt.title("Anomaly and gate state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "gate_timeline.png")
    plt.close()

    payload = {
        "precision": eval_res.precision,
        "recall": eval_res.recall,
        "avg_detection_delay": eval_res.avg_detection_delay,
        "drawdown_reduction": eval_res.drawdown_reduction,
    }
    (args.out_dir / "gate_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out_dir / "report.md").write_text(
        "\n".join(
            [
                "# Risk Gate Simulation Report",
                "",
                f"- precision: {eval_res.precision:.4f}",
                f"- recall: {eval_res.recall:.4f}",
                f"- detection_delay: {eval_res.avg_detection_delay:.4f}",
                f"- drawdown_reduction: {eval_res.drawdown_reduction:.4f}",
                "",
                "![equity comparison](plots/equity_comparison.png)",
                "![gate timeline](plots/gate_timeline.png)",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
