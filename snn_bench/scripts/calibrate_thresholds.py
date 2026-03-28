from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from snn_bench.sentinel.calibration import save_threshold_config, tune_thresholds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate sentinel anomaly thresholds")
    p.add_argument("--scores", type=Path, required=True, help="NPY anomaly score array")
    p.add_argument("--stress-labels", type=Path, required=True, help="NPY binary stress labels")
    p.add_argument("--target-fpr", type=float, default=0.05)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scores = np.load(args.scores)
    stress = np.load(args.stress_labels)
    cfg = tune_thresholds(scores, stress, target_fpr=args.target_fpr)
    save_threshold_config(args.out, cfg)
    print(json.dumps(cfg.__dict__, indent=2))


if __name__ == "__main__":
    main()
