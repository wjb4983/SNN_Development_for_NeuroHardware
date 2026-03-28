from __future__ import annotations

import argparse
import json
from pathlib import Path

from snn_bench.multistream.train import run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multistream SNN/ANN lead-lag models")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model-type", choices=["snn", "ann"], default="snn")
    p.add_argument("--ann-mode", choices=["lstm", "tcn"], default="lstm")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args.config, model_type=args.model_type, ann_mode=args.ann_mode)
    print(json.dumps({"ablation_tag": result["ablation_tag"], "folds": len(result["fold_reports"])}, indent=2))


if __name__ == "__main__":
    main()
