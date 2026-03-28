from __future__ import annotations

import argparse
from pathlib import Path

from snn_bench.execution_policy.dataset import build_sequence_payload, save_preprocessed_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess execution event logs into sequence payload.")
    p.add_argument("--events", required=True, type=Path, help="Input event log CSV/JSONL")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory for payload")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--lookback-events", type=int, default=50)
    p.add_argument("--no-normalize", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_sequence_payload(
        args.events,
        top_k=args.top_k,
        lookback_events=args.lookback_events,
        normalize=not args.no_normalize,
    )
    path = save_preprocessed_payload(payload, args.out_dir)
    print(f"saved payload: {path}")


if __name__ == "__main__":
    main()
