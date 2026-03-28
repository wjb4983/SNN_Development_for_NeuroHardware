from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from snn_bench.execution_policy.dataset import SequenceDataset, load_preprocessed_payload
from snn_bench.execution_policy.eval import evaluate_policy
from snn_bench.execution_policy.model import ANNBaselinePolicy, RecurrentSpikingPolicy
from snn_bench.execution_policy.schema import ACTIONS, SIZE_BUCKETS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate execution policy checkpoints.")
    p.add_argument("--payload", required=True, type=Path)
    p.add_argument("--meta", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--model", choices=["snn", "ann"], default="snn")
    p.add_argument("--hidden-dim", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_preprocessed_payload(args.payload, args.meta)
    ds = SequenceDataset(payload, window=args.window, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    if args.model == "snn":
        model = RecurrentSpikingPolicy(
            input_dim=payload.features.shape[1],
            hidden_dim=args.hidden_dim,
            num_actions=len(ACTIONS),
            num_sizes=len(SIZE_BUCKETS),
            value_head=True,
        )
    else:
        model = ANNBaselinePolicy(
            input_dim=payload.features.shape[1],
            hidden_dim=args.hidden_dim,
            num_actions=len(ACTIONS),
            num_sizes=len(SIZE_BUCKETS),
        )

    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=False)
    metrics = evaluate_policy(model, loader, out_dir=args.out_dir)
    print(metrics)


if __name__ == "__main__":
    main()
