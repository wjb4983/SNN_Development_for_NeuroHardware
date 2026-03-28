from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from snn_bench.execution_policy.dataset import SequenceDataset, load_preprocessed_payload
from snn_bench.execution_policy.model import ANNBaselinePolicy, RecurrentSpikingPolicy
from snn_bench.execution_policy.schema import ACTIONS, SIZE_BUCKETS
from snn_bench.execution_policy.trainers import train_behavior_cloning


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train execution policy via behavior cloning.")
    p.add_argument("--payload", required=True, type=Path)
    p.add_argument("--meta", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model", choices=["snn", "ann"], default="snn")
    p.add_argument("--hidden-dim", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_preprocessed_payload(args.payload, args.meta)
    ds = SequenceDataset(payload, window=args.window, stride=args.stride)
    n_train = int(0.8 * len(ds))
    n_val = max(len(ds) - n_train, 1)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(7))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = payload.features.shape[1]
    if args.model == "snn":
        model = RecurrentSpikingPolicy(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_actions=len(ACTIONS),
            num_sizes=len(SIZE_BUCKETS),
            value_head=True,
        )
    else:
        model = ANNBaselinePolicy(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_actions=len(ACTIONS),
            num_sizes=len(SIZE_BUCKETS),
        )

    ckpt = train_behavior_cloning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out_dir,
    )
    print(f"behavior cloning checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
