from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from snn_bench.execution_policy.constraints import ConstraintConfig
from snn_bench.execution_policy.dataset import SequenceDataset, load_preprocessed_payload
from snn_bench.execution_policy.model import RecurrentSpikingPolicy
from snn_bench.execution_policy.schema import ACTIONS, SIZE_BUCKETS
from snn_bench.execution_policy.trainers import train_actor_critic_replay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optional RL fine-tuning for execution policy.")
    p.add_argument("--payload", required=True, type=Path)
    p.add_argument("--meta", required=True, type=Path)
    p.add_argument("--bc-checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-participation-rate", type=float, default=0.2)
    p.add_argument("--max-order-rate", type=float, default=20.0)
    p.add_argument("--cancel-throttle", type=float, default=10.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_preprocessed_payload(args.payload, args.meta)
    ds = SequenceDataset(payload, window=args.window, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = RecurrentSpikingPolicy(
        input_dim=payload.features.shape[1],
        hidden_dim=args.hidden_dim,
        num_actions=len(ACTIONS),
        num_sizes=len(SIZE_BUCKETS),
        value_head=True,
    )
    model.load_state_dict(torch.load(args.bc_checkpoint, map_location="cpu"))

    ckpt = train_actor_critic_replay(
        model=model,
        replay_loader=loader,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out_dir,
        constraints=ConstraintConfig(
            max_participation_rate=args.max_participation_rate,
            max_order_rate_per_sec=args.max_order_rate,
            cancel_throttle_per_sec=args.cancel_throttle,
        ),
    )
    print(f"rl checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
