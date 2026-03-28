from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constraints import ConstraintConfig, ConstraintLayer
from .schema import ACTIONS


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_behavior_cloning(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    best_ckpt = out_dir / "policy_bc_best.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for x, act, size in train_loader:
            x = x.to(device)
            act = act.to(device)
            size = size.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred["action_logits"], act) + 0.5 * F.cross_entropy(pred["size_logits"], size)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += float(loss.item()) * len(x)
            n += len(x)

        train_loss = total / max(n, 1)
        val_loss = evaluate_bc_loss(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), best_ckpt)

    (out_dir / "bc_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_ckpt


def evaluate_bc_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, act, size in loader:
            x = x.to(device)
            act = act.to(device)
            size = size.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred["action_logits"], act) + 0.5 * F.cross_entropy(pred["size_logits"], size)
            total += float(loss.item()) * len(x)
            n += len(x)
    return total / max(n, 1)


def train_actor_critic_replay(
    model: torch.nn.Module,
    replay_loader: DataLoader,
    epochs: int,
    lr: float,
    out_dir: Path,
    constraints: ConstraintConfig,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "value_head"):
        raise ValueError("Actor-critic stage requires model with value head")

    device = _device()
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    layer = ConstraintLayer(constraints)
    action_idx = {a: i for i, a in enumerate(ACTIONS)}
    ckpt = out_dir / "policy_rl_last.pt"

    for _ in range(epochs):
        model.train()
        t = 0.0
        for x, act, _size in replay_loader:
            x = x.to(device)
            act = act.to(device)
            pred = model(x)
            constrained_logits = layer.action_mask(
                now_s=t,
                est_participation=0.1,
                action_logits=pred["action_logits"],
                action_idx=action_idx,
            )
            logp = F.log_softmax(constrained_logits, dim=-1)
            chosen_logp = logp.gather(1, act.unsqueeze(1)).squeeze(1)

            mid = x[:, -1, 1]
            spread = x[:, -1, 0].abs() + 1e-4
            reward = torch.where(act == action_idx["cross"], -spread, mid * 0.001)
            value = pred["value"]
            adv = (reward - value).detach()

            actor_loss = -(chosen_logp * adv).mean()
            critic_loss = 0.5 * F.mse_loss(value, reward)
            entropy = -(torch.exp(logp) * logp).sum(dim=-1).mean()
            loss = actor_loss + critic_loss - 0.01 * entropy

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            actions = torch.argmax(constrained_logits.detach(), dim=-1)
            for a in actions.tolist():
                layer.record_action(ACTIONS[a], t)
            t += 0.01

    torch.save(model.state_dict(), ckpt)
    return ckpt
