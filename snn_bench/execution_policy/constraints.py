from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ConstraintConfig:
    max_participation_rate: float = 0.2
    max_order_rate_per_sec: float = 20.0
    cancel_throttle_per_sec: float = 10.0


class ConstraintLayer:
    """Rule-based action filtering for safety + market microstructure controls."""

    def __init__(self, config: ConstraintConfig) -> None:
        self.cfg = config
        self.order_ts: deque[float] = deque()
        self.cancel_ts: deque[float] = deque()

    @staticmethod
    def _prune(q: deque[float], now_s: float) -> None:
        while q and (now_s - q[0]) > 1.0:
            q.popleft()

    def action_mask(
        self,
        now_s: float,
        est_participation: float,
        action_logits: torch.Tensor,
        action_idx: dict[str, int],
    ) -> torch.Tensor:
        self._prune(self.order_ts, now_s)
        self._prune(self.cancel_ts, now_s)
        mask = torch.zeros_like(action_logits)

        if est_participation >= self.cfg.max_participation_rate:
            for a in ["join_bid", "join_ask", "improve", "cross"]:
                mask[..., action_idx[a]] = -1e9

        if len(self.order_ts) >= self.cfg.max_order_rate_per_sec:
            for a in ["join_bid", "join_ask", "improve", "cross"]:
                mask[..., action_idx[a]] = -1e9

        if len(self.cancel_ts) >= self.cfg.cancel_throttle_per_sec:
            mask[..., action_idx["cancel"]] = -1e9

        return action_logits + mask

    def record_action(self, action_name: str, now_s: float) -> None:
        if action_name in {"join_bid", "join_ask", "improve", "cross", "cancel"}:
            self.order_ts.append(now_s)
        if action_name == "cancel":
            self.cancel_ts.append(now_s)
