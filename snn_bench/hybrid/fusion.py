from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np


@dataclass
class FusionOutput:
    score: np.ndarray
    confidence: np.ndarray
    slow_component: np.ndarray
    fast_component: np.ndarray


class WeightedFusion:
    def __init__(self, slow_weight: float = 0.55, fast_weight: float = 0.45) -> None:
        total = slow_weight + fast_weight
        self.slow_weight = slow_weight / total
        self.fast_weight = fast_weight / total

    def blend(
        self,
        slow_score: np.ndarray,
        slow_conf: np.ndarray,
        fast_score: np.ndarray,
        fast_conf: np.ndarray,
    ) -> FusionOutput:
        w_slow = self.slow_weight * np.clip(slow_conf, 0.05, 1.0)
        w_fast = self.fast_weight * np.clip(fast_conf, 0.05, 1.0)
        w_total = w_slow + w_fast + 1e-8

        slow_component = (w_slow / w_total) * slow_score
        fast_component = (w_fast / w_total) * fast_score
        score = slow_component + fast_component
        confidence = np.clip((w_slow + w_fast) / (self.slow_weight + self.fast_weight), 0.0, 1.0)
        return FusionOutput(
            score=score.astype(np.float32),
            confidence=confidence.astype(np.float32),
            slow_component=slow_component.astype(np.float32),
            fast_component=fast_component.astype(np.float32),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"slow_weight": self.slow_weight, "fast_weight": self.fast_weight}, f)

    @classmethod
    def load(cls, path: Path) -> "WeightedFusion":
        with path.open("rb") as f:
            state = pickle.load(f)
        return cls(slow_weight=state["slow_weight"], fast_weight=state["fast_weight"])
