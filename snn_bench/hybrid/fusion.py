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
    snn_trust: np.ndarray | None = None
    regime_confidence: np.ndarray | None = None


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


class RegimeAwareFusion(WeightedFusion):
    def __init__(
        self,
        slow_weight: float = 0.55,
        fast_weight: float = 0.45,
        regime_conf_sensitivity: float = 0.7,
        low_confidence_threshold: float = 0.5,
        low_confidence_snn_scale: float = 0.75,
        regime_scale_by_state: list[float] | None = None,
    ) -> None:
        super().__init__(slow_weight=slow_weight, fast_weight=fast_weight)
        self.regime_conf_sensitivity = float(np.clip(regime_conf_sensitivity, 0.0, 1.0))
        self.low_confidence_threshold = float(np.clip(low_confidence_threshold, 0.0, 1.0))
        self.low_confidence_snn_scale = float(np.clip(low_confidence_snn_scale, 0.0, 1.0))
        self.regime_scale_by_state = (
            np.asarray(regime_scale_by_state, dtype=np.float32)
            if regime_scale_by_state is not None
            else None
        )

    @staticmethod
    def _entropy_confidence(regime_posteriors: np.ndarray) -> np.ndarray:
        clipped = np.clip(regime_posteriors, 1e-8, 1.0)
        ent = -np.sum(clipped * np.log(clipped), axis=1)
        n_states = max(2, regime_posteriors.shape[1])
        max_ent = np.log(float(n_states))
        return np.clip(1.0 - (ent / max(max_ent, 1e-8)), 0.0, 1.0)

    def blend(
        self,
        slow_score: np.ndarray,
        slow_conf: np.ndarray,
        fast_score: np.ndarray,
        fast_conf: np.ndarray,
        regime_posteriors: np.ndarray | None = None,
        calibration_weights: np.ndarray | None = None,
    ) -> FusionOutput:
        if regime_posteriors is None:
            return super().blend(slow_score, slow_conf, fast_score, fast_conf)

        if regime_posteriors.ndim != 2:
            raise ValueError("regime_posteriors must be a 2D array with shape (n_samples, n_states)")
        if regime_posteriors.shape[0] != slow_score.shape[0]:
            raise ValueError("regime_posteriors first dimension must match score length")

        regime_conf = self._entropy_confidence(regime_posteriors)
        snn_trust = np.clip(
            1.0 + self.regime_conf_sensitivity * (regime_conf - 0.5) * 2.0,
            0.25,
            2.0,
        )
        low_conf_mask = regime_conf < self.low_confidence_threshold
        snn_trust = np.where(low_conf_mask, snn_trust * self.low_confidence_snn_scale, snn_trust)

        if calibration_weights is not None:
            if calibration_weights.ndim != 2 or calibration_weights.shape != regime_posteriors.shape:
                raise ValueError("calibration_weights must have the same shape as regime_posteriors")
            regime_scale = np.sum(regime_posteriors * calibration_weights, axis=1)
        elif self.regime_scale_by_state is not None:
            if self.regime_scale_by_state.shape[0] != regime_posteriors.shape[1]:
                raise ValueError("regime_scale_by_state length must match number of regime states")
            regime_scale = np.sum(regime_posteriors * self.regime_scale_by_state[None, :], axis=1)
        else:
            regime_scale = np.ones_like(regime_conf, dtype=np.float32)

        adjusted_fast_conf = np.clip(fast_conf * snn_trust * regime_scale, 0.01, 1.5)
        adjusted_slow_conf = np.clip(slow_conf / np.maximum(snn_trust, 1e-8), 0.01, 1.5)
        out = super().blend(
            slow_score=slow_score,
            slow_conf=adjusted_slow_conf,
            fast_score=fast_score,
            fast_conf=adjusted_fast_conf,
        )
        out.snn_trust = snn_trust.astype(np.float32)
        out.regime_confidence = regime_conf.astype(np.float32)
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "fusion_type": "regime_aware",
                    "slow_weight": self.slow_weight,
                    "fast_weight": self.fast_weight,
                    "regime_conf_sensitivity": self.regime_conf_sensitivity,
                    "low_confidence_threshold": self.low_confidence_threshold,
                    "low_confidence_snn_scale": self.low_confidence_snn_scale,
                    "regime_scale_by_state": self.regime_scale_by_state,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "RegimeAwareFusion":
        with path.open("rb") as f:
            state = pickle.load(f)
        return cls(
            slow_weight=state["slow_weight"],
            fast_weight=state["fast_weight"],
            regime_conf_sensitivity=state.get("regime_conf_sensitivity", 0.7),
            low_confidence_threshold=state.get("low_confidence_threshold", 0.5),
            low_confidence_snn_scale=state.get("low_confidence_snn_scale", 0.75),
            regime_scale_by_state=state.get("regime_scale_by_state"),
        )
