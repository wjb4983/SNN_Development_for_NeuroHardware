from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from snn_bench.sentinel.gate import GateConfig


@dataclass(slots=True)
class ThresholdConfig:
    warning_on: float
    warning_off: float
    block_on: float
    block_off: float
    target_fpr: float

    def to_gate_config(self, min_warning_steps: int = 3, min_block_steps: int = 2) -> GateConfig:
        return GateConfig(
            warning_on=self.warning_on,
            warning_off=self.warning_off,
            block_on=self.block_on,
            block_off=self.block_off,
            min_warning_steps=min_warning_steps,
            min_block_steps=min_block_steps,
        )


def tune_thresholds(scores: np.ndarray, stress_labels: np.ndarray, target_fpr: float = 0.05) -> ThresholdConfig:
    normal_scores = scores[stress_labels == 0]
    if normal_scores.size == 0:
        normal_scores = scores
    warning_on = float(np.quantile(normal_scores, 1.0 - target_fpr))
    block_on = float(np.quantile(normal_scores, 1.0 - target_fpr * 0.2))
    warning_off = float(max(0.0, warning_on * 0.85))
    block_off = float(max(warning_off, block_on * 0.9))
    return ThresholdConfig(
        warning_on=warning_on,
        warning_off=warning_off,
        block_on=block_on,
        block_off=block_off,
        target_fpr=float(target_fpr),
    )


def save_threshold_config(path: Path, cfg: ThresholdConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def load_threshold_config(path: Path) -> ThresholdConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ThresholdConfig(**payload)
