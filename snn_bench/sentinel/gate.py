from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class GateState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    BLOCK = "BLOCK"


@dataclass(slots=True)
class GateConfig:
    warning_on: float
    warning_off: float
    block_on: float
    block_off: float
    min_warning_steps: int = 3
    min_block_steps: int = 2


class RiskGate:
    def __init__(self, cfg: GateConfig) -> None:
        self.cfg = cfg
        self.state = GateState.NORMAL
        self._warning_count = 0
        self._block_count = 0

    def update(self, anomaly_score: float) -> GateState:
        if anomaly_score >= self.cfg.block_on:
            self._block_count += 1
        else:
            self._block_count = 0

        if anomaly_score >= self.cfg.warning_on:
            self._warning_count += 1
        else:
            self._warning_count = 0

        if self.state != GateState.BLOCK and self._block_count >= self.cfg.min_block_steps:
            self.state = GateState.BLOCK
            return self.state

        if self.state == GateState.BLOCK:
            if anomaly_score <= self.cfg.block_off:
                self.state = GateState.WARNING if anomaly_score >= self.cfg.warning_off else GateState.NORMAL
            return self.state

        if self.state == GateState.NORMAL and self._warning_count >= self.cfg.min_warning_steps:
            self.state = GateState.WARNING
        elif self.state == GateState.WARNING and anomaly_score <= self.cfg.warning_off:
            self.state = GateState.NORMAL
        return self.state

    def run(self, anomaly_scores: np.ndarray) -> list[GateState]:
        states: list[GateState] = []
        for score in anomaly_scores:
            states.append(self.update(float(score)))
        return states
