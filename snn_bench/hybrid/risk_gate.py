from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    BLOCK = "BLOCK"


@dataclass
class RiskGateOutput:
    state: np.ndarray
    leverage: np.ndarray
    anomaly_score: np.ndarray
    regime_score: np.ndarray


class RiskGate:
    def __init__(
        self,
        warning_vol: float = 0.0014,
        block_vol: float = 0.0022,
        warning_anomaly: float = 2.2,
        block_anomaly: float = 3.2,
        warning_leverage: float = 0.5,
    ) -> None:
        self.warning_vol = warning_vol
        self.block_vol = block_vol
        self.warning_anomaly = warning_anomaly
        self.block_anomaly = block_anomaly
        self.warning_leverage = warning_leverage

    def evaluate(self, market: pd.DataFrame, fast_features: pd.DataFrame) -> RiskGateOutput:
        vol = market["ret_1"].rolling(30, min_periods=2).std().fillna(0.0)
        flow = fast_features["trade_flow"]
        z = (flow - flow.rolling(90, min_periods=10).mean()) / (flow.rolling(90, min_periods=10).std() + 1e-8)
        anomaly = np.abs(z.fillna(0.0))

        state = np.full(len(market), RiskState.NORMAL.value, dtype=object)
        leverage = np.ones(len(market), dtype=np.float32)

        warning = (vol >= self.warning_vol) | (anomaly >= self.warning_anomaly)
        block = (vol >= self.block_vol) | (anomaly >= self.block_anomaly)
        state[warning] = RiskState.WARNING.value
        leverage[warning] = self.warning_leverage
        state[block] = RiskState.BLOCK.value
        leverage[block] = 0.0

        return RiskGateOutput(
            state=state,
            leverage=leverage,
            anomaly_score=anomaly.values.astype(np.float32),
            regime_score=vol.values.astype(np.float32),
        )
