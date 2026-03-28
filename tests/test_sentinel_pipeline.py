from __future__ import annotations

import numpy as np

from snn_bench.sentinel.calibration import tune_thresholds
from snn_bench.sentinel.gate import GateState, RiskGate


def test_thresholds_ordering() -> None:
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.2, 0.05, 500), rng.normal(0.8, 0.1, 100)])
    stress = np.array([0] * 500 + [1] * 100, dtype=np.int64)
    cfg = tune_thresholds(scores, stress, target_fpr=0.05)
    assert cfg.warning_off <= cfg.warning_on <= cfg.block_on
    assert cfg.block_off <= cfg.block_on


def test_gate_hysteresis_transitions() -> None:
    gate = RiskGate(
        cfg=tune_thresholds(
            scores=np.array([0.1, 0.2, 0.3, 0.9, 1.1, 0.2]),
            stress_labels=np.array([0, 0, 0, 1, 1, 0]),
            target_fpr=0.5,
        ).to_gate_config(min_warning_steps=1, min_block_steps=1)
    )
    states = gate.run(np.array([0.1, 0.8, 1.2, 0.9, 0.2]))
    assert GateState.WARNING in states
    assert GateState.BLOCK in states
