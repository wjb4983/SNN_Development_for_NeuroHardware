from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

FEATURE_COLUMNS = [
    "spread_jumps",
    "depth_collapses",
    "realized_volatility",
    "trade_cancel_intensity",
    "feed_latency_gap",
]


class RollingNormalizer:
    def __init__(self, window: int = 256, eps: float = 1e-6) -> None:
        self.window = max(4, int(window))
        self.eps = float(eps)

    def transform(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = frame.copy()
        for col in columns:
            mean = out[col].rolling(self.window, min_periods=4).mean().shift(1)
            std = out[col].rolling(self.window, min_periods=4).std(ddof=0).shift(1)
            out[col] = (out[col] - mean) / (std + self.eps)
        return out.fillna(0.0)


class SentinelDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, regime_labels: np.ndarray, stress_labels: np.ndarray, seq_len: int = 32) -> None:
        self.features = features.astype(np.float32)
        self.regime_labels = regime_labels.astype(np.int64)
        self.stress_labels = stress_labels.astype(np.int64)
        self.seq_len = max(4, int(seq_len))
        if len(self.features) < self.seq_len:
            raise ValueError("not enough rows for configured seq_len")

    def __len__(self) -> int:
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = idx + self.seq_len
        x = torch.from_numpy(self.features[idx:end])
        regime = torch.tensor(self.regime_labels[end - 1], dtype=torch.long)
        stress = torch.tensor(self.stress_labels[end - 1], dtype=torch.long)
        return x, regime, stress


@dataclass(slots=True)
class SentinelDataModule:
    frame: pd.DataFrame
    feature_columns: list[str]
    regime_column: str
    stress_column: str
    seq_len: int = 32
    batch_size: int = 128
    split_ratio: float = 0.8

    def build(self) -> tuple[DataLoader, DataLoader, dict[str, np.ndarray]]:
        arr_x = self.frame[self.feature_columns].to_numpy(dtype=np.float32)
        arr_regime = self.frame[self.regime_column].to_numpy(dtype=np.int64)
        arr_stress = self.frame[self.stress_column].to_numpy(dtype=np.int64)
        cut = int(len(arr_x) * self.split_ratio)
        cut = min(max(cut, self.seq_len + 1), len(arr_x) - (self.seq_len + 1))

        train_ds = SentinelDataset(arr_x[:cut], arr_regime[:cut], arr_stress[:cut], seq_len=self.seq_len)
        val_ds = SentinelDataset(arr_x[cut - self.seq_len :], arr_regime[cut - self.seq_len :], arr_stress[cut - self.seq_len :], seq_len=self.seq_len)
        train = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        arrays = {
            "features": arr_x,
            "regime": arr_regime,
            "stress": arr_stress,
            "cut": np.array([cut], dtype=np.int64),
        }
        return train, val, arrays


def load_stream_csv(path: Path, *, normalization_window: int = 256) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"input stream missing required features: {missing}")

    if "regime_label" not in frame.columns:
        frame["regime_label"] = 0
    if "stress_label" not in frame.columns:
        frame["stress_label"] = (frame["regime_label"] > 0).astype(int)

    frame["regime_label"] = frame["regime_label"].astype(int)
    frame["stress_label"] = frame["stress_label"].astype(int)

    normalizer = RollingNormalizer(window=normalization_window)
    frame = normalizer.transform(frame, FEATURE_COLUMNS)
    return frame.reset_index(drop=True)
