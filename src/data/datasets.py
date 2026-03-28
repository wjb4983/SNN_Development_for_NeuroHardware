from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class LOBSequenceDataset(Dataset):
    """Windowed sequence dataset for LOB event features."""

    features: np.ndarray
    labels: np.ndarray
    window: int
    stride: int = 1

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be [T, C]")
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels length mismatch")
        self.indices = np.arange(self.window, len(self.features), self.stride)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        end = self.indices[idx]
        start = end - self.window
        x = self.features[start:end]
        y = self.labels[end]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
