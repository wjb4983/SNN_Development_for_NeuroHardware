from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class BinaryDirectionDataset(Dataset):
    """Minimal dataset wrapper for next-bar direction labels."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
