from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedEmbargoWalkForward:
    """Walk-forward splitter with training purge and post-train embargo."""

    n_splits: int = 4
    purge_window: int = 10
    embargo_pct: float = 0.01

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if n_samples <= self.n_splits + 1:
            raise ValueError("n_samples too small for requested n_splits")

        fold_size = n_samples // (self.n_splits + 1)
        idx = np.arange(n_samples)
        embargo = max(1, int(round(n_samples * self.embargo_pct)))

        for split_id in range(self.n_splits):
            train_end = (split_id + 1) * fold_size
            purged_train_end = max(0, train_end - self.purge_window)
            valid_start = min(n_samples, train_end + embargo)
            valid_end = min(n_samples, valid_start + fold_size)

            train_idx = idx[:purged_train_end]
            valid_idx = idx[valid_start:valid_end]
            if len(train_idx) == 0 or len(valid_idx) == 0:
                continue
            yield train_idx, valid_idx
