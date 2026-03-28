from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PurgedWalkForwardSplit:
    n_splits: int = 4
    embargo_pct: float = 0.01
    purge_window: int = 64

    def split(self, n_samples: int):
        fold = n_samples // (self.n_splits + 1)
        indices = np.arange(n_samples)
        embargo = max(1, int(n_samples * self.embargo_pct))

        for i in range(self.n_splits):
            train_end = fold * (i + 1)
            valid_start = train_end + embargo
            valid_end = min(valid_start + fold, n_samples)

            train_idx = indices[: max(0, train_end - self.purge_window)]
            valid_idx = indices[valid_start:valid_end]
            if len(valid_idx) > 0:
                yield train_idx, valid_idx
