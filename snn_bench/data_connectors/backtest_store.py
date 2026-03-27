from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class BacktestBarStoreConnector:
    """Reads backtest index metadata and yearly NPZ bar stores."""

    def __init__(self, backtest_root: Path) -> None:
        self.backtest_root = Path(backtest_root)

    def _timeframe_dir(self, safe_ticker: str, timeframe: str) -> Path:
        return self.backtest_root / safe_ticker / timeframe

    def load_index(self, safe_ticker: str, timeframe: str) -> dict:
        idx_file = self._timeframe_dir(safe_ticker, timeframe) / "index.json"
        with idx_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_year(self, safe_ticker: str, timeframe: str, year: int) -> pd.DataFrame:
        tf_dir = self._timeframe_dir(safe_ticker, timeframe)
        npz_path = tf_dir / f"{safe_ticker}_{timeframe}_{year}.npz"
        blob = np.load(npz_path)
        out = pd.DataFrame(
            {
                "t": blob["t"],
                "o": blob["o"],
                "h": blob["h"],
                "l": blob["l"],
                "c": blob["c"],
                "v": blob["v"],
                "n": blob["n"],
            }
        )
        return out
