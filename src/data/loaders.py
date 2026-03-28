from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_GENERIC_COLUMNS = {
    "timestamp",
    "bid_price_1",
    "ask_price_1",
    "bid_size_1",
    "ask_size_1",
}


@dataclass
class FI2010Loader:
    """Load FI-2010 benchmark files into a canonical event dataframe."""

    path: Path

    def load(self) -> pd.DataFrame:
        path = Path(self.path)
        if path.suffix.lower() in {".csv", ".txt"}:
            raw = pd.read_csv(path, header=None)
        else:
            raw = pd.DataFrame(np.loadtxt(path))

        if raw.shape[1] < 40:
            raise ValueError("FI-2010 file should include at least 40 LOB feature columns.")

        out = pd.DataFrame()
        out["timestamp"] = np.arange(len(raw), dtype=np.int64)
        for level in range(1, 6):
            bid_px_col = (level - 1) * 4
            bid_sz_col = bid_px_col + 1
            ask_px_col = bid_px_col + 2
            ask_sz_col = bid_px_col + 3
            out[f"bid_price_{level}"] = raw.iloc[:, bid_px_col]
            out[f"bid_size_{level}"] = raw.iloc[:, bid_sz_col]
            out[f"ask_price_{level}"] = raw.iloc[:, ask_px_col]
            out[f"ask_size_{level}"] = raw.iloc[:, ask_sz_col]

        label_col = raw.shape[1] - 1
        out["label"] = raw.iloc[:, label_col].map({1: 0, 2: 1, 3: 2}).fillna(1).astype(int)
        return out


@dataclass
class GenericEventCSVLoader:
    """Load a generic event CSV and enforce required schema."""

    path: Path
    parse_dates: bool = True

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        missing = REQUIRED_GENERIC_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required generic event columns: {sorted(missing)}")

        if self.parse_dates:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if df["timestamp"].isna().all():
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)

        numeric_candidates: Iterable[str] = [
            c for c in df.columns if c != "timestamp" and ("price" in c or "size" in c or c in {"trade_qty", "cancel_qty"})
        ]
        for col in numeric_candidates:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df.sort_values("timestamp").reset_index(drop=True)


def load_lob_dataframe(source: str, path: str | Path) -> pd.DataFrame:
    source_l = source.lower()
    if source_l == "fi2010":
        return FI2010Loader(Path(path)).load()
    if source_l in {"generic", "csv"}:
        return GenericEventCSVLoader(Path(path)).load()
    raise ValueError(f"Unsupported data source {source}. Expected one of: fi2010, generic")
