from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class SnapshotCacheConnector:
    """Loads cached snapshot JSON files keyed by SAFE_TICKER."""

    def __init__(self, primary_dir: Path, fallback_dir: Path | None = None) -> None:
        self.primary_dir = Path(primary_dir)
        self.fallback_dir = Path(fallback_dir) if fallback_dir else None

    def _candidate_paths(self, safe_ticker: str) -> list[Path]:
        candidates = [self.primary_dir / f"{safe_ticker}.json"]
        if self.fallback_dir:
            candidates.append(self.fallback_dir / f"{safe_ticker}.json")
        return candidates

    def load_raw(self, safe_ticker: str) -> dict[str, Any]:
        for path in self._candidate_paths(safe_ticker):
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
        raise FileNotFoundError(f"No snapshot cache found for {safe_ticker}")

    def load_frame(self, safe_ticker: str) -> pd.DataFrame:
        raw = self.load_raw(safe_ticker)
        if isinstance(raw, list):
            return pd.DataFrame(raw)
        if "data" in raw and isinstance(raw["data"], list):
            return pd.DataFrame(raw["data"])
        return pd.json_normalize(raw)
