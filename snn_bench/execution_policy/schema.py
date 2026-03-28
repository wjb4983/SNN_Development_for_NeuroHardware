from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

ACTIONS = ["join_bid", "join_ask", "improve", "cross", "cancel", "hold"]
SIZE_BUCKETS = ["tiny", "small", "medium", "large"]


class EventType(str, Enum):
    MARKET_BOOK = "market_book"
    MARKET_TRADE = "market_trade"
    MARKET_CANCEL = "market_cancel"
    OWN_PLACEMENT = "own_placement"
    OWN_QUEUE = "own_queue"
    OWN_FILL = "own_fill"


@dataclass(slots=True)
class EventRecord:
    ts_ns: int
    event_type: EventType
    side: str | None
    price: float | None
    size: float | None
    level: int | None
    bid_prices: list[float]
    bid_sizes: list[float]
    ask_prices: list[float]
    ask_sizes: list[float]
    own_order_id: str | None
    queue_position: float | None
    fill_size: float | None
    action: str | None
    size_bucket: str | None


REQUIRED_COLUMNS = {
    "ts_ns",
    "event_type",
    "bid_prices",
    "bid_sizes",
    "ask_prices",
    "ask_sizes",
}


class EventLogParser:
    """Parser for csv/jsonl event logs with market + own-order events."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    @staticmethod
    def _parse_vec(raw: Any, k: int) -> list[float]:
        if raw is None:
            values: list[float] = []
        elif isinstance(raw, list):
            values = [float(v) for v in raw]
        elif isinstance(raw, str):
            stripped = raw.strip()
            if not stripped:
                values = []
            else:
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        values = [float(v) for v in parsed]
                    else:
                        values = [float(x) for x in stripped.split(",")]
                except json.JSONDecodeError:
                    values = [float(x) for x in stripped.split(",")]
        else:
            values = [float(raw)]
        if len(values) < k:
            values.extend([0.0] * (k - len(values)))
        return values[:k]

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            return None
        return float(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
            return None
        return int(value)

    def load_frame(self, path: Path | str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in {".jsonl", ".ndjson"}:
            df = pd.read_json(p, lines=True)
        else:
            raise ValueError(f"Unsupported log format: {p.suffix}")

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for col in [
            "side",
            "price",
            "size",
            "level",
            "own_order_id",
            "queue_position",
            "fill_size",
            "action",
            "size_bucket",
        ]:
            if col not in df.columns:
                df[col] = None

        df["event_type"] = df["event_type"].map(lambda x: EventType(str(x)))
        df["ts_ns"] = df["ts_ns"].astype("int64")
        df["bid_prices"] = df["bid_prices"].map(lambda x: self._parse_vec(x, self.top_k))
        df["bid_sizes"] = df["bid_sizes"].map(lambda x: self._parse_vec(x, self.top_k))
        df["ask_prices"] = df["ask_prices"].map(lambda x: self._parse_vec(x, self.top_k))
        df["ask_sizes"] = df["ask_sizes"].map(lambda x: self._parse_vec(x, self.top_k))

        for c in ["price", "size", "queue_position", "fill_size"]:
            df[c] = df[c].map(self._optional_float)
        df["level"] = df["level"].map(self._optional_int)

        df = df.sort_values("ts_ns").reset_index(drop=True)
        return df

    def to_records(self, frame: pd.DataFrame) -> list[EventRecord]:
        records: list[EventRecord] = []
        for row in frame.itertuples(index=False):
            records.append(
                EventRecord(
                    ts_ns=int(row.ts_ns),
                    event_type=row.event_type if isinstance(row.event_type, EventType) else EventType(str(row.event_type)),
                    side=row.side,
                    price=row.price,
                    size=row.size,
                    level=row.level,
                    bid_prices=list(row.bid_prices),
                    bid_sizes=list(row.bid_sizes),
                    ask_prices=list(row.ask_prices),
                    ask_sizes=list(row.ask_sizes),
                    own_order_id=row.own_order_id,
                    queue_position=row.queue_position,
                    fill_size=row.fill_size,
                    action=row.action,
                    size_bucket=row.size_bucket,
                )
            )
        return records
