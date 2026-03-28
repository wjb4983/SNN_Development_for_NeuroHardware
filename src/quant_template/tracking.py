from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    run_id: str
    model_name: str
    split: int
    metrics: dict[str, float]
    params: dict[str, Any]
    timestamp_utc: str


class LocalTracker:
    """Filesystem tracker storing per-run JSON and append-only CSV summaries."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root_dir / "experiment_log.csv"

    def log(self, run_id: str, model_name: str, split: int, metrics: dict[str, float], params: dict[str, Any]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        record = ExperimentRecord(
            run_id=run_id,
            model_name=model_name,
            split=split,
            metrics=metrics,
            params=params,
            timestamp_utc=ts,
        )
        run_dir = self.root_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / f"split_{split}.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2)

        row = {
            "run_id": run_id,
            "timestamp_utc": ts,
            "model_name": model_name,
            "split": split,
            **{f"metric_{k}": v for k, v in metrics.items()},
        }
        self._append_csv(row)

    def _append_csv(self, row: dict[str, Any]) -> None:
        existing_header = []
        if self.csv_path.exists():
            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])

        fieldnames = list(dict.fromkeys(existing_header + list(row.keys())))
        rows = []
        if self.csv_path.exists():
            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        rows.append({k: row.get(k, "") for k in fieldnames})
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
