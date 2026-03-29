from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import tempfile
import unittest

from snn_bench.configs.settings import BenchmarkConfig, DataPaths
from snn_bench.scripts.cache_market_data import _cache_is_current


class CacheMarketDataSkipTest(unittest.TestCase):
    def test_cache_is_current_when_index_and_options_match_today(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_paths = DataPaths(snapshot_dir=root / "snap", backtest_root=root / "backtest")
            cfg = BenchmarkConfig(ticker="SPY", timeframe="1Min", data_paths=data_paths)
            safe_ticker = "SPY"
            today = date(2026, 3, 29)

            (data_paths.snapshot_dir / f"{safe_ticker}.json").parent.mkdir(parents=True, exist_ok=True)
            (data_paths.snapshot_dir / f"{safe_ticker}.json").write_text("[]", encoding="utf-8")

            timeframe_dir = data_paths.backtest_root / safe_ticker / cfg.timeframe
            timeframe_dir.mkdir(parents=True, exist_ok=True)
            (timeframe_dir / "index.json").write_text(
                json.dumps(
                    {
                        "ticker": safe_ticker,
                        "timeframe": cfg.timeframe,
                        "rows_total": 123,
                        "stock_years": 5,
                        "updated_at": today.isoformat(),
                    }
                ),
                encoding="utf-8",
            )

            options_path = data_paths.snapshot_dir / "options" / f"{safe_ticker}_options.json"
            options_path.parent.mkdir(parents=True, exist_ok=True)
            options_path.write_text(
                json.dumps(
                    {
                        "ticker": safe_ticker,
                        "as_of": today.isoformat(),
                        "window_start": "2024-03-29",
                        "contracts": [],
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(_cache_is_current(cfg, safe_ticker, today=today, stock_years=5, option_years=2))


if __name__ == "__main__":
    unittest.main()
