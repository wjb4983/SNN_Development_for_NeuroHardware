from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from snn_bench.configs.settings import BenchmarkConfig
from snn_bench.data_connectors.massive_client import MassiveClient
from snn_bench.utils.secrets import load_massive_api_key


MAIN_INDEX_TICKERS: tuple[str, ...] = (
    "SPY",  # S&P 500 proxy
    "QQQ",  # Nasdaq-100 proxy
    "DIA",  # Dow Jones proxy
    "IWM",  # Russell 2000 proxy
)

TOP_100_MARKET_CAP_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK.B", "AVGO",
    "LLY", "WMT", "JPM", "V", "XOM", "UNH", "MA", "COST", "ORCL", "HD",
    "PG", "JNJ", "BAC", "NFLX", "ABBV", "CRM", "KO", "CVX", "MRK", "AMD",
    "PEP", "TMO", "MCD", "ACN", "CSCO", "ADBE", "LIN", "WFC", "DIS", "ABT",
    "DHR", "QCOM", "TXN", "VZ", "INTU", "PM", "CAT", "NKE", "RTX", "PFE",
    "NOW", "IBM", "INTC", "AMGN", "SPGI", "UNP", "COP", "GS", "MS", "LOW",
    "BX", "HON", "PLD", "BLK", "NEE", "SYK", "AMAT", "GE", "MDT", "LMT",
    "DE", "TJX", "BMY", "CB", "AXP", "ISRG", "SCHW", "AMT", "C", "ADP",
    "MO", "BKNG", "TMUS", "MMC", "GILD", "SO", "ETN", "PANW", "VRTX", "CI",
    "ADI", "MDLZ", "REGN", "DUK", "ELV", "PGR", "ZTS", "KLAC", "CME", "SNPS",
)


def _api_ticker(ticker: str) -> str:
    return ticker.upper()


def _safe_ticker(ticker: str) -> str:
    return "".join(ch for ch in ticker.upper() if ch.isalnum() or ch in ("-", "_", "."))


def _to_npz_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "t": np.array([r.get("t", 0) for r in rows]),
        "o": np.array([r.get("o", 0.0) for r in rows], dtype=float),
        "h": np.array([r.get("h", 0.0) for r in rows], dtype=float),
        "l": np.array([r.get("l", 0.0) for r in rows], dtype=float),
        "c": np.array([r.get("c", 0.0) for r in rows], dtype=float),
        "v": np.array([r.get("v", 0.0) for r in rows], dtype=float),
        "n": np.array([r.get("n", 0.0) for r in rows], dtype=float),
    }


def _cache_single_ticker(
    cfg: BenchmarkConfig,
    ticker: str,
    stock_years: int,
    option_years: int,
    client: MassiveClient,
) -> dict:
    api_ticker = _api_ticker(ticker)
    safe_ticker = _safe_ticker(ticker)
    today = date.today()
    stock_start = today - timedelta(days=365 * stock_years)
    option_start = today - timedelta(days=365 * option_years)

    stock_rows = client.fetch_daily_bars(api_ticker, stock_start, today)
    snapshot_path = cfg.data_paths.snapshot_dir / f"{safe_ticker}.json"
    MassiveClient.save_json(snapshot_path, stock_rows)

    timeframe_dir = cfg.data_paths.backtest_root / safe_ticker / cfg.timeframe
    timeframe_dir.mkdir(parents=True, exist_ok=True)

    by_year: dict[int, list[dict]] = {}
    for row in stock_rows:
        ts_ms = int(row.get("t", 0))
        year = date.fromtimestamp(ts_ms / 1000).year if ts_ms else today.year
        by_year.setdefault(year, []).append(row)

    years = sorted(by_year)
    for year in years:
        arrays = _to_npz_arrays(by_year[year])
        np.savez(timeframe_dir / f"{safe_ticker}_{cfg.timeframe}_{year}.npz", **arrays)

    index_payload = {
        "ticker": safe_ticker,
        "timeframe": cfg.timeframe,
        "years": years,
        "rows_total": len(stock_rows),
        "updated_at": today.isoformat(),
    }
    (timeframe_dir / "index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    option_rows = client.fetch_options_snapshots(api_ticker, as_of=today)
    filtered_options = [r for r in option_rows if r.get("details", {}).get("expiration_date", "") >= option_start.isoformat()]
    options_path = cfg.data_paths.snapshot_dir / "options" / f"{safe_ticker}_options.json"
    MassiveClient.save_json(
        options_path,
        {
            "ticker": safe_ticker,
            "as_of": today.isoformat(),
            "window_start": option_start.isoformat(),
            "contracts": filtered_options,
        },
    )

    return {
        "ticker": safe_ticker,
        "stock_rows": len(stock_rows),
        "stock_years": stock_years,
        "option_rows": len(filtered_options),
        "option_years": option_years,
        "snapshot_path": str(snapshot_path),
        "backtest_index": str(timeframe_dir / "index.json"),
        "options_path": str(options_path),
    }


def _resolve_tickers(universe: str, ticker: str | None) -> list[str]:
    if universe == "single":
        if not ticker:
            raise ValueError("--ticker is required when --universe=single")
        return [_api_ticker(ticker)]
    if universe == "indices":
        return list(MAIN_INDEX_TICKERS)
    if universe == "top100":
        return list(TOP_100_MARKET_CAP_TICKERS)
    if universe == "all":
        return list(dict.fromkeys([*MAIN_INDEX_TICKERS, *TOP_100_MARKET_CAP_TICKERS]))
    raise ValueError(f"Unsupported universe: {universe}")


def cache_market_data(cfg: BenchmarkConfig, stock_years: int, option_years: int, universe: str, ticker: str | None) -> dict:
    api_key = load_massive_api_key(cfg.massive_api_key_file)
    client = MassiveClient(api_key=api_key)
    tickers = _resolve_tickers(universe=universe, ticker=ticker)
    summaries: list[dict] = []
    for symbol in tickers:
        symbol_cfg = BenchmarkConfig(
            ticker=symbol,
            timeframe=cfg.timeframe,
            massive_api_key_file=cfg.massive_api_key_file,
            data_paths=cfg.data_paths,
        )
        summaries.append(
            _cache_single_ticker(
                cfg=symbol_cfg,
                ticker=symbol,
                stock_years=stock_years,
                option_years=option_years,
                client=client,
            )
        )
    return {
        "universe": universe,
        "tickers_requested": len(tickers),
        "tickers_cached": len(summaries),
        "summaries": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and cache stock + option data")
    parser.add_argument("--ticker", default=None, help="Ticker used when --universe=single")
    parser.add_argument(
        "--universe",
        default="all",
        choices=("single", "indices", "top100", "all"),
        help="Preset ticker universe to cache",
    )
    parser.add_argument("--timeframe", default="1D")
    parser.add_argument("--stock-years", type=int, default=5)
    parser.add_argument("--option-years", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(ticker=args.ticker or "AAPL", timeframe=args.timeframe)
    summary = cache_market_data(
        cfg,
        stock_years=args.stock_years,
        option_years=args.option_years,
        universe=args.universe,
        ticker=args.ticker,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
