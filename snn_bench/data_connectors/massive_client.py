from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


class MassiveClient:
    """Minimal HTTP client for Massive/Polygon-style REST endpoints."""

    def __init__(self, api_key: str, base_url: str = "https://api.massive.com") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = dict(params or {})
        query["apiKey"] = self.api_key
        url = f"{self.base_url}{path}?{urlencode(query)}"
        with urlopen(url, timeout=30) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    def fetch_daily_bars(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
        data = self._get(path, {"adjusted": "true", "sort": "asc", "limit": 50000})
        return data.get("results", [])

    def fetch_options_snapshots(self, ticker: str, as_of: date, max_pages: int = 30) -> list[dict[str, Any]]:
        path = f"/v3/snapshot/options/{ticker}"
        params: dict[str, Any] = {"limit": 250, "order": "asc", "sort": "expiration_date", "as_of": as_of.isoformat()}
        rows: list[dict[str, Any]] = []

        for _ in range(max_pages):
            data = self._get(path, params)
            rows.extend(data.get("results", []))
            next_url = data.get("next_url")
            if not next_url:
                break
            if next_url.startswith(self.base_url):
                path, _, query = next_url[len(self.base_url) :].partition("?")
                params = dict(item.split("=") for item in query.split("&") if "=" in item and not item.startswith("apiKey="))
            else:
                break

        return rows

    @staticmethod
    def save_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
