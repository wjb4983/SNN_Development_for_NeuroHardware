from __future__ import annotations

from datetime import date
import unittest

from snn_bench.data_connectors.massive_client import MassiveClient


class _FakeMassiveClient(MassiveClient):
    def __init__(self) -> None:
        super().__init__(api_key="dummy", base_url="https://api.massive.com")
        self.calls: list[tuple[str, dict]] = []

    def _get(self, path: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        self.calls.append((path, params))
        cursor = params.get("cursor")
        if cursor is None:
            return {
                "results": [{"t": 1}, {"t": 2}],
                "next_url": "https://api.massive.com/v2/aggs/ticker/SPY/range/1/minute/2021-01-01/2021-01-02?cursor=abc&apiKey=dummy",
            }
        if cursor == "abc":
            return {"results": [{"t": 3}], "next_url": None}
        return {"results": []}


class MassiveClientPaginationTest(unittest.TestCase):
    def test_fetch_bars_paginates_until_next_url_exhausted(self) -> None:
        client = _FakeMassiveClient()
        rows = client.fetch_bars("SPY", start=date(2021, 1, 1), end=date(2021, 1, 2), timespan="minute")
        self.assertEqual([r["t"] for r in rows], [1, 2, 3])
        self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
