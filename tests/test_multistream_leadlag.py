from __future__ import annotations

import pandas as pd
import torch

from snn_bench.multistream.data import causal_synchronize
from snn_bench.multistream.models import MultiStreamSNN


def test_causal_sync_no_lookahead() -> None:
    target = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01T00:00:01Z", "2025-01-01T00:00:02Z"], utc=True),
            "event_type": ["trade", "trade"],
            "price": [100.0, 101.0],
            "size": [1.0, 1.0],
            "side": ["buy", "sell"],
            "max_staleness_ms": [500, 500],
        }
    )
    src = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01T00:00:01.500Z", "2025-01-01T00:00:02.500Z"], utc=True),
            "event_type": ["trade", "trade"],
            "price": [50.0, 51.0],
            "size": [1.0, 1.0],
            "side": ["buy", "sell"],
            "max_staleness_ms": [600, 600],
        }
    )
    aligned = causal_synchronize({"TGT": target, "SRC": src}, target_asset="TGT")
    # First target event cannot use future source event at 1.500.
    assert pd.isna(aligned.loc[0, "SRC_price"])


def test_multistream_snn_forward() -> None:
    model = MultiStreamSNN(per_asset_dim=6, n_assets=4, encoder_dim=8, fusion_dim=10, decay=0.9, top_k_edges=4, n_horizons=2)
    x = torch.randn(16, 12, 4, 6)
    logits, couplings, fused = model(x)
    assert logits.shape == (16, 2)
    assert couplings.shape == (4, 4)
    assert fused.shape[1] == 12
