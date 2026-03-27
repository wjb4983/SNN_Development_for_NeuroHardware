from __future__ import annotations

import numpy as np
import pandas as pd


class BasicFeaturePipeline:
    """Converts OHLCV bars into simple normalized features + target."""

    feature_columns = ["ret_1", "hl_spread", "oc_spread", "vol_z"]

    def transform(self, bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        bars = bars.copy()
        bars["ret_1"] = bars["c"].pct_change().fillna(0.0)
        bars["hl_spread"] = (bars["h"] - bars["l"]) / bars["o"].replace(0, np.nan)
        bars["oc_spread"] = (bars["c"] - bars["o"]) / bars["o"].replace(0, np.nan)
        vol = bars["v"].astype(float)
        bars["vol_z"] = (vol - vol.mean()) / (vol.std() + 1e-8)
        bars = bars.fillna(0.0)

        y = (bars["ret_1"].shift(-1) > 0).astype(np.float32).fillna(0.0).to_numpy()
        x = bars[self.feature_columns].astype(np.float32).to_numpy()
        return x, y
