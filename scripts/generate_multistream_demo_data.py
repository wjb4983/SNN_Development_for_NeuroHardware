from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gen_asset(name: str, base: float, n: int, freq_ms: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq=f"{freq_ms}ms", tz="UTC")
    rw = np.cumsum(rng.normal(0.0, 0.0015, size=n))
    price = base * (1 + rw)
    size = rng.lognormal(mean=4.0, sigma=0.5, size=n)
    side = rng.choice(["buy", "sell"], p=[0.52, 0.48], size=n)
    evt = rng.choice(["trade", "quote", "book"], p=[0.6, 0.3, 0.1], size=n)
    return pd.DataFrame({"timestamp": ts, "event_type": evt, "price": price, "size": size, "side": side})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("examples/multistream_data"))
    p.add_argument("--rows", type=int, default=5000)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("ES_F", 5000.0, 1, 11),
        ("SPY", 500.0, 2, 13),
        ("ZN_F", 112.0, 3, 17),
        ("DXY", 102.0, 4, 23),
    ]
    for name, base, seed, freq in specs:
        df = gen_asset(name, base, args.rows, freq_ms=freq, seed=seed)
        df.to_csv(args.out_dir / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
