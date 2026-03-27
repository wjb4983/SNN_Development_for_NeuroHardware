from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class DataPaths(BaseModel):
    """Filesystem locations for local quant data caches."""

    snapshot_dir: Path = Field(default=Path("src/data"))
    backtest_root: Path = Field(default=Path("src/data/backtest_cache"))
    external_snapshot_dir: Path = Field(default=Path("../stoptions_analyzer/src/data"))


class SmokeConfig(BaseModel):
    enabled: bool = False
    sample_size: int = 512
    epochs: int = 1


class ModelSelectionConfig(BaseModel):
    name: str = "mlp"
    params: dict = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Top-level benchmark configuration."""

    ticker: str = "AAPL"
    timeframe: str = "1D"
    start_year: int | None = None
    end_year: int | None = None
    batch_size: int = 32
    epochs: int = 1
    lr: float = 1e-3
    seed: int = 7
    deterministic: bool = True
    massive_api_key_file: Path = Path("/etc/Massive/api-key")
    run_name: str = "default"
    model: ModelSelectionConfig = ModelSelectionConfig()
    smoke: SmokeConfig = SmokeConfig()
    data_paths: DataPaths = DataPaths()
