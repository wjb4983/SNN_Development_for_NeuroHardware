from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class StreamConfig:
    asset: str
    path: Path
    max_staleness_ms: int = 250


@dataclass(slots=True)
class FeatureConfig:
    event_type_vocab: list[str] = field(default_factory=lambda: ["trade", "quote", "book"])
    price_col: str = "price"
    size_col: str = "size"
    side_col: str = "side"


@dataclass(slots=True)
class TrainConfig:
    seed: int = 7
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    horizons_s: tuple[int, ...] = (1, 5)
    walk_forward_folds: int = 4
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    transaction_cost_bps: float = 1.0


@dataclass(slots=True)
class ModelConfig:
    hidden_dim: int = 32
    encoder_dim: int = 24
    fusion_dim: int = 64
    recurrent_decay: float = 0.9
    dropout: float = 0.1
    top_k_edges: int = 8


@dataclass(slots=True)
class DatasetConfig:
    target_asset: str
    streams: list[StreamConfig]
    feature: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass(slots=True)
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: Path = Path("artifacts/multistream")
