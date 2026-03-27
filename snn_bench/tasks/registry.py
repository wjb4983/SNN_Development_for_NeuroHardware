from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from snn_bench.feature_pipelines.basic_features import BasicFeaturePipeline
from snn_bench.tasks.performance_realism import (
    assert_no_lookahead,
    build_direction_distribution_targets,
    iv_skew_movement_labels,
    next_window_realized_vol,
)


TaskBuilder = Callable[[pd.DataFrame, dict], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class TaskSpec:
    path: Path
    raw: dict

    @property
    def name(self) -> str:
        return str(self.raw["task_name"])

    @property
    def task_type(self) -> str:
        return str(self.raw.get("type", "classification"))


class TaskRegistry:
    """Loads YAML task definitions and maps them to concrete target builders."""

    def __init__(self, task_dir: str | Path = "snn_bench/configs/tasks") -> None:
        self.task_dir = Path(task_dir)
        self._builders: dict[str, TaskBuilder] = {
            "direction_5m_distribution": self._build_direction_distribution,
            "direction_30m_distribution": self._build_direction_distribution,
            "realized_vol_30m": self._build_realized_vol,
            "options_iv_skew_movement": self._build_options_iv_skew,
            "next_bar_direction": self._build_next_bar_direction,
        }

    def available_tasks(self) -> list[str]:
        return sorted(self._builders)

    def resolve(self, task_name: str | None = None, task_config: str | Path | None = None) -> TaskSpec:
        if task_name and task_config:
            raise ValueError("Specify only one of task_name or task_config")

        if task_config:
            path = Path(task_config)
            if not path.exists():
                raise ValueError(f"Task config not found: {path}")
        elif task_name:
            path = self.task_dir / f"{task_name}.yaml"
            if not path.exists():
                matched = self._find_by_task_name(task_name)
                if not matched:
                    known = ", ".join(self.available_tasks())
                    raise ValueError(f"Unknown task '{task_name}'. Available tasks: {known}")
                path = matched
        else:
            path = self.task_dir / "direction_5m.yaml"

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = TaskSpec(path=path, raw=raw)
        if spec.name not in self._builders:
            known = ", ".join(self.available_tasks())
            raise ValueError(f"Task '{spec.name}' is not implemented. Available tasks: {known}")
        return spec

    def build_dataset(self, bars: pd.DataFrame, spec: TaskSpec) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._builders[spec.name](bars, spec.raw)
        if len(x) == 0 or len(y) == 0:
            raise ValueError(f"Task '{spec.name}' produced an empty dataset")
        if len(x) != len(y):
            raise ValueError("Feature/label rows are not aligned")
        return x.astype(np.float32), y

    def _find_by_task_name(self, task_name: str) -> Path | None:
        for p in sorted(self.task_dir.glob("*.yaml")):
            payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if payload.get("task_name") == task_name:
                return p
        return None

    def _build_next_bar_direction(self, bars: pd.DataFrame, _: dict) -> tuple[np.ndarray, np.ndarray]:
        return BasicFeaturePipeline().transform(bars)

    def _build_direction_distribution(self, bars: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
        horizon = int(cfg.get("horizon_minutes", 5))
        direction_cfg = (cfg.get("labeling") or {}).get("direction") or {}
        neutral_band_bps = float(direction_cfg.get("neutral_band_bps", 0.0))

        x_raw, _ = BasicFeaturePipeline().transform(bars)
        frame = bars.copy()
        frame.index = pd.RangeIndex(len(frame))
        targets, _ = build_direction_distribution_targets(
            close=frame["c"].astype(float),
            horizon=horizon,
            neutral_band_bps=neutral_band_bps,
            bins=int(((cfg.get("labeling") or {}).get("distribution") or {}).get("bins", 5)),
        )

        aligned = pd.DataFrame({"y": targets["direction_label"]}).dropna()
        keep = aligned.index.to_numpy(dtype=int)
        # Current model zoo is binary; map {-1,0,1} -> {0,1} where only positive direction is class 1.
        y = (aligned["y"] > 0).to_numpy(dtype=np.int64)
        return x_raw[keep], y

    def _build_realized_vol(self, bars: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
        target_cfg = (((cfg.get("labeling") or {}).get("target") or {}))
        window = int(target_cfg.get("window_minutes", cfg.get("horizon_minutes", 30)))

        x_raw, _ = BasicFeaturePipeline().transform(bars)
        frame = bars.copy()
        frame.index = pd.RangeIndex(len(frame))
        rv = next_window_realized_vol(frame["c"].astype(float), window=window)

        aligned = pd.DataFrame({"y": rv}).dropna()
        keep = aligned.index.to_numpy(dtype=int)
        y = aligned["y"].to_numpy(dtype=np.float32)
        return x_raw[keep], y

    def _build_options_iv_skew(self, bars: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
        horizon = int(cfg.get("horizon_minutes", 30))
        threshold = float((cfg.get("labeling") or {}).get("movement_threshold", 0.01))

        frame = bars.copy()
        frame.index = pd.RangeIndex(len(frame))
        x_raw, _ = BasicFeaturePipeline().transform(frame)
        skew = frame["c"].pct_change().rolling(5, min_periods=1).mean().fillna(0.0)
        labels = iv_skew_movement_labels(skew=skew, horizon=horizon, threshold=threshold)

        aligned = pd.DataFrame({"y": labels}).dropna()
        keep = aligned.index.to_numpy(dtype=int)
        y = (aligned["y"] > 0).to_numpy(dtype=np.int64)
        return x_raw[keep], y


def validate_task_model_compatibility(spec: TaskSpec, model_name: str) -> None:
    if spec.task_type == "regression":
        raise ValueError(
            f"Task '{spec.name}' is regression, but '{model_name}' currently supports binary classification outputs in train.py"
        )



def assert_aligned_not_empty(x: np.ndarray, y: np.ndarray) -> None:
    if len(x) == 0 or len(y) == 0:
        raise ValueError("empty dataset")
    if len(x) != len(y):
        raise ValueError("dataset rows are not aligned")

    features = pd.DataFrame(x)
    labels = pd.DataFrame({"y": y})
    features.index = pd.RangeIndex(len(features))
    labels.index = pd.RangeIndex(len(labels))
    assert_no_lookahead(features, labels)
