from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import build_feature_frame
from .schema import ACTIONS, SIZE_BUCKETS, EventLogParser


@dataclass(slots=True)
class SequencePayload:
    features: np.ndarray
    action_targets: np.ndarray
    size_targets: np.ndarray
    ts_ns: np.ndarray
    feature_names: list[str]


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, payload: SequencePayload, window: int = 64, stride: int = 1):
        if payload.features.shape[0] < window:
            raise ValueError(f"Not enough events ({payload.features.shape[0]}) for window={window}")
        self.payload = payload
        self.window = window
        self.indices = list(range(window - 1, payload.features.shape[0], stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = self.indices[idx]
        start = end - self.window + 1
        x = self.payload.features[start : end + 1]
        action = self.payload.action_targets[end]
        size = self.payload.size_targets[end]
        return (
            torch.from_numpy(x).float(),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(size, dtype=torch.long),
        )


def _fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def build_sequence_payload(
    events_path: Path | str,
    top_k: int = 5,
    lookback_events: int = 50,
    normalize: bool = True,
) -> SequencePayload:
    parser = EventLogParser(top_k=top_k)
    events = parser.load_frame(events_path)
    feat_df = build_feature_frame(events, top_k=top_k, lookback_events=lookback_events)

    if "action" not in events.columns:
        events["action"] = None
    if "size_bucket" not in events.columns:
        events["size_bucket"] = None

    action_map = {a: i for i, a in enumerate(ACTIONS)}
    size_map = {s: i for i, s in enumerate(SIZE_BUCKETS)}

    actions = events["action"].map(lambda x: action_map.get(str(x), action_map["hold"]))
    sizes = events["size_bucket"].map(lambda x: size_map.get(str(x), size_map["tiny"]))

    drop_cols = ["ts_ns"]
    x_df = feat_df.drop(columns=drop_cols)
    x = x_df.to_numpy(dtype=np.float32)
    if normalize:
        m, s = _fit_standardizer(x)
        x = (x - m) / s

    return SequencePayload(
        features=x,
        action_targets=actions.to_numpy(dtype=np.int64),
        size_targets=sizes.to_numpy(dtype=np.int64),
        ts_ns=feat_df["ts_ns"].to_numpy(dtype=np.int64),
        feature_names=list(x_df.columns),
    )


def save_preprocessed_payload(payload: SequencePayload, out_dir: Path | str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "sequence_payload.npz"
    np.savez_compressed(
        data_path,
        features=payload.features,
        action_targets=payload.action_targets,
        size_targets=payload.size_targets,
        ts_ns=payload.ts_ns,
    )
    meta = {"feature_names": payload.feature_names}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return data_path


def load_preprocessed_payload(data_path: Path | str, meta_path: Path | str) -> SequencePayload:
    npz = np.load(data_path)
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    return SequencePayload(
        features=np.asarray(npz["features"], dtype=np.float32),
        action_targets=np.asarray(npz["action_targets"], dtype=np.int64),
        size_targets=np.asarray(npz["size_targets"], dtype=np.int64),
        ts_ns=np.asarray(npz["ts_ns"], dtype=np.int64),
        feature_names=[str(x) for x in meta["feature_names"]],
    )
