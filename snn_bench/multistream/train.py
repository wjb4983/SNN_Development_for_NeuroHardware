from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from snn_bench.multistream.data import causal_synchronize, load_event_streams, make_multi_horizon_targets
from snn_bench.multistream.eval import directional_metrics, latency_adjusted_throughput, pnl_proxy
from snn_bench.multistream.explain import coupling_edge_importance, temporal_attribution_snapshots
from snn_bench.multistream.features import build_feature_matrix, drop_nan_targets
from snn_bench.multistream.models import MultiStreamANNBaseline, MultiStreamSNN, balanced_bce_loss, class_weights_from_targets
from snn_bench.multistream.schema import DatasetConfig, ExperimentConfig, FeatureConfig, ModelConfig, StreamConfig, TrainConfig


def _to_config(raw: dict[str, Any]) -> ExperimentConfig:
    streams = [StreamConfig(asset=s["asset"], path=Path(s["path"]), max_staleness_ms=int(s.get("max_staleness_ms", 250))) for s in raw["dataset"]["streams"]]
    dataset = DatasetConfig(target_asset=raw["dataset"]["target_asset"], streams=streams, feature=FeatureConfig(**raw["dataset"].get("feature", {})))
    return ExperimentConfig(
        dataset=dataset,
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        output_dir=Path(raw.get("output_dir", "artifacts/multistream")),
    )


def _window_tensor(x: np.ndarray, y: np.ndarray, seq_len: int = 32) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(x)):
        xs.append(x[i - seq_len : i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _adaptive_seq_len(n_rows: int, preferred: int = 32) -> int:
    if n_rows <= 1:
        return 1
    # Keep context reasonably long when possible, but never longer than the data allows.
    return max(1, min(preferred, n_rows - 1))


def _walk_forward_indices(n: int, folds: int, train_ratio: float, val_ratio: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if n < 3 or folds <= 0:
        return []

    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("invalid split ratios: require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1")

    min_fold_size = 3
    while True:
        tr = int(min_fold_size * train_ratio)
        va = int(min_fold_size * val_ratio)
        te = min_fold_size - tr - va
        if min(tr, va, te) > 0:
            break
        min_fold_size += 1

    max_folds = max(1, n // min_fold_size)
    effective_folds = max(1, min(folds, max_folds))
    fold_size = max(1, n // effective_folds)
    splits = []
    for f in range(effective_folds):
        start = f * fold_size
        end = n if f == effective_folds - 1 else (f + 1) * fold_size
        fold_idx = np.arange(start, end)
        tr_end = start + int(len(fold_idx) * train_ratio)
        va_end = tr_end + int(len(fold_idx) * val_ratio)
        train_idx = np.arange(start, tr_end)
        val_idx = np.arange(tr_end, va_end)
        test_idx = np.arange(va_end, end)
        if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
            continue
        splits.append((train_idx, val_idx, test_idx))
    return splits


def _fallback_split_indices(n: int, train_ratio: float, val_ratio: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if n < 3:
        return []
    tr_end = int(n * train_ratio)
    va_end = tr_end + int(n * val_ratio)
    tr_end = min(max(1, tr_end), n - 2)
    va_end = min(max(tr_end + 1, va_end), n - 1)
    train_idx = np.arange(0, tr_end)
    val_idx = np.arange(tr_end, va_end)
    test_idx = np.arange(va_end, n)
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        return []
    return [(train_idx, val_idx, test_idx)]


def _train_one(
    model: torch.nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    train_cfg: TrainConfig,
    is_snn: bool,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    class_w = torch.tensor(class_weights_from_targets(y_train), device=device)

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    val_x = torch.tensor(x_val, device=device)
    val_y = torch.tensor(y_val, device=device)
    loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    best_val = float("inf")
    best_state = None
    hist = {"train_loss": [], "val_loss": [], "class_weights": class_w.detach().cpu().tolist()}

    for _ in range(train_cfg.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            if is_snn:
                logits, _, _ = model(xb)
            else:
                logits = model(xb)
            loss = balanced_bce_loss(logits, yb, class_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            if is_snn:
                val_logits, _, _ = model(val_x)
            else:
                val_logits = model(val_x)
            val_loss = balanced_bce_loss(val_logits, val_y, class_w).item()
        hist["train_loss"].append(float(np.mean(losses) if losses else 0.0))
        hist["val_loss"].append(float(val_loss))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def run_experiment(config_path: Path, *, model_type: str = "snn", ann_mode: str = "lstm") -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = _to_config(raw)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    streams = load_event_streams(cfg.dataset)
    aligned = causal_synchronize(streams, cfg.dataset.target_asset)
    aligned = make_multi_horizon_targets(aligned, cfg.train.horizons_s)
    y_cols = [f"y_{hz}s_direction" for hz in cfg.train.horizons_s]
    x_flat, event_vocab, feature_names = build_feature_matrix(aligned, cfg.dataset.feature)
    y = aligned[y_cols].to_numpy(dtype=np.float32)
    x_flat, y = drop_nan_targets(x_flat, y)
    if len(x_flat) == 0:
        max_hz = max(cfg.train.horizons_s) if cfg.train.horizons_s else 0
        raise ValueError(
            "no valid multistream training rows after target construction; "
            f"all rows were dropped (max_horizon_s={max_hz}). "
            "Check stream CSV timestamps cover enough forward time for label horizons "
            "and ensure config dataset.streams points to event CSV files."
        )

    n_assets = len(cfg.dataset.streams)
    per_asset_dim = x_flat.shape[1] // n_assets
    x = x_flat[:, : per_asset_dim * n_assets].reshape(len(x_flat), n_assets, per_asset_dim)
    seq_len = _adaptive_seq_len(len(x), preferred=32)
    x_seq, y_seq = _window_tensor(x, y, seq_len=seq_len)

    splits = _walk_forward_indices(len(x_seq), cfg.train.walk_forward_folds, cfg.train.train_ratio, cfg.train.val_ratio)
    if not splits:
        splits = _fallback_split_indices(len(x_seq), cfg.train.train_ratio, cfg.train.val_ratio)
    if not splits:
        raise ValueError(
            f"walk-forward split produced no folds after fallback "
            f"(rows={len(x)}, windowed_rows={len(x_seq)}, seq_len={seq_len}); increase data size"
        )

    fold_reports = []
    for fold_id, (tr, va, te) in enumerate(splits):
        if model_type == "snn":
            model = MultiStreamSNN(
                per_asset_dim=per_asset_dim,
                n_assets=n_assets,
                encoder_dim=cfg.model.encoder_dim,
                fusion_dim=cfg.model.fusion_dim,
                decay=cfg.model.recurrent_decay,
                top_k_edges=cfg.model.top_k_edges,
                n_horizons=len(cfg.train.horizons_s),
            )
            is_snn = True
        else:
            model = MultiStreamANNBaseline(
                per_asset_dim=per_asset_dim,
                n_assets=n_assets,
                hidden_dim=cfg.model.hidden_dim,
                n_horizons=len(cfg.train.horizons_s),
                mode=ann_mode,
            )
            is_snn = False

        hist = _train_one(model, x_seq[tr], y_seq[tr], x_seq[va], y_seq[va], cfg.train, is_snn=is_snn)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        xt = torch.tensor(x_seq[te], device=device)
        yt = y_seq[te]
        latencies = []
        model.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            if is_snn:
                logits, coupling, fused = model(xt)
            else:
                logits = model(xt)
                coupling = torch.zeros((n_assets, n_assets), device=device)
                fused = xt.reshape(xt.shape[0], xt.shape[1], -1).requires_grad_(True)
            dt = (time.perf_counter() - t0) * 1e3 / max(1, len(xt))
            latencies = [dt] * len(xt)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

        horizon_metrics = {}
        for i, hz in enumerate(cfg.train.horizons_s):
            dm = directional_metrics(yt[:, i].astype(int), probs[:, i])
            pm = pnl_proxy(yt[:, i], probs[:, i], transaction_cost_bps=cfg.train.transaction_cost_bps)
            horizon_metrics[f"{hz}s"] = {**dm, **pm}

        latency = latency_adjusted_throughput(np.asarray(latencies), target_budget_ms=5.0)
        report = {
            "fold": fold_id,
            "train_history": hist,
            "metrics": horizon_metrics,
            "latency": latency,
        }
        if is_snn:
            logits_for_attr, _, fused_for_attr = model(torch.tensor(x_seq[te][: min(64, len(te))], device=device))
            report["explainability"] = {
                "coupling_edges": coupling_edge_importance(coupling, [s.asset for s in cfg.dataset.streams])[:20],
                "temporal_attribution": temporal_attribution_snapshots(fused_for_attr, logits_for_attr),
            }
        fold_reports.append(report)

    out_dir = cfg.output_dir / f"{model_type}_{ann_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    payload = {
        "config": asdict(cfg),
        "event_vocab": event_vocab,
        "feature_names": feature_names,
        "fold_reports": fold_reports,
        "ablation_tag": f"{model_type}_{ann_mode}",
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
