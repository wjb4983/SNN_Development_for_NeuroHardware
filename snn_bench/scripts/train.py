from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from snn_bench.configs.settings import BenchmarkConfig, ModelSelectionConfig, SmokeConfig, TaskConfig
from snn_bench.data_connectors.backtest_store import BacktestBarStoreConnector
from snn_bench.data_connectors.snapshot_cache import SnapshotCacheConnector
from snn_bench.eval.reporting import generate_run_report
from snn_bench.models import ModelSpec, ModelZoo, save_prediction_artifacts, set_global_seed
from snn_bench.tasks.registry import TaskRegistry, assert_aligned_not_empty, validate_task_model_compatibility
from snn_bench.utils.secrets import load_massive_api_key


def _available_years(index_blob: dict) -> list[int]:
    years = index_blob.get("years") or []
    if not years:
        raise ValueError("index.json does not contain years")
    return sorted(int(y) for y in years)


def _load_cfg_from_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _apply_smoke(cfg: BenchmarkConfig, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if not cfg.smoke.enabled:
        return x, y, cfg.epochs
    limit = min(len(x), int(cfg.smoke.sample_size))
    return x[:limit], y[:limit], int(cfg.smoke.epochs)


def _split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    split_mode: str = "random",
    test_size: float = 0.2,
    walk_forward_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split_mode == "walk_forward":
        cut = int(len(x) * walk_forward_ratio)
        cut = min(max(1, cut), len(x) - 1)
        return x[:cut], x[cut:], y[:cut], y[cut:]
    if np.issubdtype(y.dtype, np.floating):
        return train_test_split(x, y, test_size=test_size, random_state=seed, shuffle=True)
    y_int = y.astype(np.int64)
    return train_test_split(
        x,
        y_int,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y_int if len(np.unique(y_int)) > 1 else None,
    )


def run_training(
    cfg: BenchmarkConfig,
    out_dir: Path,
    max_years: int = 0,
    *,
    task_name: str | None = None,
    task_config: Path | None = None,
    split_mode: str = "random",
    walk_forward_ratio: float = 0.8,
) -> dict:
    set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    load_massive_api_key(cfg.massive_api_key_file)

    snapshot = SnapshotCacheConnector(cfg.data_paths.snapshot_dir, cfg.data_paths.external_snapshot_dir)
    bars = BacktestBarStoreConnector(cfg.data_paths.backtest_root)

    rows_snapshot = len(snapshot.load_frame(cfg.ticker))
    years = _available_years(bars.load_index(cfg.ticker, cfg.timeframe))
    selected_years = years if max_years <= 0 else years[:max_years]
    frames = [bars.load_year(cfg.ticker, cfg.timeframe, y) for y in selected_years]
    frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    task_registry = TaskRegistry()
    spec = task_registry.resolve(task_name=task_name, task_config=task_config)
    validate_task_model_compatibility(spec, cfg.model.name)

    x, y = task_registry.build_dataset(frame, spec)
    assert_aligned_not_empty(x, y)
    x, y, epochs = _apply_smoke(cfg, x, y)

    strategy = str(cfg.model.params.get("training_strategy", "classification"))
    y_for_training = y.astype(np.float32) if strategy == "volatility_regression" else y.astype(np.int64)
    x_train, x_test, y_train, y_test = _split_dataset(
        x,
        y_for_training,
        seed=cfg.seed,
        split_mode=split_mode,
        test_size=0.2,
        walk_forward_ratio=walk_forward_ratio,
    )

    model_spec = ModelSpec(name=cfg.model.name, family="zoo", params={**cfg.model.params, "seed": cfg.seed})
    model = ModelZoo.create(model_spec, input_dim=x_train.shape[1])
    train_info = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        scheduler=str(cfg.model.params.get("scheduler", "none")),
        grad_clip_norm=float(cfg.model.params.get("grad_clip_norm", 0.0)),
        early_stopping_patience=int(cfg.model.params.get("early_stopping_patience", 0)),
        mixed_precision=bool(cfg.model.params.get("mixed_precision", False)),
        aux_objective=str(cfg.model.params.get("aux_objective", "none")),
        aux_weight=float(cfg.model.params.get("aux_weight", 0.0)),
        val_split=float(cfg.model.params.get("val_split", 0.1)),
    )

    y_prob = model.predict_proba(x_test)
    eval_metrics = model.evaluate(x_test, y_test)

    task_meta = {
        "name": spec.name,
        "horizon": cfg.task.horizon,
        "label_type": cfg.task.label_type,
        "classes": cfg.task.classes,
        "label_semantics": cfg.task.label_semantics,
        "task_config": str(spec.path),
        "evaluation": (spec.raw.get("evaluation") or {}),
    }
    run_id = (
        f"{cfg.run_name}_{cfg.model.name}_{spec.name}_{cfg.task.horizon}_"
        f"{cfg.ticker}_{cfg.timeframe}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = run_dir / f"{cfg.model.name}_checkpoint.bin"
    model.save_checkpoint(checkpoint)
    pred_artifact = save_prediction_artifacts(
        run_dir,
        cfg.model.name,
        y_test,
        y_prob,
        target_summary={
            "task_name": spec.name,
            "horizon": cfg.task.horizon,
            "label_type": cfg.task.label_type,
            "classes": cfg.task.classes,
            "label_semantics": cfg.task.label_semantics,
        },
    )

    metrics = {
        "run_id": run_id,
        "ticker": cfg.ticker,
        "timeframe": cfg.timeframe,
        "years": selected_years,
        "rows_snapshot": rows_snapshot,
        "rows_train_total": int(len(x)),
        "rows_train": int(len(x_train)),
        "rows_eval": int(len(x_test)),
        "split_mode": split_mode,
        "walk_forward_ratio": walk_forward_ratio if split_mode == "walk_forward" else None,
        "epochs": epochs,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "model": cfg.model.name,
        "model_params": cfg.model.params,
        "task": task_meta,
        "train_info": train_info,
        "eval": eval_metrics,
        "checkpoint": str(checkpoint),
        "predictions": str(pred_artifact),
        "run_dir": str(run_dir),
    }
    metrics_path = run_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    try:
        report_path = generate_run_report(run_dir)
        metrics["report"] = str(report_path)
    except Exception as exc:  # noqa: BLE001 - report generation should not block training artifacts
        metrics["report_error"] = str(exc)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model-zoo benchmark model")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--timeframe", default="1D")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--max-years", type=int, default=0, help="0=all available years; otherwise use earliest N years")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-sample-size", type=int, default=256)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-name", default=None, help="Task name from task YAML, e.g. direction_5m_distribution")
    parser.add_argument("--task-config", type=Path, default=None, help="Path to task YAML config")
    parser.add_argument("--split-mode", choices=["random", "walk_forward"], default="random")
    parser.add_argument("--walk-forward-ratio", type=float, default=0.8, help="Train ratio for walk-forward split mode")
    parser.add_argument("--scheduler", choices=["none", "cosine", "one_cycle"], default="none")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--loss", choices=["default", "focal", "class_balanced", "huber"], default="default")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--training-strategy", choices=["classification", "multiclass", "ordinal", "volatility_regression"], default="classification")
    parser.add_argument("--aux-objective", choices=["none", "reconstruction", "contrastive"], default="none")
    parser.add_argument("--aux-weight", type=float, default=0.0)
    parser.add_argument("--num-classes", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_data: dict = {}
    if args.config:
        cfg_data = _load_cfg_from_yaml(args.config)

    cfg = BenchmarkConfig.model_validate(
        {
            "ticker": args.ticker,
            "timeframe": args.timeframe,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "run_name": args.run_name,
            "seed": args.seed,
            "task_name": args.task_name,
            "task_config": str(args.task_config) if args.task_config else None,
            "model": ModelSelectionConfig(
                name=args.model,
                params={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "scheduler": args.scheduler,
                    "grad_clip_norm": args.grad_clip_norm,
                    "early_stopping_patience": args.early_stopping_patience,
                    "mixed_precision": args.mixed_precision,
                    "loss": args.loss,
                    "label_smoothing": args.label_smoothing,
                    "training_strategy": args.training_strategy,
                    "aux_objective": args.aux_objective,
                    "aux_weight": args.aux_weight,
                    "num_classes": args.num_classes,
                    "output_dim": args.num_classes,
                },
            ),
            "task": TaskConfig(),
            "smoke": SmokeConfig(enabled=args.smoke, sample_size=args.smoke_sample_size, epochs=args.smoke_epochs),
            **cfg_data,
        }
    )

    effective_task_name = cfg.task_name or args.task_name
    effective_task_config = Path(cfg.task_config) if cfg.task_config else args.task_config

    metrics = run_training(
        cfg,
        Path(args.out_dir),
        max_years=args.max_years,
        task_name=effective_task_name,
        task_config=effective_task_config,
        split_mode=args.split_mode,
        walk_forward_ratio=args.walk_forward_ratio,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
