from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from snn_bench.configs.settings import BenchmarkConfig
from snn_bench.eval.reporting import generate_run_report
from snn_bench.scripts.train import run_training


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_run_config(defaults: dict[str, Any], run_item: dict[str, Any]) -> dict[str, Any]:
    if "name" not in run_item:
        raise ValueError("Each run in the experiment manifest must include a 'name' field")
    run_name = str(run_item["name"]).strip()
    if not run_name:
        raise ValueError("Run 'name' cannot be empty")

    payload = {k: v for k, v in run_item.items() if k != "name"}
    merged = _deep_merge(defaults, payload)
    merged["run_name"] = run_name
    return merged


def _metric_value(payload: dict[str, Any], dotted_key: str) -> float | None:
    cursor: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    if isinstance(cursor, (int, float)):
        return float(cursor)
    return None


def _metric_direction(metric_name: str) -> str:
    low_is_better = {"logloss", "brier", "calibration", "mse", "rmse", "mae", "max_drawdown", "turnover"}
    return "asc" if metric_name.lower() in low_is_better else "desc"


def _task_primary_metrics(metrics: dict[str, Any]) -> tuple[str, str | None]:
    evaluation = ((metrics.get("task") or {}).get("evaluation") or {})
    primary_ml = str(evaluation.get("primary_ml_metric") or "")
    if not primary_ml:
        eval_payload = metrics.get("eval") or {}
        for candidate in ("roc_auc", "auc", "accuracy", "f1", "f1_macro", "rmse", "mse"):
            if candidate in eval_payload:
                primary_ml = candidate
                break
    primary_trading = evaluation.get("primary_trading_metric")
    if isinstance(primary_trading, str) and primary_trading.strip():
        return primary_ml, primary_trading
    return primary_ml, None


def _build_leaderboard(completed: list[dict[str, Any]], metric_key: str, metric_type: str, direction: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in completed:
        value = _metric_value(item, metric_key)
        if value is None:
            continue
        rows.append(
            {
                "run_id": item.get("run_id"),
                "run_name": str(item.get("run_id", "")).split("_")[0],
                "model": item.get("model"),
                "task_name": (item.get("task") or {}).get("name"),
                "metric_key": metric_key,
                "metric_type": metric_type,
                "metric_value": value,
            }
        )
    reverse = direction == "desc"
    rows.sort(key=lambda r: r["metric_value"], reverse=reverse)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def run_experiments(manifest_path: Path, out_dir: Path, max_years: int = 0, stop_on_error: bool = False) -> dict[str, Any]:
    manifest = _load_yaml(manifest_path)
    defaults = manifest.get("defaults") or {}
    runs = manifest.get("runs") or []

    if not runs:
        raise ValueError(f"No runs found in experiment manifest: {manifest_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for run_item in runs:
        cfg_dict = _build_run_config(defaults, run_item)
        run_name = cfg_dict["run_name"]
        try:
            cfg = BenchmarkConfig.model_validate(cfg_dict)
            metrics = run_training(cfg, out_dir=out_dir, max_years=max_years)
            run_dir = Path(metrics.get("run_dir", out_dir / metrics["run_id"]))
            try:
                report_path = generate_run_report(run_dir)
                metrics["report"] = str(report_path)
                (run_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - report generation should not break experiment sweeps
                metrics["report_error"] = str(exc)
                (run_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            completed.append(metrics)
            print(f"[ok] completed run: {run_name}")
        except Exception as exc:  # noqa: BLE001 - keep sweep execution resilient
            failures.append({"run_name": run_name, "error": str(exc)})
            print(f"[error] failed run: {run_name} -> {exc}")
            if stop_on_error:
                break

    primary_ml_metric = ""
    primary_trading_metric = ""
    if completed:
        primary_ml_metric, maybe_trading = _task_primary_metrics(completed[0])
        primary_trading_metric = maybe_trading or ""

    ml_direction = _metric_direction(primary_ml_metric) if primary_ml_metric else "desc"
    trading_direction = _metric_direction(primary_trading_metric) if primary_trading_metric else "desc"
    ml_leaderboard = _build_leaderboard(completed, primary_ml_metric, "ml", ml_direction) if primary_ml_metric else []
    trading_leaderboard = (
        _build_leaderboard(completed, f"eval.trading.{primary_trading_metric}", "trading", trading_direction)
        if primary_trading_metric
        else []
    )
    summary = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "total_runs": len(runs),
        "completed_runs": len(completed),
        "failed_runs": len(failures),
        "failures": failures,
        "run_ids": [item["run_id"] for item in completed],
        "primary_metrics": {
            "ml": primary_ml_metric,
            "trading": primary_trading_metric or None,
        },
        "leaderboards": {
            "ml": ml_leaderboard,
            "trading": trading_leaderboard,
        },
        "comparison_tables": {
            "ml": {
                "columns": ["rank", "run_id", "model", "task_name", "metric_value"],
                "rows": ml_leaderboard,
            },
            "trading": {
                "columns": ["rank", "run_id", "model", "task_name", "metric_value"],
                "rows": trading_leaderboard,
            },
        },
    }
    summary_path = out_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-model experiment sweep from one YAML file")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to experiment YAML manifest")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--max-years", type=int, default=0, help="0=all available years; otherwise earliest N years")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop sweep after the first failed run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiments(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        max_years=args.max_years,
        stop_on_error=args.stop_on_error,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
