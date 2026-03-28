from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from .backtest import BacktestConfig, EventDrivenBacktester
from .interfaces import QuantModel
from .logging_utils import setup_logger
from .metrics import ml_metrics, trading_kpis
from .seed import set_global_seed
from .splits import PurgedEmbargoWalkForward
from .tracking import LocalTracker


@dataclass
class ExperimentArtifacts:
    run_id: str
    run_dir: Path
    summary_metrics: dict[str, float]


def run_experiment(
    model: QuantModel,
    X: np.ndarray,
    y: np.ndarray,
    prices: np.ndarray,
    output_dir: str | Path,
    seed: int = 42,
    splitter: PurgedEmbargoWalkForward | None = None,
    backtest_cfg: BacktestConfig | None = None,
) -> ExperimentArtifacts:
    """Standardized train/eval loop that works with ANN or SNN model interfaces."""
    set_global_seed(seed)
    splitter = splitter or PurgedEmbargoWalkForward()
    backtest_cfg = backtest_cfg or BacktestConfig()

    run_id = f"{model.name}_{uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("quant_template", run_dir)
    tracker = LocalTracker(Path(output_dir) / "tracking")
    backtester = EventDrivenBacktester(backtest_cfg)

    all_fold_metrics: list[dict[str, float]] = []
    for split_id, (train_idx, val_idx) in enumerate(splitter.split(len(X))):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        val_prices = prices[val_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        prob = model.predict_proba(X_val)

        ml = ml_metrics(y_val, pred, prob)
        signal = np.where(pred > 0, 1.0, -1.0)
        bt = backtester.run(val_prices, signal)
        tkpi = trading_kpis(bt["returns"], np.sign(signal), cost_per_turnover_bps=backtest_cfg.fee_bps + backtest_cfg.spread_bps)

        fold_metrics = {**ml, **{k: v for k, v in tkpi.items() if isinstance(v, float)}}
        all_fold_metrics.append(fold_metrics)
        tracker.log(run_id=run_id, model_name=model.name, split=split_id, metrics=fold_metrics, params={"seed": seed})
        logger.info("split=%s metrics=%s", split_id, fold_metrics)

    keys = sorted({k for m in all_fold_metrics for k in m})
    summary: dict[str, float] = {}
    for k in keys:
        values = np.array([m.get(k, np.nan) for m in all_fold_metrics], dtype=float)
        if np.isnan(values).all():
            summary[f"mean_{k}"] = float("nan")
        else:
            summary[f"mean_{k}"] = float(np.nanmean(values))
    tracker.log(run_id=run_id, model_name=model.name, split=-1, metrics=summary, params={"seed": seed, "summary": True})
    logger.info("summary=%s", summary)

    return ExperimentArtifacts(run_id=run_id, run_dir=run_dir, summary_metrics=summary)
