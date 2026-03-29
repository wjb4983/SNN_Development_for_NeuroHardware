from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from snn_bench.hybrid.backtest import run_backtest
from snn_bench.hybrid.fast_model_snn import FastSNNModel
from snn_bench.hybrid.feature_pipeline import generate_synthetic_hybrid_data
from snn_bench.hybrid.fusion import RegimeAwareFusion, WeightedFusion
from snn_bench.hybrid.risk_gate import RiskGate
from snn_bench.hybrid.slow_model_ann import SlowANNModel


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _data_from_config(cfg: dict):
    data_cfg = cfg.get("data", {})
    return generate_synthetic_hybrid_data(
        n_steps=int(data_cfg.get("n_steps", 5000)),
        seed=int(data_cfg.get("seed", 7)),
    )


def cmd_train_slow(config_path: Path) -> None:
    cfg = _load_config(config_path)
    ds = _data_from_config(cfg)
    ann_cfg = cfg.get("slow_model", {})
    model = SlowANNModel(
        hidden_layer_sizes=tuple(ann_cfg.get("hidden_layer_sizes", [32, 16])),
        alpha=float(ann_cfg.get("alpha", 1e-4)),
        random_state=int(cfg.get("reproducibility", {}).get("seed", 7)),
    )
    model.fit(ds.slow_features, ds.target)
    out = Path(cfg["artifacts"]["slow_model_path"])
    model.save(out)
    print(json.dumps({"saved": str(out), "rows": len(ds.slow_features)}))


def cmd_train_fast(config_path: Path) -> None:
    cfg = _load_config(config_path)
    ds = _data_from_config(cfg)
    fast_cfg = cfg.get("fast_model", {})
    model = FastSNNModel(
        input_dim=ds.fast_features.shape[1],
        hidden_dim=int(fast_cfg.get("hidden_dim", 24)),
        lr=float(fast_cfg.get("lr", 1e-3)),
        epochs=int(fast_cfg.get("epochs", 35)),
        seed=int(cfg.get("reproducibility", {}).get("seed", 7)),
    )
    model.fit(ds.fast_features, ds.target)
    out = Path(cfg["artifacts"]["fast_model_path"])
    model.save(out)
    print(json.dumps({"saved": str(out), "rows": len(ds.fast_features)}))


def cmd_train_fusion(config_path: Path) -> None:
    cfg = _load_config(config_path)
    out = Path(cfg["artifacts"]["fusion_model_path"])
    fusion_cfg = cfg.get("fusion", {})
    fusion_type = str(fusion_cfg.get("type", "weighted")).lower()
    if fusion_type == "regime_aware":
        model = RegimeAwareFusion(
            slow_weight=float(fusion_cfg.get("slow_weight", 0.55)),
            fast_weight=float(fusion_cfg.get("fast_weight", 0.45)),
            regime_conf_sensitivity=float(fusion_cfg.get("regime_conf_sensitivity", 0.7)),
            low_confidence_threshold=float(fusion_cfg.get("low_confidence_threshold", 0.5)),
            low_confidence_snn_scale=float(fusion_cfg.get("low_confidence_snn_scale", 0.75)),
            regime_scale_by_state=fusion_cfg.get("regime_scale_by_state"),
        )
    else:
        model = WeightedFusion(
            slow_weight=float(fusion_cfg.get("slow_weight", 0.55)),
            fast_weight=float(fusion_cfg.get("fast_weight", 0.45)),
        )
    model.save(out)
    print(json.dumps({"saved": str(out), "method": fusion_type}))


def _load_fusion_model(path: Path) -> WeightedFusion | RegimeAwareFusion:
    with path.open("rb") as f:
        state = pickle.load(f)
    if state.get("fusion_type") == "regime_aware":
        return RegimeAwareFusion.load(path)
    return WeightedFusion.load(path)


def _load_markov_posteriors(path: Path, expected_rows: int) -> np.ndarray:
    if path.suffix == ".npy":
        posterior = np.load(path)
    else:
        payload = np.load(path)
        key = "posterior" if "posterior" in payload else "regime_posteriors"
        posterior = payload[key]
    if posterior.shape[0] != expected_rows:
        raise ValueError(
            f"Markov posterior rows ({posterior.shape[0]}) must match backtest rows ({expected_rows})"
        )
    return posterior.astype(np.float32)


def cmd_backtest_hybrid(config_path: Path) -> None:
    cfg = _load_config(config_path)
    ds = _data_from_config(cfg)
    artifacts = cfg["artifacts"]

    slow = SlowANNModel.load(Path(artifacts["slow_model_path"]))
    fast = FastSNNModel.load(Path(artifacts["fast_model_path"]))
    fusion = _load_fusion_model(Path(artifacts["fusion_model_path"]))

    slow_pred = slow.predict(ds.slow_features)
    fast_pred = fast.predict(ds.fast_features)

    risk_cfg = cfg.get("risk_gate", {})
    gate = RiskGate(
        warning_vol=float(risk_cfg.get("warning_vol", 0.0014)),
        block_vol=float(risk_cfg.get("block_vol", 0.0022)),
        warning_anomaly=float(risk_cfg.get("warning_anomaly", 2.2)),
        block_anomaly=float(risk_cfg.get("block_anomaly", 3.2)),
        warning_leverage=float(risk_cfg.get("warning_leverage", 0.5)),
    )
    gate_out = gate.evaluate(ds.market, ds.fast_features)

    regime_cfg = cfg.get("regime_fusion", {})
    markov_path = regime_cfg.get("markov_posterior_path")
    calibration_path = regime_cfg.get("calibration_weights_path")
    posteriors = None
    calibration_weights = None
    if markov_path:
        posteriors = _load_markov_posteriors(Path(markov_path), expected_rows=len(ds.market))
    if calibration_path:
        calibration_weights = _load_markov_posteriors(Path(calibration_path), expected_rows=len(ds.market))

    if isinstance(fusion, RegimeAwareFusion):
        fused = fusion.blend(
            slow_score=slow_pred.score,
            slow_conf=slow_pred.confidence,
            fast_score=fast_pred.score,
            fast_conf=fast_pred.confidence,
            regime_posteriors=posteriors,
            calibration_weights=calibration_weights,
        )
    else:
        fused = fusion.blend(
            slow_score=slow_pred.score,
            slow_conf=slow_pred.confidence,
            fast_score=fast_pred.score,
            fast_conf=fast_pred.confidence,
        )

    bt_cfg = cfg.get("backtest", {})
    bt = run_backtest(
        ds.market,
        fused_score=fused.score,
        fused_conf=fused.confidence,
        risk_state=gate_out.state,
        leverage=gate_out.leverage,
        slow_component=fused.slow_component,
        fast_component=fused.fast_component,
        score_to_pos=float(bt_cfg.get("score_to_pos", 75.0)),
        max_position=float(bt_cfg.get("max_position", 1.0)),
        max_turnover_step=float(bt_cfg.get("max_turnover_step", 0.35)),
        cooldown_steps=int(bt_cfg.get("cooldown_steps", 5)),
        tx_cost_bps=float(bt_cfg.get("tx_cost_bps", 1.5)),
        slippage_bps=float(bt_cfg.get("slippage_bps", 0.75)),
        latency_steps=int(bt_cfg.get("latency_steps", 1)),
    )

    report = {
        "metrics": bt.metrics,
        "attribution": bt.attribution,
        "risk_state_counts": {
            state: int((gate_out.state == state).sum()) for state in ["NORMAL", "WARNING", "BLOCK"]
        },
    }
    out = Path(artifacts.get("backtest_report_path", "artifacts/hybrid/backtest_report.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid ANN+SNN quant stack CLI")
    sp = p.add_subparsers(dest="command", required=True)

    for command in ["train_slow", "train_fast", "train_fusion", "backtest_hybrid"]:
        cp = sp.add_parser(command)
        cp.add_argument("--config", type=Path, required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train_slow":
        cmd_train_slow(args.config)
    elif args.command == "train_fast":
        cmd_train_fast(args.config)
    elif args.command == "train_fusion":
        cmd_train_fusion(args.config)
    elif args.command == "backtest_hybrid":
        cmd_backtest_hybrid(args.config)


if __name__ == "__main__":
    main()
