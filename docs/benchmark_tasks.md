# Benchmark Task Matrix

This matrix maps benchmark tasks to business decisions and specifies required acceptance criteria.

| Task | Business decision | Prediction target | Horizon | Data requirements | Primary ML metric | Primary trading metric | Acceptance thresholds |
|---|---|---|---|---|---|---|---|
| **Short-horizon direction** (`direction_5m_distribution`) | Intraday directional positioning / sizing for rapid execution | Binary direction label (`up` vs `down_or_flat`) from 5-minute forward return with neutral band filtering | 5 minutes | Minute bars (`o/h/l/c/v`) + engineered momentum/volatility features, point-in-time aligned labels | `roc_auc` (maximize) | `sharpe` (maximize) | `roc_auc >= 0.55`, `f1 >= 0.52`, `sharpe >= 0.40`, `max_drawdown >= -0.08` |
| **Volatility forecasting** (`realized_vol_30m`) | Dynamic risk budgeting, leverage scaling, and stop-width selection | Next-window realized volatility estimate | 30 minutes | Minute bars with stable realized-vol target construction and horizon-safe rolling windows | `rmse` (minimize) | N/A (informational task) | `rmse <= 0.020`, `mae <= 0.014` |
| **Options skew movement** (`options_iv_skew_movement`) | Relative-value options positioning and hedge rebalance trigger | Direction of skew proxy movement above threshold | 30 minutes | Underlier bars + skew proxy feature stream, leakage-safe future shift labels | `roc_auc` (maximize) | `net_pnl` (maximize) | `roc_auc >= 0.54`, `f1 >= 0.50`, `net_pnl >= 0`, `max_drawdown >= -0.10` |
| **Regime classification** (`regime_classification` planned) | Switch strategy family / risk caps by inferred market regime | Regime class (trend, mean-reverting, high-vol, low-vol) | 1 day (rolling updates) | OHLCV bars, realized vol, cross-asset breadth, macro/event calendar tags | `f1_macro` (maximize) | `sharpe` of regime-conditioned strategy (maximize) | `f1_macro >= 0.45`, `regime_precision >= 0.50`, `sharpe >= 0.30` |

## Experiment policy additions

1. **Required baseline set** in experiment manifests:
   - `naive_persistence`
   - `logreg`
   - `gbm`
   - `mlp`
   - SNN variants (`snntorch_lif`, `snntorch_alif`, `norse_recurrent_lsnn`, `spikingjelly_temporal_conv`)
2. Experiment summaries must include **comparison tables** and **leaderboards** ranked by task primary metric(s).
3. Walk-forward experiments must pass leakage guardrails (`no_leakage_walkforward_check`) and index alignment assertions.
