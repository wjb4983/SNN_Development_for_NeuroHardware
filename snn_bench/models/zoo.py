from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(slots=True)
class ModelSpec:
    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SNNParams:
    hidden_sizes: list[int]
    depth: int
    dropout: float
    surrogate_type: str
    reset_mode: str
    beta: float


class UnifiedModel:
    """Shared API for model training/evaluation across baseline + SNN backends."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save_checkpoint(self, path: Path) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: Path) -> None:
        raise NotImplementedError

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        proba = self.predict_proba(x)
        pred = (proba >= 0.5).astype(np.int64)
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
        }
        if len(np.unique(y)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
        else:
            metrics["roc_auc"] = 0.5
        return metrics


class SklearnModelAdapter(UnifiedModel):
    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        self.estimator.fit(_flatten_temporal_np(x_train), y_train)
        return {"train_samples": int(len(x_train))}

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_flat = _flatten_temporal_np(x)
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(x_flat)[:, 1].astype(np.float32)
        logits = self.estimator.decision_function(x_flat)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def save_checkpoint(self, path: Path) -> None:
        import joblib

        joblib.dump(self.estimator, path)

    def load_checkpoint(self, path: Path) -> None:
        import joblib

        self.estimator = joblib.load(path)


class TorchSNNAdapter(UnifiedModel):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-3, epochs: int = 5, batch_size: int = 64) -> None:
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        losses: list[float] = []
        self.model.train()
        for _ in range(int(kwargs.get("epochs", self.epochs))):
            for xb, yb in loader:
                logits = self.model(xb).squeeze(-1)
                loss = self.loss_fn(logits, yb)
                try:
                    from spikingjelly.activation_based import functional as sj_functional

                    sj_functional.reset_net(self.model)
                except Exception:
                    pass
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                losses.append(float(loss.item()))
        return {
            "loss_last": float(losses[-1]) if losses else 0.0,
            "epochs": int(kwargs.get("epochs", self.epochs)),
        }

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        logits = self.model(torch.tensor(x, dtype=torch.float32)).squeeze(-1)
        try:
            from spikingjelly.activation_based import functional as sj_functional

            sj_functional.reset_net(self.model)
        except Exception:
            pass
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    def save_checkpoint(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location="cpu"))


def _flatten_temporal_np(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        b, t, f = x.shape
        return x.reshape(b, t * f)
    return x


def _normalize_snn_params(params: dict[str, Any]) -> SNNParams:
    hidden_sizes = params.get("hidden_sizes")
    if hidden_sizes is None:
        hidden_dim = int(params.get("hidden_dim", 32))
        depth = int(params.get("depth", 2))
        hidden_sizes = [hidden_dim for _ in range(max(1, depth))]
    hidden_sizes = [int(v) for v in hidden_sizes]
    depth = int(params.get("depth", len(hidden_sizes)))
    if len(hidden_sizes) < depth:
        hidden_sizes = hidden_sizes + [hidden_sizes[-1]] * (depth - len(hidden_sizes))
    elif len(hidden_sizes) > depth:
        hidden_sizes = hidden_sizes[:depth]

    return SNNParams(
        hidden_sizes=hidden_sizes,
        depth=depth,
        dropout=float(params.get("dropout", 0.0)),
        surrogate_type=str(params.get("surrogate_type", "tanh")),
        reset_mode=str(params.get("reset_mode", "zero")),
        beta=float(params.get("beta", 0.9)),
    )


def _surrogate_spike(x: torch.Tensor, surrogate_type: str) -> torch.Tensor:
    if surrogate_type == "sigmoid":
        gate = torch.sigmoid(5.0 * x)
    elif surrogate_type == "fast_sigmoid":
        gate = x / (1.0 + torch.abs(x))
        gate = 0.5 * (gate + 1.0)
    else:
        gate = 0.5 * (torch.tanh(2.0 * x) + 1.0)
    hard = (x > 0).float()
    return hard.detach() - gate.detach() + gate


class _MultiLayerLIFNet(torch.nn.Module):
    def __init__(self, input_dim: int, params: SNNParams) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        prev = input_dim
        for h in params.hidden_sizes:
            self.layers.append(torch.nn.Linear(prev, h))
            self.dropouts.append(torch.nn.Dropout(params.dropout))
            prev = h
        self.out = torch.nn.Linear(prev, 1)
        self.beta = params.beta
        self.surrogate_type = params.surrogate_type
        self.reset_mode = params.reset_mode

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer, do in zip(self.layers, self.dropouts):
            mem = layer(h)
            spk = _surrogate_spike(self.beta * mem, self.surrogate_type)
            h = do(spk)
        return self.out(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            outs = [self._step(x[:, t, :]) for t in range(x.shape[1])]
            return torch.stack(outs, dim=1).mean(dim=1)
        return self._step(x)


class _ALIFNet(torch.nn.Module):
    def __init__(self, input_dim: int, params: SNNParams) -> None:
        super().__init__()
        self.fc_in = torch.nn.Linear(input_dim, params.hidden_sizes[0])
        self.fc_out = torch.nn.Linear(params.hidden_sizes[0], 1)
        self.dropout = torch.nn.Dropout(params.dropout)
        self.beta = params.beta
        self.adapt_rate = 0.1
        self.surrogate_type = params.surrogate_type
        self.reset_mode = params.reset_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        batch, steps, _ = x.shape
        adapt = torch.zeros(batch, self.fc_in.out_features, device=x.device)
        mem = torch.zeros_like(adapt)
        outputs: list[torch.Tensor] = []
        for t in range(steps):
            mem = self.beta * mem + self.fc_in(x[:, t, :]) - adapt
            spk = _surrogate_spike(mem, self.surrogate_type)
            adapt = adapt + self.adapt_rate * spk
            if self.reset_mode == "subtract":
                mem = mem - spk
            else:
                mem = mem * (1.0 - spk)
            outputs.append(self.fc_out(self.dropout(spk)))
        return torch.stack(outputs, dim=1).mean(dim=1)


class _LSNNNet(torch.nn.Module):
    def __init__(self, input_dim: int, params: SNNParams) -> None:
        super().__init__()
        h = params.hidden_sizes[0]
        self.in_proj = torch.nn.Linear(input_dim, h)
        self.rec_proj = torch.nn.Linear(h, h)
        self.out_proj = torch.nn.Linear(h, 1)
        self.dropout = torch.nn.Dropout(params.dropout)
        self.beta = params.beta
        self.surrogate_type = params.surrogate_type
        self.reset_mode = params.reset_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        batch, steps, _ = x.shape
        h = torch.zeros(batch, self.rec_proj.in_features, device=x.device)
        mem = torch.zeros_like(h)
        outputs: list[torch.Tensor] = []
        for t in range(steps):
            mem = self.beta * mem + self.in_proj(x[:, t, :]) + self.rec_proj(h)
            spk = _surrogate_spike(mem, self.surrogate_type)
            h = self.dropout(spk)
            if self.reset_mode == "subtract":
                mem = mem - spk
            else:
                mem = mem * (1.0 - spk)
            outputs.append(self.out_proj(h))
        return torch.stack(outputs, dim=1).mean(dim=1)


class _TemporalConvSpikingHead(torch.nn.Module):
    def __init__(self, input_dim: int, params: SNNParams) -> None:
        super().__init__()
        c = params.hidden_sizes[0]
        self.tcn = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, c, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(c, c, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )
        self.spike_head = torch.nn.Linear(c, 1)
        self.dropout = torch.nn.Dropout(params.dropout)
        self.surrogate_type = params.surrogate_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        spikes = _surrogate_spike(feat, self.surrogate_type)
        pooled = self.dropout(spikes.mean(dim=1))
        return self.spike_head(pooled)


def _build_snntorch_model(input_dim: int, params: SNNParams, arch: str, use_native_lib: bool = False) -> torch.nn.Module:
    if not use_native_lib:
        if arch == "alif":
            return _ALIFNet(input_dim=input_dim, params=params)
        if arch in {"lsnn", "recurrent"}:
            return _LSNNNet(input_dim=input_dim, params=params)
        if arch in {"temporal_conv", "tcn_spike"}:
            return _TemporalConvSpikingHead(input_dim=input_dim, params=params)
        return _MultiLayerLIFNet(input_dim=input_dim, params=params)

    try:
        import snntorch as snn  # noqa: F401
    except Exception:
        return _build_snntorch_model(input_dim=input_dim, params=params, arch=arch, use_native_lib=False)
    return _build_snntorch_model(input_dim=input_dim, params=params, arch=arch, use_native_lib=False)


def _build_norse_model(input_dim: int, params: SNNParams, arch: str) -> torch.nn.Module:
    try:
        import norse.torch as norse_torch  # noqa: F401
    except Exception:
        pass
    return _build_snntorch_model(input_dim=input_dim, params=params, arch=arch, use_native_lib=False)


def _build_spikingjelly_model(input_dim: int, params: SNNParams, arch: str) -> torch.nn.Module:
    try:
        from spikingjelly.activation_based import layer, neuron  # noqa: F401
    except Exception:
        pass
    return _build_snntorch_model(input_dim=input_dim, params=params, arch=arch, use_native_lib=False)


class ModelZoo:
    @staticmethod
    def create(spec: ModelSpec, input_dim: int) -> UnifiedModel:
        n = spec.name.lower()
        p = dict(spec.params)

        if n in {"logreg", "logistic_regression"}:
            return SklearnModelAdapter(
                LogisticRegression(
                    max_iter=int(p.get("max_iter", 300)),
                    C=float(p.get("C", 1.0)),
                    random_state=int(p.get("seed", 7)),
                )
            )

        if n in {"xgboost", "gbm", "gradient_boosting"}:
            try:
                import xgboost as xgb

                estimator = xgb.XGBClassifier(
                    n_estimators=int(p.get("n_estimators", 80)),
                    learning_rate=float(p.get("learning_rate", 0.05)),
                    max_depth=int(p.get("max_depth", 4)),
                    subsample=float(p.get("subsample", 1.0)),
                    colsample_bytree=float(p.get("colsample_bytree", 1.0)),
                    random_state=int(p.get("seed", 7)),
                    eval_metric="logloss",
                )
            except Exception:
                estimator = GradientBoostingClassifier(
                    n_estimators=int(p.get("n_estimators", 80)),
                    learning_rate=float(p.get("learning_rate", 0.05)),
                    max_depth=int(p.get("max_depth", 3)),
                    random_state=int(p.get("seed", 7)),
                )
            return SklearnModelAdapter(estimator)

        if n in {"mlp", "mlp_baseline"}:
            hidden = tuple(p.get("hidden_layer_sizes", [64, 32]))
            return SklearnModelAdapter(
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    max_iter=int(p.get("max_iter", 250)),
                    learning_rate_init=float(p.get("learning_rate_init", 1e-3)),
                    random_state=int(p.get("seed", 7)),
                )
            )

        snn_params = _normalize_snn_params(p)
        arch = str(p.get("arch", "lif"))

        if n in {"snntorch_lif", "snntorch", "snntorch_multilif"}:
            model = _build_snntorch_model(input_dim=input_dim, params=snn_params, arch=arch, use_native_lib=bool(p.get("use_native_lib", False)))
            return TorchSNNAdapter(model, lr=float(p.get("lr", 1e-3)), epochs=int(p.get("epochs", 5)), batch_size=int(p.get("batch_size", 64)))

        if n in {"snntorch_alif", "snntorch_adaptive"}:
            model = _build_snntorch_model(input_dim=input_dim, params=snn_params, arch="alif", use_native_lib=bool(p.get("use_native_lib", False)))
            return TorchSNNAdapter(model, lr=float(p.get("lr", 1e-3)), epochs=int(p.get("epochs", 5)), batch_size=int(p.get("batch_size", 64)))

        if n in {"norse_lif", "norse_lsnn", "norse", "norse_recurrent_lsnn"}:
            model = _build_norse_model(input_dim=input_dim, params=snn_params, arch="lsnn" if "lsnn" in n else arch)
            return TorchSNNAdapter(model, lr=float(p.get("lr", 1e-3)), epochs=int(p.get("epochs", 5)), batch_size=int(p.get("batch_size", 64)))

        if n in {"spikingjelly_lif", "spikingjelly", "spikingjelly_temporal_conv"}:
            use_arch = "temporal_conv" if "temporal" in n else arch
            model = _build_spikingjelly_model(input_dim=input_dim, params=snn_params, arch=use_arch)
            return TorchSNNAdapter(model, lr=float(p.get("lr", 1e-3)), epochs=int(p.get("epochs", 5)), batch_size=int(p.get("batch_size", 64)))

        if n in {"tcn_spike", "temporal_conv_spike"}:
            model = _TemporalConvSpikingHead(input_dim=input_dim, params=snn_params)
            return TorchSNNAdapter(model, lr=float(p.get("lr", 1e-3)), epochs=int(p.get("epochs", 5)), batch_size=int(p.get("batch_size", 64)))

        raise ValueError(f"Unknown model: {spec.name}")


def save_prediction_artifacts(
    out_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_summary: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = target_summary or {
        "task_name": "next_bar_direction",
        "horizon": "1_bar",
        "label_type": "binary",
        "classes": ["down_or_flat", "up"],
        "label_semantics": "1 if next-bar close-to-close return > 0, else 0",
    }
    payload = {
        "model": model_name,
        "target_summary": summary,
        "y_true": y_true.astype(np.float32).tolist(),
        "y_prob": y_prob.astype(np.float32).tolist(),
    }
    path = out_dir / f"{model_name}_predictions.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
