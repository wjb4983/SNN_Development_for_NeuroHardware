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


class UnifiedModel:
    """Shared API for model training/evaluation across baseline + SNN backends."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save_checkpoint(self, path: Path) -> None:
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
        self.estimator.fit(x_train, y_train)
        return {"train_samples": int(len(x_train))}

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(x)[:, 1].astype(np.float32)
        logits = self.estimator.decision_function(x)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def save_checkpoint(self, path: Path) -> None:
        import joblib

        joblib.dump(self.estimator, path)


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


class _SimpleLIFNet(torch.nn.Module):
    """Fallback lightweight pseudo-LIF implementation."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, beta: float = 0.9) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem = self.fc1(x)
        spk = (torch.tanh(mem * self.beta) > 0).float()
        return self.fc2(spk)


def _build_snntorch_model(input_dim: int, hidden_dim: int, use_native_lib: bool = False) -> torch.nn.Module:
    if not use_native_lib:
        return _SimpleLIFNet(input_dim=input_dim, hidden_dim=hidden_dim)
    try:
        import snntorch as snn
    except Exception:
        return _SimpleLIFNet(input_dim=input_dim, hidden_dim=hidden_dim)

    class SnnTorchNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.lif1 = snn.Leaky(beta=0.95, init_hidden=False)
            self.fc2 = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            cur1 = self.fc1(x)
            spk1, _ = self.lif1(cur1, None)
            return self.fc2(spk1)

    return SnnTorchNet()


def _build_norse_model(input_dim: int, hidden_dim: int) -> torch.nn.Module:
    try:
        import norse.torch as norse_torch
    except Exception:
        return _SimpleLIFNet(input_dim=input_dim, hidden_dim=hidden_dim)

    class NorseNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.lif = norse_torch.LIFCell()
            self.fc2 = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.fc1(x)
            spk, _ = self.lif(z)
            return self.fc2(spk)

    return NorseNet()


def _build_spikingjelly_model(input_dim: int, hidden_dim: int) -> torch.nn.Module:
    try:
        from spikingjelly.activation_based import layer, neuron
    except Exception:
        return _SimpleLIFNet(input_dim=input_dim, hidden_dim=hidden_dim)

    class SpikingJellyNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                layer.Linear(input_dim, hidden_dim),
                neuron.LIFNode(tau=2.0),
                layer.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return SpikingJellyNet()


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

        if n in {"snntorch_lif", "snntorch"}:
            model = _build_snntorch_model(input_dim=input_dim, hidden_dim=int(p.get("hidden_dim", 32)), use_native_lib=bool(p.get("use_native_lib", False)))
            return TorchSNNAdapter(
                model,
                lr=float(p.get("lr", 1e-3)),
                epochs=int(p.get("epochs", 5)),
                batch_size=int(p.get("batch_size", 64)),
            )

        if n in {"norse_lif", "norse_lsnn", "norse"}:
            model = _build_norse_model(input_dim=input_dim, hidden_dim=int(p.get("hidden_dim", 32)))
            return TorchSNNAdapter(
                model,
                lr=float(p.get("lr", 1e-3)),
                epochs=int(p.get("epochs", 5)),
                batch_size=int(p.get("batch_size", 64)),
            )

        if n in {"spikingjelly_lif", "spikingjelly"}:
            model = _build_spikingjelly_model(input_dim=input_dim, hidden_dim=int(p.get("hidden_dim", 32)))
            return TorchSNNAdapter(
                model,
                lr=float(p.get("lr", 1e-3)),
                epochs=int(p.get("epochs", 5)),
                batch_size=int(p.get("batch_size", 64)),
            )

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
