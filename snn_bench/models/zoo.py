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
    output_dim: int


@dataclass(slots=True)
class TrainingDiagnostics:
    epoch: int
    train_loss: float
    val_loss: float | None
    calibration_proxy: float
    class_balance_proxy: float


class TrainingStrategy:
    def __init__(self, *, output_dim: int = 1, loss_name: str = "default", label_smoothing: float = 0.0, class_balance_beta: float = 0.999) -> None:
        self.output_dim = output_dim
        self.loss_name = loss_name
        self.label_smoothing = max(0.0, min(0.4, label_smoothing))
        self.class_balance_beta = class_balance_beta

    def build_target(self, yb: torch.Tensor) -> torch.Tensor:
        return yb

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, *, y_all: np.ndarray | None = None) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def diagnostics(self, probs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
        raise NotImplementedError


class BinaryClassificationStrategy(TrainingStrategy):
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, *, y_all: np.ndarray | None = None) -> torch.Tensor:
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = (1.0 - self.label_smoothing) * targets + 0.5 * self.label_smoothing
        if self.loss_name == "focal":
            return _binary_focal_loss(logits, targets)
        if self.loss_name == "class_balanced" and y_all is not None:
            counts = np.bincount(y_all.astype(np.int64), minlength=2).astype(np.float32)
            pos_weight = _class_balanced_weights(counts, beta=self.class_balance_beta)[1]
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device))
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def diagnostics(self, probs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
        calibration = float(torch.mean(torch.abs(probs - targets.float())).item())
        class_balance = float(torch.mean((probs >= 0.5).float()).item())
        return calibration, class_balance


class MulticlassClassificationStrategy(TrainingStrategy):
    def build_target(self, yb: torch.Tensor) -> torch.Tensor:
        return yb.long()

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, *, y_all: np.ndarray | None = None) -> torch.Tensor:
        if self.loss_name == "focal":
            return _multiclass_focal_loss(logits, targets)
        if self.loss_name == "class_balanced" and y_all is not None:
            counts = np.bincount(y_all.astype(np.int64), minlength=self.output_dim).astype(np.float32)
            weights = torch.tensor(_class_balanced_weights(counts, beta=self.class_balance_beta), dtype=logits.dtype, device=logits.device)
            return torch.nn.functional.cross_entropy(logits, targets, weight=weights, label_smoothing=self.label_smoothing)
        return torch.nn.functional.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def diagnostics(self, probs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
        picked = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        calibration = float(torch.mean(torch.abs(1.0 - picked)).item())
        class_balance = float(torch.std(probs.mean(dim=0), unbiased=False).item())
        return calibration, class_balance


class VolatilityRegressionStrategy(TrainingStrategy):
    def loss(self, logits: torch.Tensor, targets: torch.Tensor, *, y_all: np.ndarray | None = None) -> torch.Tensor:
        targets = targets.float()
        if self.loss_name == "huber":
            return torch.nn.functional.huber_loss(logits, targets, delta=1.0)
        return torch.nn.functional.mse_loss(logits, targets)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.relu(logits)

    def diagnostics(self, probs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
        calibration = float(torch.mean(torch.abs(probs - targets.float())).item())
        class_balance = float(torch.std(probs, unbiased=False).item())
        return calibration, class_balance


def _class_balanced_weights(counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    effective_num = 1.0 - np.power(beta, np.maximum(counts, 1.0))
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    return weights / np.mean(weights)


def _binary_focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(targets > 0.5, alpha, 1.0 - alpha)
    loss = alpha_t * ((1.0 - pt) ** gamma) * bce
    return loss.mean()


def _multiclass_focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def _auxiliary_objective_loss(inputs: torch.Tensor, logits: torch.Tensor, *, objective: str) -> torch.Tensor:
    if objective == "reconstruction":
        if inputs.ndim == 3:
            target = inputs.mean(dim=1)
        else:
            target = inputs
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        expanded = logits
        if logits.shape[-1] != target.shape[-1]:
            repeat = int(np.ceil(target.shape[-1] / max(1, logits.shape[-1])))
            expanded = logits.repeat(1, repeat)[:, : target.shape[-1]]
        return torch.nn.functional.mse_loss(expanded, target)
    if objective == "contrastive":
        flat = inputs.reshape(inputs.shape[0], -1)
        norm_feat = torch.nn.functional.normalize(flat, dim=-1)
        norm_logits = torch.nn.functional.normalize(logits.reshape(logits.shape[0], -1), dim=-1)
        sim = torch.sum(norm_feat[:, : norm_logits.shape[1]] * norm_logits, dim=-1)
        return 1.0 - sim.mean()
    return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)


def _select_training_strategy(strategy_name: str, *, output_dim: int, loss_name: str, label_smoothing: float, class_balance_beta: float) -> TrainingStrategy:
    if strategy_name in {"multiclass", "ordinal"} or output_dim > 1:
        return MulticlassClassificationStrategy(
            output_dim=output_dim,
            loss_name=loss_name,
            label_smoothing=label_smoothing,
            class_balance_beta=class_balance_beta,
        )
    if strategy_name in {"volatility_regression", "regression"}:
        return VolatilityRegressionStrategy(
            output_dim=1,
            loss_name=loss_name,
            label_smoothing=0.0,
            class_balance_beta=class_balance_beta,
        )
    return BinaryClassificationStrategy(
        output_dim=1,
        loss_name=loss_name,
        label_smoothing=label_smoothing,
        class_balance_beta=class_balance_beta,
    )


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
        if proba.ndim == 2 and proba.shape[1] > 1:
            pred = np.argmax(proba, axis=1).astype(np.int64)
            metrics = {
                "accuracy": float(accuracy_score(y, pred)),
                "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            }
        elif np.issubdtype(y.dtype, np.floating):
            mse = float(np.mean((proba.reshape(-1) - y.reshape(-1)) ** 2))
            metrics = {"mse": mse, "rmse": float(np.sqrt(mse))}
        else:
            pred = (proba >= 0.5).astype(np.int64)
            metrics = {
                "accuracy": float(accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred, zero_division=0)),
            }
            if len(np.unique(y)) > 1:
                metrics["roc_auc"] = float(roc_auc_score(y, proba))
            else:
                metrics["roc_auc"] = 0.5
        return metrics


class NaivePersistenceAdapter(UnifiedModel):
    """Simple persistence baseline for binary classification.

    Predicts the last observed train label probability for every future sample.
    """

    def __init__(self, confidence: float = 0.7) -> None:
        self.last_label = 0
        self.confidence = min(max(float(confidence), 0.5), 0.999)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        if len(y_train) == 0:
            raise ValueError("NaivePersistenceAdapter requires non-empty labels")
        self.last_label = int(np.asarray(y_train).reshape(-1)[-1] > 0)
        return {"train_samples": int(len(x_train)), "last_label": self.last_label}

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = self.confidence if self.last_label == 1 else (1.0 - self.confidence)
        return np.full(shape=(len(x),), fill_value=p, dtype=np.float32)

    def save_checkpoint(self, path: Path) -> None:
        payload = {"last_label": self.last_label, "confidence": self.confidence}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_checkpoint(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.last_label = int(payload.get("last_label", 0))
        self.confidence = float(payload.get("confidence", 0.7))


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
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 64,
        *,
        output_dim: int = 1,
        strategy: str = "classification",
        loss_name: str = "default",
        label_smoothing: float = 0.0,
        class_balance_beta: float = 0.999,
    ) -> None:
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.strategy_name = strategy
        self.training_strategy = _select_training_strategy(
            strategy_name=strategy,
            output_dim=output_dim,
            loss_name=loss_name,
            label_smoothing=label_smoothing,
            class_balance_beta=class_balance_beta,
        )
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        epochs = int(kwargs.get("epochs", self.epochs))
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        val_split = float(kwargs.get("val_split", 0.1))
        early_stopping_patience = int(kwargs.get("early_stopping_patience", 0))
        grad_clip_norm = float(kwargs.get("grad_clip_norm", 0.0))
        scheduler_name = str(kwargs.get("scheduler", "none")).lower()
        use_amp = bool(kwargs.get("mixed_precision", False) and device.type == "cuda")
        aux_objective = str(kwargs.get("aux_objective", "none")).lower()
        aux_weight = float(kwargs.get("aux_weight", 0.0))

        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        target_dtype = torch.float32 if isinstance(self.training_strategy, (BinaryClassificationStrategy, VolatilityRegressionStrategy)) else torch.long
        y_tensor = torch.tensor(y_train, dtype=target_dtype)

        train_len = len(x_tensor)
        val_len = int(train_len * val_split) if val_split > 0 else 0
        if val_len >= train_len:
            val_len = max(0, train_len - 1)
        if val_len > 0:
            perm = torch.randperm(train_len)
            val_idx = perm[:val_len]
            train_idx = perm[val_len:]
            train_ds = torch.utils.data.TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
            val_ds = torch.utils.data.TensorDataset(x_tensor[val_idx], y_tensor[val_idx])
        else:
            train_ds = torch.utils.data.TensorDataset(x_tensor, y_tensor)
            val_ds = None

        loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=max(1, epochs))
        elif scheduler_name in {"one_cycle", "onecycle"}:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.lr, epochs=max(1, epochs), steps_per_epoch=max(1, len(loader)))

        history: list[dict[str, float]] = []
        best_val = float("inf")
        stale_epochs = 0

        self.model.train()
        for epoch in range(epochs):
            epoch_losses: list[float] = []
            epoch_probs: list[torch.Tensor] = []
            epoch_targets: list[torch.Tensor] = []
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                self.optim.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self.model(xb)
                    logits = logits.squeeze(-1) if self.output_dim == 1 else logits
                    targets = self.training_strategy.build_target(yb)
                    base_loss = self.training_strategy.loss(logits, targets, y_all=y_train)
                    aux_loss = _auxiliary_objective_loss(xb, logits, objective=aux_objective)
                    loss = base_loss + aux_weight * aux_loss
                try:
                    from spikingjelly.activation_based import functional as sj_functional

                    sj_functional.reset_net(self.model)
                except Exception:
                    pass
                scaler.scale(loss).backward()
                scaler.unscale_(self.optim)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                scaler.step(self.optim)
                scaler.update()
                if scheduler_name in {"one_cycle", "onecycle"} and scheduler is not None:
                    scheduler.step()
                epoch_losses.append(float(loss.item()))
                epoch_probs.append(self.training_strategy.predict(logits.detach()).cpu())
                epoch_targets.append(targets.detach().cpu())

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            train_probs = torch.cat(epoch_probs) if epoch_probs else torch.zeros(1)
            train_targets = torch.cat(epoch_targets) if epoch_targets else torch.zeros(1)
            calibration, class_balance = self.training_strategy.diagnostics(train_probs, train_targets)

            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_losses: list[float] = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = self.model(xb)
                        logits = logits.squeeze(-1) if self.output_dim == 1 else logits
                        targets = self.training_strategy.build_target(yb)
                        val_losses.append(float(self.training_strategy.loss(logits, targets, y_all=y_train).item()))
                self.model.train()
                val_loss = float(np.mean(val_losses)) if val_losses else None

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "calibration_proxy": calibration,
                    "class_balance_proxy": class_balance,
                }
            )
            if scheduler_name == "cosine" and scheduler is not None:
                scheduler.step()
            if val_loss is not None and early_stopping_patience > 0:
                if val_loss < best_val:
                    best_val = val_loss
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= early_stopping_patience:
                        break

        return {
            "loss_last": float(history[-1]["train_loss"]) if history else 0.0,
            "epochs": len(history),
            "diagnostics": history,
        }

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        device = next(self.model.parameters()).device
        self.model.eval()
        logits = self.model(torch.tensor(x, dtype=torch.float32, device=device))
        logits = logits.squeeze(-1) if self.output_dim == 1 else logits
        try:
            from spikingjelly.activation_based import functional as sj_functional

            sj_functional.reset_net(self.model)
        except Exception:
            pass
        probs = self.training_strategy.predict(logits)
        return probs.cpu().numpy().astype(np.float32)

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
        output_dim=max(1, int(params.get("output_dim", params.get("num_classes", 1)))),
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
        self.out = torch.nn.Linear(prev, params.output_dim)
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
        self.fc_out = torch.nn.Linear(params.hidden_sizes[0], params.output_dim)
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
        self.out_proj = torch.nn.Linear(h, params.output_dim)
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
        self.spike_head = torch.nn.Linear(c, params.output_dim)
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

        if n in {"naive_persistence", "persistence"}:
            return NaivePersistenceAdapter(confidence=float(p.get("confidence", 0.7)))

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
        strategy = str(p.get("training_strategy", "classification"))
        loss_name = str(p.get("loss", "default"))
        label_smoothing = float(p.get("label_smoothing", 0.0))
        class_balance_beta = float(p.get("class_balance_beta", 0.999))

        def _make_torch_adapter(model: torch.nn.Module) -> TorchSNNAdapter:
            return TorchSNNAdapter(
                model,
                lr=float(p.get("lr", 1e-3)),
                epochs=int(p.get("epochs", 5)),
                batch_size=int(p.get("batch_size", 64)),
                output_dim=snn_params.output_dim,
                strategy=strategy,
                loss_name=loss_name,
                label_smoothing=label_smoothing,
                class_balance_beta=class_balance_beta,
            )

        if n in {"snntorch_lif", "snntorch", "snntorch_multilif"}:
            model = _build_snntorch_model(input_dim=input_dim, params=snn_params, arch=arch, use_native_lib=bool(p.get("use_native_lib", False)))
            return _make_torch_adapter(model)

        if n in {"snntorch_alif", "snntorch_adaptive"}:
            model = _build_snntorch_model(input_dim=input_dim, params=snn_params, arch="alif", use_native_lib=bool(p.get("use_native_lib", False)))
            return _make_torch_adapter(model)

        if n in {"norse_lif", "norse_lsnn", "norse", "norse_recurrent_lsnn"}:
            model = _build_norse_model(input_dim=input_dim, params=snn_params, arch="lsnn" if "lsnn" in n else arch)
            return _make_torch_adapter(model)

        if n in {"spikingjelly_lif", "spikingjelly", "spikingjelly_temporal_conv"}:
            use_arch = "temporal_conv" if "temporal" in n else arch
            model = _build_spikingjelly_model(input_dim=input_dim, params=snn_params, arch=use_arch)
            return _make_torch_adapter(model)

        if n in {"tcn_spike", "temporal_conv_spike"}:
            model = _TemporalConvSpikingHead(input_dim=input_dim, params=snn_params)
            return _make_torch_adapter(model)

        raise ValueError(f"Unknown model: {spec.name}")


def save_prediction_artifacts(
    out_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_summary: dict[str, Any] | None = None,
    reference_close: np.ndarray | None = None,
    reference_next_close: np.ndarray | None = None,
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
    if reference_close is not None:
        payload["reference_close"] = reference_close.astype(np.float32).tolist()
    if reference_next_close is not None:
        payload["reference_next_close"] = reference_next_close.astype(np.float32).tolist()
    path = out_dir / f"{model_name}_predictions.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
