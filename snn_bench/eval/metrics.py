from __future__ import annotations

from typing import Any

import numpy as np


def binary_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)
    return float((preds == labels).mean())


def _as_spike_array(spikes_or_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    arr = np.asarray(spikes_or_probs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    binary = (arr >= threshold).astype(np.float32)
    return binary


def spike_sparsity(spikes_or_probs: np.ndarray, threshold: float = 0.5) -> float:
    spikes = _as_spike_array(spikes_or_probs, threshold=threshold)
    return float(1.0 - spikes.mean())


def firing_rate_distribution(spikes_or_probs: np.ndarray, dt_ms: float = 1.0, threshold: float = 0.5) -> dict[str, float]:
    spikes = _as_spike_array(spikes_or_probs, threshold=threshold)
    rates_hz = spikes.mean(axis=0) * (1000.0 / max(dt_ms, 1e-6))
    return {
        "mean_hz": float(np.mean(rates_hz)),
        "std_hz": float(np.std(rates_hz)),
        "p10_hz": float(np.percentile(rates_hz, 10)),
        "p50_hz": float(np.percentile(rates_hz, 50)),
        "p90_hz": float(np.percentile(rates_hz, 90)),
    }


def temporal_precision(spikes_or_probs: np.ndarray, threshold: float = 0.5) -> float:
    spikes = _as_spike_array(spikes_or_probs, threshold=threshold)
    precision_scores: list[float] = []
    for n in range(spikes.shape[1]):
        idx = np.flatnonzero(spikes[:, n] > 0)
        if idx.size < 3:
            continue
        isi = np.diff(idx)
        cv = float(np.std(isi) / (np.mean(isi) + 1e-6))
        precision_scores.append(1.0 / (1.0 + cv))
    if not precision_scores:
        return 0.0
    return float(np.mean(precision_scores))


def stability_metrics(spikes_or_probs: np.ndarray, window: int = 16, threshold: float = 0.5) -> dict[str, float]:
    spikes = _as_spike_array(spikes_or_probs, threshold=threshold)
    if spikes.shape[0] < 2:
        return {"firing_rate_drift": 0.0, "rolling_rate_std": 0.0, "spike_balance": 0.0}

    step_rate = spikes.mean(axis=1)
    w = max(2, min(window, len(step_rate)))
    kernel = np.ones(w, dtype=np.float32) / w
    rolling = np.convolve(step_rate, kernel, mode="valid")
    drift = float(abs(rolling[-1] - rolling[0])) if rolling.size > 1 else 0.0
    return {
        "firing_rate_drift": drift,
        "rolling_rate_std": float(np.std(rolling)) if rolling.size else 0.0,
        "spike_balance": float(np.std(spikes.mean(axis=0))),
    }


def bio_plausibility_metrics(
    spikes_or_probs: np.ndarray,
    *,
    dt_ms: float = 1.0,
    threshold: float = 0.5,
    stability_window: int = 16,
) -> dict[str, Any]:
    rate_dist = firing_rate_distribution(spikes_or_probs, dt_ms=dt_ms, threshold=threshold)
    stable = stability_metrics(spikes_or_probs, window=stability_window, threshold=threshold)
    return {
        "spike_sparsity": spike_sparsity(spikes_or_probs, threshold=threshold),
        "firing_rate_distribution": rate_dist,
        "temporal_precision": temporal_precision(spikes_or_probs, threshold=threshold),
        "stability": stable,
    }
