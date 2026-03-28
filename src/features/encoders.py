from __future__ import annotations

import numpy as np


def rate_code(x: np.ndarray, timesteps: int = 16, max_rate: float = 0.9, seed: int = 7) -> np.ndarray:
    """Bernoulli rate coding for normalized inputs."""
    rng = np.random.default_rng(seed)
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-9)
    probs = np.clip(x_n * max_rate, 0.0, max_rate)
    spikes = rng.random((timesteps, *x.shape)) < probs
    return spikes.astype(np.float32)


def ttfs_code(x: np.ndarray, timesteps: int = 16) -> np.ndarray:
    """Time-to-first-spike coding where larger magnitude spikes earlier."""
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-9)
    first_fire = np.floor((1.0 - x_n) * (timesteps - 1)).astype(int)
    spikes = np.zeros((timesteps, *x.shape), dtype=np.float32)
    for t in range(timesteps):
        spikes[t] = (first_fire == t).astype(np.float32)
    return spikes
