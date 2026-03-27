from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class BioPlausibleConfig:
    hidden_dim: int = 64
    output_dim: int = 1
    neuron_model: str = "lif"  # lif | alif | adex_like
    tau_m: float = 20.0
    tau_syn: float = 8.0
    refractory_steps: int = 2
    v_th: float = 1.0
    v_reset: float = 0.0
    dt: float = 1.0
    adaptation_tau: float = 120.0
    adaptation_strength: float = 0.05
    adex_delta_t: float = 0.15
    adex_exp_scale: float = 0.04
    stdp_rule: str = "pair"  # pair | triplet | reward_modulated
    tau_pre: float = 20.0
    tau_post: float = 20.0
    tau_pre_long: float = 80.0
    tau_post_long: float = 80.0
    a_plus: float = 0.01
    a_minus: float = 0.012
    reward_scale: float = 0.5
    reward_signal: float = 0.0
    eligibility_tau: float = 100.0
    synaptic_delay_steps: int = 0
    conductance_based: bool = False
    e_exc: float = 1.2
    homeostasis_rate_target_hz: float = 8.0
    homeostasis_reg_strength: float = 0.002
    threshold_adaptation_rate: float = 0.01

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "BioPlausibleConfig":
        normalized = dict(params)
        if "hidden_sizes" in normalized and normalized["hidden_sizes"]:
            normalized["hidden_dim"] = int(normalized["hidden_sizes"][0])
        if "output_dim" in normalized:
            normalized["output_dim"] = int(normalized["output_dim"])
        return cls(**{k: v for k, v in normalized.items() if k in cls.__dataclass_fields__})


class BioPlausibleSNN(torch.nn.Module):
    """Differentiable proxy for biologically inspired neuron/synapse dynamics."""

    def __init__(self, input_dim: int, config: BioPlausibleConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.in_proj = torch.nn.Linear(input_dim, h)
        self.rec_proj = torch.nn.Linear(h, h)
        self.out_proj = torch.nn.Linear(h, config.output_dim)
        self._last_state: dict[str, float] = {}

    def _surrogate_spike(self, membrane: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        smooth = torch.sigmoid(6.0 * (membrane - threshold))
        hard = (membrane >= threshold).float()
        return hard.detach() - smooth.detach() + smooth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        cfg = self.config
        batch, steps, _ = x.shape
        hidden = self.in_proj.out_features
        device = x.device

        mem = torch.zeros(batch, hidden, device=device)
        syn = torch.zeros_like(mem)
        adapt = torch.zeros_like(mem)
        pre_trace = torch.zeros_like(mem)
        post_trace = torch.zeros_like(mem)
        pre_trace_long = torch.zeros_like(mem)
        post_trace_long = torch.zeros_like(mem)
        eligibility = torch.zeros_like(mem)
        refractory = torch.zeros_like(mem)

        delay_steps = max(0, int(cfg.synaptic_delay_steps))
        delay_buffer = [torch.zeros_like(mem) for _ in range(delay_steps + 1)]

        decay_m = torch.exp(torch.tensor(-cfg.dt / max(cfg.tau_m, 1e-3), device=device))
        decay_syn = torch.exp(torch.tensor(-cfg.dt / max(cfg.tau_syn, 1e-3), device=device))
        decay_pre = torch.exp(torch.tensor(-cfg.dt / max(cfg.tau_pre, 1e-3), device=device))
        decay_post = torch.exp(torch.tensor(-cfg.dt / max(cfg.tau_post, 1e-3), device=device))
        decay_pre_long = torch.exp(torch.tensor(-cfg.dt / max(cfg.tau_pre_long, 1e-3), device=device))
        decay_post_long = torch.exp(torch.tensor(-cfg.dt / max(cfg.tau_post_long, 1e-3), device=device))
        decay_elig = torch.exp(torch.tensor(-cfg.dt / max(cfg.eligibility_tau, 1e-3), device=device))
        decay_adapt = torch.exp(torch.tensor(-cfg.dt / max(cfg.adaptation_tau, 1e-3), device=device))

        outputs: list[torch.Tensor] = []
        spike_history: list[torch.Tensor] = []

        base_threshold = torch.full((batch, hidden), float(cfg.v_th), device=device)

        for t in range(steps):
            input_current = self.in_proj(x[:, t, :])
            delayed_spike = delay_buffer.pop(0)
            recurrent_drive = self.rec_proj(delayed_spike)

            syn = decay_syn * syn + input_current + recurrent_drive
            if cfg.conductance_based:
                syn = syn * (cfg.e_exc - mem)

            dynamic_threshold = base_threshold + adapt
            if cfg.neuron_model == "adex_like":
                exp_term = cfg.adex_exp_scale * (
                    torch.exp(torch.clamp((mem - dynamic_threshold) / max(cfg.adex_delta_t, 1e-3), min=-6.0, max=6.0)) - 1.0
                )
                mem = decay_m * mem + syn + exp_term
            else:
                mem = decay_m * mem + syn

            can_spike = (refractory <= 0).float()
            spikes = self._surrogate_spike(mem, dynamic_threshold) * can_spike
            spike_history.append(spikes)

            if cfg.neuron_model in {"alif", "adex_like"}:
                adapt = decay_adapt * adapt + cfg.adaptation_strength * spikes
            else:
                adapt = torch.zeros_like(adapt)

            pre_trace = decay_pre * pre_trace + spikes
            post_trace = decay_post * post_trace + spikes
            pre_trace_long = decay_pre_long * pre_trace_long + spikes
            post_trace_long = decay_post_long * post_trace_long + spikes

            stdp_drive = cfg.a_plus * pre_trace * post_trace - cfg.a_minus * post_trace
            if cfg.stdp_rule == "triplet":
                stdp_drive = stdp_drive + 0.5 * cfg.a_plus * pre_trace_long * post_trace
            elif cfg.stdp_rule == "reward_modulated":
                stdp_drive = stdp_drive * (1.0 + cfg.reward_scale * cfg.reward_signal)

            eligibility = decay_elig * eligibility + stdp_drive
            homeo_target = cfg.homeostasis_rate_target_hz * (cfg.dt / 1000.0)
            rate_error = spikes.mean(dim=0, keepdim=True) - homeo_target
            dynamic_threshold = dynamic_threshold + cfg.threshold_adaptation_rate * rate_error

            mem = mem * (1.0 - spikes) + cfg.v_reset * spikes
            mem = mem - cfg.homeostasis_reg_strength * rate_error
            refractory = torch.where(spikes > 0, torch.full_like(refractory, float(cfg.refractory_steps)), torch.clamp(refractory - 1.0, min=0.0))

            delay_buffer.append(spikes)
            outputs.append(self.out_proj(spikes + 0.1 * eligibility))

        stacked = torch.stack(outputs, dim=1)
        mean_spikes = torch.stack(spike_history, dim=1).mean()
        self._last_state = {
            "mean_spike_rate": float(mean_spikes.detach().cpu()),
            "mean_threshold": float(dynamic_threshold.mean().detach().cpu()),
            "eligibility_norm": float(eligibility.abs().mean().detach().cpu()),
        }
        return stacked.mean(dim=1)

    def get_last_state_summary(self) -> dict[str, float]:
        return dict(self._last_state)
