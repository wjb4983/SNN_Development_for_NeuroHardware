from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class HardwareProfile:
    name: str
    description: str
    max_fan_in: int
    max_fan_out: int
    weight_precision_bits: int
    state_precision_bits: int
    max_neurons_per_layer: int
    max_layers: int
    max_timesteps: int
    supported_ops: list[str]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "HardwareProfile":
        return cls(
            name=str(payload.get("name", "custom_profile")),
            description=str(payload.get("description", "")),
            max_fan_in=int(payload.get("max_fan_in", 256)),
            max_fan_out=int(payload.get("max_fan_out", 256)),
            weight_precision_bits=int(payload.get("weight_precision_bits", 8)),
            state_precision_bits=int(payload.get("state_precision_bits", 16)),
            max_neurons_per_layer=int(payload.get("max_neurons_per_layer", 1024)),
            max_layers=int(payload.get("max_layers", 8)),
            max_timesteps=int(payload.get("max_timesteps", 128)),
            supported_ops=[str(op) for op in payload.get("supported_ops", ["linear", "lif", "readout"])],
        )


_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "loihi2_like": {
        "name": "loihi2_like",
        "description": "Loihi2-inspired constraints for inference deployment planning.",
        "max_fan_in": 4096,
        "max_fan_out": 4096,
        "weight_precision_bits": 8,
        "state_precision_bits": 16,
        "max_neurons_per_layer": 8192,
        "max_layers": 64,
        "max_timesteps": 512,
        "supported_ops": ["linear", "conv1d", "lif", "alif", "readout"],
    },
    "generic_edge_neuromorphic": {
        "name": "generic_edge_neuromorphic",
        "description": "Conservative edge-device profile for low-power deployments.",
        "max_fan_in": 512,
        "max_fan_out": 512,
        "weight_precision_bits": 6,
        "state_precision_bits": 8,
        "max_neurons_per_layer": 2048,
        "max_layers": 16,
        "max_timesteps": 128,
        "supported_ops": ["linear", "lif", "readout"],
    },
}


def available_profile_names() -> list[str]:
    return sorted(_BUILTIN_PROFILES.keys())


def load_hardware_profile(target_profile: str, profile_yaml: Path | None = None) -> HardwareProfile:
    if profile_yaml is not None:
        payload = yaml.safe_load(profile_yaml.read_text(encoding="utf-8")) or {}
        return HardwareProfile.from_mapping(payload)

    if target_profile in _BUILTIN_PROFILES:
        return HardwareProfile.from_mapping(_BUILTIN_PROFILES[target_profile])

    default_path = Path("snn_bench/configs/hardware") / f"{target_profile}.yaml"
    if default_path.exists():
        payload = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
        return HardwareProfile.from_mapping(payload)

    names = ", ".join(available_profile_names())
    raise ValueError(f"Unknown target profile '{target_profile}'. Available profiles: {names}")
