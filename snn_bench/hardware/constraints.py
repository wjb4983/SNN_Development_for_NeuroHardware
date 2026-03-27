from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .profiles import HardwareProfile


@dataclass(slots=True)
class ConstraintResult:
    name: str
    passed: bool
    observed: Any
    expected: Any
    remediation: str


def evaluate_constraints(export_meta: dict[str, Any], profile: HardwareProfile) -> list[ConstraintResult]:
    fan_in = int(export_meta.get("max_fan_in", 0))
    fan_out = int(export_meta.get("max_fan_out", 0))
    neurons = int(export_meta.get("max_neurons_per_layer", 0))
    timesteps = int(export_meta.get("timesteps", 0))
    weight_bits = int(export_meta.get("quantization_bits", 32))
    layers = int(export_meta.get("layer_count", 0))
    ops = {str(op) for op in export_meta.get("ops", [])}

    checks: list[ConstraintResult] = [
        ConstraintResult(
            name="fan_in",
            passed=fan_in <= profile.max_fan_in,
            observed=fan_in,
            expected=f"<= {profile.max_fan_in}",
            remediation="Reduce dense connectivity or partition layer projections into smaller blocks.",
        ),
        ConstraintResult(
            name="fan_out",
            passed=fan_out <= profile.max_fan_out,
            observed=fan_out,
            expected=f"<= {profile.max_fan_out}",
            remediation="Use sparse projections, pruning, or split output channels across cores.",
        ),
        ConstraintResult(
            name="precision",
            passed=weight_bits <= profile.weight_precision_bits,
            observed=f"{weight_bits}-bit",
            expected=f"<= {profile.weight_precision_bits}-bit",
            remediation="Apply post-training quantization or quantization-aware training for weights.",
        ),
        ConstraintResult(
            name="max_neurons_per_layer",
            passed=neurons <= profile.max_neurons_per_layer,
            observed=neurons,
            expected=f"<= {profile.max_neurons_per_layer}",
            remediation="Reduce hidden width, apply structured pruning, or split layers into stages.",
        ),
        ConstraintResult(
            name="layer_count",
            passed=layers <= profile.max_layers,
            observed=layers,
            expected=f"<= {profile.max_layers}",
            remediation="Collapse repeated blocks or distill to a shallower architecture.",
        ),
        ConstraintResult(
            name="timesteps",
            passed=timesteps <= profile.max_timesteps,
            observed=timesteps,
            expected=f"<= {profile.max_timesteps}",
            remediation="Lower simulation steps, shorten temporal window, or use faster dynamics.",
        ),
    ]

    unsupported_ops = sorted(op for op in ops if op not in set(profile.supported_ops))
    checks.append(
        ConstraintResult(
            name="supported_ops",
            passed=not unsupported_ops,
            observed=sorted(ops),
            expected=profile.supported_ops,
            remediation=(
                "Replace unsupported ops with profile-compatible primitives or insert conversion shims."
                if unsupported_ops
                else "No remediation required."
            ),
        )
    )
    return checks
