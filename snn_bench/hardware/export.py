from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .profiles import HardwareProfile


def _resolve_train_metrics_path(artifact_path: Path) -> Path:
    if artifact_path.is_dir():
        metrics_path = artifact_path / "train_metrics.json"
        if metrics_path.exists():
            return metrics_path
    if artifact_path.name == "train_metrics.json" and artifact_path.exists():
        return artifact_path
    raise FileNotFoundError(f"Unable to locate train_metrics.json from artifact path: {artifact_path}")


def _estimate_graph(train_metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    params = (train_metrics.get("model_params") or {}).copy()
    hidden_sizes = params.get("hidden_sizes")
    if not hidden_sizes:
        hidden_dim = int(params.get("hidden_dim", 64))
        depth = int(params.get("depth", 1))
        hidden_sizes = [hidden_dim for _ in range(max(depth, 1))]
    hidden_sizes = [int(v) for v in hidden_sizes]

    input_dim = int(train_metrics.get("train_info", {}).get("input_dim", 32))
    output_dim = int(params.get("output_dim", params.get("num_classes", 1)))
    quant_bits = int(params.get("quantization_bits", 8))
    timesteps = int(params.get("timesteps", 32))

    layer_widths = [input_dim, *hidden_sizes, output_dim]
    nodes = []
    edges = []
    max_fan_in = 0
    max_fan_out = 0

    for idx, width in enumerate(layer_widths):
        op = "readout" if idx == len(layer_widths) - 1 else ("lif" if idx > 0 else "input")
        nodes.append({"id": idx, "name": f"layer_{idx}", "op": op, "neurons": int(width)})

    for idx in range(len(layer_widths) - 1):
        src = layer_widths[idx]
        dst = layer_widths[idx + 1]
        max_fan_in = max(max_fan_in, src)
        max_fan_out = max(max_fan_out, dst)
        edges.append(
            {
                "source": idx,
                "target": idx + 1,
                "op": "linear",
                "shape": [int(src), int(dst)],
                "weights": int(src * dst),
            }
        )

    metadata = {
        "layer_count": len(layer_widths) - 1,
        "max_fan_in": int(max_fan_in),
        "max_fan_out": int(max_fan_out),
        "max_neurons_per_layer": int(max(layer_widths)),
        "quantization_bits": quant_bits,
        "timesteps": timesteps,
        "ops": sorted({edge["op"] for edge in edges} | {node["op"] for node in nodes if node["op"] != "input"}),
    }
    return nodes, edges, metadata


def export_graph_and_metadata(
    artifact_path: Path | str,
    out_dir: Path | str,
    profile: HardwareProfile,
    *,
    quantization_bits: int | None = None,
    max_neurons_per_layer: int | None = None,
    supported_ops: list[str] | None = None,
) -> dict[str, Any]:
    artifact = Path(artifact_path)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    metrics_path = _resolve_train_metrics_path(artifact)
    train_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    nodes, edges, export_meta = _estimate_graph(train_metrics)
    if quantization_bits is not None:
        export_meta["quantization_bits"] = int(quantization_bits)
    if max_neurons_per_layer is not None:
        export_meta["max_neurons_per_layer"] = int(max_neurons_per_layer)
    if supported_ops:
        export_meta["ops"] = sorted({*export_meta.get("ops", []), *[str(op) for op in supported_ops]})

    export_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": train_metrics.get("run_id", metrics_path.parent.name),
        "model": train_metrics.get("model"),
        "target_profile": profile.name,
        "profile_summary": {
            "max_fan_in": profile.max_fan_in,
            "max_fan_out": profile.max_fan_out,
            "weight_precision_bits": profile.weight_precision_bits,
            "max_neurons_per_layer": profile.max_neurons_per_layer,
            "max_timesteps": profile.max_timesteps,
            "supported_ops": profile.supported_ops,
        },
        "graph": {"nodes": nodes, "edges": edges},
        "metadata": export_meta,
    }

    export_path = output / "neuromorphic_graph_export.json"
    export_path.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
    return {"export_path": export_path, "payload": export_payload}
