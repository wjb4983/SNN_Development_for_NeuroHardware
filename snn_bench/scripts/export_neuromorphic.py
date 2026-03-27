from __future__ import annotations

import argparse
import json
from pathlib import Path

from snn_bench.hardware import emit_deployment_report, export_graph_and_metadata, load_hardware_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained artifact for neuromorphic deployment checks")
    parser.add_argument("--artifact", type=Path, required=True, help="Path to run directory or train_metrics.json")
    parser.add_argument("--target-profile", default="loihi2_like", help="Hardware target profile name")
    parser.add_argument("--profile-yaml", type=Path, default=None, help="Optional custom hardware profile YAML")
    parser.add_argument("--quantization", type=int, default=None, help="Override quantization bits for export metadata")
    parser.add_argument("--max-neurons-per-layer", type=int, default=None, help="Override max neurons per layer")
    parser.add_argument(
        "--supported-op",
        action="append",
        default=None,
        help="Add op(s) to exported supported ops metadata. Repeatable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination for export + deployment report files. Defaults to <artifact_dir>/hardware",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_hardware_profile(args.target_profile, profile_yaml=args.profile_yaml)

    artifact_dir = args.artifact if args.artifact.is_dir() else args.artifact.parent
    out_dir = args.out_dir or (artifact_dir / "hardware")

    export_out = export_graph_and_metadata(
        artifact_path=args.artifact,
        out_dir=out_dir,
        profile=profile,
        quantization_bits=args.quantization,
        max_neurons_per_layer=args.max_neurons_per_layer,
        supported_ops=args.supported_op,
    )
    report_out = emit_deployment_report(export_out["payload"], profile=profile, out_dir=out_dir)

    print(
        json.dumps(
            {
                "target_profile": profile.name,
                "export_path": str(export_out["export_path"]),
                "deployment_report_json": str(report_out["json_report"]),
                "deployment_report_markdown": str(report_out["markdown_report"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
