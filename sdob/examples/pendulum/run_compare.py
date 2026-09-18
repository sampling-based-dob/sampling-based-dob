from __future__ import annotations

import argparse
from pathlib import Path

from sdob.examples.pendulum.config import load_experiment_config
from sdob.examples.pendulum.workspace import BenchmarkWorkspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the five-controller inverted-pendulum comparison and render a GIF.",
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "default.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None)
    parser.add_argument("--noise-sigma", type=float, default=None)
    parser.add_argument("--sampling-control-cost", type=float, default=None)
    parser.add_argument("--likelihood-alpha", type=float, default=None)
    parser.add_argument("--eta-min", type=float, default=None)
    parser.add_argument("--eta-max", type=float, default=None)
    parser.add_argument("--action-cost", type=float, default=None)
    parser.add_argument("--disturbance-compensation-cost", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--initial-theta", type=float, default=None)
    parser.add_argument("--initial-theta-dot", type=float, default=None)
    parser.add_argument(
        "--disturbance",
        choices=["constant_bias", "sinusoidal_bias", "step_bias"],
        default=None,
    )
    parser.add_argument("--disturbance-value", type=float, default=None)
    parser.add_argument("--style", choices=["gym", "schematic"], default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--skip-gif", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = vars(args).copy()
    config_path = overrides.pop("config")
    config = load_experiment_config(
        config_path,
        overrides={name.replace("-", "_"): value for name, value in overrides.items()},
    )
    print(f"Running inverted pendulum comparison from {config_path}", flush=True)
    print(f"Output directory: {config.output_dir}", flush=True)
    output = BenchmarkWorkspace(config).run()

    print("Inverted pendulum MPPI/DOB/SDOB comparison")
    print("=" * 52)
    print(output.result.summary.sort_values("total_cost").to_string(index=False))
    print(f"\nOutputs saved to: {config.output_dir}")
    print(f"Plot saved to: {output.plot_path}")
    if output.gif_path is not None:
        print(f"GIF saved to: {output.gif_path}")


if __name__ == "__main__":
    main()
