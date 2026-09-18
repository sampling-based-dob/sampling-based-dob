"""Run a config-driven SDOB navigation comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .benchmark import CompareRunner
from .config import load_experiment_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MPPI/DOB/SDOB/Tube-MPPI/RMPPI navigation comparison.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "compare_8obs.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenario", choices=["four", "4obs", "eight", "8obs", "chaotic"], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to the YAML value.",
    )
    parser.add_argument(
        "--disturbance",
        choices=["none", "sinusoidal_slip", "sinusoidal_input", "constant_input"],
        default=None,
    )
    parser.add_argument("--slip-amplitude", type=float, default=None)
    parser.add_argument("--slip-frequency", type=float, default=None)
    parser.add_argument("--input-disturbance-frequency", type=float, default=None)
    parser.add_argument("--perception-radius", type=float, default=None)
    parser.add_argument("--sdob-radius", type=float, default=None)
    parser.add_argument("--lookahead-dist", type=float, default=None)
    parser.add_argument("--num-cost-queries", type=int, default=None)
    parser.add_argument("--gif-skip", type=int, default=None)
    parser.add_argument("--max-gif-frames", type=int, default=None)
    parser.add_argument("--skip-gif", action="store_true", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    overrides = vars(args).copy()
    config_path = overrides.pop("config")
    if overrides.get("device") is None and torch.cuda.is_available():
        overrides["device"] = None
    config = load_experiment_config(config_path, overrides=overrides)

    print(f"Running SDOB navigation comparison from {config_path}", flush=True)
    print(f"Output directory: {config.output_dir}", flush=True)
    output = CompareRunner(config).run()

    print("SDOB navigation MPPI/DOB/SDOB comparison")
    print("=" * 56)
    for result in output.results:
        m = result.metrics
        hit = "not reached"
        if m["first_hit_time"] == m["first_hit_time"]:
            hit = f"{m['first_hit_time']:.2f}s"
        print(
            f"{result.name:10s}  final={m['final_error']:.3f}  "
            f"tail={m['tail_error']:.3f}  mean safety={m['mean_safety']:.3f}  "
            f"collision={'YES' if m['collision'] else 'no'}  hit={hit}"
        )

    print(f"\nSummary saved to: {output.summary_csv}")
    print(f"Trajectories saved to: {output.trajectories_csv}")
    print(f"Plot saved to: {output.figure_path}")
    if output.gif_path is not None:
        print(f"GIF saved to: {output.gif_path}")


if __name__ == "__main__":
    main()
