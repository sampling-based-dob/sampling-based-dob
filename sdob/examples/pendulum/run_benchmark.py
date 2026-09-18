from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import replace

import numpy as np

from .benchmark import make_controllers, plot_result, run_suite, save_result
from .dynamics import PendulumConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inverted-pendulum MPPI/DOB/SDOB/Tube-MPPI/RMPPI comparisons.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pendulum/default"))
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--sampling-control-cost", type=float, default=0.01)
    parser.add_argument("--likelihood-alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-theta", type=float, default=0.35)
    parser.add_argument("--initial-theta-dot", type=float, default=0.0)
    parser.add_argument(
        "--disturbance",
        choices=["constant_bias", "sinusoidal_bias", "step_bias"],
        default="constant_bias",
    )
    parser.add_argument("--disturbance-value", type=float, default=0.75)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--max-torque", type=float, default=None)
    parser.add_argument("--gym-pendulum", action="store_true", help="Use Gymnasium Pendulum-v1 dynamics and limits.")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PendulumConfig.gym_pendulum() if args.gym_pendulum else PendulumConfig()
    if args.dt is not None:
        config = replace(config, dt=args.dt)
    if args.max_torque is not None:
        config = replace(config, max_torque=args.max_torque)
    controllers = make_controllers(
        config,
        num_samples=args.num_samples,
        horizon=args.horizon,
        sampling_control_cost=args.sampling_control_cost,
        likelihood_alpha=args.likelihood_alpha,
        seed=args.seed,
    )
    result = run_suite(
        controllers=controllers,
        config=config,
        steps=args.steps,
        initial_state=np.array([args.initial_theta, args.initial_theta_dot], dtype=float),
        disturbance_name=args.disturbance,
        disturbance_value=args.disturbance_value,
        seed=args.seed,
    )
    save_result(result, args.output_dir)
    plot_path = None if args.no_plot else plot_result(result, args.output_dir)

    print("Inverted pendulum DOB/SDOB benchmark")
    print("=" * 48)
    print(result.summary.sort_values("total_cost").to_string(index=False))
    print(f"\nOutputs saved to: {args.output_dir}")
    if plot_path is not None:
        print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
