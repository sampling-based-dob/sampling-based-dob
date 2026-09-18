from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .animate import animate_trajectories
from .benchmark import BenchmarkResult, plot_result, run_suite, save_result, with_nominal_mppi_baseline
from .config import ExperimentConfig
from .controllers import MPPIController
from .dynamics import PendulumConfig
from .interfaces import ControllerBase
from .mppi import MPPIConfig


@dataclass(frozen=True)
class RunnerOutput:
    result: BenchmarkResult
    summary_path: Path
    trajectory_path: Path
    plot_path: Path
    gif_path: Path | None


class PendulumEnvRunner:
    """Executes a controller suite on the pendulum task and writes artifacts."""

    def __init__(self, pendulum: PendulumConfig, config: ExperimentConfig) -> None:
        self.pendulum = pendulum
        self.config = config

    def run(self, controllers: list[ControllerBase]) -> RunnerOutput:
        baseline_result = run_suite(
            controllers=[
                MPPIController(
                    self.pendulum,
                    MPPIConfig(
                        horizon=self.config.horizon,
                        num_samples=self.config.num_samples,
                        lambda_=self.config.lambda_,
                        noise_sigma=self.config.noise_sigma,
                        adaptive_temperature=self.config.adaptive_temperature,
                        eta_min=self.config.eta_min,
                        eta_max=self.config.eta_max,
                        lambda_decrease=self.config.lambda_decrease,
                        lambda_increase=self.config.lambda_increase,
                        lambda_min=self.config.lambda_min,
                        lambda_max=self.config.lambda_max,
                        sampling_control_cost=self.config.sampling_control_cost,
                        likelihood_alpha=self.config.likelihood_alpha,
                        device=self.config.device,
                        seed=self.config.seed,
                    ),
                )
            ],
            config=self.pendulum,
            steps=self.config.steps,
            initial_state=np.array([self.config.initial_theta, self.config.initial_theta_dot], dtype=float),
            disturbance_name="constant_bias",
            disturbance_value=0.0,
            seed=self.config.seed,
            progress_label="no-disturbance baseline",
            device=self.config.device,
        )
        result = run_suite(
            controllers=controllers,
            config=self.pendulum,
            steps=self.config.steps,
            initial_state=np.array([self.config.initial_theta, self.config.initial_theta_dot], dtype=float),
            disturbance_name=self.config.disturbance,
            disturbance_value=self.config.disturbance_value,
            seed=self.config.seed,
            progress_label="disturbed comparison",
            device=self.config.device,
        )
        result = with_nominal_mppi_baseline(result, baseline_result)
        save_result(result, self.config.output_dir)
        plot_path = plot_result(result, self.config.output_dir)
        trajectory_path = self.config.output_dir / "trajectories.csv"
        gif_path = None
        if not self.config.skip_gif:
            gif_path = self.config.output_dir / "comparison_gym_dhat.gif"
            print(f"[render] writing GIF to {gif_path}...", flush=True)
            animate_trajectories(
                trajectory_path,
                gif_path,
                fps=self.config.fps,
                stride=self.config.stride,
                dpi=self.config.dpi,
                style=self.config.style,
            )
        return RunnerOutput(
            result=result,
            summary_path=self.config.output_dir / "summary.csv",
            trajectory_path=trajectory_path,
            plot_path=plot_path,
            gif_path=gif_path,
        )
