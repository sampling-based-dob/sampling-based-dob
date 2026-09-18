from __future__ import annotations

from sdob.workspace import BaseWorkspace

from .config import ExperimentConfig, load_experiment_config
from .dynamics import PendulumConfig
from .env_runner import PendulumEnvRunner, RunnerOutput
from .factory import make_controllers


class BenchmarkWorkspace(BaseWorkspace):
    """Owns construction and execution for one benchmark experiment."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config, config.output_dir)
        self.config = config
        self.pendulum = PendulumConfig.gym_pendulum(
            action_cost=config.action_cost,
            disturbance_compensation_cost=config.disturbance_compensation_cost,
        )
        self.controllers = make_controllers(
            self.pendulum,
            num_samples=config.num_samples,
            horizon=config.horizon,
            lambda_=config.lambda_,
            noise_sigma=config.noise_sigma,
            adaptive_temperature=config.adaptive_temperature,
            eta_min=config.eta_min,
            eta_max=config.eta_max,
            lambda_decrease=config.lambda_decrease,
            lambda_increase=config.lambda_increase,
            lambda_min=config.lambda_min,
            lambda_max=config.lambda_max,
            sampling_control_cost=config.sampling_control_cost,
            likelihood_alpha=config.likelihood_alpha,
            device=config.device,
            seed=config.seed,
        )
        self.env_runner = PendulumEnvRunner(self.pendulum, config)

    def run(self) -> RunnerOutput:
        return self.env_runner.run(self.controllers)
