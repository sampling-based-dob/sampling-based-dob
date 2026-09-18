from __future__ import annotations

from .controllers import DOBMPPIController, SDOBMPPIController, MPPIController, RMPPIController, TubeMPPIController
from .dynamics import PendulumConfig
from .interfaces import ControllerBase
from .mppi import MPPIConfig


def make_controllers(
    config: PendulumConfig,
    num_samples: int = 512,
    horizon: int = 25,
    lambda_: float = 0.35,
    noise_sigma: float = 1.2,
    adaptive_temperature: bool = True,
    eta_min: float = 0.05,
    eta_max: float = 0.1,
    lambda_decrease: float = 0.9,
    lambda_increase: float = 1.2,
    lambda_min: float = 1.0e-3,
    lambda_max: float = 1.0e3,
    sampling_control_cost: float = 0.01,
    likelihood_alpha: float = 0.0,
    device: str = "cpu",
    seed: int = 0,
) -> list[ControllerBase]:
    base = MPPIConfig(
        horizon=horizon,
        num_samples=num_samples,
        lambda_=lambda_,
        noise_sigma=noise_sigma,
        adaptive_temperature=adaptive_temperature,
        eta_min=eta_min,
        eta_max=eta_max,
        lambda_decrease=lambda_decrease,
        lambda_increase=lambda_increase,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        sampling_control_cost=sampling_control_cost,
        likelihood_alpha=likelihood_alpha,
        device=device,
        seed=seed,
    )
    return [
        MPPIController(config, MPPIConfig(**{**base.__dict__, "seed": seed + 1})),
        DOBMPPIController(config, MPPIConfig(**{**base.__dict__, "seed": seed + 2})),
        SDOBMPPIController(config, MPPIConfig(**{**base.__dict__, "seed": seed + 3})),
        TubeMPPIController(config, MPPIConfig(**{**base.__dict__, "seed": seed + 4})),
        RMPPIController(config, MPPIConfig(**{**base.__dict__, "seed": seed + 5})),
    ]
