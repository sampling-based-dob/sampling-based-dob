from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sdob.observer import score_state_effect_candidates

from .dynamics import PendulumConfig, pendulum_step_np, wrap_angle_np, wrap_angle_torch


def _is_velocity_saturated(config: PendulumConfig, state: np.ndarray, feedback: np.ndarray, margin: float) -> bool:
    threshold = float(config.max_speed) * float(margin)
    if threshold <= 0.0:
        return False
    theta_dot = float(np.asarray(state, dtype=float)[1])
    next_dot = float(np.asarray(feedback, dtype=float)[1])
    return abs(theta_dot) >= threshold or abs(next_dot) >= threshold


def _clamp_rate(rate: float, rate_limit: float) -> float:
    limit = abs(float(rate_limit))
    if limit <= 0.0:
        return 0.0
    return float(np.clip(rate, -limit, limit))


@dataclass(frozen=True)
class DisturbanceEstimate:
    mean: float
    sigma: float
    measurement: float
    rate: float = 0.0

    def forecast(
        self,
        horizon: int,
        dt: float,
        lower: float | None = None,
        upper: float | None = None,
        max_time: float | None = None,
    ) -> np.ndarray:
        steps = np.arange(max(1, int(horizon)), dtype=float) * float(dt)
        if max_time is not None:
            steps = np.minimum(steps, float(max_time))
        values = float(self.mean) + steps * float(self.rate)
        if lower is not None or upper is not None:
            values = np.clip(values, lower if lower is not None else -np.inf, upper if upper is not None else np.inf)
        return values.astype(float)


class ClassicalDOBObserver:
    """Matched torque-bias observer from one-step acceleration residual."""

    def __init__(
        self,
        config: PendulumConfig,
        alpha: float = 0.75,
        sigma_alpha: float = 0.2,
        rate_limit: float = 1.2,
        saturation_margin: float = 0.98,
    ) -> None:
        self.config = config
        self.alpha = alpha
        self.sigma_alpha = sigma_alpha
        self.rate_limit = rate_limit
        self.saturation_margin = saturation_margin
        self.mean = 0.0
        self.rate = 0.0
        self.var = 0.1**2
        self.initialized = False

    def reset(self) -> None:
        self.mean = 0.0
        self.rate = 0.0
        self.var = 0.1**2
        self.initialized = False

    def update(self, state: np.ndarray, action: np.ndarray | float, feedback: np.ndarray) -> DisturbanceEstimate:
        dt = self.config.dt
        theta, theta_dot = np.asarray(state, dtype=float)
        next_dot = float(feedback[1])
        torque = float(np.asarray(action).reshape(-1)[0])
        observed_ddot = (next_dot - theta_dot) / dt
        inertia = self.config.mass * self.config.length * self.config.length
        if self.config.dynamics_model == "gym":
            nominal_ddot = (
                3.0 * self.config.gravity / (2.0 * self.config.length) * np.sin(theta)
                + 3.0 * torque / inertia
            )
            measurement = inertia * (observed_ddot - nominal_ddot) / 3.0
        else:
            nominal_ddot = (
                self.config.gravity / self.config.length * np.sin(theta)
                + (torque - self.config.damping * theta_dot) / inertia
            )
            measurement = inertia * (observed_ddot - nominal_ddot)
        if _is_velocity_saturated(self.config, state, feedback, self.saturation_margin):
            return DisturbanceEstimate(
                float(self.mean),
                float(np.sqrt(max(self.var, 1.0e-8))),
                float(measurement),
                float(self.rate),
            )
        previous_mean = self.mean
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * measurement
        if self.initialized:
            raw_rate = (self.mean - previous_mean) / max(dt, 1.0e-8)
            raw_rate = _clamp_rate(raw_rate, self.rate_limit)
            self.rate = _clamp_rate((1.0 - self.alpha) * self.rate + self.alpha * raw_rate, self.rate_limit)
        else:
            self.rate = 0.0
            self.initialized = True
        residual = measurement - self.mean
        self.var = (1.0 - self.sigma_alpha) * self.var + self.sigma_alpha * residual * residual
        return DisturbanceEstimate(
            float(self.mean),
            float(np.sqrt(max(self.var, 1.0e-8))),
            float(measurement),
            float(self.rate),
        )


class ParticleSDOBObserver:
    """One-step particle observer over input-equivalent torque disturbance."""

    def __init__(
        self,
        config: PendulumConfig,
        d_min: float = -2.0,
        d_max: float = 2.0,
        num_particles: int = 256,
        lambda_d: float = 0.003,
        beta: float = 0.8,
        rate_limit: float = 1.2,
        saturation_margin: float = 0.98,
    ) -> None:
        self.config = config
        self.particles = np.linspace(d_min, d_max, num_particles, dtype=float)
        self.lambda_d = lambda_d
        self.beta = beta
        self.rate_limit = rate_limit
        self.saturation_margin = saturation_margin
        self.mean = 0.0
        self.rate = 0.0
        self.sigma = 0.25
        self.initialized = False

    def reset(self) -> None:
        self.mean = 0.0
        self.rate = 0.0
        self.sigma = 0.25
        self.initialized = False

    def update(self, state: np.ndarray, action: np.ndarray | float, feedback: np.ndarray) -> DisturbanceEstimate:
        predictions = []
        for disturbance in self.particles:
            pred = pendulum_step_np(state, action, self.config, disturbance=float(disturbance))
            predictions.append(pred)
        candidates_t = torch.as_tensor(self.particles[:, None], dtype=torch.double)
        predictions_t = torch.as_tensor(np.asarray(predictions), dtype=torch.double)
        feedback_t = torch.as_tensor(feedback, dtype=torch.double)

        def pendulum_residual(observed: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
            residual = observed - predicted
            return torch.stack((wrap_angle_torch(residual[:, 0]), residual[:, 1]), dim=-1)

        _, _, weights_t = score_state_effect_candidates(
            candidates_t,
            predictions_t,
            feedback_t,
            temperature=max(self.lambda_d, 1.0e-8),
            prediction_weight=torch.tensor([np.sqrt(20.0), np.sqrt(40.0)], dtype=torch.double),
            regularization_weight=torch.zeros(1, dtype=torch.double),
            residual_function=pendulum_residual,
        )
        weights = weights_t.detach().cpu().numpy()
        posterior_mean = float(np.sum(weights * self.particles))
        posterior_var = float(np.sum(weights * (self.particles - posterior_mean) ** 2))
        if _is_velocity_saturated(self.config, state, feedback, self.saturation_margin):
            return DisturbanceEstimate(float(self.mean), float(self.sigma), posterior_mean, float(self.rate))
        previous_mean = self.mean
        self.mean = (1.0 - self.beta) * self.mean + self.beta * posterior_mean
        if self.initialized:
            raw_rate = (self.mean - previous_mean) / max(self.config.dt, 1.0e-8)
            raw_rate = _clamp_rate(raw_rate, self.rate_limit)
            self.rate = _clamp_rate((1.0 - self.beta) * self.rate + self.beta * raw_rate, self.rate_limit)
        else:
            self.rate = 0.0
            self.initialized = True
        self.sigma = (1.0 - self.beta) * self.sigma + self.beta * np.sqrt(max(posterior_var, 1.0e-8))
        return DisturbanceEstimate(float(self.mean), float(self.sigma), posterior_mean, float(self.rate))
