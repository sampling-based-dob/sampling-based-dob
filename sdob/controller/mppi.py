"""A task-independent Model Predictive Path Integral controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .base import RolloutModel, TrajectoryCost


def gibbs_weights(costs: torch.Tensor, temperature: float, dim: int = -1) -> torch.Tensor:
    """Convert sampled costs into normalized exponential weights."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    shifted = costs - costs.amin(dim=dim, keepdim=True)
    logits = -shifted / float(temperature)
    return torch.softmax(logits, dim=dim)


def effective_sample_size(weights: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return the effective number of samples represented by the weights."""
    return 1.0 / weights.square().sum(dim=dim).clamp_min(1.0e-12)


def _as_tuple(value: float | Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),) * size
    result = tuple(float(item) for item in value)
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} values")
    return result


@dataclass(frozen=True)
class MPPIConfig:
    """Configuration shared by task-specific MPPI applications."""

    horizon: int
    num_samples: int
    action_dim: int
    noise_sigma: float | tuple[float, ...]
    action_min: float | tuple[float, ...]
    action_max: float | tuple[float, ...]
    temperature: float = 1.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 0
    adaptive_temperature: bool = False
    eta_min: float = 0.05
    eta_max: float = 0.10
    temperature_decrease: float = 0.9
    temperature_increase: float = 1.2
    temperature_min: float = 1.0e-3
    temperature_max: float = 1.0e3

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.num_samples <= 0 or self.action_dim <= 0:
            raise ValueError("horizon, num_samples, and action_dim must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        sigma = _as_tuple(self.noise_sigma, self.action_dim, "noise_sigma")
        lower = _as_tuple(self.action_min, self.action_dim, "action_min")
        upper = _as_tuple(self.action_max, self.action_dim, "action_max")
        if any(item < 0.0 for item in sigma):
            raise ValueError("noise_sigma must be nonnegative")
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("each action_min value must be smaller than action_max")


@dataclass(frozen=True)
class MPPIResult:
    """Diagnostics returned by one receding-horizon MPPI update."""

    action: torch.Tensor
    control_sequence: torch.Tensor
    trajectories: torch.Tensor
    costs: torch.Tensor
    weights: torch.Tensor
    expected_cost: float
    temperature: float
    effective_sample_size: float


class MPPIController:
    """Generic batched MPPI using a forward model and trajectory-cost callback."""

    def __init__(
        self,
        model: RolloutModel,
        trajectory_cost: TrajectoryCost,
        config: MPPIConfig,
    ) -> None:
        self.model = model
        self.trajectory_cost = trajectory_cost
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self.noise_sigma = torch.tensor(
            _as_tuple(config.noise_sigma, config.action_dim, "noise_sigma"),
            dtype=self.dtype,
            device=self.device,
        )
        self.action_min = torch.tensor(
            _as_tuple(config.action_min, config.action_dim, "action_min"),
            dtype=self.dtype,
            device=self.device,
        )
        self.action_max = torch.tensor(
            _as_tuple(config.action_max, config.action_dim, "action_max"),
            dtype=self.dtype,
            device=self.device,
        )
        self.control_sequence = torch.zeros(
            config.horizon,
            config.action_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.temperature = float(config.temperature)
        self.last_eta: float | None = None

    def reset(self) -> None:
        self.control_sequence.zero_()
        self.temperature = float(self.config.temperature)
        self.last_eta = None

    def command(
        self,
        state: torch.Tensor,
        context: Any = None,
        state_disturbance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.plan(state, context, state_disturbance).action

    @torch.no_grad()
    def plan(
        self,
        state: torch.Tensor,
        context: Any = None,
        state_disturbance: torch.Tensor | None = None,
    ) -> MPPIResult:
        state = torch.as_tensor(state, dtype=self.dtype, device=self.device).reshape(-1)
        noise = torch.randn(
            self.config.num_samples,
            self.config.horizon,
            self.config.action_dim,
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
        ) * self.noise_sigma
        controls = self.control_sequence.unsqueeze(0) + noise
        controls = torch.maximum(torch.minimum(controls, self.action_max), self.action_min)
        controls[0] = self.control_sequence

        trajectories = self._rollout(state, controls, state_disturbance)
        costs = self.trajectory_cost(trajectories, controls, context)
        costs = torch.as_tensor(costs, dtype=self.dtype, device=self.device).reshape(-1)
        if costs.shape[0] != self.config.num_samples:
            raise ValueError("trajectory_cost must return one cost per sampled trajectory")
        if not torch.isfinite(costs).all():
            raise ValueError("trajectory_cost returned a non-finite value")

        used_temperature = self.temperature
        weights = gibbs_weights(costs, used_temperature)
        optimized = torch.sum(weights[:, None, None] * controls, dim=0).detach()
        action = optimized[0].clone()
        expected_cost = float(torch.sum(weights * costs).detach().cpu())
        ess = float(effective_sample_size(weights).detach().cpu())
        self.control_sequence = optimized.clone()
        self._adapt_temperature(weights)
        self._shift_control_sequence()
        return MPPIResult(
            action=action,
            control_sequence=optimized,
            trajectories=trajectories,
            costs=costs,
            weights=weights,
            expected_cost=expected_cost,
            temperature=used_temperature,
            effective_sample_size=ess,
        )

    def _rollout(
        self,
        state: torch.Tensor,
        controls: torch.Tensor,
        state_disturbance: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = controls.shape[0]
        current = state.unsqueeze(0).expand(batch, -1)
        states = [current]
        for step in range(self.config.horizon):
            disturbance = self._disturbance_at(state_disturbance, step, batch)
            current = self.model.step(current, controls[:, step], disturbance)
            states.append(current)
        return torch.stack(states, dim=1)

    def _disturbance_at(
        self,
        disturbance: torch.Tensor | None,
        step: int,
        batch: int,
    ) -> torch.Tensor | None:
        if disturbance is None:
            return None
        value = torch.as_tensor(disturbance, dtype=self.dtype, device=self.device)
        if value.ndim >= 2 and value.shape[0] == self.config.horizon:
            value = value[step]
        if value.ndim == 1:
            value = value.unsqueeze(0).expand(batch, -1)
        return value

    def _adapt_temperature(self, weights: torch.Tensor) -> None:
        if not self.config.adaptive_temperature:
            return
        eta = effective_sample_size(weights) / weights.numel()
        self.last_eta = float(eta.detach().cpu())
        if self.last_eta > self.config.eta_max:
            self.temperature *= self.config.temperature_decrease
        elif self.last_eta < self.config.eta_min:
            self.temperature *= self.config.temperature_increase
        self.temperature = float(
            min(
                max(self.temperature, self.config.temperature_min),
                self.config.temperature_max,
            )
        )

    def _shift_control_sequence(self) -> None:
        terminal = self.control_sequence[-1].clone()
        self.control_sequence[:-1] = self.control_sequence[1:].clone()
        self.control_sequence[-1] = terminal
