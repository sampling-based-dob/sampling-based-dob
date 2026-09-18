from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from sdob.controller import effective_sample_size, gibbs_weights

from .dynamics import (
    PendulumConfig,
    pendulum_step_torch,
    running_cost_torch,
    terminal_cost_torch,
    wrap_angle_torch,
)
from .tracking import tracking_feedback_from_gain_np


@dataclass(frozen=True)
class MPPIConfig:
    horizon: int = 25
    num_samples: int = 512
    lambda_: float = 0.35
    noise_sigma: float = 1.2
    device: str = "cpu"
    seed: int = 0
    risk_weight: float = 0.0
    adaptive_temperature: bool = True
    eta_min: float = 0.05
    eta_max: float = 0.1
    lambda_decrease: float = 0.9
    lambda_increase: float = 1.2
    lambda_min: float = 1.0e-3
    lambda_max: float = 1.0e3
    sampling_control_cost: float = 0.01
    likelihood_alpha: float = 0.0


@dataclass(frozen=True)
class MPPIPlan:
    action: float
    cost: float
    nominal_action: float | None = None
    real_action: float | None = None


class BaseMPPI(ABC):
    """Common interface for receding-horizon sampling planners."""

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def command(
        self,
        state: np.ndarray,
        rollout_disturbance: float | np.ndarray | torch.Tensor = 0.0,
        disturbance_scenarios: list[float] | None = None,
    ) -> float:
        return self.plan(
            state,
            rollout_disturbance=rollout_disturbance,
            disturbance_scenarios=disturbance_scenarios,
        ).action

    @abstractmethod
    def plan(
        self,
        state: np.ndarray,
        rollout_disturbance: float = 0.0,
        disturbance_scenarios: list[float] | None = None,
    ) -> MPPIPlan:
        raise NotImplementedError


class TorchMPPI(BaseMPPI):
    """Small MPPI implementation with a pytorch_mppi-like dynamics/cost interface."""

    def __init__(self, pendulum: PendulumConfig, config: MPPIConfig) -> None:
        self.pendulum = pendulum
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.double
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.seed)
        self.u_min = -pendulum.max_torque
        self.u_max = pendulum.max_torque
        self.U = torch.zeros(config.horizon, 1, dtype=self.dtype, device=self.device)
        self.lambda_ = float(config.lambda_)
        self.last_eta: float | None = None
        self.lambda_history = [self.lambda_]
        self.previous_action = 0.0
        self.last_rollout_disturbance = 0.0

    def reset(self) -> None:
        self.U.zero_()
        self.lambda_ = float(self.config.lambda_)
        self.last_eta = None
        self.lambda_history = [self.lambda_]
        self.previous_action = 0.0
        self.last_rollout_disturbance = 0.0

    def plan(
        self,
        state: np.ndarray,
        rollout_disturbance: float | np.ndarray | torch.Tensor = 0.0,
        disturbance_scenarios: list[float] | None = None,
    ) -> MPPIPlan:
        if disturbance_scenarios:
            costs = self._scenario_costs(state, disturbance_scenarios)
        else:
            self._recenter_action_prior_for_disturbance(rollout_disturbance)
            noise = self._sample_noise()
            actions = (self.U.unsqueeze(0) + noise).clamp(self.u_min, self.u_max)
            costs = self._rollout_costs(state, actions, rollout_disturbance)
            weights = self._update_nominal(actions, costs)
            action = float(self.U[0, 0].detach().cpu())
            cost = float(torch.sum(weights * costs).detach().cpu())
            self.previous_action = action
            self._shift_nominal()
            return MPPIPlan(action=action, cost=cost, nominal_action=action, real_action=action)

        noise, actions, aggregate_cost = costs
        weights = self._update_nominal(actions, aggregate_cost)
        action = float(self.U[0, 0].detach().cpu())
        cost = float(torch.sum(weights * aggregate_cost).detach().cpu())
        self.previous_action = action
        self._shift_nominal()
        return MPPIPlan(action=action, cost=cost, nominal_action=action, real_action=action)

    def command_augmented(
        self,
        actual_state: np.ndarray,
        nominal_state: np.ndarray,
        tracking_gain: np.ndarray,
        tracking_cost_weight: float,
        value_function_threshold: float,
        shift_after: bool = True,
    ) -> MPPIPlan:
        noise = self._sample_noise()
        actions = (self.U.unsqueeze(0) + noise).clamp(self.u_min, self.u_max)
        mixed_cost, real_cost, real_actions = self._augmented_rollout_costs(
            actual_state,
            nominal_state,
            actions,
            tracking_gain=tracking_gain,
            tracking_cost_weight=tracking_cost_weight,
            value_function_threshold=value_function_threshold,
        )
        nominal_weights = self._weights_from_costs(mixed_cost)
        real_weights = self._weights_from_costs(real_cost)
        self.U = torch.sum(nominal_weights.view(-1, 1, 1) * actions, dim=0).detach()
        self._adapt_temperature_from_weights(nominal_weights)
        real_sequence = torch.sum(real_weights.view(-1, 1, 1) * real_actions, dim=0).detach()
        nominal_action = float(self.U[0, 0].detach().cpu())
        real_nominal_action = float(real_sequence[0, 0].detach().cpu())
        feedback = tracking_feedback_from_gain_np(actual_state, nominal_state, tracking_gain)
        action = float(np.clip(real_nominal_action + feedback, self.u_min, self.u_max))
        cost = float(torch.sum(real_weights * real_cost).detach().cpu())
        self.previous_action = action
        if shift_after:
            self._shift_nominal()
        return MPPIPlan(action=action, cost=cost, nominal_action=nominal_action, real_action=real_nominal_action)

    def first_nominal_action(self) -> float:
        return float(self.U[0, 0].detach().cpu())

    def _sample_noise(self) -> torch.Tensor:
        shape = (self.config.num_samples, self.config.horizon, 1)
        return torch.randn(shape, generator=self.generator, dtype=self.dtype, device=self.device) * self.config.noise_sigma

    def _first_disturbance_value(self, disturbance: float | np.ndarray | torch.Tensor) -> float:
        if torch.is_tensor(disturbance):
            flat = disturbance.detach().reshape(-1)
            return float(flat[0].cpu()) if flat.numel() else 0.0
        array = np.asarray(disturbance, dtype=float).reshape(-1)
        return float(array[0]) if array.size else 0.0

    def _disturbance_at(
        self,
        disturbance: float | np.ndarray | torch.Tensor,
        step: int,
    ) -> float | torch.Tensor:
        if torch.is_tensor(disturbance):
            tensor = disturbance.to(dtype=self.dtype, device=self.device)
            if tensor.ndim == 0:
                return tensor
            index = min(step, tensor.shape[0] - 1)
            return tensor[index]
        array = np.asarray(disturbance, dtype=float)
        if array.ndim == 0:
            return float(array)
        index = min(step, array.shape[0] - 1)
        return float(array[index])

    def _recenter_action_prior_for_disturbance(self, rollout_disturbance: float | np.ndarray | torch.Tensor) -> None:
        first_disturbance = self._first_disturbance_value(rollout_disturbance)
        delta = first_disturbance - self.last_rollout_disturbance
        if abs(delta) > 1.0e-12:
            self.U = (self.U - delta).clamp(self.u_min, self.u_max).detach()
        self.last_rollout_disturbance = first_disturbance

    def _uses_disturbance_compensation_cost(self, disturbance: float | np.ndarray | torch.Tensor) -> bool:
        if self.pendulum.disturbance_compensation_cost <= 0.0:
            return False
        if isinstance(disturbance, torch.Tensor):
            return bool(torch.any(torch.abs(disturbance) > 1.0e-12).detach().cpu())
        return bool(np.any(np.abs(np.asarray(disturbance, dtype=float)) > 1.0e-12))

    def _rollout_costs(
        self,
        state: np.ndarray,
        actions: torch.Tensor,
        disturbance: float | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        k = actions.shape[0]
        x = torch.as_tensor(state, dtype=self.dtype, device=self.device).view(1, 2).repeat(k, 1)
        total = torch.zeros(k, dtype=self.dtype, device=self.device)
        prev_u = torch.full((k, 1), self.previous_action, dtype=self.dtype, device=self.device)
        use_disturbance_compensation = self._uses_disturbance_compensation_cost(disturbance)
        for t in range(self.config.horizon):
            step_disturbance = self._disturbance_at(disturbance, t)
            u = actions[:, t, :]
            total = total + running_cost_torch(x, u, self.pendulum)
            if self.pendulum.action_smoothness_cost > 0.0:
                total = total + self.pendulum.action_smoothness_cost * (u[:, 0] - prev_u[:, 0]).square()
            if use_disturbance_compensation:
                total = total + self.pendulum.disturbance_compensation_cost * (u[:, 0] + step_disturbance).square()
            prev_u = u
            x = pendulum_step_torch(x, u, self.pendulum, disturbance=step_disturbance)
        return total + terminal_cost_torch(x, self.pendulum) + self._likelihood_ratio_cost(actions)

    def _augmented_rollout_costs(
        self,
        actual_state: np.ndarray,
        nominal_state: np.ndarray,
        actions: torch.Tensor,
        tracking_gain: np.ndarray,
        tracking_cost_weight: float,
        value_function_threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = actions.shape[0]
        x_real = torch.as_tensor(actual_state, dtype=self.dtype, device=self.device).view(1, 2).repeat(k, 1)
        x_nom = torch.as_tensor(nominal_state, dtype=self.dtype, device=self.device).view(1, 2).repeat(k, 1)
        nominal_total = torch.zeros(k, dtype=self.dtype, device=self.device)
        real_total = torch.zeros(k, dtype=self.dtype, device=self.device)
        feedback_total = torch.zeros(k, dtype=self.dtype, device=self.device)
        real_actions = torch.empty_like(actions)
        gain = torch.as_tensor(tracking_gain, dtype=self.dtype, device=self.device).view(1, 2)
        prev_u = torch.full((k, 1), self.previous_action, dtype=self.dtype, device=self.device)
        nominal_lr_total = self._likelihood_ratio_cost(actions)
        for t in range(self.config.horizon):
            nominal_u = actions[:, t, :]
            angle_error = wrap_angle_torch(x_nom[:, 0] - x_real[:, 0]).view(-1, 1)
            vel_error = (x_nom[:, 1] - x_real[:, 1]).view(-1, 1)
            error = torch.cat((angle_error, vel_error), dim=1)
            k_fb = torch.sum(gain * error, dim=1, keepdim=True)
            real_u = (nominal_u + k_fb).clamp(self.u_min, self.u_max)
            real_actions[:, t, :] = real_u
            nominal_total = nominal_total + running_cost_torch(x_nom, nominal_u, self.pendulum)
            real_total = real_total + running_cost_torch(x_real, real_u, self.pendulum)
            if self.pendulum.action_smoothness_cost > 0.0:
                real_total = real_total + self.pendulum.action_smoothness_cost * (real_u[:, 0] - prev_u[:, 0]).square()
            prev_u = real_u
            feedback_total = feedback_total + self._feedback_sampling_cost(k_fb)
            if tracking_cost_weight > 0.0:
                feedback_total = feedback_total + tracking_cost_weight * k_fb[:, 0].square()
            x_nom = pendulum_step_torch(x_nom, nominal_u, self.pendulum, disturbance=0.0)
            x_real = pendulum_step_torch(x_real, real_u, self.pendulum, disturbance=0.0)
        nominal_total = nominal_total + terminal_cost_torch(x_nom, self.pendulum)
        real_total = real_total + terminal_cost_torch(x_real, self.pendulum)
        real_lr_total = self._likelihood_ratio_cost(real_actions)
        threshold = torch.as_tensor(value_function_threshold, dtype=self.dtype, device=self.device)
        feedback_cost = real_total + feedback_total
        mixed_cost = 0.5 * nominal_total + 0.5 * torch.maximum(torch.minimum(feedback_cost, threshold), nominal_total)
        mixed_cost = mixed_cost + nominal_lr_total
        real_cost = real_total + real_lr_total
        return mixed_cost, real_cost, real_actions

    def _scenario_costs(self, state: np.ndarray, scenarios: list[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = self._sample_noise()
        actions = (self.U.unsqueeze(0) + noise).clamp(self.u_min, self.u_max)
        scenario_costs = torch.stack(
            [self._rollout_costs(state, actions, float(disturbance)) for disturbance in scenarios],
            dim=1,
        )
        aggregate = scenario_costs.mean(dim=1) + self.config.risk_weight * scenario_costs.std(dim=1, unbiased=False)
        return noise, actions, aggregate

    def nominal_sequence_cost(self, state: np.ndarray) -> float:
        actions = self.U.unsqueeze(0)
        cost = self._rollout_costs(state, actions, disturbance=0.0)
        return float(cost[0].detach().cpu())

    def _update_nominal(self, actions: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
        weights = self._weights_from_costs(costs)
        self.U = torch.sum(weights.view(-1, 1, 1) * actions, dim=0).detach()
        self._adapt_temperature_from_weights(weights)
        return weights

    def _weights_from_costs(self, costs: torch.Tensor) -> torch.Tensor:
        return gibbs_weights(costs, max(self.lambda_, 1.0e-8))

    def _sampling_cost_scale(self) -> float:
        if self.config.sampling_control_cost <= 0.0 or self.config.likelihood_alpha >= 1.0:
            return 0.0
        variance = max(float(self.config.noise_sigma) ** 2, 1.0e-12)
        return (
            0.5
            * self.lambda_
            * (1.0 - float(self.config.likelihood_alpha))
            * float(self.config.sampling_control_cost)
            / variance
        )

    def _likelihood_ratio_cost(self, actions: torch.Tensor) -> torch.Tensor:
        scale = self._sampling_cost_scale()
        if scale == 0.0:
            return torch.zeros(actions.shape[0], dtype=self.dtype, device=self.device)
        mean = self.U.view(1, self.config.horizon, 1)
        return scale * torch.sum(mean * (mean - 2.0 * actions), dim=(1, 2))

    def _feedback_sampling_cost(self, feedback: torch.Tensor) -> torch.Tensor:
        scale = self._sampling_cost_scale()
        if scale == 0.0:
            if feedback.ndim == 3:
                return torch.zeros(feedback.shape[0], dtype=self.dtype, device=self.device)
            return torch.zeros(feedback.shape[0], dtype=self.dtype, device=self.device)
        if feedback.ndim == 3:
            return scale * torch.sum(feedback.square(), dim=(1, 2))
        return scale * torch.sum(feedback.square(), dim=1)

    def _adapt_temperature_from_weights(self, weights: torch.Tensor) -> None:
        if not self.config.adaptive_temperature:
            return
        ess = effective_sample_size(weights)
        eta = ess / weights.numel()
        self.last_eta = float(eta.detach().cpu())
        if self.last_eta > self.config.eta_max:
            self.lambda_ *= self.config.lambda_decrease
        elif self.last_eta < self.config.eta_min:
            self.lambda_ *= self.config.lambda_increase
        self.lambda_ = float(min(max(self.lambda_, self.config.lambda_min), self.config.lambda_max))
        self.lambda_history.append(self.lambda_)

    def shift_nominal(self, steps: int = 1) -> None:
        self._shift_nominal(steps)

    def _shift_nominal(self, steps: int = 1) -> None:
        steps = int(steps)
        if steps <= 0:
            return
        steps = min(steps, self.config.horizon)
        terminal_action = self.U[-1].clone()
        if steps < self.config.horizon:
            self.U[:-steps] = self.U[steps:].clone()
        self.U[-steps:] = terminal_action
