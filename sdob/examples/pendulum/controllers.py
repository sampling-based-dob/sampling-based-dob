from __future__ import annotations

from dataclasses import replace

import numpy as np

from .dynamics import PendulumConfig, pendulum_step_np, wrap_angle_np
from .interfaces import ControllerBase
from .mppi import BaseMPPI, MPPIConfig, MPPIPlan, TorchMPPI
from .observers import ClassicalDOBObserver, DisturbanceEstimate, ParticleSDOBObserver
from .tracking import tracking_feedback_control_np, tracking_feedback_from_gain_np


def disturbance_forecast(estimate: DisturbanceEstimate, pendulum: PendulumConfig, horizon: int) -> np.ndarray:
    return estimate.forecast(
        horizon=horizon,
        dt=pendulum.dt,
        lower=-pendulum.max_torque,
        upper=pendulum.max_torque,
        max_time=5.0 * pendulum.dt,
    )


class MPPIController(ControllerBase):
    name = "MPPI"

    def __init__(self, pendulum: PendulumConfig, mppi_config: MPPIConfig) -> None:
        self.pendulum = pendulum
        self.planner = TorchMPPI(pendulum, mppi_config)

    def reset(self) -> None:
        self.planner.reset()

    def act(self, state: np.ndarray, step: int) -> float:
        return self.planner.command(state, rollout_disturbance=0.0)


class DOBMPPIController(ControllerBase):
    name = "MPPI+DOB"

    def __init__(self, pendulum: PendulumConfig, mppi_config: MPPIConfig) -> None:
        self.pendulum = pendulum
        self.planner = TorchMPPI(pendulum, mppi_config)
        self.observer = ClassicalDOBObserver(pendulum)
        self.estimate = DisturbanceEstimate(0.0, 0.1, 0.0)

    def reset(self) -> None:
        self.planner.reset()
        self.observer.reset()
        self.estimate = DisturbanceEstimate(0.0, 0.1, 0.0)

    def act(self, state: np.ndarray, step: int) -> float:
        rollout_disturbance = disturbance_forecast(self.estimate, self.pendulum, self.planner.config.horizon)
        return self.planner.command(state, rollout_disturbance=rollout_disturbance)

    def observe(self, state: np.ndarray, action: float, feedback: np.ndarray, step: int) -> None:
        self.estimate = self.observer.update(state, action, feedback)

    def info(self) -> dict[str, float]:
        return {"d_hat": self.estimate.mean, "d_dot_hat": self.estimate.rate, "d_sigma": self.estimate.sigma}


class SDOBMPPIController(ControllerBase):
    name = "MPPI+SDOB"

    def __init__(self, pendulum: PendulumConfig, mppi_config: MPPIConfig) -> None:
        self.pendulum = pendulum
        self.planner = TorchMPPI(pendulum, mppi_config)
        self.observer = ParticleSDOBObserver(pendulum)
        self.estimate = DisturbanceEstimate(0.0, 0.25, 0.0)

    def reset(self) -> None:
        self.planner.reset()
        self.observer.reset()
        self.estimate = DisturbanceEstimate(0.0, 0.25, 0.0)

    def act(self, state: np.ndarray, step: int) -> float:
        rollout_disturbance = disturbance_forecast(self.estimate, self.pendulum, self.planner.config.horizon)
        return self.planner.command(state, rollout_disturbance=rollout_disturbance)

    def observe(self, state: np.ndarray, action: float, feedback: np.ndarray, step: int) -> None:
        self.estimate = self.observer.update(state, action, feedback)

    def info(self) -> dict[str, float]:
        return {"d_hat": self.estimate.mean, "d_dot_hat": self.estimate.rate, "d_sigma": self.estimate.sigma}


class TubeMPPIController(ControllerBase):
    name = "Tube-MPPI"

    def __init__(
        self,
        pendulum: PendulumConfig,
        mppi_config: MPPIConfig,
        kp: float = 4.0,
        kd: float = 1.2,
        reset_threshold: float = -float("inf"),
        tracking_mode: str = "lqr",
        lqr_horizon: int = 20,
        lqr_q: tuple[float, float] = (30.0, 3.0),
        lqr_r: float = 0.1,
    ) -> None:
        self.pendulum = replace(pendulum, disturbance_compensation_cost=0.0)
        self.nominal_planner = TorchMPPI(self.pendulum, mppi_config)
        self.actual_planner = TorchMPPI(self.pendulum, replace(mppi_config, seed=mppi_config.seed + 1009))
        self.kp = kp
        self.kd = kd
        self.reset_threshold = reset_threshold
        self.tracking_mode = tracking_mode
        self.lqr_horizon = lqr_horizon
        self.lqr_q = lqr_q
        self.lqr_r = lqr_r
        self.nominal_state = np.zeros(2, dtype=float)
        self.last_nominal_action = 0.0
        self.last_real_action = 0.0
        self.last_tracking_feedback = 0.0
        self.last_tracking_gain = np.array([kp, kd], dtype=float)
        self.used_actual_reset = False
        self.initialized = False

    def reset(self) -> None:
        self.nominal_planner.reset()
        self.actual_planner.reset()
        self.nominal_state = np.zeros(2, dtype=float)
        self.last_nominal_action = 0.0
        self.last_real_action = 0.0
        self.last_tracking_feedback = 0.0
        self.last_tracking_gain = np.array([self.kp, self.kd], dtype=float)
        self.used_actual_reset = False
        self.initialized = False

    def act(self, state: np.ndarray, step: int) -> float:
        if not self.initialized:
            self.nominal_state = np.asarray(state, dtype=float).copy()
            self.initialized = True
        nominal_plan = self.nominal_planner.plan(self.nominal_state, rollout_disturbance=0.0)
        actual_plan = self.actual_planner.plan(state, rollout_disturbance=0.0)
        self.used_actual_reset = actual_plan.cost <= nominal_plan.cost + self.reset_threshold
        if self.used_actual_reset:
            self.nominal_state = np.asarray(state, dtype=float).copy()
            self.nominal_planner.U = self.actual_planner.U.clone()
            nominal_u = actual_plan.action
        else:
            nominal_u = nominal_plan.action
        feedback, gain = tracking_feedback_control_np(
            self.pendulum,
            state,
            self.nominal_state,
            nominal_u,
            self.kp,
            self.kd,
            self.tracking_mode,
            self.lqr_horizon,
            self.lqr_q,
            self.lqr_r,
        )
        self.last_nominal_action = nominal_u
        self.last_tracking_feedback = feedback
        self.last_tracking_gain = gain
        return float(np.clip(nominal_u + feedback, -self.pendulum.max_torque, self.pendulum.max_torque))

    def observe(self, state: np.ndarray, action: float, feedback: np.ndarray, step: int) -> None:
        self.nominal_state = pendulum_step_np(self.nominal_state, self.last_nominal_action, self.pendulum, disturbance=0.0)

    def info(self) -> dict[str, float]:
        return {
            "nominal_theta": float(self.nominal_state[0]),
            "nominal_theta_dot": float(self.nominal_state[1]),
            "tracking_feedback": float(self.last_tracking_feedback),
            "tracking_gain_theta": float(self.last_tracking_gain[0]),
            "tracking_gain_theta_dot": float(self.last_tracking_gain[1]),
            "tracking_mode": self.tracking_mode,
            "used_actual_reset": float(self.used_actual_reset),
        }


class RMPPIController(ControllerBase):
    name = "RMPPI"

    def __init__(
        self,
        pendulum: PendulumConfig,
        mppi_config: MPPIConfig,
        disturbance_bound: float = 0.9,
        kp: float = 4.0,
        kd: float = 1.2,
        tracking_cost_weight: float = 0.04,
        free_energy_threshold: float = 6.0,
        value_function_threshold: float = 25.0,
        num_nominal_candidates: int = 9,
        allow_real_nominal_projection: bool = True,
        nominal_projection_penalty: float = 500.0,
        nominal_jump_penalty: float = 0.0,
        tracking_mode: str = "lqr",
        lqr_horizon: int = 20,
        lqr_q: tuple[float, float] = (30.0, 3.0),
        lqr_r: float = 0.1,
    ) -> None:
        self.pendulum = replace(pendulum, disturbance_compensation_cost=0.0)
        robust_config = MPPIConfig(
            horizon=mppi_config.horizon,
            num_samples=mppi_config.num_samples,
            lambda_=mppi_config.lambda_,
            noise_sigma=mppi_config.noise_sigma,
            device=mppi_config.device,
            seed=mppi_config.seed,
            risk_weight=mppi_config.risk_weight,
            adaptive_temperature=mppi_config.adaptive_temperature,
            eta_min=mppi_config.eta_min,
            eta_max=mppi_config.eta_max,
            lambda_decrease=mppi_config.lambda_decrease,
            lambda_increase=mppi_config.lambda_increase,
            lambda_min=mppi_config.lambda_min,
            lambda_max=mppi_config.lambda_max,
            sampling_control_cost=mppi_config.sampling_control_cost,
            likelihood_alpha=mppi_config.likelihood_alpha,
        )
        self.planner = TorchMPPI(self.pendulum, robust_config)
        self.disturbance_bound = disturbance_bound
        self.kp = kp
        self.kd = kd
        self.tracking_cost_weight = tracking_cost_weight
        self.free_energy_threshold = free_energy_threshold
        self.value_function_threshold = value_function_threshold
        self.num_nominal_candidates = max(3, int(num_nominal_candidates))
        if self.num_nominal_candidates % 2 == 0:
            self.num_nominal_candidates += 1
        self.allow_real_nominal_projection = allow_real_nominal_projection
        self.nominal_projection_penalty = nominal_projection_penalty
        self.nominal_jump_penalty = nominal_jump_penalty
        self.tracking_mode = tracking_mode
        self.lqr_horizon = lqr_horizon
        self.lqr_q = lqr_q
        self.lqr_r = lqr_r
        self.nominal_state = np.zeros(2, dtype=float)
        self.last_nominal_action = 0.0
        self.last_real_action = 0.0
        self.last_tracking_feedback = 0.0
        self.last_tracking_gain = np.array([kp, kd], dtype=float)
        self.last_nominal_candidate_count = 0
        self.last_selected_nominal_index = -1
        self.last_selected_nominal_stride = 0
        self.last_selected_nominal_cost = float("nan")
        self.last_selected_nominal_score = float("nan")
        self.last_selected_nominal_source = "unset"
        self.last_selected_nominal_reason = "unset"
        self.nominal_stride = 0
        self.last_nominal_stride = 0
        self.initialized = False

    def reset(self) -> None:
        self.planner.reset()
        self.nominal_state = np.zeros(2, dtype=float)
        self.last_nominal_action = 0.0
        self.last_real_action = 0.0
        self.last_tracking_feedback = 0.0
        self.last_tracking_gain = np.array([self.kp, self.kd], dtype=float)
        self.last_nominal_candidate_count = 0
        self.last_selected_nominal_index = -1
        self.last_selected_nominal_stride = 0
        self.last_selected_nominal_cost = float("nan")
        self.last_selected_nominal_score = float("nan")
        self.last_selected_nominal_source = "unset"
        self.last_selected_nominal_reason = "unset"
        self.nominal_stride = 0
        self.last_nominal_stride = 0
        self.initialized = False

    def act(self, state: np.ndarray, step: int) -> float:
        if not self.initialized:
            self.nominal_state = np.asarray(state, dtype=float).copy()
            self.initialized = True
        self.planner.shift_nominal(self.nominal_stride)
        self.last_nominal_stride = self.nominal_stride
        self.nominal_stride = 0
        prior_nominal_action = self.planner.first_nominal_action()
        _, tracking_gain = tracking_feedback_control_np(
            self.pendulum,
            state,
            self.nominal_state,
            prior_nominal_action,
            self.kp,
            self.kd,
            self.tracking_mode,
            self.lqr_horizon,
            self.lqr_q,
            self.lqr_r,
        )
        plan = self.planner.command_augmented(
            actual_state=state,
            nominal_state=self.nominal_state,
            tracking_gain=tracking_gain,
            tracking_cost_weight=self.tracking_cost_weight,
            value_function_threshold=self.value_function_threshold,
            shift_after=False,
        )
        self.last_tracking_feedback = tracking_feedback_from_gain_np(state, self.nominal_state, tracking_gain)
        self.last_tracking_gain = tracking_gain
        nominal_action = plan.nominal_action if plan.nominal_action is not None else plan.action - self.last_tracking_feedback
        self.last_nominal_action = float(np.clip(nominal_action, -self.pendulum.max_torque, self.pendulum.max_torque))
        real_action = plan.real_action if plan.real_action is not None else plan.action - self.last_tracking_feedback
        self.last_real_action = float(np.clip(real_action, -self.pendulum.max_torque, self.pendulum.max_torque))
        return plan.action

    def observe(self, state: np.ndarray, action: float, feedback: np.ndarray, step: int) -> None:
        propagated = pendulum_step_np(self.nominal_state, self.last_nominal_action, self.pendulum, disturbance=0.0)
        self.nominal_state, self.nominal_stride = self._select_nominal_state_and_stride(feedback, propagated)

    def _candidate_free_energy(self, candidate: np.ndarray) -> float:
        return self.planner.nominal_sequence_cost(candidate)

    def _state_distance_sq(self, a: np.ndarray, b: np.ndarray) -> float:
        angle_error = float(wrap_angle_np(float(a[0] - b[0])))
        velocity_error = float(a[1] - b[1])
        return angle_error * angle_error + 0.05 * velocity_error * velocity_error

    def _interpolate_state(self, start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        angle_delta = float(wrap_angle_np(end[0] - start[0]))
        return np.array(
            [
                float(wrap_angle_np(start[0] + fraction * angle_delta)),
                float(start[1] + fraction * (end[1] - start[1])),
            ],
            dtype=float,
        )

    def _nominal_state_candidates(self, feedback: np.ndarray, propagated: np.ndarray) -> list[np.ndarray]:
        return [candidate for candidate, _ in self._nominal_state_candidates_with_strides(feedback, propagated)]

    def _nominal_state_candidates_with_strides(self, feedback: np.ndarray, propagated: np.ndarray) -> list[tuple[np.ndarray, int]]:
        feedback = np.asarray(feedback, dtype=float)
        propagated = np.asarray(propagated, dtype=float)
        previous = np.asarray(self.nominal_state, dtype=float)
        half_count = self.num_nominal_candidates // 2
        candidates: list[tuple[np.ndarray, int]] = []
        for i in range(half_count + 1):
            fraction = i / half_count
            stride = int(np.floor(fraction + 0.5))
            candidates.append((self._interpolate_state(previous, propagated, fraction), stride))
        for i in range(1, half_count + 1):
            fraction = i / half_count
            candidates.append((self._interpolate_state(propagated, feedback, fraction), 1))
        return candidates

    def _select_nominal_state(self, feedback: np.ndarray, propagated: np.ndarray) -> np.ndarray:
        selected, _ = self._select_nominal_state_and_stride(feedback, propagated)
        return selected

    def _select_nominal_state_and_stride(self, feedback: np.ndarray, propagated: np.ndarray) -> tuple[np.ndarray, int]:
        candidate_pairs = self._nominal_state_candidates_with_strides(feedback, propagated)
        candidates = [candidate for candidate, _ in candidate_pairs]
        costs = np.array([self._candidate_free_energy(candidate) for candidate in candidates], dtype=float)
        half_count = self.num_nominal_candidates // 2
        search_stop = len(candidates) if self.allow_real_nominal_projection else half_count + 1
        search_indices = np.arange(search_stop)
        scores = costs.copy()
        if self.nominal_projection_penalty > 0.0:
            scores += self.nominal_projection_penalty * np.array(
                [self._state_distance_sq(candidate, propagated) for candidate in candidates],
                dtype=float,
            )
        if self.nominal_jump_penalty > 0.0:
            previous = np.asarray(self.nominal_state, dtype=float)
            scores += self.nominal_jump_penalty * np.array(
                [self._state_distance_sq(candidate, previous) for candidate in candidates],
                dtype=float,
            )
        feasible_indices = search_indices[costs[:search_stop] < self.free_energy_threshold]
        if feasible_indices.size:
            selected_index = int(feasible_indices[np.argmin(scores[feasible_indices])])
            selected_reason = "soft_threshold"
        else:
            selected_index = int(search_indices[np.argmin(scores[:search_stop])])
            selected_reason = "soft_min_cost"
        self.last_nominal_candidate_count = len(candidates)
        self.last_selected_nominal_index = selected_index
        self.last_selected_nominal_stride = int(candidate_pairs[selected_index][1])
        self.last_selected_nominal_cost = float(costs[selected_index])
        self.last_selected_nominal_score = float(scores[selected_index])
        self.last_selected_nominal_source = (
            "previous_to_propagated" if selected_index <= half_count else "propagated_to_feedback"
        )
        self.last_selected_nominal_reason = selected_reason
        return candidates[selected_index].copy(), int(candidate_pairs[selected_index][1])

    def info(self) -> dict[str, float]:
        return {
            "nominal_theta": float(self.nominal_state[0]),
            "nominal_theta_dot": float(self.nominal_state[1]),
            "nominal_action": float(self.last_nominal_action),
            "real_action": float(self.last_real_action),
            "tracking_feedback": float(self.last_tracking_feedback),
            "tracking_gain_theta": float(self.last_tracking_gain[0]),
            "tracking_gain_theta_dot": float(self.last_tracking_gain[1]),
            "tracking_mode": self.tracking_mode,
            "nominal_candidate_count": float(self.last_nominal_candidate_count),
            "selected_nominal_index": float(self.last_selected_nominal_index),
            "selected_nominal_stride": float(self.last_selected_nominal_stride),
            "selected_nominal_cost": float(self.last_selected_nominal_cost),
            "selected_nominal_score": float(self.last_selected_nominal_score),
            "selected_nominal_source": self.last_selected_nominal_source,
            "selected_nominal_reason": self.last_selected_nominal_reason,
            "num_nominal_candidates": float(self.num_nominal_candidates),
            "nominal_stride": float(self.last_nominal_stride),
            "pending_nominal_stride": float(self.nominal_stride),
            "value_function_threshold": float(self.value_function_threshold),
            "allow_real_nominal_projection": float(self.allow_real_nominal_projection),
            "nominal_projection_penalty": float(self.nominal_projection_penalty),
            "nominal_jump_penalty": float(self.nominal_jump_penalty),
        }
