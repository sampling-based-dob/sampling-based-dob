from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

os.environ.setdefault("PYTORCH_JIT", "0")

import torch


@dataclass(frozen=True)
class PendulumConfig:
    """Continuous-torque inverted pendulum parameters."""

    dynamics_model: str = "simple"
    dt: float = 0.05
    gravity: float = 9.81
    mass: float = 1.0
    length: float = 1.0
    damping: float = 0.06
    max_torque: float = 2.5
    max_speed: float = 12.0
    angle_cost: float = 8.0
    velocity_cost: float = 0.25
    action_cost: float = 0.03
    action_smoothness_cost: float = 0.02
    disturbance_compensation_cost: float = 0.0
    terminal_angle_cost: float = 18.0
    terminal_velocity_cost: float = 0.8

    @classmethod
    def gym_pendulum(cls, **overrides: float | str) -> "PendulumConfig":
        values: dict[str, float | str] = {
            "dynamics_model": "gym",
            "dt": 0.05,
            "gravity": 10.0,
            "mass": 1.0,
            "length": 1.0,
            "damping": 0.0,
            "max_torque": 2.0,
            "max_speed": 8.0,
            "angle_cost": 1.0,
            "velocity_cost": 0.1,
            "action_cost": 0.0,
            "action_smoothness_cost": 0.03,
            "disturbance_compensation_cost": 2.0,
            "terminal_angle_cost": 6.0,
            "terminal_velocity_cost": 0.4,
        }
        values.update(overrides)
        return cls(**values)


def wrap_angle_np(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def pendulum_step_np(
    state: np.ndarray,
    action: np.ndarray | float,
    config: PendulumConfig,
    disturbance: float = 0.0,
) -> np.ndarray:
    theta, theta_dot = np.asarray(state, dtype=float)
    torque = float(np.asarray(action).reshape(-1)[0])
    torque = float(np.clip(torque, -config.max_torque, config.max_torque))
    if config.dynamics_model == "gym":
        theta_ddot = (
            3.0 * config.gravity / (2.0 * config.length) * np.sin(theta)
            + 3.0 * (torque + disturbance) / (config.mass * config.length * config.length)
        )
    else:
        inertia = config.mass * config.length * config.length
        theta_ddot = (
            config.gravity / config.length * np.sin(theta)
            + (torque + disturbance - config.damping * theta_dot) / inertia
        )
    next_dot = float(np.clip(theta_dot + config.dt * theta_ddot, -config.max_speed, config.max_speed))
    next_theta = float(wrap_angle_np(theta + config.dt * next_dot))
    return np.array([next_theta, next_dot], dtype=float)


def pendulum_step_torch(
    state: torch.Tensor,
    action: torch.Tensor,
    config: PendulumConfig,
    disturbance: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    theta = state[..., 0]
    theta_dot = state[..., 1]
    torque = action[..., 0].clamp(-config.max_torque, config.max_torque)
    if not torch.is_tensor(disturbance):
        disturbance = torch.as_tensor(disturbance, dtype=state.dtype, device=state.device)
    disturbance = disturbance.to(dtype=state.dtype, device=state.device)
    if config.dynamics_model == "gym":
        theta_ddot = (
            3.0 * config.gravity / (2.0 * config.length) * torch.sin(theta)
            + 3.0 * (torque + disturbance) / (config.mass * config.length * config.length)
        )
    else:
        inertia = config.mass * config.length * config.length
        theta_ddot = (
            config.gravity / config.length * torch.sin(theta)
            + (torque + disturbance - config.damping * theta_dot) / inertia
        )
    next_dot = (theta_dot + config.dt * theta_ddot).clamp(-config.max_speed, config.max_speed)
    next_theta = wrap_angle_torch(theta + config.dt * next_dot)
    return torch.stack((next_theta, next_dot), dim=-1)


def running_cost_torch(state: torch.Tensor, action: torch.Tensor, config: PendulumConfig) -> torch.Tensor:
    angle = wrap_angle_torch(state[..., 0])
    theta_dot = state[..., 1]
    torque = action[..., 0]
    return (
        config.angle_cost * angle.square()
        + config.velocity_cost * theta_dot.square()
        + config.action_cost * torque.square()
    )


def terminal_cost_torch(state: torch.Tensor, config: PendulumConfig) -> torch.Tensor:
    angle = wrap_angle_torch(state[..., 0])
    theta_dot = state[..., 1]
    return config.terminal_angle_cost * angle.square() + config.terminal_velocity_cost * theta_dot.square()


def state_cost_np(state: np.ndarray, action: float, config: PendulumConfig) -> float:
    angle = float(wrap_angle_np(float(state[0])))
    return float(
        config.angle_cost * angle * angle
        + config.velocity_cost * float(state[1]) ** 2
        + config.action_cost * action * action
    )
