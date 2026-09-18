from __future__ import annotations

import numpy as np

from .dynamics import PendulumConfig, pendulum_step_np, wrap_angle_np


def tracking_feedback_np(actual_state: np.ndarray, nominal_state: np.ndarray, kp: float, kd: float) -> float:
    angle_error = float(wrap_angle_np(nominal_state[0] - actual_state[0]))
    vel_error = float(nominal_state[1] - actual_state[1])
    return kp * angle_error + kd * vel_error


def tracking_feedback_from_gain_np(actual_state: np.ndarray, nominal_state: np.ndarray, gain: np.ndarray) -> float:
    error = np.array(
        [
            float(wrap_angle_np(nominal_state[0] - actual_state[0])),
            float(nominal_state[1] - actual_state[1]),
        ],
        dtype=float,
    )
    return float(np.asarray(gain, dtype=float).reshape(2) @ error)


def linearize_pendulum_np(
    config: PendulumConfig,
    state: np.ndarray,
    action: float,
    state_eps: float = 1.0e-5,
    action_eps: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=float)
    a_mat = np.zeros((2, 2), dtype=float)
    for idx in range(2):
        delta = np.zeros(2, dtype=float)
        delta[idx] = state_eps
        plus = pendulum_step_np(state + delta, action, config, disturbance=0.0)
        minus = pendulum_step_np(state - delta, action, config, disturbance=0.0)
        a_mat[:, idx] = np.array(
            [
                float(wrap_angle_np(plus[0] - minus[0])),
                float(plus[1] - minus[1]),
            ],
            dtype=float,
        ) / (2.0 * state_eps)

    plus_u = pendulum_step_np(state, action + action_eps, config, disturbance=0.0)
    minus_u = pendulum_step_np(state, action - action_eps, config, disturbance=0.0)
    b_mat = (
        np.array(
            [
                float(wrap_angle_np(plus_u[0] - minus_u[0])),
                float(plus_u[1] - minus_u[1]),
            ],
            dtype=float,
        ).reshape(2, 1)
        / (2.0 * action_eps)
    )
    return a_mat, b_mat


def lqr_tracking_gain_np(
    config: PendulumConfig,
    nominal_state: np.ndarray,
    nominal_action: float,
    horizon: int = 20,
    q_diag: tuple[float, float] = (4.0, 0.3),
    r_value: float = 10.0,
) -> np.ndarray:
    a_mat, b_mat = linearize_pendulum_np(config, nominal_state, nominal_action)
    q_mat = np.diag(q_diag)
    p_mat = q_mat.copy()
    r_mat = np.array([[r_value]], dtype=float)
    gain = np.array([[0.0, 0.0]], dtype=float)
    for _ in range(max(1, horizon)):
        s_mat = r_mat + b_mat.T @ p_mat @ b_mat
        gain = np.linalg.solve(s_mat, b_mat.T @ p_mat @ a_mat)
        p_mat = q_mat + a_mat.T @ p_mat @ (a_mat - b_mat @ gain)
    return gain.reshape(2)


def tracking_feedback_control_np(
    config: PendulumConfig,
    actual_state: np.ndarray,
    nominal_state: np.ndarray,
    nominal_action: float,
    kp: float,
    kd: float,
    mode: str,
    lqr_horizon: int,
    lqr_q: tuple[float, float],
    lqr_r: float,
) -> tuple[float, np.ndarray]:
    if mode == "pd":
        gain = np.array([kp, kd], dtype=float)
    elif mode == "lqr":
        gain = lqr_tracking_gain_np(config, nominal_state, nominal_action, lqr_horizon, lqr_q, lqr_r)
    else:
        raise ValueError(f"tracking_mode must be 'pd' or 'lqr', got {mode!r}")
    return tracking_feedback_from_gain_np(actual_state, nominal_state, gain), gain
