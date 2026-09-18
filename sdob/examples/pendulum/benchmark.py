from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .dynamics import PendulumConfig, pendulum_step_np, state_cost_np, wrap_angle_np
from .factory import make_controllers
from .interfaces import ControllerBase


def _synchronize_device(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


@dataclass(frozen=True)
class BenchmarkResult:
    summary: pd.DataFrame
    trajectories: pd.DataFrame


def disturbance_profile(name: str, value: float) -> callable:
    if name == "constant_bias":
        return lambda step, state: value
    if name == "sinusoidal_bias":
        return lambda step, state: value * np.sin(0.06 * step)
    if name == "step_bias":
        return lambda step, state: 0.0 if step < 40 else value
    raise ValueError(f"Unknown disturbance profile: {name}")


def run_suite(
    controllers: list[ControllerBase],
    config: PendulumConfig,
    steps: int,
    initial_state: np.ndarray,
    disturbance_name: str,
    disturbance_value: float,
    seed: int = 0,
    progress_label: str | None = None,
    device: str = "cpu",
) -> BenchmarkResult:
    np.random.default_rng(seed)
    disturbance_fn = disturbance_profile(disturbance_name, disturbance_value)
    trajectory_rows: list[dict[str, float | str | int]] = []
    summary_rows: list[dict[str, float | str]] = []

    for controller_index, controller in enumerate(controllers, start=1):
        if progress_label is not None:
            print(
                f"[{progress_label}] {controller_index}/{len(controllers)} running {controller.name}...",
                flush=True,
            )
        controller.reset()
        state = np.asarray(initial_state, dtype=float).copy()
        total_cost = 0.0
        control_energy = 0.0
        abs_angles: list[float] = []
        d_errors: list[float] = []
        d_dot_errors: list[float] = []
        planning_times_ms: list[float] = []
        previous_disturbance: float | None = None

        for step in range(steps):
            _synchronize_device(device)
            planning_start = time.perf_counter()
            action = float(controller.act(state.copy(), step))
            _synchronize_device(device)
            planning_time_ms = 1000.0 * (time.perf_counter() - planning_start)
            planning_times_ms.append(planning_time_ms)
            disturbance = float(disturbance_fn(step, state))
            feedback = pendulum_step_np(state, action, config, disturbance=disturbance)
            controller.observe(state.copy(), action, feedback.copy(), step)
            cost = state_cost_np(feedback, action, config)
            total_cost += cost
            control_energy += action * action * config.dt
            angle_abs = abs(float(wrap_angle_np(feedback[0])))
            abs_angles.append(angle_abs)
            info = controller.info()
            d_hat = float(info.get("d_hat", np.nan))
            d_dot_hat = float(info.get("d_dot_hat", np.nan))
            if np.isfinite(d_hat):
                d_errors.append(d_hat - disturbance)
            if previous_disturbance is not None and np.isfinite(d_dot_hat):
                true_rate = (disturbance - previous_disturbance) / max(config.dt, 1.0e-8)
                d_dot_errors.append(d_dot_hat - true_rate)
            previous_disturbance = disturbance
            trajectory_rows.append(
                {
                    "controller": controller.name,
                    "step": step,
                    "time": step * config.dt,
                    "theta": float(feedback[0]),
                    "theta_abs": angle_abs,
                    "theta_dot": float(feedback[1]),
                    "action": action,
                    "disturbance": disturbance,
                    "cost": cost,
                    "planning_time_ms": planning_time_ms,
                    "d_hat": d_hat,
                    "d_dot_hat": d_dot_hat,
                    "d_sigma": float(info.get("d_sigma", np.nan)),
                    "nominal_theta": float(info.get("nominal_theta", np.nan)),
                    "nominal_theta_dot": float(info.get("nominal_theta_dot", np.nan)),
                    "selected_nominal_index": float(info.get("selected_nominal_index", np.nan)),
                    "selected_nominal_stride": float(info.get("selected_nominal_stride", np.nan)),
                    "selected_nominal_cost": float(info.get("selected_nominal_cost", np.nan)),
                    "selected_nominal_score": float(info.get("selected_nominal_score", np.nan)),
                    "selected_nominal_source": str(info.get("selected_nominal_source", "")),
                    "selected_nominal_reason": str(info.get("selected_nominal_reason", "")),
                }
            )
            state = feedback

        tail = np.asarray(abs_angles[-max(1, steps // 5) :], dtype=float)
        d_error_array = np.asarray(d_errors, dtype=float)
        d_dot_error_array = np.asarray(d_dot_errors, dtype=float)
        d_tail = d_error_array[-max(1, len(d_error_array) // 5) :] if d_error_array.size else d_error_array
        summary_rows.append(
            {
                "controller": controller.name,
                "disturbance": disturbance_name,
                "disturbance_value": disturbance_value,
                "total_cost": total_cost,
                "rms_angle": float(np.sqrt(np.mean(np.square(abs_angles)))),
                "final_abs_angle": float(abs_angles[-1]),
                "tail_mean_abs_angle": float(np.mean(tail)),
                "max_abs_angle": float(np.max(abs_angles)),
                "control_energy": control_energy,
                "d_mae": float(np.mean(np.abs(d_error_array))) if d_error_array.size else float("nan"),
                "d_rmse": float(np.sqrt(np.mean(np.square(d_error_array)))) if d_error_array.size else float("nan"),
                "d_tail_mae": float(np.mean(np.abs(d_tail))) if d_tail.size else float("nan"),
                "d_dot_mae": float(np.mean(np.abs(d_dot_error_array))) if d_dot_error_array.size else float("nan"),
                "d_dot_rmse": (
                    float(np.sqrt(np.mean(np.square(d_dot_error_array)))) if d_dot_error_array.size else float("nan")
                ),
                "planning_time_mean_ms": float(np.mean(planning_times_ms)),
                "planning_time_p95_ms": float(np.quantile(planning_times_ms, 0.95)),
                "success": bool(np.mean(tail) < 0.12),
            }
        )

    return BenchmarkResult(summary=pd.DataFrame(summary_rows), trajectories=pd.DataFrame(trajectory_rows))


def with_nominal_mppi_baseline(
    result: BenchmarkResult,
    baseline: BenchmarkResult,
    baseline_name: str = "MPPI (no disturbance)",
) -> BenchmarkResult:
    baseline_summary = baseline.summary.copy()
    baseline_trajectories = baseline.trajectories.copy()
    baseline_summary["controller"] = baseline_name
    baseline_trajectories["controller"] = baseline_name
    summary = pd.concat([baseline_summary, result.summary.copy()], ignore_index=True)
    trajectories = pd.concat([baseline_trajectories, result.trajectories.copy()], ignore_index=True)
    return BenchmarkResult(summary=summary, trajectories=trajectories)


def save_result(result: BenchmarkResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_dir / "summary.csv", index=False)
    result.trajectories.to_csv(output_dir / "trajectories.csv", index=False)


def plot_result(result: BenchmarkResult, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for controller, frame in result.trajectories.groupby("controller"):
        time = frame["time"].to_numpy(dtype=float)
        axes[0].plot(time, frame["theta_abs"].to_numpy(dtype=float), label=controller)
        axes[1].plot(time, frame["action"].to_numpy(dtype=float), label=controller)
        if frame["d_hat"].notna().any():
            axes[2].plot(time, frame["d_hat"].to_numpy(dtype=float), label=controller)
    axes[0].set_ylabel("|theta| rad")
    axes[1].set_ylabel("torque")
    axes[2].set_ylabel("d_hat")
    axes[2].set_xlabel("time [s]")
    axes[0].legend(ncol=3, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = output_dir / "comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
