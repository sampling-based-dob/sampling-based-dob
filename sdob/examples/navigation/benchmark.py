"""Five-controller benchmark runner for the SDOB navigation toy simulator."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import torch

from .core import (
    ChaoticObstacle,
    DiffDriveDynamics,
    DynamicObstacle,
    MPPIController,
    SamplingDisturbanceObserver,
    interpolate_query_values,
)


DEFAULT_CONTROLLER_NAMES = ["MPPI", "MPPI+DOB", "MPPI+SDOB", "Tube-MPPI", "RMPPI"]
COLLISION_TOL = 1e-6
OBS_COLORS = [
    "#E53935",
    "#FB8C00",
    "#8E24AA",
    "#00838F",
    "#3949AB",
    "#7CB342",
    "#D81B60",
    "#6D4C41",
]
CONTROLLER_COLORS = {
    "MPPI": "#6E6E6E",
    "MPPI+DOB": "#1976D2",
    "MPPI+SDOB": "#2E7D32",
    "Tube-MPPI": "#8E24AA",
    "RMPPI": "#F57C00",
}


def _synchronize_device(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


@dataclass
class CompareConfig:
    scenario: str = "four"
    steps: int = 320
    dt: float = 0.05
    horizon: int = 30
    num_samples: int = 500
    seed: int = 42
    device: str = "cpu"
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
        / "outputs"
        / "compare"
        / "4obs"
    )
    make_gif: bool = True
    skip_gif: bool = False
    gif_skip: int = 3
    max_gif_frames: int = 200
    goal_radius: float = 0.35
    start: Tuple[float, float, float] = (0.0, 0.0, 0.7854)
    goal: Tuple[float, float, float] = (8.0, 8.0, 0.0)
    state_disturbance: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    disturbance: str = "constant_input"
    slip_amplitude: float = 0.0
    slip_frequency: float = 0.35
    slip_phase: float = 0.0
    input_disturbance_bias: Tuple[float, float] = (0.12, 0.24)
    input_disturbance_amplitude: Tuple[float, float] = (0.08, 0.16)
    input_disturbance_frequency: float = 0.08
    input_disturbance_phase: Tuple[float, float] = (0.0, 1.5708)
    perception_radius: Optional[float] = 3.0
    sdob_radius: Optional[float] = None
    lookahead_dist: float = 3.0
    num_cost_queries: int = 500

    def __post_init__(self):
        if self.sdob_radius is None and self.perception_radius is not None:
            self.sdob_radius = 0.8 * self.perception_radius
        if self.perception_radius is not None and self.perception_radius <= 0.0:
            raise ValueError("perception_radius must be positive")
        if self.sdob_radius is not None and self.sdob_radius <= 0.0:
            raise ValueError("sdob_radius must be positive")
        if (
            self.perception_radius is not None
            and self.sdob_radius is not None
            and self.sdob_radius > self.perception_radius
        ):
            raise ValueError("sdob_radius must not exceed perception_radius")


@dataclass
class ControllerResult:
    name: str
    states: np.ndarray
    controls: np.ndarray
    obstacles: List[List[Tuple[np.ndarray, float]]]
    min_safety: np.ndarray
    flow_errors: np.ndarray
    state_effect_errors: np.ndarray
    cost_forecast_nominal_errors: np.ndarray
    cost_forecast_sdob_errors: np.ndarray
    planning_times_ms: np.ndarray
    lambda_history: np.ndarray
    last_eta: Optional[float]
    metrics: Dict[str, float]


@dataclass
class CompareOutput:
    results: List[ControllerResult]
    summary_csv: Path
    trajectories_csv: Path
    figure_path: Path
    gif_path: Optional[Path]


def create_obstacles(scenario: str, device: str = "cpu"):
    """Create deterministic moving-obstacle scenarios."""
    scenario = scenario.lower()
    if scenario in {"four", "4obs", "few", "small", "multi"}:
        bounds = [-1.2, 10.2, -1.2, 10.2]
        return [
            DynamicObstacle([6.8, 0.8], 0.50, [-0.62, 0.72], bounds, device),
            DynamicObstacle([0.7, 3.2], 0.50, [0.76, 0.30], bounds, device),
            DynamicObstacle([7.6, 4.7], 0.50, [-0.72, 0.05], bounds, device),
            DynamicObstacle([3.0, 8.4], 0.50, [0.28, -0.72], bounds, device),
        ]

    if scenario in {"three", "3obs"}:
        bounds = [-1.2, 10.2, -1.2, 10.2]
        return [
            DynamicObstacle([4.1, 1.0], 0.48, [-0.12, 0.72], bounds, device),
            DynamicObstacle([2.2, 5.4], 0.50, [0.72, -0.15], bounds, device),
            DynamicObstacle([7.4, 4.6], 0.48, [-0.62, 0.18], bounds, device),
        ]

    if scenario in {"eight", "8obs"}:
        bounds = [-2, 12, -2, 12]
        return [
            DynamicObstacle([7.2, -0.4], 0.48, [-0.65, 0.82], bounds, device),
            DynamicObstacle([-0.4, 2.8], 0.48, [0.85, 0.35], bounds, device),
            DynamicObstacle([8.4, 3.2], 0.48, [-0.82, 0.28], bounds, device),
            DynamicObstacle([3.2, -0.8], 0.48, [0.28, 0.90], bounds, device),
            DynamicObstacle([2.2, 8.8], 0.48, [0.35, -0.75], bounds, device),
            DynamicObstacle([-0.6, 6.6], 0.48, [0.95, -0.25], bounds, device),
            DynamicObstacle([6.5, 9.0], 0.48, [-0.45, -0.72], bounds, device),
            DynamicObstacle([9.6, 7.6], 0.48, [-0.80, -0.20], bounds, device),
        ]

    if scenario == "chaotic":
        bounds = [-1.2, 10.2, -1.2, 10.2]
        return [
            ChaoticObstacle([7.5, -0.7], 0.48, [-0.70, 0.95], bounds,
                            [0.42, 0.34], [1.25, 0.85], [0.10, 0.90], device),
            ChaoticObstacle([-0.8, 3.0], 0.46, [1.00, 0.55], bounds,
                            [0.35, 0.45], [0.75, 1.45], [1.30, 0.20], device),
            ChaoticObstacle([8.8, 2.6], 0.50, [-1.05, 0.25], bounds,
                            [0.45, 0.60], [1.60, 0.95], [0.60, 1.80], device),
            ChaoticObstacle([3.1, -0.9], 0.44, [0.20, 1.05], bounds,
                            [0.60, 0.32], [1.00, 1.70], [2.00, 0.30], device),
            ChaoticObstacle([1.0, 9.2], 0.48, [0.65, -0.95], bounds,
                            [0.50, 0.45], [1.40, 0.80], [1.70, 1.10], device),
            ChaoticObstacle([-0.9, 6.5], 0.46, [1.05, -0.20], bounds,
                            [0.25, 0.55], [1.90, 1.15], [0.40, 2.40], device),
            ChaoticObstacle([6.3, 9.3], 0.50, [-0.35, -1.00], bounds,
                            [0.65, 0.25], [0.80, 1.80], [2.50, 0.70], device),
            ChaoticObstacle([9.4, 6.4], 0.52, [-0.95, -0.30], bounds,
                            [0.30, 0.65], [1.55, 1.05], [1.00, 2.10], device),
        ]

    raise ValueError(f"Unknown scenario {scenario!r}; expected few, eight, or chaotic")


def _snapshot_obstacles(obstacles):
    return [(obs.pos.detach().cpu().numpy().copy(), float(obs.radius)) for obs in obstacles]


def _clone_obstacle_dicts(obstacles):
    return [{"pos": obs["pos"].clone(), "radius": obs["radius"]} for obs in obstacles]


def _shift_obstacle_dicts(obstacles, dx, dy):
    if not obstacles:
        return []
    offset = torch.tensor([dx, dy], device=obstacles[0]["pos"].device)
    return [{"pos": obs["pos"] + offset, "radius": obs["radius"]} for obs in obstacles]


def cost_flow_evaluation_points(cost_flow, *, offset: float = 0.05):
    """Create held-out local queries offset from the observer's grid points."""
    if cost_flow is None:
        return None
    query_grid = cost_flow.get("query_points")
    if query_grid is None or query_grid.numel() == 0:
        return None
    offsets = query_grid.new_tensor(
        [
            [offset, 0.0],
            [-offset, 0.0],
            [0.0, offset],
            [0.0, -offset],
        ]
    )
    return (query_grid.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 2)


def _obstacle_query_cost(
    query_points,
    obstacles,
    *,
    obstacle_decay: float,
    obstacle_margin: float,
):
    costs = torch.zeros(
        query_points.shape[0],
        device=query_points.device,
        dtype=query_points.dtype,
    )
    for obstacle in obstacles:
        center = obstacle["pos"][:2].to(
            device=query_points.device,
            dtype=query_points.dtype,
        )
        distance = torch.norm(query_points - center.unsqueeze(0), dim=-1).clamp(min=1e-6)
        costs += torch.exp(
            -obstacle_decay
            * (distance - float(obstacle["radius"]) - obstacle_margin)
        )
    return costs


def _interpolate_cost_flow(query_points, cost_flow):
    velocity = interpolate_query_values(
        query_points,
        cost_flow,
        "query_velocities",
    )
    if velocity is None:
        return None
    return velocity * float(cost_flow.get("dt", 1.0))


def cost_forecast_errors(
    previous_obstacles,
    current_obstacles,
    cost_flow,
    *,
    query_points,
    obstacle_decay: float,
    obstacle_margin: float,
):
    """Measure one-step cost prediction on queries not used by the observer."""
    observed_cost = _obstacle_query_cost(
        query_points,
        current_obstacles,
        obstacle_decay=obstacle_decay,
        obstacle_margin=obstacle_margin,
    )
    nominal_cost = _obstacle_query_cost(
        query_points,
        previous_obstacles,
        obstacle_decay=obstacle_decay,
        obstacle_margin=obstacle_margin,
    )
    nominal_error = float(torch.mean(torch.abs(nominal_cost - observed_cost)).item())
    if cost_flow is None:
        return nominal_error, math.nan

    local_displacement = _interpolate_cost_flow(query_points, cost_flow)
    if local_displacement is None:
        return nominal_error, math.nan
    compensated_cost = _obstacle_query_cost(
        query_points - local_displacement,
        previous_obstacles,
        obstacle_decay=obstacle_decay,
        obstacle_margin=obstacle_margin,
    )
    compensated_error = float(
        torch.mean(torch.abs(compensated_cost - observed_cost)).item()
    )
    return nominal_error, compensated_error


class StateEffectDOB:
    """Small matched state-effect observer used by DOB and SDOB baselines."""

    def __init__(self, dynamics: DiffDriveDynamics, alpha: float = 0.6, clip: float = 0.08):
        self.dynamics = dynamics
        self.alpha = alpha
        self.clip = clip
        self.d_hat = torch.zeros(3, device=dynamics.device)

    def update(self, prev_state, action, feedback_state):
        pred = self.dynamics.step(prev_state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
        residual = feedback_state - pred
        residual[2] = torch.atan2(torch.sin(residual[2]), torch.cos(residual[2]))
        residual = torch.clamp(residual, -self.clip, self.clip)
        self.d_hat = self.alpha * residual + (1.0 - self.alpha) * self.d_hat

    def estimate(self):
        return self.d_hat.clone()


class ControllerAgent:
    def __init__(self, name, planner, dynamics, state_observer=None, cost_observer=None):
        self.name = name
        self.planner = planner
        self.dynamics = dynamics
        self.state_observer = state_observer
        self.cost_observer = cost_observer
        self.last_cost_flow = None

    def act(self, state, goal, obstacles, step_i):
        state_disturbance = (
            self.state_observer.estimate() if self.state_observer is not None else None
        )
        self.last_cost_flow = (
            self.cost_observer.estimate(state, obstacles)
            if self.cost_observer is not None
            else None
        )
        action = self.planner.step(
            state,
            goal,
            obstacles,
            cost_flow=self.last_cost_flow,
            state_disturbance=state_disturbance,
        )
        return action, {"cost_flow": self.last_cost_flow}

    def observe(self, prev_state, action, feedback_state):
        if self.state_observer is not None:
            self.state_observer.update(prev_state, action, feedback_state)


class RMPPIAgent(ControllerAgent):
    """Scenario-risk MPPI baseline using uncertain obstacle positions."""

    def __init__(self, name, planner, dynamics, offsets):
        super().__init__(name, planner, dynamics)
        self.offsets = offsets

    def act(self, state, goal, obstacles, step_i):
        scenarios = [_clone_obstacle_dicts(obstacles)]
        scenarios.extend(_shift_obstacle_dicts(obstacles, dx, dy) for dx, dy in self.offsets)
        action = self.planner.step(
            state,
            goal,
            obstacles,
            obstacle_scenarios=scenarios,
        )
        return action, {"cost_flow": None}


class CompareRunner:
    def __init__(self, config: CompareConfig):
        self.config = config
        self.config.output_dir = Path(self.config.output_dir)
        self.run_dir = self.config.output_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.goal = torch.tensor(self.config.goal, dtype=torch.float32, device=self.config.device)
        self.start = torch.tensor(self.config.start, dtype=torch.float32, device=self.config.device)
        self.actual_state_disturbance = torch.tensor(
            self.config.state_disturbance,
            dtype=torch.float32,
            device=self.config.device,
        )
        self.input_disturbance_amplitude = torch.tensor(
            self.config.input_disturbance_amplitude,
            dtype=torch.float32,
            device=self.config.device,
        )
        self.input_disturbance_bias = torch.tensor(
            self.config.input_disturbance_bias,
            dtype=torch.float32,
            device=self.config.device,
        )
        self.input_disturbance_phase = torch.tensor(
            self.config.input_disturbance_phase,
            dtype=torch.float32,
            device=self.config.device,
        )

    def run(self) -> CompareOutput:
        results = []
        for name in DEFAULT_CONTROLLER_NAMES:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
            dynamics = self._make_dynamics()
            agent = self._make_agent(name, dynamics)
            obstacles = create_obstacles(self.config.scenario, device=self.config.device)
            results.append(self._run_one(agent, dynamics, obstacles))

        summary_csv = self.run_dir / "summary.csv"
        trajectories_csv = self.run_dir / "trajectories.csv"
        figure_path = self.run_dir / "comparison.png"
        gif_path = (
            self.run_dir / "comparison.gif"
            if self.config.make_gif and not self.config.skip_gif
            else None
        )

        self._write_summary(summary_csv, results)
        self._write_trajectories(trajectories_csv, results)
        self._plot_comparison(figure_path, results)
        if gif_path is not None:
            self._make_animation(gif_path, results)

        return CompareOutput(
            results=results,
            summary_csv=summary_csv,
            trajectories_csv=trajectories_csv,
            figure_path=figure_path,
            gif_path=gif_path,
        )

    def _make_dynamics(self):
        return DiffDriveDynamics(dt=self.config.dt, device=self.config.device)

    def _perceived_obstacles(self, state, obstacles):
        radius = self.config.perception_radius
        if radius is None:
            return obstacles
        return [
            obstacle
            for obstacle in obstacles
            if torch.norm(obstacle["pos"][:2] - state[:2]) - obstacle["radius"]
            <= radius
        ]

    def _state_disturbance(self, state, step_i):
        if self.config.disturbance in {"none", "sinusoidal_input", "constant_input"}:
            return self.actual_state_disturbance.clone()
        if self.config.disturbance != "sinusoidal_slip":
            raise ValueError(
                f"unknown disturbance {self.config.disturbance!r}; "
                "expected none, sinusoidal_slip, sinusoidal_input, or constant_input"
            )

        t = step_i * self.config.dt
        magnitude = self.config.slip_amplitude * math.sin(
            2.0 * math.pi * self.config.slip_frequency * t
            + self.config.slip_phase
        )
        heading = state[2]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)])
        disturbance_xy = magnitude * direction
        return torch.cat(
            [disturbance_xy, state.new_tensor([0.0])]
        ) + self.actual_state_disturbance

    def _input_disturbance(self, step_i):
        if self.config.disturbance == "constant_input":
            return self.input_disturbance_bias.clone()
        if self.config.disturbance != "sinusoidal_input":
            return self.input_disturbance_amplitude.new_zeros(2)

        t = step_i * self.config.dt
        phase = (
            2.0
            * math.pi
            * self.config.input_disturbance_frequency
            * t
            + self.input_disturbance_phase
        )
        return self.input_disturbance_amplitude * torch.sin(phase)

    def _executed_action(self, action, input_disturbance):
        executed = action + input_disturbance
        executed = executed.clone()
        executed[0] = torch.clamp(executed[0], 0.0, 1.0)
        executed[1] = torch.clamp(executed[1], -2.0, 2.0)
        return executed

    def _mppi_kwargs(self):
        return {
            "horizon": self.config.horizon,
            "num_samples": self.config.num_samples,
            "lambda_": 1.0,
            "goal_weight": 25.0,
            "running_goal_weight": 2.5,
            "obstacle_weight": 60.0,
            "obstacle_decay": 5.5,
            "obstacle_margin": 0.05,
            "control_weight": 0.05,
            "tail_goal_weight": 30.0,
            "tail_goal_steps": 8,
            "flow_weight": 2000.0,
            "flow_mode": "mean",
            "flow_synthesis": "map_update",
            "adaptive_temperature": True,
            "eta_min": 0.05,
            "eta_max": 0.1,
            "device": self.config.device,
        }

    def _make_agent(self, name, dynamics):
        kwargs = self._mppi_kwargs()
        if name == "MPPI":
            return ControllerAgent(name, MPPIController(dynamics, **kwargs), dynamics)

        if name == "MPPI+DOB":
            return ControllerAgent(
                name,
                MPPIController(dynamics, **kwargs),
                dynamics,
                state_observer=StateEffectDOB(dynamics),
            )

        if name == "MPPI+SDOB":
            flow_order = 2 if self.config.scenario == "chaotic" else 1
            cost_observer = SamplingDisturbanceObserver(
                num_samples=self.config.num_cost_queries,
                sigma=[1.2, 1.2, 0.1],
                alpha=1.0,
                alpha_dot=0.6,
                scale=35.0,
                obstacle_decay=kwargs["obstacle_decay"],
                obstacle_margin=kwargs["obstacle_margin"],
                num_clusters=5,
                lookahead_dist=self.config.lookahead_dist,
                n_lookahead_centers=6,
                perception_radius=self.config.perception_radius,
                sdob_radius=self.config.sdob_radius,
                dt=self.config.dt,
                flow_order=flow_order,
                flow_velocity_alpha=0.6 if flow_order == 2 else 1.0,
                flow_accel_alpha=0.3,
                flow_accel_clip=0.30,
                device=self.config.device,
            )
            if self.config.scenario == "chaotic":
                kwargs["flow_transport_gain"] = 0.5
            return ControllerAgent(
                name,
                MPPIController(dynamics, **kwargs),
                dynamics,
                state_observer=StateEffectDOB(dynamics),
                cost_observer=cost_observer,
            )

        if name == "Tube-MPPI":
            kwargs.update(
                {
                    "obstacle_weight": kwargs["obstacle_weight"] * 1.7,
                    "control_weight": kwargs["control_weight"] * 1.4,
                }
            )
            return ControllerAgent(name, MPPIController(dynamics, **kwargs), dynamics)

        if name == "RMPPI":
            kwargs.update(
                {
                    "risk_weight": 1.2,
                    "obstacle_weight": kwargs["obstacle_weight"] * 1.25,
                }
            )
            offsets = [
                (0.35, 0.0),
                (-0.35, 0.0),
                (0.0, 0.35),
                (0.0, -0.35),
                (0.25, 0.25),
                (0.25, -0.25),
                (-0.25, 0.25),
                (-0.25, -0.25),
            ]
            return RMPPIAgent(name, MPPIController(dynamics, **kwargs), dynamics, offsets)

        raise ValueError(f"Unknown controller {name!r}")

    def _run_one(self, agent, dynamics, obstacles):
        state = self.start.clone()
        states = [state.detach().cpu().numpy().copy()]
        controls = []
        obstacle_log = [_snapshot_obstacles(obstacles)]
        min_safety = []
        flow_errors = []
        state_effect_errors = []
        cost_forecast_nominal_errors = []
        cost_forecast_sdob_errors = []
        planning_times_ms = []
        prev_obs_dicts = None
        first_hit_step = None

        for step_i in range(self.config.steps):
            world_obs_dicts = [obs.to_dict() for obs in obstacles]
            obs_dicts = self._perceived_obstacles(state, world_obs_dicts)
            prev_state = state.clone()
            _synchronize_device(self.config.device)
            planning_start = time.perf_counter()
            action, diagnostics = agent.act(state, self.goal, obs_dicts, step_i)
            _synchronize_device(self.config.device)
            planning_times_ms.append(1000.0 * (time.perf_counter() - planning_start))
            flow_errors.append(
                self._cost_flow_error(
                    diagnostics.get("cost_flow"),
                    prev_obs_dicts,
                    world_obs_dicts,
                )
            )

            true_disturbance = self._state_disturbance(prev_state, step_i)
            true_input_disturbance = self._input_disturbance(step_i)
            executed_action = self._executed_action(action, true_input_disturbance)
            state = dynamics.step(
                state.unsqueeze(0),
                executed_action.unsqueeze(0),
                disturbance=true_disturbance,
            ).squeeze(0)
            for obs in obstacles:
                obs.step(self.config.dt)
            next_obs_dicts = [obs.to_dict() for obs in obstacles]
            evaluation_points = cost_flow_evaluation_points(
                diagnostics.get("cost_flow")
            )
            if evaluation_points is None:
                nominal_forecast_error, sdob_forecast_error = math.nan, math.nan
            else:
                nominal_forecast_error, sdob_forecast_error = cost_forecast_errors(
                    obs_dicts,
                    next_obs_dicts,
                    diagnostics.get("cost_flow"),
                    query_points=evaluation_points,
                    obstacle_decay=agent.planner.obstacle_decay,
                    obstacle_margin=agent.planner.obstacle_margin,
                )
            cost_forecast_nominal_errors.append(nominal_forecast_error)
            cost_forecast_sdob_errors.append(sdob_forecast_error)

            agent.observe(prev_state, action, state)
            nominal_next = dynamics.step(
                prev_state.unsqueeze(0),
                action.unsqueeze(0),
                disturbance=true_disturbance,
            ).squeeze(0)
            disturbed_next = dynamics.step(
                prev_state.unsqueeze(0),
                executed_action.unsqueeze(0),
                disturbance=true_disturbance,
            ).squeeze(0)
            true_state_effect = disturbed_next - nominal_next
            true_state_effect[2] = torch.atan2(
                torch.sin(true_state_effect[2]),
                torch.cos(true_state_effect[2]),
            )
            state_effect_errors.append(
                self._state_effect_error(agent, true_state_effect)
            )
            prev_obs_dicts = world_obs_dicts

            current_safety = min(
                torch.norm(state[:2] - obs.pos).item() - float(obs.radius)
                for obs in obstacles
            )
            min_safety.append(current_safety)
            controls.append(action.detach().cpu().numpy().copy())
            states.append(state.detach().cpu().numpy().copy())
            obstacle_log.append(_snapshot_obstacles(obstacles))

            if first_hit_step is None:
                err = torch.norm(state[:2] - self.goal[:2]).item()
                if err < self.config.goal_radius:
                    first_hit_step = step_i

        states_arr = np.asarray(states)
        controls_arr = np.asarray(controls)
        safety_arr = np.asarray(min_safety)
        flow_err_arr = np.asarray(flow_errors, dtype=float)
        state_err_arr = np.asarray(state_effect_errors, dtype=float)
        cost_nominal_arr = np.asarray(cost_forecast_nominal_errors, dtype=float)
        cost_sdob_arr = np.asarray(cost_forecast_sdob_errors, dtype=float)
        planning_time_arr = np.asarray(planning_times_ms, dtype=float)
        metrics = self._metrics(
            states_arr,
            controls_arr,
            safety_arr,
            flow_err_arr,
            state_err_arr,
            cost_nominal_arr,
            cost_sdob_arr,
            planning_time_arr,
            first_hit_step,
        )

        return ControllerResult(
            name=agent.name,
            states=states_arr,
            controls=controls_arr,
            obstacles=obstacle_log,
            min_safety=safety_arr,
            flow_errors=flow_err_arr,
            state_effect_errors=state_err_arr,
            cost_forecast_nominal_errors=cost_nominal_arr,
            cost_forecast_sdob_errors=cost_sdob_arr,
            planning_times_ms=planning_time_arr,
            lambda_history=np.asarray(agent.planner.lambda_history, dtype=float),
            last_eta=agent.planner.last_eta,
            metrics=metrics,
        )

    def _state_effect_error(self, agent, true_disturbance):
        if agent.state_observer is None:
            return math.nan
        estimate = agent.state_observer.estimate()
        return float(torch.norm(estimate - true_disturbance).detach().cpu().item())

    def _metrics(
        self,
        states,
        controls,
        safety,
        flow_errors,
        state_effect_errors,
        cost_forecast_nominal_errors,
        cost_forecast_sdob_errors,
        planning_times_ms,
        first_hit_step,
    ):
        goal_xy = np.asarray(self.config.goal[:2], dtype=float)
        traj_err = np.linalg.norm(states[:, :2] - goal_xy.reshape(1, 2), axis=1)
        tail_n = max(1, len(traj_err) // 5)
        finite_flow = flow_errors[np.isfinite(flow_errors)]
        finite_state = state_effect_errors[np.isfinite(state_effect_errors)]
        finite_nominal = cost_forecast_nominal_errors[
            np.isfinite(cost_forecast_nominal_errors)
        ]
        finite_sdob = cost_forecast_sdob_errors[np.isfinite(cost_forecast_sdob_errors)]
        collision_steps = np.flatnonzero(safety < -COLLISION_TOL)
        first_collision_step = int(collision_steps[0]) if collision_steps.size else None
        collision_before_goal = first_collision_step is not None and (
            first_hit_step is None or first_collision_step <= first_hit_step
        )
        success = first_hit_step is not None and not collision_before_goal
        terminal_step = (
            first_collision_step
            if collision_before_goal
            else first_hit_step if success else len(controls) - 1
        )
        terminal_controls = controls[: terminal_step + 1]
        terminal_states = states[: terminal_step + 2]
        path_length = float(
            np.linalg.norm(np.diff(terminal_states[:, :2], axis=0), axis=1).sum()
        )
        direct_distance = float(np.linalg.norm(goal_xy - states[0, :2]))
        control_tv = (
            float(np.linalg.norm(np.diff(terminal_controls, axis=0), axis=1).mean())
            if len(terminal_controls) > 1
            else 0.0
        )
        nominal_forecast_mae = (
            float(finite_nominal.mean()) if finite_nominal.size else math.nan
        )
        sdob_forecast_mae = float(finite_sdob.mean()) if finite_sdob.size else math.nan
        forecast_reduction = (
            100.0 * (1.0 - sdob_forecast_mae / nominal_forecast_mae)
            if np.isfinite(sdob_forecast_mae) and nominal_forecast_mae > 0.0
            else math.nan
        )
        return {
            "final_error": float(traj_err[-1]),
            "min_error": float(traj_err.min()),
            "tail_error": float(traj_err[-tail_n:].mean()),
            "mean_safety": float(safety.mean()) if safety.size else math.nan,
            "min_safety": float(safety.min()) if safety.size else math.nan,
            "collision": float(safety.min() < -COLLISION_TOL) if safety.size else 0.0,
            "success": float(success),
            "collision_before_goal": float(collision_before_goal),
            "terminal_step": float(terminal_step),
            "time_to_goal": (
                float((first_hit_step + 1) * self.config.dt) if success else math.nan
            ),
            "path_length": path_length,
            "path_ratio": (
                path_length / direct_distance if direct_distance > 0.0 else math.nan
            ),
            "control_tv": control_tv,
            "first_hit_step": float(first_hit_step) if first_hit_step is not None else math.nan,
            "first_hit_time": (
                float((first_hit_step + 1) * self.config.dt)
                if first_hit_step is not None
                else math.nan
            ),
            "cost_flow_estimation_error": (
                float(finite_flow.mean()) if finite_flow.size else math.nan
            ),
            "state_effect_estimation_error": (
                float(finite_state.mean()) if finite_state.size else math.nan
            ),
            "cost_forecast_nominal_mae": nominal_forecast_mae,
            "cost_forecast_sdob_mae": sdob_forecast_mae,
            "cost_forecast_reduction": forecast_reduction,
            "planning_time_mean_ms": float(planning_times_ms.mean()),
            "planning_time_p95_ms": float(np.quantile(planning_times_ms, 0.95)),
        }

    def _cost_flow_error(self, cost_flow, prev_obstacles, curr_obstacles):
        if cost_flow is None or prev_obstacles is None:
            return math.nan

        query_points = cost_flow.get("query_points")
        query_weights = cost_flow.get("query_weights")
        velocities = cost_flow.get("query_velocities")
        if query_points is None or velocities is None or query_points.numel() == 0:
            return math.nan

        if query_weights is None:
            query_weights = torch.ones(
                query_points.shape[0],
                dtype=query_points.dtype,
                device=query_points.device,
            )
        active = query_weights > 1e-8
        if not torch.any(active):
            return math.nan
        query_points = query_points[active]
        velocities = velocities[active]
        query_weights = query_weights[active]

        prev_pos = torch.stack([obs["pos"] for obs in prev_obstacles], dim=0)
        curr_pos = torch.stack([obs["pos"] for obs in curr_obstacles], dim=0)
        true_displacements = curr_pos - prev_pos
        nearest = torch.cdist(query_points, curr_pos).argmin(dim=1)
        predicted_displacements = velocities * float(cost_flow.get("dt", self.config.dt))
        err = torch.norm(
            predicted_displacements - true_displacements[nearest],
            dim=1,
        )
        weighted_error = (query_weights * err).sum() / query_weights.sum()
        return float(weighted_error.detach().cpu().item())

    def _write_summary(self, path, results):
        fields = [
            "controller",
            "final_error",
            "min_error",
            "tail_error",
            "mean_safety",
            "min_safety",
            "collision",
            "first_hit_step",
            "first_hit_time",
            "cost_flow_estimation_error",
            "state_effect_estimation_error",
            "success",
            "collision_before_goal",
            "terminal_step",
            "time_to_goal",
            "path_length",
            "path_ratio",
            "control_tv",
            "cost_forecast_nominal_mae",
            "cost_forecast_sdob_mae",
            "cost_forecast_reduction",
            "planning_time_mean_ms",
            "planning_time_p95_ms",
            "lambda_final",
            "eta_last",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for result in results:
                row = {"controller": result.name}
                row.update(result.metrics)
                row["lambda_final"] = float(result.lambda_history[-1])
                row["eta_last"] = (
                    float(result.last_eta) if result.last_eta is not None else math.nan
                )
                writer.writerow(row)

    def _write_trajectories(self, path, results):
        fields = [
            "controller",
            "step",
            "time",
            "x",
            "y",
            "theta",
            "v",
            "omega",
            "min_safety",
            "cost_flow_estimation_error",
            "state_effect_estimation_error",
            "cost_forecast_nominal_mae",
            "cost_forecast_sdob_mae",
            "planning_time_ms",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for result in results:
                for step_i, control in enumerate(result.controls):
                    state = result.states[step_i + 1]
                    row = {
                        "controller": result.name,
                        "step": step_i,
                        "time": (step_i + 1) * self.config.dt,
                        "x": state[0],
                        "y": state[1],
                        "theta": state[2],
                        "v": control[0],
                        "omega": control[1],
                        "min_safety": result.min_safety[step_i],
                        "cost_flow_estimation_error": result.flow_errors[step_i],
                        "state_effect_estimation_error": result.state_effect_errors[step_i],
                        "cost_forecast_nominal_mae": (
                            result.cost_forecast_nominal_errors[step_i]
                        ),
                        "cost_forecast_sdob_mae": (
                            result.cost_forecast_sdob_errors[step_i]
                        ),
                        "planning_time_ms": result.planning_times_ms[step_i],
                    }
                    writer.writerow(row)

    def _setup_map(self, ax):
        ax.set_xlim(-1.0, 10.0)
        ax.set_ylim(-1.0, 10.0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.18)
        ax.plot(self.config.start[0], self.config.start[1], "ks", ms=6, label="start")
        ax.plot(self.config.goal[0], self.config.goal[1], "r*", ms=13, label="goal")

    def _draw_obstacles(self, ax, obstacle_snapshot, alpha=0.24):
        for i, (pos, radius) in enumerate(obstacle_snapshot):
            color = OBS_COLORS[i % len(OBS_COLORS)]
            ax.add_patch(Circle(pos, radius, fc=color, ec=color, alpha=alpha, lw=1.2))

    def _plot_comparison(self, path, results):
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(
            f"SDOB Navigation Benchmark | {self.config.scenario}",
            fontsize=14,
            fontweight="bold",
        )

        ax = axes[0, 0]
        self._setup_map(ax)
        self._draw_obstacles(ax, results[0].obstacles[-1], alpha=0.18)
        for result in results:
            color = CONTROLLER_COLORS[result.name]
            ax.plot(result.states[:, 0], result.states[:, 1], lw=2.0, color=color,
                    label=result.name)
        ax.set_title("Trajectories")
        ax.legend(fontsize=8, loc="upper left")

        ax = axes[0, 1]
        for result in results:
            t = np.arange(len(result.min_safety)) * self.config.dt
            ax.plot(t, result.min_safety, color=CONTROLLER_COLORS[result.name],
                    lw=1.6, label=result.name)
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.45)
        ax.axhline(-COLLISION_TOL, color="tab:red", lw=0.8, ls=":", alpha=0.7)
        ax.set_title("Safety Margin")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("distance to nearest obstacle (m)")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        names = [result.name for result in results]
        final_errors = [result.metrics["final_error"] for result in results]
        tail_errors = [result.metrics["tail_error"] for result in results]
        x = np.arange(len(names))
        ax.bar(x - 0.18, final_errors, 0.36, label="final")
        ax.bar(x + 0.18, tail_errors, 0.36, label="tail")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, ha="right")
        ax.set_title("Goal Error")
        ax.set_ylabel("m")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        for result in results:
            if np.isfinite(result.flow_errors).any():
                t = np.arange(len(result.flow_errors)) * self.config.dt
                ax.plot(t, result.flow_errors, color=CONTROLLER_COLORS[result.name],
                        lw=1.6, label=result.name)
        ax.set_title("Cost-Flow Estimation Error")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("m per step")
        if ax.lines:
            ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _make_animation(self, path, results):
        max_frames = max(len(result.states) for result in results)
        skip = max(1, self.config.gif_skip)
        if self.config.max_gif_frames > 0:
            skip = max(skip, math.ceil(max_frames / self.config.max_gif_frames))
        frame_ids = list(range(0, max_frames, skip))
        fig, ax = plt.subplots(figsize=(7, 7))

        def update(frame_idx):
            ax.clear()
            self._setup_map(ax)
            ref = results[0]
            obs_idx = min(frame_idx, len(ref.obstacles) - 1)
            self._draw_obstacles(ax, ref.obstacles[obs_idx], alpha=0.18)
            for result in results:
                idx = min(frame_idx, len(result.states) - 1)
                color = CONTROLLER_COLORS[result.name]
                ax.plot(result.states[: idx + 1, 0], result.states[: idx + 1, 1],
                        color=color, lw=1.9, label=result.name)
                ax.plot(result.states[idx, 0], result.states[idx, 1], "o",
                        color=color, ms=5)
            ax.set_title(f"{self.config.scenario} | t={frame_idx * self.config.dt:.2f}s")
            ax.legend(fontsize=8, loc="upper left")

        animation = FuncAnimation(fig, update, frames=frame_ids, interval=50)
        animation.save(path, writer=PillowWriter(fps=20))
        plt.close(fig)
