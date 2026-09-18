from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for real le-wm / stable-worldmodel SDOB comparisons."""

    task: str = "reacher"
    env_name: str = "swm/ReacherDMControl-v0"
    world_model_name: str = "reacher/lewm"
    lewm_root: Path = Path("third_party/le-wm")
    stable_worldmodel_root: Path = Path("third_party/stable-worldmodel")
    output_dir: Path = Path("outputs/reacher/reacher_lewm")
    write_media: bool = True
    device: str = "cpu"

    steps: int = 225
    img_size: int = 224
    seed: int = 42
    task_seed: int | None = None
    target_delta: tuple[float, ...] = (0.55, -0.45)

    horizon: int = 5
    receding_horizon: int = 1
    action_block: int = 5
    num_samples: int = 300
    cem_steps: int = 10
    topk: int = 0
    var_scale: float = 1.0
    mppi_temperature: float = 0.5
    adaptive_temperature: bool = True
    eta_min: float = 0.05
    eta_max: float = 0.2
    mppi_temperature_decrease: float = 0.9
    mppi_temperature_increase: float = 1.2
    mppi_min_temperature: float = 0.01
    mppi_max_temperature: float = 1.0
    mppi_action_clamp: bool = True
    mppi_noise_decay: float = 0.6
    mppi_min_variance: float = 0.0
    mppi_mean_update_rate: float = 0.3
    mppi_cost_normalize: bool = True
    action_reg_weight: float = 1.0
    smoothness_reg_weight: float = 1.0
    reacher_progress_weight: float = 0.0
    state_progress_weight: float = 0.0
    state_goal_cost_weight: float = 0.0
    state_goal_step_scale: float = 5.0
    state_goal_cost_steps: int = 1
    trajectory_cost_weight: float = 1.0
    trajectory_cost_steps: int = 5
    trajectory_cost_mode: str = "replace"

    dob_observer_mode: str = "particle_mean"
    dob_action_prior_rollout_gain: float = 0.8
    sdob_action_prior_rollout_gain: float = 0.85
    dob_action_prior_forecast_order: int = 0
    dob_action_prior_filter_alpha: float = 0.1
    dob_action_prior_max_norm: float = 0.25
    dob_action_prior_particle_samples: int = 256
    dob_action_prior_particle_std: float = 0.35
    dob_action_prior_particle_temperature: float = 0.05
    dob_action_prior_particle_process_noise: float = 0.002
    dob_action_prior_particle_prior_weight: float = 0.5
    dob_action_prior_particle_magnitude_prior_weight: float = 0.02
    dob_action_prior_particle_resample_ess_fraction: float = 0.3
    dob_action_prior_particle_action_clamp: bool = True
    dob_action_prior_window_size: int = 5


    disturbance_mode: str = "action_bias"
    disturb_action_bias: tuple[float, ...] = (0.15, -0.1)
    disturbance_amplitude: float = 0.15
    disturbance_frequency: float = 0.08


def _coerce_config_value(name: str, value: Any) -> Any:
    if name in {"output_dir", "lewm_root", "stable_worldmodel_root"}:
        return Path(value)
    if name in {"disturb_action_bias", "target_delta"}:
        return tuple(value)
    return value


def load_experiment_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"experiment config must be a mapping: {path}")

    valid_names = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(data).difference(valid_names))
    if unknown:
        raise ValueError(f"unknown experiment config keys in {path}: {unknown}")

    values = {name: _coerce_config_value(name, value) for name, value in data.items()}
    config = ExperimentConfig(**values)
    if overrides:
        clean_overrides = {
            name: _coerce_config_value(name, value)
            for name, value in overrides.items()
            if value is not None and name in valid_names
        }
        config = replace(config, **clean_overrides)

    if (
        type(config.dob_action_prior_forecast_order) is not int
        or config.dob_action_prior_forecast_order not in {0, 1}
    ):
        raise ValueError(
            "dob_action_prior_forecast_order must be exactly integer 0 or 1"
        )

    return config
