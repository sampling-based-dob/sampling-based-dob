from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one inverted-pendulum comparison run."""

    output_dir: Path = Path("outputs/pendulum/direct_compare")
    steps: int = 160
    num_samples: int = 512
    horizon: int = 20
    lambda_: float = 0.35
    noise_sigma: float = 1.2
    adaptive_temperature: bool = True
    eta_min: float = 0.05
    eta_max: float = 0.1
    lambda_decrease: float = 0.9
    lambda_increase: float = 1.2
    lambda_min: float = 1.0e-3
    lambda_max: float = 1.0e3
    sampling_control_cost: float = 0.01
    likelihood_alpha: float = 0.0
    action_cost: float = 0.0
    disturbance_compensation_cost: float = 0.0
    device: str = "cpu"
    seed: int = 7
    initial_theta: float = 0.35
    initial_theta_dot: float = 0.0
    disturbance: str = "constant_bias"
    disturbance_value: float = 0.75
    style: str = "gym"
    fps: int = 12
    stride: int = 2
    dpi: int = 115
    skip_gif: bool = False


def _coerce_config_value(name: str, value: Any) -> Any:
    if name == "output_dir":
        return Path(value)
    return value


def load_experiment_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
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
    if not overrides:
        return config

    clean_overrides = {
        name: _coerce_config_value(name, value)
        for name, value in overrides.items()
        if value is not None and name in valid_names
    }
    return replace(config, **clean_overrides)
