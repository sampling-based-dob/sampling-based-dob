"""Configuration loading for SDOB navigation comparisons."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml

from .benchmark import CompareConfig


def _coerce_config_value(name: str, value: Any) -> Any:
    if name == "output_dir":
        return Path(value)
    if name in {
        "start",
        "goal",
        "state_disturbance",
        "input_disturbance_bias",
        "input_disturbance_amplitude",
        "input_disturbance_phase",
    }:
        return tuple(value)
    return value


def load_experiment_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> CompareConfig:
    """Load an sdob.examples.pendulum-style YAML config."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"experiment config must be a mapping: {path}")

    valid_names = {field.name for field in fields(CompareConfig)}
    unknown = sorted(set(data).difference(valid_names))
    if unknown:
        raise ValueError(f"unknown experiment config keys in {path}: {unknown}")

    values = {name: _coerce_config_value(name, value) for name, value in data.items()}
    config = CompareConfig(**values)
    if not overrides:
        return config

    clean_overrides = {
        name: _coerce_config_value(name, value)
        for name, value in overrides.items()
        if value is not None and name in valid_names
    }
    return replace(config, **clean_overrides)
