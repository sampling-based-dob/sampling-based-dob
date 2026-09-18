"""Load the packaged controller and observer configuration templates."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import yaml


def load_config(relative_path: str) -> dict[str, Any]:
    """Load a YAML template from ``sdob/config``."""
    resource = files(__package__).joinpath(relative_path)
    if not resource.is_file():
        raise FileNotFoundError(f"unknown SDOB config: {relative_path}")
    content = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"SDOB config must contain a mapping: {relative_path}")
    return content


__all__ = ["load_config"]
