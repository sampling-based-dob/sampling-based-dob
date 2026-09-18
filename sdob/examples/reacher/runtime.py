"""Locate separately installed upstream sources without machine-specific paths."""

import os
import sys
from pathlib import Path


def ensure_runtime_paths(config=None):
    root = Path(__file__).resolve().parents[3]
    sources = (
        ("SDOB_LEWM_ROOT", "lewm_root", "le-wm", "jepa.py"),
        ("SDOB_SWM_ROOT", "stable_worldmodel_root", "stable-worldmodel", "stable_worldmodel/__init__.py"),
    )
    for variable, field, folder, marker in reversed(sources):
        configured = Path(getattr(config, field, root / "third_party" / folder))
        if not configured.is_absolute():
            configured = root / configured
        path = Path(os.environ.get(variable, configured)).resolve()
        if not (path / marker).is_file():
            raise FileNotFoundError(
                f"Missing upstream source: {path / marker}. "
                "Run `python scripts/setup_reacher.py` from the checkout, "
                f"or set {variable}."
            )
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
