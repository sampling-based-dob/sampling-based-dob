"""Small workspace lifecycle inspired by task-independent policy repositories."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseWorkspace(ABC):
    """Own task construction, execution, and an output directory."""

    def __init__(self, config: Any, output_dir: str | Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir)

    @abstractmethod
    def run(self) -> Any:
        """Run one configured experiment."""
