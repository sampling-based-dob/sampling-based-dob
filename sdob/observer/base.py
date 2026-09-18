"""Base observer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseObserver(ABC):
    """Minimal lifecycle shared by both SDOB channels."""

    @abstractmethod
    def reset(self) -> None:
        """Clear the online estimate."""
