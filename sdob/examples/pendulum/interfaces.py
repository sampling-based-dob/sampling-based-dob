from __future__ import annotations

import numpy as np


class ControllerBase:
    """Runtime interface shared by benchmark controllers."""

    name = "controller"

    def reset(self) -> None:
        raise NotImplementedError

    def act(self, state: np.ndarray, step: int) -> float:
        raise NotImplementedError

    def observe(self, state: np.ndarray, action: float, feedback: np.ndarray, step: int) -> None:
        pass

    def info(self) -> dict[str, float]:
        return {}
